"""Fit how a real detector sees the master (:mod:`de_ebsd.index.appearance`) on a scan.

The scan is indexed against the phases' default masters (the Monte Carlo's energy
weights, as Autopilot simulates them), its best-scored patterns spread over the
grains are chosen (a fifth of the grains held out), and the appearance -- energy
weights, point spread, contrast, residual background -- is fitted on the indexer's
own corrected patterns. Printed: the mean NCC before and after on the patterns
fitted on and on those held out, the held-out rough-search margin, and the sample
indexed again against the fitted master. Saved to ``--out``:

* ``appearance_<phase>.json``: the fitted parameters and the fit's numbers
  (:func:`de_ebsd.index.appearance.load_appearance` reads the parameters back);
* ``appearance_<phase>.png``: the fitted energy weights against the Monte Carlo's,
  and a few patterns -- real, simulated with the default master, with the fitted;
* the energy-resolved masters it simulated, for the next time.

From an Autopilot run folder (an EBSD Acquire with indexing on, its movie saved
as MRC or UP2 on this machine)::

    uv run python scripts/fit_appearance.py D:/data/2026-11-17/101500_ebsd_acquire

which reads ``dataset.json`` (the scan shape, the movie's path, the indexing
settings: pattern center, beam energy, tilts, the phases' CIF files, the step),
``dataset/camera_<first>.npy`` (which movie frame is which scan point) and the movie.
Anything the run folder lacks can be given on the command line, or the whole scan
without a run folder::

    uv run python scripts/fit_appearance.py --patterns scan.npy --scan 200 300 \\
        --cif Ni.cif --pc 0.50 0.45 0.60 --kv 20 --sample-tilt 70

The masters must be energy-resolved: they are simulated from the CIFs (a minute or
so each on a GPU, kept in ``--out``), or given with ``--master`` as files saved from
``MasterPatternSimulator.compute(energy_resolved=True)``.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

MRC_MODES = {0: "u1", 1: "i2", 2: "f4", 6: "u2"}
UP_TYPES = {".up1": "u1", ".up2": "<u2"}


# --- reading a run folder -------------------------------------------------------


def open_movie(path: Path) -> np.ndarray:
    """A saved movie ``(frames, H, W)`` as a read-only memory map: MRC, UP1, UP2 or
    a NumPy ``.npy``."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, mmap_mode="r")
    with open(path, "rb") as f:
        head = f.read(1024)
    if suffix == ".mrc":
        endian = ">" if head[212] == 0x11 else "<"
        nx, ny, nz, mode = struct.unpack(endian + "4i", head[:16])
        (extended,) = struct.unpack(endian + "i", head[92:96])
        dtype = np.dtype(endian + MRC_MODES[mode])
        return np.memmap(path, dtype, "r", offset=1024 + max(0, extended), shape=(nz, ny, nx))
    if suffix in UP_TYPES:
        _, width, height, offset = struct.unpack("<4I", head[:16])
        dtype = np.dtype(UP_TYPES[suffix])
        frames = (path.stat().st_size - offset) // (width * height * dtype.itemsize)
        return np.memmap(path, dtype, "r", offset=offset, shape=(frames, height, width))
    sys.exit(f"{path}: cannot read {suffix} movies; save the scan as MRC or UP2, or as .npy")


class ScanFrames:
    """A movie's frames in scan order: ``frames[points]`` reads the frames of scan
    *points* (row-major), whichever movie frames they were saved as."""

    def __init__(self, movie, frame_of_point):
        self.movie = movie
        self.frame_of_point = frame_of_point
        self.shape = (len(frame_of_point), *movie.shape[1:])

    def __len__(self):
        return len(self.frame_of_point)

    def __getitem__(self, points):
        frames = self.frame_of_point[points]
        order = np.argsort(frames, kind="stable")
        out = np.empty((len(frames), *self.movie.shape[1:]), dtype=self.movie.dtype)
        out[order] = self.movie[np.maximum(frames[order], 0)]
        out[frames < 0] = 0
        return out


def read_run(folder: Path):
    """What a run folder holds for re-indexing: ``(frames, shape, settings)``."""
    record = folder / "dataset.json"
    if not record.exists():
        sys.exit(f"{folder}: no dataset.json; give --patterns and --scan instead")
    dataset = json.loads(record.read_text())
    scan = dataset.get("scan", {})
    shape = tuple(int(v) for v in (scan.get("navigation_shape") or scan.get("field")))
    files = dataset.get("detectors", {}).get("camera", {}).get("files", [])
    if not files:
        note = dataset.get("detectors", {}).get("camera", {}).get("note", "")
        sys.exit(f"{folder}: no movie in the record ({note}); give --patterns instead")
    if len(files) > 1:
        print(f"note: the movie is {len(files)} files; reading the first", file=sys.stderr)
    entry = files[0]
    movie_path = Path(entry["path"])
    if not movie_path.is_absolute():
        movie_path = folder / movie_path
    if not movie_path.exists():
        sys.exit(f"{movie_path}: the movie is not on this machine (the server saved it)")
    movie = open_movie(movie_path)
    index = np.load(folder / entry["index"])
    frame_of_point = np.full(shape[0] * shape[1], -1, dtype=np.int64)
    # the first pass's frame of each point, the index a row per movie frame
    for row in range(len(index) - 1, -1, -1):
        point = int(index["point"][row])
        if 0 <= point < len(frame_of_point) and int(index["pass"][row]) == 0:
            frame_of_point[point] = row
    missing = int((frame_of_point < 0).sum())
    if missing:
        print(f"note: {missing} scan points have no frame; they are left out", file=sys.stderr)
    settings = {}
    for derived in dataset.get("derived", []):
        if derived.get("name") == "orientation_map":
            settings = derived.get("parameters") or {}
    return ScanFrames(movie, frame_of_point), shape, settings


# --- the masters ----------------------------------------------------------------


def resolved_master(cif: str, kv: float, tilt: float, npx: int, cache: Path):
    """A phase's energy-resolved master, as Autopilot simulates it (half the beam
    energy up, 1 keV bins, the Monte Carlo's weights kept rather than summed in),
    kept in *cache*."""
    import de_ebsd

    name = f"{Path(cif).stem}_{kv:g}kV_{tilt:g}deg_{npx}px_resolved.h5"
    path = cache / name
    if path.exists():
        return de_ebsd.MasterPattern.load(str(path))
    print(f"simulating {Path(cif).name}'s energy-resolved master at {kv:g} kV, {npx} px ...")
    start = time.perf_counter()
    crystal = de_ebsd.Crystal.from_cif(cif)
    floor = kv - round(kv * 0.5)
    depth = de_ebsd.DepthDistribution.from_simulation(
        crystal, kv, e_min_kev=floor, e_bin_kev=1.0, tilt_deg=tilt, device="cuda", seed=0
    )
    simulator = de_ebsd.MasterPatternSimulator(
        crystal,
        de_ebsd.Microscope(energy=kv * 1e3, sample_tilt=tilt),
        npx=npx,
        dmin=0.05,
        backend=de_ebsd.BlochScatteringMatrixBackend(cdtype=torch.complex64),
        monte_carlo=depth,
    )
    master = simulator.compute(device="cuda", energy_resolved=True).to("cpu")
    master.metadata.pop("crystal", None)  # an object; save() would write its repr
    cache.mkdir(parents=True, exist_ok=True)
    master.save(str(path))
    print(f"  {time.perf_counter() - start:.0f} s, kept as {path}")
    return master


# --- the figure ---------------------------------------------------------------------


def figure(path, fit, resolved, detector, rows, chosen_g, title):
    """The fitted energy weights against the Monte Carlo's, and four held-out (or
    fitted) patterns: real, the default master's, the fitted master's."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; no figure", file=sys.stderr)
        return
    from de_ebsd.index.appearance import _without_residual
    from de_ebsd.index.preprocess import normalize_rows

    a = fit.appearance
    picks = list(fit.held_out[:4]) or list(fit.train[:4])
    shape = tuple(detector.shape)
    real = _without_residual(rows[picks], a.residual_background(), shape)
    before_master = fit.default.master(resolved)
    after_master = a.master(resolved)

    def simulate(master, g, det):
        from orix.quaternion import Rotation

        projected = master.project(det, Rotation.from_matrix(np.asarray(g)), rescale=False)
        return normalize_rows(projected.flatten(1).float().cpu())

    before = simulate(before_master, chosen_g[picks], detector)
    after = simulate(after_master, fit.orientations[picks], a.detector(detector))
    fig = plt.figure(figsize=(12, 3 + 2.6 * len(picks)), constrained_layout=True)
    grid = fig.add_gridspec(1 + len(picks), 3)
    ax = fig.add_subplot(grid[0, :])
    ax.plot(a.energies_kev, a.prior_weights, "o-", label="Monte Carlo")
    ax.plot(a.energies_kev, a.energy_weights, "s-", label="fitted")
    ax.set_xlabel("energy (keV)")
    ax.set_ylabel("weight")
    s = fit.summary()
    ax.set_title(
        f"{title}: held-out NCC {s['held_out_before']:.3f} -> {s['held_out_after']:.3f}; "
        f"PSF {a.psf_sigma_deg:.2f} deg, gamma {a.gamma:.2f}, high pass "
        f"{a.high_pass_fraction:.2f}"
    )
    ax.legend()
    real = real.cpu()
    for i in range(len(picks)):
        for j, (label, image) in enumerate(
            (("real", real[i]), ("default master", before[i]), ("fitted master", after[i]))
        ):
            ax = fig.add_subplot(grid[1 + i, j])
            ax.imshow(image.view(shape).numpy(), cmap="gray")
            score = "" if j == 0 else f", NCC {float((image * real[i]).sum()):.3f}"
            ax.set_title(f"{label}{score}", fontsize=9)
            ax.set_axis_off()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --- the fit ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", nargs="?", type=Path, help="an Autopilot run folder")
    parser.add_argument(
        "--patterns", type=Path, help="the scan's patterns instead (.npy/.mrc/.up2), in scan order"
    )
    parser.add_argument("--scan", type=int, nargs=2, metavar=("ROWS", "COLS"))
    parser.add_argument(
        "--master", nargs="+", help="each phase's energy-resolved master .h5, in phase order"
    )
    parser.add_argument("--cif", nargs="+", help="each phase's CIF, to simulate its master")
    parser.add_argument(
        "--pc", type=float, nargs=3, metavar=("X", "Y", "Z"), help="pattern center (Bruker)"
    )
    parser.add_argument("--kv", type=float, help="beam energy, kV")
    parser.add_argument("--sample-tilt", type=float)
    parser.add_argument("--detector-tilt", type=float)
    parser.add_argument("--step", type=float, help="dictionary step, degrees (the run's, else 3)")
    parser.add_argument("--npx", type=int, default=300, help="masters' Lambert half-width")
    parser.add_argument("--refine-side", type=int, default=128)
    parser.add_argument(
        "--sample", type=int, default=20000, help="scan points indexed to choose patterns from"
    )
    parser.add_argument("--count", type=int, default=300, help="patterns fitted (with held out)")
    parser.add_argument("--per-grain", type=int, default=8, help="most patterns from one grain")
    parser.add_argument("--restarts", type=int, default=1, help="fits from random starts")
    parser.add_argument("--halo", action="store_true", help="fit a wide second PSF Gaussian")
    parser.add_argument(
        "--pattern-center", action="store_true", help="refine the pattern center too"
    )
    parser.add_argument("--single-crystal", action="store_true", help="blur the static background")
    parser.add_argument("--out", type=Path, default=Path("appearance_out"))
    args = parser.parse_args()

    from de_ebsd import EBSDDetector, MasterPattern
    from de_ebsd.index.appearance import fit_appearance, select_patterns
    from de_ebsd.index.io import phase_from_cif
    from de_ebsd.index.live import LiveIndexer, fitted_refine_steps_deg, fused_fit_available

    if not torch.cuda.is_available():
        sys.exit("this needs a CUDA GPU")
    device = torch.device("cuda")
    if args.run is not None:
        frames, shape, settings = read_run(args.run)
    elif args.patterns is not None and args.scan:
        movie = open_movie(args.patterns)
        frames, shape, settings = ScanFrames(movie, np.arange(len(movie))), tuple(args.scan), {}
    else:
        parser.error("give a run folder, or --patterns with --scan")
    if len(frames) != shape[0] * shape[1]:
        sys.exit(f"{len(frames)} patterns for a {shape[0]} x {shape[1]} scan")
    geometry = settings.get("geometry", {})

    def setting(given, key, default=None):
        value = given if given is not None else geometry.get(key, default)
        if value is None:
            sys.exit(f"the run folder does not say the {key}; give it on the command line")
        return value

    pc = tuple(args.pc) if args.pc else tuple(setting(None, k) for k in ("pc_x", "pc_y", "pc_z"))
    kv = float(setting(args.kv, "beam_energy"))
    sample_tilt = float(setting(args.sample_tilt, "sample_tilt", 70.0))
    detector_tilt = float(setting(args.detector_tilt, "detector_tilt", 0.0))
    step = float(args.step or settings.get("step_deg") or 3.0)
    args.out.mkdir(parents=True, exist_ok=True)
    cifs = args.cif or [p.get("cif") for p in settings.get("phases", [])]
    if args.master:
        resolved = [MasterPattern.load(p) for p in args.master]
        for path, master in zip(args.master, resolved, strict=True):
            if len(master.energies_kev) < 2 or master.energy_weights is None:
                print(f"note: {path} is not energy-resolved; its energy weights are not fitted")
    else:
        if not cifs or not all(cifs):
            sys.exit("no master patterns: give --master, or --cif to simulate them")
        resolved = [resolved_master(c, kv, sample_tilt, args.npx, args.out) for c in cifs]
    if not cifs or not all(cifs) or len(cifs) != len(resolved):
        sys.exit("give each phase's CIF (--cif), for its point group")
    point_groups = [phase_from_cif(c).point_group for c in cifs]
    names = [Path(c).stem for c in cifs]
    frame_shape = tuple(frames.shape[1:])
    detector = EBSDDetector(
        shape=frame_shape,
        pattern_center=pc,
        sample_tilt=sample_tilt,
        tilt=detector_tilt,
        azimuthal=float(geometry.get("azimuthal", 0.0)),
        twist=float(geometry.get("twist", 0.0)),
    )
    fit_orientations = fused_fit_available(device)

    def indexer_for(masters, residual=None):
        return LiveIndexer(
            [m.to(device) for m in masters],
            point_groups,
            detector,
            resolution=step,
            search_shape=32,
            refine_shape=args.refine_side,
            refine_steps_deg=fitted_refine_steps_deg(device)
            if fit_orientations
            else (2.0, 1.0, 0.5, 0.25, 0.12),
            search_blur_deg=step / 2.0,
            device=device,
            residual_background=residual,
        )

    present = np.flatnonzero(getattr(frames, "frame_of_point", np.arange(len(frames))) >= 0)
    points = present[np.linspace(0, len(present) - 1, min(args.sample, len(present))).astype(int)]
    every = max(1, len(present) // 2048)
    background = torch.as_tensor(np.asarray(frames[present[::every]]).astype(np.float32))
    blur = 0.1 if args.single_crystal else None

    def index(indexer):
        indexer.restart_static(background.to(device), blur=blur)
        out = []
        for lo in range(0, len(points), 2048):
            batch = np.asarray(frames[points[lo : lo + 2048]]).astype(np.float32)
            found = indexer.index(torch.as_tensor(batch, device=device), fit=fit_orientations)
            out.append([c.cpu().numpy() for c in found])
        return [np.concatenate(c) for c in zip(*out, strict=True)]

    print(
        f"{shape[0]} x {shape[1]} scan, {frame_shape[1]} x {frame_shape[0]} px patterns, "
        f"{len(resolved)} phase(s), pattern center {pc}, {kv:g} kV, tilt {sample_tilt:g}; "
        f"indexing {len(points)} points against the default masters ({step:g} deg) ..."
    )
    defaults = [m.collapse() for m in resolved]
    indexer = indexer_for(defaults)
    start = time.perf_counter()
    phase, g, score, _ = index(indexer)
    print(f"  {time.perf_counter() - start:.1f} s, mean score {score.mean():.4f}")

    fitted, residual = list(defaults), None
    for number, (name, master, group) in enumerate(zip(names, resolved, point_groups, strict=True)):
        mine = np.flatnonzero(phase == number)
        if len(mine) < 50:
            print(f"{name}: {len(mine)} patterns indexed; too few to fit")
            continue
        train, held = select_patterns(
            g[mine], score[mine], group, count=args.count, per_grain=args.per_grain
        )
        chosen = mine[np.concatenate([train, held])]
        rows = []
        for lo in range(0, len(chosen), 512):
            batch = np.asarray(frames[points[chosen[lo : lo + 512]]]).astype(np.float32)
            rows.append(indexer.prepare(torch.as_tensor(batch, device=device))[1])
        rows = torch.cat(rows).view(len(chosen), *indexer.refine_shape)
        print(f"\n{name}: fitting on {len(train)} patterns, {len(held)} held out ...")
        fit = fit_appearance(
            master.to(device),
            rows,
            g[chosen],
            detector,
            train=np.arange(len(train)),
            held_out=np.arange(len(train), len(chosen)),
            point_group=group,
            margin_resolution_deg=step,
            halo=args.halo,
            fit_pattern_center=args.pattern_center,
            restarts=args.restarts,
            progress=None,
        )
        print(fit.report())
        fit.save(args.out / f"appearance_{name}.json")
        figure(
            args.out / f"appearance_{name}.png",
            fit,
            master.to(device),
            replace(detector, shape=tuple(indexer.refine_shape)),
            rows.flatten(1),
            g[chosen],
            name,
        )
        print(f"saved {args.out / f'appearance_{name}.json'} and .png")
        fitted[number] = fit.appearance.master(master.to(device))
        if residual is None:
            residual = fit.appearance.residual_background()

    print("\nindexing the same points against the fitted masters ...")
    start = time.perf_counter()
    phase_after, g_after, score_after, _ = index(indexer_for(fitted, residual))
    print(
        f"  {time.perf_counter() - start:.1f} s, mean score {score.mean():.4f} -> "
        f"{score_after.mean():.4f}"
    )
    turn = g_after @ np.transpose(g, (0, 2, 1))
    changed = []
    for number, group in enumerate(point_groups):
        same = (phase == number) & (phase_after == number)
        operators = group.to_matrix()
        traces = np.einsum("kij,nij->nk", operators, turn[same]).max(-1)
        angle = np.degrees(np.arccos(np.clip((traces - 1) / 2, -1, 1)))
        changed.append(int((angle > 5).sum()))
    moved = sum(changed) + int((phase != phase_after).sum())
    print(f"  {moved} of {len(points)} points moved more than 5 degrees or changed phase")


if __name__ == "__main__":
    main()
