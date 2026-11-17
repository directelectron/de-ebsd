"""Speckle on a real scan: index its saved patterns with one candidate, with several,
and with several plus the post-scan neighbor retry, and compare the maps.

Speckle is the isolated points whose orientation agrees with none of their four
neighbors (more than 5 degrees from each, symmetry reduced): almost always a
pattern the search sent into the wrong basin. For each setting this prints the
speckle fraction, the patterns indexed a second, and how many points changed
from the one-candidate map.

From an Autopilot run folder (an EBSD Acquire with indexing on, its movie saved
as MRC or UP2 on this machine)::

    uv run python scripts/compare_speckle.py D:/data/2026-11-16/101500_ebsd_acquire

which reads, from the run folder:

* ``dataset.json``: the scan shape, the movie's path, and the indexing settings
  the run used (pattern center, beam energy, tilts, phases' CIF files, step);
* ``dataset/camera_<first>.npy``: which movie frame is which scan point;
* the movie itself (``..._movie.mrc`` / ``.up2``).

Master patterns are not in a run folder: give each phase's with ``--master``
(:meth:`de_ebsd.sim.MasterPattern.save`), or the script simulates them from the
phases' CIF files as Autopilot does (a minute or so each on a GPU) and keeps
them in ``--out`` for the next time. Anything the run folder lacks can be given
on the command line instead, or the whole scan without a run folder::

    uv run python scripts/compare_speckle.py --patterns scan.npy --scan 200 300 \\
        --cif Ni.cif --pc 0.50 0.45 0.60 --kv 20 --sample-tilt 70

The maps are saved to ``--out`` (``speckle_maps.npz``: each setting's ``g``,
``phase``, ``score``, ``entry`` and the ``speckled`` points), for a look at where
the speckle was.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
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
        print(f"note: {missing} scan points have no frame; they index as blank", file=sys.stderr)
    settings = {}
    for derived in dataset.get("derived", []):
        if derived.get("name") == "orientation_map":
            settings = derived.get("parameters") or {}
    return ScanFrames(movie, frame_of_point), shape, settings


# --- the masters ----------------------------------------------------------------


def simulate_master(cif: str, kv: float, tilt: float, npx: int, cache: Path):
    """A phase's master pattern as Autopilot simulates it, kept in *cache*."""
    import de_ebsd
    from de_ebsd.diffraction.prism import BlochPrismBackend

    name = f"{Path(cif).stem}_{kv:g}kV_{tilt:g}deg_{npx}px.h5"
    path = cache / name
    if path.exists():
        return de_ebsd.MasterPattern.load(str(path))
    print(f"simulating {Path(cif).name}'s master at {kv:g} kV, {npx} px ...", flush=True)
    crystal = de_ebsd.Crystal.from_cif(cif)
    floor = kv - round(kv * 0.5)
    energies = de_ebsd.DepthDistribution.from_simulation(
        crystal, kv, e_min_kev=floor, e_bin_kev=1.0, tilt_deg=tilt, device="cuda", seed=0
    )
    simulator = de_ebsd.MasterPatternSimulator(
        crystal,
        de_ebsd.Microscope(energy=kv * 1e3, sample_tilt=tilt),
        npx=npx,
        dmin=0.05,
        backend=BlochPrismBackend(cdtype=torch.complex64),
        monte_carlo=energies,
    )
    resolved = simulator.compute(device="cuda", energy_resolved=True).to("cpu")
    master = de_ebsd.MasterPattern(resolved.data.sum(0, keepdim=True), np.array([kv]), {})
    cache.mkdir(parents=True, exist_ok=True)
    master.save(str(path))
    return master


# --- indexing -------------------------------------------------------------------


def index_scan(indexer, frames, chunk, fit):
    """The whole scan indexed, a chunk of frames at a time:
    ``(phase, g, score, entry, patterns a second)``."""
    columns, seconds = [], 0.0
    for lo in range(0, len(frames), chunk):
        points = np.arange(lo, min(len(frames), lo + chunk))
        batch = torch.as_tensor(np.asarray(frames[points]).astype(np.float32)).to(indexer.device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        found = indexer.index(batch, fit=fit)
        torch.cuda.synchronize()
        seconds += time.perf_counter() - start
        columns.append([c.cpu().numpy() for c in found])
    phase, g, score, entry = (np.concatenate(c) for c in zip(*columns, strict=True))
    return phase, g, score, entry, len(frames) / seconds


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
        "--master", nargs="+", help="each phase's master pattern .h5, in phase order"
    )
    parser.add_argument("--cif", nargs="+", help="each phase's CIF, to simulate its master")
    parser.add_argument(
        "--pc", type=float, nargs=3, metavar=("X", "Y", "Z"), help="pattern center (Bruker)"
    )
    parser.add_argument("--kv", type=float, help="beam energy, kV")
    parser.add_argument("--sample-tilt", type=float)
    parser.add_argument("--detector-tilt", type=float)
    parser.add_argument("--step", type=float, help="dictionary step, degrees (the run's, else 3)")
    parser.add_argument(
        "--npx", type=int, default=300, help="simulated masters' Lambert half-width"
    )
    parser.add_argument("--refine-side", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=4, help="the several-candidate setting's")
    parser.add_argument("--threshold", type=float, default=5.0, help="speckle threshold, degrees")
    parser.add_argument("--single-crystal", action="store_true", help="blur the static background")
    parser.add_argument("--no-fit", action="store_true", help="index without the orientation fit")
    parser.add_argument("--chunk", type=int, default=4096, help="frames read at a time")
    parser.add_argument("--out", type=Path, default=Path("speckle_out"))
    args = parser.parse_args()

    from de_ebsd import EBSDDetector, MasterPattern
    from de_ebsd.index.live import LiveIndexer, fitted_refine_steps_deg, retry_speckle
    from de_ebsd.index.neighbors import _matrices, _operators, _pair_angles, speckled

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
    if args.master:
        masters = [MasterPattern.load(p) for p in args.master]
        cifs = args.cif or [p.get("cif") for p in settings.get("phases", [])]
    else:
        cifs = args.cif or [p.get("cif") for p in settings.get("phases", [])]
        if not cifs or not all(cifs):
            sys.exit("no master patterns: give --master, or --cif to simulate them")
        masters = [simulate_master(c, kv, sample_tilt, args.npx, args.out) for c in cifs]
    if not cifs or not all(cifs) or len(cifs) != len(masters):
        sys.exit("give each phase's CIF (--cif), for its point group")
    from de_ebsd.index.io import phase_from_cif

    point_groups = [phase_from_cif(c).point_group for c in cifs]
    frame_shape = tuple(frames.shape[1:])
    detector = EBSDDetector(
        shape=frame_shape,
        pattern_center=pc,
        sample_tilt=sample_tilt,
        tilt=detector_tilt,
        azimuthal=float(geometry.get("azimuthal", 0.0)),
        twist=float(geometry.get("twist", 0.0)),
    )
    fit = not args.no_fit
    indexer = LiveIndexer(
        [m.to(device) for m in masters],
        point_groups,
        detector,
        resolution=step,
        search_shape=32,
        refine_shape=args.refine_side,
        refine_steps_deg=fitted_refine_steps_deg(device) if fit else (2.0, 1.0, 0.5, 0.25, 0.12),
        device=device,
    )
    print(
        f"{shape[0]} x {shape[1]} scan, {frame_shape[1]} x {frame_shape[0]} px patterns, "
        f"{len(masters)} phase(s), pattern center {pc}, {kv:g} kV, tilt {sample_tilt:g}, "
        f"{step:g} deg dictionary ({len(indexer.rotations)} entries), fit={fit}"
    )
    # the static background from frames all over the scan, as a live scan's ends up
    every = max(1, len(frames) // 2048)
    background = torch.as_tensor(
        np.asarray(frames[np.arange(0, len(frames), every)]).astype(np.float32)
    )
    blur = 0.1 if args.single_crystal else None

    def restart():
        indexer.restart_static(background.to(device), blur=blur)

    groups = indexer.point_groups

    def map_of(g):
        return _matrices(g.reshape(*shape, 3, 3))

    def report(name, phase, g, rate, reference=None, extra=""):
        mask = speckled(map_of(g), groups, phase.reshape(shape), threshold_deg=args.threshold)
        changed = ""
        if reference is not None:
            ref_phase, ref_g = reference
            angle = np.full(len(phase), np.inf)
            same = phase == ref_phase
            for p, group in enumerate(groups):
                where = same & (phase == p)
                angle[where] = _pair_angles(
                    g[where].astype(np.float64), ref_g[where].astype(np.float64), _operators(group)
                )
            changed = f"{int((angle > args.threshold).sum()):>9}"
        print(f"{name:<36}{100 * mask.mean():>9.2f}%{rate:>12.0f}{changed:>10}  {extra}")
        return mask

    print(f"{'setting':<36}{'speckle':>10}{'patterns/s':>12}{'changed':>10}")
    saved = {}
    restart()
    index_scan(indexer, frames, min(args.chunk, 2048), fit)  # warm the GPU
    restart()
    one = index_scan(indexer, frames, args.chunk, fit)
    saved["k1"] = (*one[:4], report("1 candidate", one[0], one[1], one[4]))
    reference = (one[0], one[1])

    indexer.set_candidates(args.candidates)
    restart()
    indexer.rechecked = 0
    several = index_scan(indexer, frames, args.chunk, fit)
    share = 100 * indexer.rechecked / len(frames)
    saved["candidates"] = (
        *several[:4],
        report(
            f"{args.candidates} candidates",
            several[0],
            several[1],
            several[4],
            reference,
            f"({share:.1f}% rechecked)",
        ),
    )
    start = time.perf_counter()
    retried = retry_speckle(
        indexer,
        frames,
        several[0],
        several[1],
        several[2],
        several[3],
        shape=shape,
        threshold_deg=args.threshold,
        fit=fit,
    )
    seconds = time.perf_counter() - start
    saved["retried"] = (
        retried.phase,
        retried.g,
        retried.score,
        retried.entry,
        report(
            f"{args.candidates} candidates + post-scan retry",
            retried.phase,
            retried.g,
            len(frames) / seconds,
            reference,
            f"({retried.tried} retried, {int(retried.changed.sum())} replaced)",
        ),
    )
    print("patterns/s of the retry is the scan's points over the retry's own time")
    out = {}
    for name, (phase, g, score, entry, mask) in saved.items():
        out.update(
            {
                f"{name}_phase": phase.reshape(shape),
                f"{name}_g": g.reshape(*shape, 3, 3),
                f"{name}_score": score.reshape(shape),
                f"{name}_entry": entry.reshape(shape),
                f"{name}_speckled": mask,
            }
        )
    np.savez_compressed(args.out / "speckle_maps.npz", **out)
    print(f"maps saved to {os.path.abspath(args.out / 'speckle_maps.npz')}")


if __name__ == "__main__":
    main()
