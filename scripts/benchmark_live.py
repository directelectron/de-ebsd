"""Live indexing speed and accuracy: the flat dictionary search against the in-plane one.

A scan of Voronoi grains is rendered from a dynamical master pattern (one saved with
:meth:`~de_ebsd.sim.MasterPattern.save`) under a backscatter falloff with counting
noise, then indexed with :class:`~de_ebsd.index.live.LiveIndexer` as a live scan is
(the static background from the scan's frames).

``speed`` (the default) times ``index(frames, fit=True)`` and ``index_with_strain``
for each search, step and batch, and prints patterns a second, where the time goes
(prepare, search -- and the in-plane search's own stages --, refine, fit), the
build time, the search's template memory, and the device memory resident after the
build and at the peak (frames excluded)::

    uv run python scripts/benchmark_live.py speed ni_master.h5 --space-group 225 \\
        --steps 2 1.5 1 --batches 1024 4096 --searches flat inplane

``accuracy`` indexes the scan for several noise seeds and prints, per search and
step, the share of patterns more than 5 degrees from the truth and the median error,
after the search alone and after ``index(fit=True)``::

    uv run python scripts/benchmark_live.py accuracy ni_master.h5 --space-group 225 \\
        --counts 16 12 --steps 2 1.5 1 --seeds 0 1 2

``--json FILE`` appends one JSON line per row. With ``--simulate ni`` (or ``ti``,
``fe``) a missing master file is simulated first (minutes on a GPU) and saved where
it was asked for. Timings depend on what else runs on the GPU: on a shared device
they are indicative only.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import time

import numpy as np
import torch

from de_ebsd import EBSDDetector, MasterPattern
from de_ebsd.core.lambert import sample_texture
from de_ebsd.core.rotation import quaternion_to_matrix
from de_ebsd.index.io import phase_from_crystal
from de_ebsd.index.live import LiveIndexer, fitted_refine_steps_deg
from de_ebsd.index.live import indexer as live_indexer
from de_ebsd.index.live.inplane import proper_rotations

#: Crystals ``--simulate`` makes masters of: (lattice, space group, atoms, Debye-Waller).
CRYSTALS = {
    "ni": (dict(a=0.35236), 225, [("Ni", (0, 0, 0))], 0.0035),
    "fe": (dict(a=0.28665), 229, [("Fe", (0, 0, 0))], 0.0035),
    "ti": (dict(a=0.29508, c=0.46855), 194, [("Ti", (1 / 3, 2 / 3, 0.25))], 0.005),
}


def simulate(name: str, path: str) -> None:
    """A 20 kV dynamical master of *name* (:data:`CRYSTALS`), energy-integrated over
    a Monte Carlo depth and energy distribution, saved to *path*."""
    import de_ebsd
    from de_ebsd.diffraction.prism import BlochPrismBackend

    lattice, space_group, atoms, debye_waller = CRYSTALS[name]
    crystal = de_ebsd.Crystal.from_parameters(
        **lattice, space_group=space_group, atoms=atoms, debye_waller=debye_waller
    )
    energies = de_ebsd.DepthDistribution.from_simulation(
        crystal, 20.0, e_min_kev=10.0, e_bin_kev=1.0, tilt_deg=70.0, device="cuda", seed=0
    )
    simulator = de_ebsd.MasterPatternSimulator(
        crystal,
        de_ebsd.Microscope(energy=20e3, sample_tilt=70.0),
        npx=300,
        dmin=0.05,
        backend=BlochPrismBackend(cdtype=torch.complex64),
        monte_carlo=energies,
    )
    result = simulator.compute(device="cuda", energy_resolved=True).to("cpu")
    metadata = dict(result.metadata, energy_resolved=False)
    MasterPattern(result.data.sum(0, keepdim=True), np.array([20.0]), metadata).save(path)


def point_group_of(space_group: int):
    from de_ebsd.crystal import Crystal

    # any crystal of the space group gives its point group
    crystal = Crystal.from_parameters(
        a=0.4, c=0.5, space_group=space_group, atoms=[("Fe", (0, 0, 0))]
    )
    return phase_from_crystal(crystal).point_group


def voronoi_scan(side: int, grains: int, seed: int, device) -> torch.Tensor:
    """``(side * side, 3, 3)``: a scan of *grains* Voronoi grains, each point turned up
    to half a degree about a random axis from its grain's orientation."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 1, (grains, 2)) * side
    y, x = np.mgrid[:side, :side]
    labels = np.argmin(
        (y[..., None] - centers[:, 0]) ** 2 + (x[..., None] - centers[:, 1]) ** 2, -1
    )
    q = rng.normal(size=(grains, 4))
    q = (q / np.linalg.norm(q, axis=1, keepdims=True))[labels.ravel()]
    axes = rng.normal(size=(len(q), 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    half = np.radians(rng.uniform(0, 0.5, len(q)))[:, None] / 2
    turn = np.concatenate([np.cos(half), np.sin(half) * axes], 1)
    w1, v1, w2, v2 = turn[:, :1], turn[:, 1:], q[:, :1], q[:, 1:]
    q = np.concatenate(
        [w1 * w2 - (v1 * v2).sum(-1, keepdims=True), w1 * v2 + w2 * v1 + np.cross(v1, v2)], -1
    )
    return quaternion_to_matrix(torch.as_tensor(q, dtype=torch.float32, device=device))


def render(master, detector, g, counts: float, seed: int, device) -> torch.Tensor:
    """Frames ``(N, H, W)`` of orientations *g*: the pattern under a backscatter
    falloff, *counts* electrons a pixel at its peak, with Poisson noise."""
    rows, cols = detector.shape
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, rows, device=device),
        torch.linspace(-1, 1, cols, device=device),
        indexing="ij",
    )
    backscatter = torch.exp(-(x**2 + (y - 0.3) ** 2) / 0.8)
    generator = torch.Generator(device=device).manual_seed(seed)
    texture = master.data[-1].to(device, torch.float32)[None]
    directions = detector.direction_cosines(device=device)
    frames = torch.empty(len(g), rows, cols, device=device)
    for lo in range(0, len(g), 512):
        v = torch.einsum("mij,pj->mpi", g[lo : lo + 512], directions)
        pattern = sample_texture(texture, v).view(-1, rows, cols)
        low = pattern.flatten(1).amin(1).view(-1, 1, 1)
        high = pattern.flatten(1).amax(1).view(-1, 1, 1)
        pattern = (pattern - low) / (high - low)
        expected = counts * backscatter * (0.6 + 0.4 * pattern)
        frames[lo : lo + 512] = torch.poisson(expected, generator=generator)
    return frames


def errors_deg(g, truth, operators) -> np.ndarray:
    """Degrees from each *g* to its *truth*, the nearer of the symmetric equivalents."""
    m = g.double() @ truth.double().transpose(1, 2)
    trace = torch.einsum("nij,sij->ns", m, operators.to(g.device)).max(1).values
    return torch.rad2deg(torch.arccos(((trace - 1) / 2).clamp(-1, 1))).cpu().numpy()


class Stages:
    """CUDA events around an indexer's steps, summed per step after a run: the
    fused path's private steps (``index(fit=True)``), the public ones
    (``index_with_strain``) and the in-plane search's stages."""

    STEPS = {
        "_prepare_in_workspace": "prepare",
        "_search_in_workspace": "search",
        "_refine_in_workspace": "refine",
        "_add_fits_in_workspace": "fit",
        "_fit_in_workspace": "fit",
        "prepare": "prepare",
        "search": "search",
        "refine": "refine",
        "add_fits": "fit",
        "refine_deformation": "fit",
    }

    def __init__(self, indexer):
        self.indexer = indexer
        self.events = []
        for method, stage in self.STEPS.items():
            setattr(indexer, method, self._timed(getattr(indexer, method), stage))
        if indexer.inplane is not None:
            self.last = None
            indexer.inplane.marker = self._mark

    def _event(self):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _timed(self, method, stage):
        def timed(*args, **kwargs):
            start = self._event()
            out = method(*args, **kwargs)
            self.events.append((stage, start, self._event()))
            return out

        return timed

    def _mark(self, stage):
        event = self._event()
        if stage != "start":
            self.events.append(("  " + stage, self.last, event))
        self.last = event

    def remove(self):
        for method in self.STEPS:
            vars(self.indexer).pop(method, None)
        if self.indexer.inplane is not None:
            self.indexer.inplane.marker = None

    def microseconds(self, patterns: int) -> dict:
        torch.cuda.synchronize()
        out = {}
        for stage, start, end in self.events:
            out[stage] = out.get(stage, 0.0) + 1e3 * start.elapsed_time(end) / patterns
        self.events = []
        return out


def build(args, master, point_group, detector, search, step, device):
    options = {}
    if search == "inplane" and args.memory_mb:
        options["search_options"] = {"memory": int(args.memory_mb * 2**20)}
    dtype = {"auto": None, "float32": torch.float32, "float16": torch.float16}[args.dtype]
    return LiveIndexer(
        [master],
        [point_group],
        detector,
        resolution=step,
        search=search,
        search_blur_deg=0.5 * step if args.blur == "half" else None,
        refine_steps_deg=fitted_refine_steps_deg(device),
        device=device,
        dtype=dtype,
        candidates=args.candidates,
        **options,
    )


def template_bytes(indexer) -> int:
    if indexer.inplane is not None:
        return indexer.inplane.template_bytes()
    return indexer.dictionary.numel() * indexer.dictionary.element_size()


def rate(function, frames, repeats: int) -> float:
    """Patterns a second: the best of *repeats* timed runs."""
    best = math.inf
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        function(frames)
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - start)
    return len(frames) / best


def speed(args, master, point_group, detector, device, write):
    truth = voronoi_scan(args.scan, args.grains, 0, device)
    frames = render(master, detector, truth, args.counts[0], 0, device)
    print(
        f"{len(frames):,} patterns of {detector.shape[0]} x {detector.shape[1]}, "
        f"{args.counts[0]:g} counts/px; µs per pattern by stage, the in-plane search's "
        "own stages indented"
    )
    for step in args.steps:
        for batch in args.batches:
            for search in args.searches:
                live_indexer.FUSED_BATCH = batch
                torch.cuda.empty_cache()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                try:
                    indexer = build(args, master, point_group, detector, search, step, device)
                except torch.cuda.OutOfMemoryError:
                    print(f"{search:8s} {step:g}° batch {batch}: out of memory building")
                    continue
                torch.cuda.synchronize()
                built = time.perf_counter() - start
                resident = torch.cuda.memory_allocated() - base
                row = dict(
                    search=search,
                    step=step,
                    batch=batch,
                    build_s=built,
                    template_mb=template_bytes(indexer) / 2**20,
                    resident_mb=resident / 2**20,
                )
                if indexer.inplane is not None:
                    row["chunk"] = indexer.inplane.chunk
                    row["search_memory_mb"] = indexer.inplane.memory / 2**20
                try:
                    indexer.restart_static(frames[::8])
                    indexer.index(frames[: 2 * batch], fit=True)
                    # nothing named may hold the indexer past this row: the next one's
                    # in-plane memory budget is read from the device's free memory
                    fitted = functools.partial(indexer.index, fit=True)
                    row["fit_rate"] = rate(fitted, frames, args.repeats)
                    half = frames[: len(frames) // 2]
                    row["strain_rate"] = rate(indexer.index_with_strain, half, 1)
                    stages = Stages(indexer)
                    indexer.index(frames, fit=True)
                    row["fit_stages_us"] = stages.microseconds(len(frames))
                    indexer.index_with_strain(half)
                    row["strain_stages_us"] = stages.microseconds(len(half))
                    stages.remove()
                    del stages, half, fitted
                except torch.cuda.OutOfMemoryError:
                    print(f"{search:8s} {step:g}° batch {batch}: out of memory indexing")
                    del indexer
                    continue
                row["peak_mb"] = (torch.cuda.max_memory_allocated() - base) / 2**20
                entries = sum(indexer.sizes)
                print(
                    f"\n{search:8s} {step:g}° batch {batch:,}: {entries:,} entries, built in "
                    f"{built:.1f} s, templates {row['template_mb']:.0f} MB, resident "
                    f"{row['resident_mb']:.0f} MB, peak {row['peak_mb']:.0f} MB\n"
                    f"  index(fit=True)     {row['fit_rate']:9,.0f} patterns/s\n"
                    f"  index_with_strain   {row['strain_rate']:9,.0f} patterns/s"
                )
                for which in ("fit", "strain"):
                    parts = row[f"{which}_stages_us"]
                    text = ", ".join(
                        f"{k.strip()} {v:.1f}" for k, v in parts.items() if k[0] != " "
                    )
                    inner = ", ".join(
                        f"{k.strip()} {v:.1f}" for k, v in parts.items() if k[0] == " "
                    )
                    print(
                        f"  {which:6s} µs/pattern: {text}"
                        + (f" | search: {inner}" if inner else "")
                    )
                write(dict(mode="speed", **row))
                del indexer
    live_indexer.FUSED_BATCH = 1024


def spread(values, digits: int) -> str:
    """Mean ± sample standard deviation."""
    values = np.asarray(values)
    deviation = values.std(ddof=1) if len(values) > 1 else 0.0
    return f"{values.mean():.{digits}f} ± {deviation:.{digits}f}"


def accuracy(args, master, point_group, detector, device, write):
    operators = proper_rotations(point_group, device)
    for step in args.steps:
        for search in args.searches:
            torch.cuda.empty_cache()
            try:
                indexer = build(args, master, point_group, detector, search, step, device)
            except torch.cuda.OutOfMemoryError:
                print(f"{search:8s} {step:g}°: out of memory building")
                continue
            for counts in args.counts:
                rows = {k: [] for k in ("search_wrong", "search_median", "wrong", "median")}
                for seed in args.seeds:
                    truth = voronoi_scan(args.scan, args.grains, seed, device)
                    frames = render(master, detector, truth, counts, seed, device)
                    indexer.restart_static(frames[::8])
                    found = []
                    for lo in range(0, len(frames), 1024):
                        search_rows, refine_rows = indexer.prepare(frames[lo : lo + 1024])
                        found.append(indexer.search(search_rows, refine_rows)[1])
                    e = errors_deg(torch.cat(found), truth, operators)
                    rows["search_wrong"].append(100 * float(np.mean(e > 5)))
                    rows["search_median"].append(float(np.median(e)))
                    indexer.restart_static(frames[::8])
                    _, g, _, _ = indexer.index(frames, fit=True)
                    e = errors_deg(g, truth, operators)
                    rows["wrong"].append(100 * float(np.mean(e > 5)))
                    rows["median"].append(float(np.median(e)))
                    del frames
                print(
                    f"{search:8s} {step:g}° {counts:g} counts ({sum(indexer.sizes):,} entries): "
                    f"search wrong {spread(rows['search_wrong'], 3)}% median "
                    f"{spread(rows['search_median'], 2)}° | fit wrong "
                    f"{spread(rows['wrong'], 3)}% median {spread(rows['median'], 3)}°",
                    flush=True,
                )
                write(dict(mode="accuracy", search=search, step=step, counts=counts, **rows))
            del indexer


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("speed", "accuracy"))
    parser.add_argument("master", help="a MasterPattern .h5 (MasterPattern.save)")
    parser.add_argument("--space-group", type=int, required=True)
    parser.add_argument("--simulate", choices=sorted(CRYSTALS), help="simulate a missing master")
    parser.add_argument("--searches", nargs="+", default=["flat", "inplane"])
    parser.add_argument("--steps", nargs="+", type=float, default=[2.0, 1.5, 1.0])
    parser.add_argument("--batches", nargs="+", type=int, default=[1024, 4096])
    parser.add_argument("--counts", nargs="+", type=float, default=[16.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--scan", type=int, default=128, help="the scan's side, in points")
    parser.add_argument("--grains", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--blur",
        choices=("half", "default"),
        default="half",
        help="the search's blur: half the step (as validated), or the indexer's default",
    )
    parser.add_argument("--dtype", choices=("auto", "float32", "float16"), default="auto")
    parser.add_argument("--memory-mb", type=float, help="the in-plane search's working memory")
    parser.add_argument("--json", help="append one JSON line per row to this file")
    parser.add_argument(
        "--candidates", type=int, default=1, help="the candidates each pattern is refined from"
    )
    args = parser.parse_args()

    if not os.path.exists(args.master):
        if not args.simulate:
            parser.error(f"{args.master} does not exist (--simulate makes it)")
        simulate(args.simulate, args.master)
    device = torch.device("cuda")
    master = MasterPattern.load(args.master).to(device)
    point_group = point_group_of(args.space_group)
    detector = EBSDDetector(
        shape=(128, 128), pattern_center=(0.5, 0.45, 0.6), sample_tilt=70.0, tilt=0.0
    )
    print(torch.cuda.get_device_name(device), f"torch {torch.__version__}")

    def write(row):
        if args.json:
            with open(args.json, "a") as out:
                out.write(json.dumps(dict(master=args.master, **row)) + "\n")

    (speed if args.mode == "speed" else accuracy)(
        args, master, point_group, detector, device, write
    )


if __name__ == "__main__":
    main()
