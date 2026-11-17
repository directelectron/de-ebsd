"""Validate the appearance fit (:mod:`de_ebsd.index.appearance`) on synthetic scans.

A scan is rendered from an energy-resolved dynamical master through a KNOWN
appearance that differs from the Monte Carlo's -- other energy weights, a point
spread, a gamma and a smooth background -- on a smooth illumination, with Poisson
noise. It is indexed with :class:`~de_ebsd.index.live.LiveIndexer` against the
default master (the Monte Carlo's weights); the best-scored patterns, spread over
the grains, are fitted; and the scan is indexed again against the fitted master.

Printed, as Markdown tables:

* the true parameters against the fitted ones, and the random restarts';
* mean NCC before and after, on the patterns fitted on and on those held out, and
  the held-out rough-search margin;
* indexing: patterns more than 5 degrees off and the median error, against the
  default, the fitted and the true master;
* the same for a scan rendered with no mismatch, where the fit should stay near the
  default.

Run on a CUDA GPU (a few minutes, most of it the master's simulation, which is
kept in ``--out``)::

    uv run python scripts/validate_appearance.py --out D:/tmp/appearance
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import de_ebsd as de
from de_ebsd.core.lambert import sample_texture
from de_ebsd.core.rotation import quaternion_to_matrix, rotation_vector_to_matrix
from de_ebsd.index.appearance import DetectorAppearance, fit_appearance, select_patterns
from de_ebsd.index.io import phase_from_crystal
from de_ebsd.index.live import LiveIndexer
from de_ebsd.index.live.background import legendre_terms


def nickel() -> de.Crystal:
    return de.Crystal.from_parameters(
        a=0.35236, space_group=225, atoms=[("Ni", (0.0, 0.0, 0.0))], debye_waller=0.0035
    )


def resolved_master(out: Path, kv: float, npx: int, device) -> de.MasterPattern:
    """Nickel's energy-resolved master, half the beam energy up in 1 keV bins."""
    path = out / f"Ni_{kv:g}kV_{npx}px_resolved.h5"
    if path.exists():
        return de.MasterPattern.load(str(path))
    print(f"simulating nickel's energy-resolved master at {kv:g} kV, npx {npx} ...", flush=True)
    start = time.perf_counter()
    crystal = nickel()
    depth = de.DepthDistribution.from_simulation(
        crystal,
        kv,
        e_min_kev=kv - round(kv * 0.5),
        e_bin_kev=1.0,
        tilt_deg=70.0,
        device=device,
        seed=0,
    )
    simulator = de.MasterPatternSimulator(
        crystal,
        de.Microscope(energy=kv * 1e3, sample_tilt=70.0),
        npx=npx,
        dmin=0.06,
        monte_carlo=depth,
    )
    master = simulator.compute(device=device, energy_resolved=True).to("cpu")
    master.metadata.pop("crystal", None)
    out.mkdir(parents=True, exist_ok=True)
    master.save(str(path))
    print(f"  {time.perf_counter() - start:.0f} s", flush=True)
    return master


def scan_orientations(grains: int, per_grain: int, spread_deg: float, seed: int) -> np.ndarray:
    """``(grains * per_grain, 3, 3)``: random grains, each a small cloud."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(grains, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    centers = quaternion_to_matrix(torch.as_tensor(q)).float()
    jitter = torch.as_tensor(rng.normal(size=(grains, per_grain, 3)), dtype=torch.float32)
    turns = rotation_vector_to_matrix(jitter * np.radians(spread_deg) / np.sqrt(3))
    return (centers[:, None] @ turns).reshape(-1, 3, 3).numpy()


def render(resolved, appearance, g, detector, counts, seed, device, chunk=256):
    """Raw frames ``(N, H, W)`` on *device*: the scan through *appearance*, on a smooth
    illumination, Poisson at *counts* per pixel."""
    texture = appearance.master(resolved.to(device)).data[0][None]
    directions = detector.direction_cosines(device=device)
    rows, cols = detector.shape
    background = 0.0
    if appearance.background.size:
        terms = legendre_terms(detector.shape, appearance.background_order, device)
        background = torch.as_tensor(appearance.background, device=device).float() @ terms
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, rows, device=device),
        torch.linspace(-1, 1, cols, device=device),
        indexing="ij",
    )
    illumination = (1.0 + 0.25 * y - 0.15 * x - 0.2 * (x * x + y * y)).flatten()
    generator = torch.Generator(device=device).manual_seed(seed)
    gt = torch.as_tensor(g, device=device, dtype=torch.float32)
    frames = torch.empty(len(g), rows, cols, device=device)
    for lo in range(0, len(g), chunk):
        v = torch.einsum("nij,pj->npi", gt[lo : lo + chunk], directions)
        values = (sample_texture(texture, v) + background).clamp_min(0) * illumination
        values = values / values.mean(-1, keepdim=True) * counts
        frames[lo : lo + chunk] = torch.poisson(values, generator=generator).view(-1, rows, cols)
    return frames


def errors_deg(found, truth, operators) -> np.ndarray:
    turn = np.asarray(found, np.float64) @ np.transpose(np.asarray(truth, np.float64), (0, 2, 1))
    traces = np.einsum("kij,nij->nk", operators, turn).max(-1)
    return np.degrees(np.arccos(np.clip((traces - 1) / 2, -1, 1)))


def index(master, residual, point_group, detector, frames, args):
    indexer = LiveIndexer(
        [master],
        [point_group],
        detector,
        resolution=args.step,
        search_shape=32,
        refine_shape=args.refine,
        search_blur_deg=args.step / 2,
        device=frames.device,
        residual_background=residual,
    )
    indexer.restart_static(frames[:: max(1, len(frames) // 1024)])
    phase, g, score, _ = indexer.index(frames)
    return indexer, g.cpu().numpy(), score.cpu().numpy()


def table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def run_case(name, truth, resolved, point_group, detector, args, device):
    operators = point_group.to_matrix()
    g_true = scan_orientations(args.grains, args.per_grain, 0.3, seed=args.seed)
    print(f"\n## {name}\n", flush=True)
    results = {}
    for counts in args.counts:
        frames = render(resolved, truth, g_true, detector, counts, args.seed, device)
        default = DetectorAppearance.default(resolved)
        start = time.perf_counter()
        indexer, g_default, score = index(
            default.master(resolved.to(device)), None, point_group, detector, frames, args
        )
        index_seconds = time.perf_counter() - start
        err_default = errors_deg(g_default, g_true, operators)

        train, held = select_patterns(g_default, score, point_group, count=args.patterns)
        chosen = np.concatenate([train, held])
        _, refine_rows = indexer.prepare(frames[torch.as_tensor(chosen, device=device)])
        refine_rows = refine_rows.view(len(chosen), *indexer.refine_shape)
        fit = fit_appearance(
            resolved.to(device),
            refine_rows,
            g_default[chosen],
            detector,
            train=np.arange(len(train)),
            held_out=np.arange(len(train), len(chosen)),
            point_group=point_group,
            margin_resolution_deg=args.step,
            restarts=args.restarts,
        )
        _, g_fitted, _ = index(
            fit.appearance.master(resolved.to(device)),
            fit.appearance.residual_background(),
            point_group,
            detector,
            frames,
            args,
        )
        err_fitted = errors_deg(g_fitted, g_true, operators)
        oracle = replace(truth, background=np.zeros(0), background_order=0)
        _, g_oracle, _ = index(
            oracle.master(resolved.to(device)), None, point_group, detector, frames, args
        )
        err_oracle = errors_deg(g_oracle, g_true, operators)
        results[counts] = (fit, err_default, err_fitted, err_oracle, index_seconds)
        print(f"{counts} counts/px: fit {fit.seconds:.1f} s, indexing {index_seconds:.1f} s")
        print(fit.report())

    fit = results[args.counts[0]][0]
    a, t = fit.appearance, truth
    rows = [
        ["energy curve P1", f"{_curve(t, 0):+.3f}", f"{_curve(a, 0):+.3f}"],
        ["energy curve P2", f"{_curve(t, 1):+.3f}", f"{_curve(a, 1):+.3f}"],
        ["PSF sigma (deg)", f"{t.psf_sigma_deg:.3f}", f"{a.psf_sigma_deg:.3f}"],
        ["gamma", f"{t.gamma:.3f}", f"{a.gamma:.3f}"],
    ]
    print(f"\nTrue against fitted, {args.counts[0]} counts/px:\n")
    print(table(["parameter", "true", "fitted"], rows))
    print("\nEnergy weights:\n")
    print(
        table(
            ["keV", "Monte Carlo", "true", "fitted"],
            [
                [f"{e:g}", f"{p:.3f}", f"{w:.3f}", f"{f:.3f}"]
                for e, p, w, f in zip(
                    a.energies_kev, a.prior_weights, t.energy_weights, a.energy_weights, strict=True
                )
            ],
        )
    )
    print("\nRandom restarts (the first starts at the default):\n")
    print(
        table(
            ["start", "train NCC", "held-out NCC", "PSF sigma", "gamma", "curve"],
            [
                [
                    i,
                    f"{r['train_ncc']:.4f}",
                    f"{r['held_out_ncc']:.4f}",
                    f"{r['appearance']['psf_sigma_deg']:.3f}",
                    f"{r['appearance']['gamma']:.3f}",
                    ", ".join(f"{c:+.2f}" for c in r["appearance"]["energy_curve"]),
                ]
                for i, r in enumerate(fit.restarts)
            ],
        )
    )
    print("\nNCC (mean) and held-out search margin (median):\n")
    rows = []
    for counts, (fit, *_rest) in results.items():
        s = fit.summary()
        rows.append(
            [
                counts,
                f"{s['train_before']:.4f} -> {s['train_after']:.4f}",
                f"{s['held_out_before']:.4f} -> {s['held_out_after']:.4f}",
                f"{s['held_out_model']:.4f}",
                f"{s.get('margin_before', float('nan')):.4f} -> "
                f"{s.get('margin_after', float('nan')):.4f}",
            ]
        )
    print(
        table(
            ["counts/px", "train NCC", "held-out NCC", "held-out, model", "held-out margin"],
            rows,
        )
    )
    print(
        f"\nIndexing {len(g_true)} patterns ({args.step:g} deg dictionary, blur "
        f"{args.step / 2:g} deg):\n"
    )
    rows = []
    for counts, (_, d, f, o, _) in results.items():
        for label, e in (("default", d), ("fitted", f), ("true appearance", o)):
            rows.append([counts, label, f"{100 * np.mean(e > 5):.2f}%", f"{np.median(e):.3f}"])
    print(table(["counts/px", "master", "wrong > 5 deg", "median error (deg)"], rows))


def _curve(appearance, k):
    return float(appearance.energy_curve[k]) if appearance.energy_curve.size > k else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("appearance_validation"))
    parser.add_argument("--kv", type=float, default=20.0)
    parser.add_argument("--npx", type=int, default=200)
    parser.add_argument("--frame", type=int, default=256, help="rendered frame side, pixels")
    parser.add_argument("--refine", type=int, default=128)
    parser.add_argument("--step", type=float, default=2.0, help="dictionary step, degrees")
    parser.add_argument("--grains", type=int, default=60)
    parser.add_argument("--per-grain", type=int, default=40)
    parser.add_argument("--patterns", type=int, default=300, help="patterns fitted on")
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--counts", type=float, nargs="+", default=[16.0, 12.0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this needs a CUDA GPU")
    device = torch.device("cuda")
    resolved = resolved_master(args.out, args.kv, args.npx, device)
    point_group = phase_from_crystal(nickel()).point_group
    detector = de.EBSDDetector(
        shape=(args.frame, args.frame), pattern_center=(0.5, 0.45, 0.6), sample_tilt=70.0
    )
    default = DetectorAppearance.default(resolved)
    truth = replace(
        default,
        energy_curve=np.array([-0.8, 0.3]),
        psf_sigma_deg=0.6,
        gamma=0.7,
        background=np.array([0.04, -0.06, 0.03, 0.0, 0.02]),
        background_order=2,
    )
    run_case("A known mismatch", truth, resolved, point_group, detector, args, device)
    run_case("No mismatch", default, resolved, point_group, detector, args, device)


if __name__ == "__main__":
    main()
