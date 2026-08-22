"""Validate the native Monte Carlo depth statistics against EMsoft ``accum_z``.

EMsoft is the oracle (validation only). This compares torchebsd's native
:func:`~torchebsd.sim.montecarlo_sim.simulate_depth_distribution` against a reference
EMsoft ``*_MC.h5`` run: per-energy-bin Pearson correlation of the exit-depth
profiles and the total backscatter yield. The native MC is run with the
reference run's own parameters (tilt, energy/depth binning), read from its
namelist. The reference file is not redistributed; point ``--mc`` (or
``TORCHEBSD_{SI,NI}_MC``) at your copy.

Usage::

    uv run python scripts/validate_montecarlo.py --material si --device cuda
    uv run python scripts/validate_montecarlo.py --material ni --mc /path/to/Ni_MC.h5
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

import torchebsd as te
from torchebsd.sim.montecarlo import DepthDistribution
from torchebsd.sim.montecarlo_sim import simulate_depth_distribution

SI_MC_PATH = os.environ.get("TORCHEBSD_SI_MC", r"D:\emsoft\EMdata\Si_MC.h5")
NI_MC_PATH = os.environ.get("TORCHEBSD_NI_MC", r"D:\emsoft\EMdata\Ni_MC.h5")

CRYSTALS = {
    "si": dict(a=0.5431, space_group=227,
               atoms=[("Si", (0.125, 0.125, 0.125))], debye_waller=0.004668),
    "ni": dict(a=0.35236, space_group=225,
               atoms=[("Ni", (0.0, 0.0, 0.0))], debye_waller=0.0035),
}


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def read_mc_params(path: str) -> dict:
    """The reference run's Monte Carlo namelist (tilt, energy/depth binning)."""
    with h5py.File(path, "r") as f:
        nml = f["NMLparameters/MCCLNameList"]
        get = lambda k: float(np.asarray(nml[k]).ravel()[0])  # noqa: E731
        return {
            "sig": get("sig"), "omega": get("omega"), "ekev": get("EkeV"),
            "ehistmin": get("Ehistmin"), "ebinsize": get("Ebinsize"),
            "depthstep": get("depthstep"), "depthmax": get("depthmax"),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--material", choices=["si", "ni"], default="si")
    parser.add_argument("--mc", default=None, help="EMsoft *_MC.h5 reference path")
    parser.add_argument("--n-electrons", type=int, default=400_000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mc_path = args.mc or {"si": SI_MC_PATH, "ni": NI_MC_PATH}[args.material]
    if not os.path.exists(mc_path):
        raise SystemExit(
            f"reference not found: {mc_path}\n"
            "Point --mc or TORCHEBSD_{SI,NI}_MC at your EMsoft *_MC.h5 file."
        )

    ref = DepthDistribution.from_emsoft(mc_path)
    p = read_mc_params(mc_path)
    crystal = te.Crystal.from_parameters(**CRYSTALS[args.material])
    print(f"reference: {ref}")
    print(f"MC params: tilt={p['sig']:g} deg, E {p['ehistmin']:g}..{p['ekev']:g} keV "
          f"/ {p['ebinsize']:g}, depth {p['depthstep']:g}..{p['depthmax']:g} nm")

    native = simulate_depth_distribution(
        crystal, p["ekev"], e_min_kev=p["ehistmin"], e_bin_kev=p["ebinsize"],
        depth_step_nm=p["depthstep"], depth_max_nm=p["depthmax"],
        n_electrons=args.n_electrons, tilt_deg=p["sig"], azimuth_deg=p["omega"],
        device=args.device, seed=args.seed,
    )

    ref_yield = ref.accum_z.sum() / ref.n_total
    nat_yield = native.accum_z.sum() / native.n_total
    print(f"\nbackscatter yield   EMsoft={ref_yield:.4f}   native={nat_yield:.4f}   "
          f"(diff {abs(ref_yield - nat_yield):+.4f})")

    print("\n  E (keV)   corr    meanz_ref  meanz_nat")
    correlations = []
    for i_e in range(ref.n_energies - 1, -1, -1):
        r = ref.accum_z[i_e].sum(axis=(1, 2))
        n = native.accum_z[i_e].sum(axis=(1, 2))
        if r.sum() == 0 or n.sum() == 0:
            continue
        c = pearson(r, n)
        correlations.append(c)
        depth = np.arange(len(r))
        mzr = (depth * r).sum() / r.sum()
        mzn = (depth * n).sum() / n.sum()
        print(f"   {ref.energies_kev[i_e]:5.0f}   {c:.4f}   {mzr:7.1f}    {mzn:7.1f}")

    print(f"\nminimum per-energy correlation: {min(correlations):.4f}")
    print("PASS" if min(correlations) > 0.99 else "BELOW 0.99")


if __name__ == "__main__":
    main()
