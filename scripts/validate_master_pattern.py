"""Validate torchebsd master patterns against EMsoft reference data.

EMsoft is the validation *oracle* only -- torchebsd computes the master pattern
itself (see DESIGN.md). This script builds the same crystal, simulates, and
correlates against a reference EMsoft master pattern:

- Ni: ``kikuchipy.data.nickel_ebsd_master_pattern_small`` (20 keV, npx = 200),
  bundled with kikuchipy, so Ni runs with no external files.
- Si: an EMsoft ``*_MC.h5`` run (20 keV, npx = 500), which also carries the
  crystal data, Bethe parameters, and Monte Carlo depth statistics of the
  reference run. Point ``--mc`` (or ``TORCHEBSD_SI_MC``) at your local copy.

The reference HDF5 files are *not* redistributed here; there is no default that
works on another machine. The comparison metric is the per-pixel Pearson
correlation of the square Lambert maps (scale invariant: normalizations
differ), the reference resampled onto the simulated grid through the same
validated interpolation used for projection, so any npx works.

The exit-depth profile that weights the Bloch depth integration comes from the
EMsoft Monte Carlo file when one is supplied; otherwise torchebsd falls back to
its own single-energy exponential model (``DepthDistribution.single_energy``),
so the simulation stays runnable with no EMsoft input. A native Monte Carlo
that removes this last runtime dependency is on the roadmap (DESIGN.md).

Usage::

    uv run python scripts/validate_master_pattern.py --material ni --backend bloch_prism
    uv run python scripts/validate_master_pattern.py --material si --backend bloch_prism \\
        --mc /path/to/Si_MC.h5 --device cuda
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import torchebsd as te
from torchebsd.sim.lambert import master_pattern_grid

# Reference EMsoft HDF5 paths. Overridable per material via the environment or
# the --mc flag; the built-in fallbacks are the maintainer's local paths and are
# not expected to exist elsewhere (the files are not redistributed).
SI_MC_PATH = os.environ.get("TORCHEBSD_SI_MC", r"D:\emsoft\EMdata\Si_MC.h5")
# Optional Monte Carlo exit-depth statistics for Ni (kikuchipy ships only the
# master pattern, not the MC run). Generate with EMsoft (EMMCOpenCL on Ni.xtal,
# same settings as Si_MC). If absent the Bloch backend falls back to torchebsd's
# own single-energy depth model, which washes out the dynamical contrast.
NI_MC_PATH = os.environ.get("TORCHEBSD_NI_MC", r"D:\emsoft\EMdata\Ni_MC.h5")


def load_reference_ni() -> tuple[te.MasterPattern, dict]:
    import kikuchipy as kp

    signal = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")
    data = torch.as_tensor(signal.data.astype(np.float32))[None]  # (1, 2, 401, 401)
    reference = te.MasterPattern(data=data, energies_kev=np.array([20.0]),
                                 metadata={"backend": "EMsoft (kikuchipy data)"})
    params = {"a": 0.35236, "space_group": 225, "sites": [("Ni", (0.0, 0.0, 0.0))],
              "debye_waller": 0.0035, "dmin": 0.05}
    return reference, params


def load_reference_si(path: str) -> tuple[te.MasterPattern, dict]:
    import h5py

    if not os.path.exists(path):
        raise SystemExit(
            f"Si reference not found: {path}\n"
            "Si validation needs the EMsoft Si_MC.h5 reference run (it supplies "
            "the reference master pattern, crystal data, and Bethe parameters).\n"
            "Point --mc or the TORCHEBSD_SI_MC environment variable at your copy."
        )
    with h5py.File(path, "r") as f:
        i_e = -1  # highest energy bin = 20 keV
        north = np.asarray(f["EMData/EBSDmaster/mLPNH"][0, i_e], dtype=np.float32)
        south = np.asarray(f["EMData/EBSDmaster/mLPSH"][0, i_e], dtype=np.float32)
        energy = float(np.asarray(f["EMData/EBSDmaster/EkeVs"])[i_e])
        dmin = float(np.asarray(f["NMLparameters/EBSDMasterNameList/dmin"])[0])
        lattice = np.asarray(f["CrystalData/LatticeParameters"])  # nm, degrees
        atom_data = np.asarray(f["CrystalData/AtomData"])  # (5, n): x y z occ B
        bethe = np.asarray(f["EMData/EBSDmaster/BetheParameters"])  # c1 c2 c3 sgdbdiff
    data = torch.as_tensor(np.stack([north, south]))[None]
    reference = te.MasterPattern(data=data, energies_kev=np.array([energy]),
                                 metadata={"backend": "EMsoft (Si_MC.h5)"})
    x, y, z, occupancy, dw = atom_data[:, 0]
    params = {
        "a": float(lattice[0]),
        "space_group": 227,
        "sites": [te.AtomSite("Si", (float(x), float(y), float(z)), float(occupancy), float(dw))],
        "debye_waller": float(dw),
        "dmin": dmin,
        "bethe": te.BetheParameters(strong=float(bethe[0]), weak=float(bethe[1]),
                                    cutoff=float(bethe[2]), sg_double=float(bethe[3])),
    }
    return reference, params


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--material", choices=["ni", "si"], default="ni")
    parser.add_argument("--npx", type=int, default=None,
                        help="Lambert half-width (default: match the reference)")
    parser.add_argument("--dmin", type=float, default=None,
                        help="dmin in nm (default: the reference run's value)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backend", default="kinematical")
    parser.add_argument(
        "--mc", default=None,
        help="EMsoft reference HDF5 for this material (Si: the reference+MC run; "
             "Ni: optional exit-depth MC to refine the depth profile). Overrides "
             "TORCHEBSD_{SI,NI}_MC and the built-in default.")
    parser.add_argument("--html", default=None, help="save a side-by-side figure to this file")
    args = parser.parse_args()

    mc_path = args.mc or {"si": SI_MC_PATH, "ni": NI_MC_PATH}[args.material]
    reference, params = (
        load_reference_ni() if args.material == "ni" else load_reference_si(mc_path)
    )
    npx = args.npx if args.npx is not None else reference.npx
    dmin = args.dmin if args.dmin is not None else params["dmin"]

    sites = params["sites"]
    crystal = te.Crystal.from_parameters(
        a=params["a"], space_group=params["space_group"], atoms=sites,
        debye_waller=params["debye_waller"],
    )
    energy_kev = float(reference.energies_kev[-1])
    microscope = te.Microscope(energy=energy_kev * 1e3)
    # The Bloch backend weights the depth integration by the exit-depth profile;
    # use the reference EMsoft Monte Carlo when available so the profile matches
    # exactly, otherwise fall back to torchebsd's own single-energy model.
    monte_carlo = None
    if mc_path and os.path.exists(mc_path):
        from torchebsd.sim.montecarlo import DepthDistribution

        monte_carlo = DepthDistribution.from_emsoft(mc_path)
    simulator = te.MasterPatternSimulator(
        crystal, microscope, npx=npx, dmin=dmin, backend=args.backend,
        monte_carlo=monte_carlo, bethe=params.get("bethe"),
    )
    print(simulator)
    print(simulator.structure_factors())

    master = simulator.compute(device=args.device, chunk_size=65_536)
    print(master)

    # Resample the reference onto the simulated grid via the shared
    # (kikuchipy-validated) interpolation, then correlate per hemisphere.
    grid = master_pattern_grid(npx, device="cpu")
    reference = reference.to("cpu")
    ours = master.to("cpu")
    ref_n = reference.sample(grid)
    ref_s = reference.sample(-grid)
    our_n = ours.data[0, 0]
    our_s = ours.data[0, 1]

    r_north = pearson(our_n, ref_n)
    r_south = pearson(our_s, ref_s)
    r_all = pearson(torch.stack([our_n, our_s]), torch.stack([ref_n, ref_s]))
    print()
    print(f"Reference: {reference.metadata['backend']}, npx={reference.npx}, "
          f"{energy_kev:g} keV")
    print(f"Simulated: backend={master.metadata['backend']}, npx={npx}, dmin={dmin} nm")
    print(f"Pearson correlation  north: {r_north:+.4f}")
    print(f"Pearson correlation  south: {r_south:+.4f}")
    print(f"Pearson correlation  total: {r_all:+.4f}")
    if master.metadata["backend"] == "kinematical":
        print("(kinematical placeholder: band positions align, band profiles do not; "
              "full parity requires the Bloch-wave backend, DESIGN.md M2)")

    if args.html:
        from torchebsd.sim import display

        figure = display.hemispheres_figure(
            our_n.numpy(), ref_n.detach().numpy(),
            title=f"{args.material.capitalize()} north: torchebsd ({master.metadata['backend']})"
                  " vs EMsoft",
        )
        figure.save_html(args.html)
        print(f"wrote {args.html}")


if __name__ == "__main__":
    main()
