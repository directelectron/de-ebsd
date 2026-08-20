"""Validate torchebsd master patterns against EMsoft reference data.

The references are dynamical (Bloch-wave) EMsoft master patterns:

- Ni: ``kikuchipy.data.nickel_ebsd_master_pattern_small`` (20 keV, npx = 200)
- Si: the EMsoft run at ``D:\\emsoft\\EMdata\\Si_MC.h5`` (20 keV, npx = 500)

The comparison metric is the per-pixel Pearson correlation of the square
Lambert maps (scale invariant: the intensity normalizations differ). The
reference is resampled onto the simulated grid through the same validated
interpolation used for projection, so any npx works.

While the Bloch-wave backend is unfinished (DESIGN.md, M2), the kinematical
backend is compared instead: band positions should agree (frame conventions
correct) while band profiles do not, so intermediate correlations are the
expected outcome. Once the Bloch backend lands, this script is the parity
gate and should approach a correlation of 1.0.

Usage::

    uv run python scripts/validate_master_pattern.py --material ni --device cuda
    uv run python scripts/validate_master_pattern.py --material si --html out.html
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import torchebsd as te
from torchebsd.lambert import master_pattern_grid

SI_MC_PATH = r"D:\emsoft\EMdata\Si_MC.h5"


def load_reference_ni() -> tuple[te.MasterPattern, dict]:
    import kikuchipy as kp

    signal = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")
    data = torch.as_tensor(signal.data.astype(np.float32))[None]  # (1, 2, 401, 401)
    reference = te.MasterPattern(data=data, energies_kev=np.array([20.0]),
                                 metadata={"backend": "EMsoft (kikuchipy data)"})
    params = {"a": 0.35236, "space_group": 225, "sites": [("Ni", (0.0, 0.0, 0.0))],
              "debye_waller": 0.0035, "dmin": 0.05}
    return reference, params


def load_reference_si() -> tuple[te.MasterPattern, dict]:
    import h5py

    with h5py.File(SI_MC_PATH, "r") as f:
        i_e = -1  # highest energy bin = 20 keV
        north = np.asarray(f["EMData/EBSDmaster/mLPNH"][0, i_e], dtype=np.float32)
        south = np.asarray(f["EMData/EBSDmaster/mLPSH"][0, i_e], dtype=np.float32)
        energy = float(np.asarray(f["EMData/EBSDmaster/EkeVs"])[i_e])
        dmin = float(np.asarray(f["NMLparameters/EBSDMasterNameList/dmin"])[0])
        lattice = np.asarray(f["CrystalData/LatticeParameters"])  # nm, degrees
        atom_data = np.asarray(f["CrystalData/AtomData"])  # (5, n): x y z occ B
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
    parser.add_argument("--html", default=None, help="save a side-by-side figure to this file")
    args = parser.parse_args()

    reference, params = load_reference_ni() if args.material == "ni" else load_reference_si()
    npx = args.npx if args.npx is not None else reference.npx
    dmin = args.dmin if args.dmin is not None else params["dmin"]

    sites = params["sites"]
    crystal = te.Crystal.from_parameters(
        a=params["a"], space_group=params["space_group"], atoms=sites,
        debye_waller=params["debye_waller"],
    )
    energy_kev = float(reference.energies_kev[-1])
    microscope = te.Microscope(energy=energy_kev * 1e3)
    simulator = te.MasterPatternSimulator(
        crystal, microscope, npx=npx, dmin=dmin, backend=args.backend
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
        from torchebsd import display

        figure = display.hemispheres_figure(
            our_n.numpy(), ref_n.detach().numpy(),
            title=f"{args.material.capitalize()} north: torchebsd ({master.metadata['backend']})"
                  " vs EMsoft",
        )
        figure.save_html(args.html)
        print(f"wrote {args.html}")


if __name__ == "__main__":
    main()
