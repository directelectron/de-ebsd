"""Rebuild the Mott elastic cross-section tables the Monte Carlo samples.

``de_ebsd.sim.montecarlo.mott`` scatters electrons by Mott (Dirac partial-wave)
differential cross sections computed by ELSEPA (Salvat, Jablonski and Powell,
Comput. Phys. Commun. 165 (2005) 157-190). ELSEPA is not part of de-ebsd and is
not run at import or simulation time: this script runs it once per element, keeps
its differential cross section at a reduced set of its own angles, and writes the
package table ``src/de_ebsd/sim/montecarlo/data/mott_elsepa.npz`` that the
Monte Carlo reads. The ELSEPA settings are recorded in the table's metadata.

Get ELSEPA from https://github.com/eScatter/elsepa (Apache-2.0), build
``elscata`` with any Fortran compiler (``make``, or ``ifx /O2 src/elscata.f
src/elsepa.f``) and point ``--elsepa-data`` (or ``ELSEPA_DATA``) at its ``data``
directory.

Usage::

    uv run python scripts/generate_mott_tables.py --elscata /path/to/elscata \\
        --elsepa-data /path/to/elsepa/data
    # one element, every ELSEPA angle, for an accuracy study:
    uv run python scripts/generate_mott_tables.py --elscata ... --elements Ni \\
        --angles all --out ni_full.npz

The settings are ELSEPA's defaults for a neutral free atom: Dirac-Fock electron
density, Fermi nuclear charge distribution, Furness-McCarthy exchange, no
correlation-polarization and no absorption potential, high-energy factorization
on. These are the settings behind NIST SRD 64 (version 4).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from de_ebsd.crystal.data.scattering_tables import ELEMENT_SYMBOLS

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "src" / "de_ebsd" / "sim" / "montecarlo" / "data" / "mott_elsepa.npz"

# The common EBSD materials (Ni, Fe, Cu, Al, Si, Ti, Co, Cr, Mo, W, Au, Zr, Mg,
# Zn) and the other elements of their usual alloys and compounds: steels and
# superalloys, oxides, carbides, nitrides, borides, sulfides, perovskites,
# III-V semiconductors and rock-forming minerals: 334 KiB. About 8 KiB an element;
# all 98 would add 0.8 MB to a 1.6 MB package, mostly for elements no one indexes.
DEFAULT_ELEMENTS = (
    "B C N O F Na Mg Al Si P S K Ca Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Sr Y Zr Nb Mo "
    "Ag In Sn Ba Hf Ta W Re Pt Au Pb"
).split()

# ELSEPA input for a neutral free atom: its defaults, written out so the table
# records exactly what ran. MCPOL/MABS 0: no correlation-polarization or
# absorption potential (inelastic channels are the stopping power's job).
ELSEPA_SETTINGS = {
    "MNUCL": (3, "nuclear charge distribution: 3 = Fermi"),
    "MELEC": (4, "electron density: 4 = Dirac-Fock"),
    "MUFFIN": (0, "0 = free atom"),
    "IELEC": (-1, "projectile: -1 = electron"),
    "MEXCH": (1, "exchange potential: 1 = Furness-McCarthy"),
    "MCPOL": (0, "correlation-polarization potential: 0 = none"),
    "MABS": (0, "absorption potential: 0 = none"),
    "IHEF": (1, "high-energy factorization: 1 = on"),
}

# 50 eV to 30 keV in steps of a factor 1.25: the depth statistics bin down to
# a few keV below the beam, and an electron is retired long before 50 eV.
DEFAULT_ENERGIES_EV = np.geomspace(50.0, 30_000.0, 30)


def elsepa_input(z: int, energies_ev) -> str:
    lines = [f"IZ     {z:d}", f"NELEC  {z:d}"]
    lines += [f"{key:<6} {value:d}" for key, (value, _) in ELSEPA_SETTINGS.items()]
    lines += [f"EV     {e:.4E}" for e in energies_ev]
    return "\n".join(lines) + "\n"


def parse_dcs(path: Path) -> dict:
    """One ``dcs_*.dat`` file: energy, total and transport cross sections, DCS."""
    text = path.read_text()

    def header(pattern):
        return float(re.search(pattern + r"\s*=\s*([0-9.E+-]+)", text).group(1))

    rows = np.array(
        [
            [float(v) for v in line.split()[:3]]
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    )
    return {
        "energy_ev": header(r"Kinetic energy"),
        "total_cm2": header(r"Total elastic cross section"),
        "transport_cm2": header(r"1st transport cross section"),
        "theta_deg": rows[:, 0],
        "dcs_cm2_sr": rows[:, 2],
    }


def run_element(z: int, energies_ev, elscata: Path, data_dir: Path, work: Path) -> list[dict]:
    """Run ELSEPA for one element and return one parsed DCS per energy, ascending."""
    folder = work / f"z{z:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ELSEPA_DATA=str(data_dir))
    result = subprocess.run(
        [str(elscata)],
        input=elsepa_input(z, energies_ev),
        cwd=folder,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    files = sorted(folder.glob("dcs_*.dat"))
    if len(files) != len(energies_ev):
        raise RuntimeError(
            f"ELSEPA wrote {len(files)} of {len(energies_ev)} files for Z={z} "
            f"(exit {result.returncode}):\n{result.stdout[-2000:]}{result.stderr[-2000:]}"
        )
    return sorted((parse_dcs(f) for f in files), key=lambda r: r["energy_ev"])


def select_angles(theta_deg: np.ndarray, n_angles: int) -> np.ndarray:
    """Indices of ``n_angles`` of ELSEPA's own angles, spread for interpolation.

    The DCS falls by many decades over the first few degrees and then varies
    slowly, so the kept angles are spaced evenly in ``ln(theta + 3 deg)``:
    logarithmic below a few degrees, close to even in angle above thirty. Of the
    offsets tried (0.3, 1, 3 and 10 degrees) 3 gave the smallest error in the
    sampled distribution. The 0 and 180 degree ends are always kept.
    """
    if n_angles >= len(theta_deg):
        return np.arange(len(theta_deg))
    offset = 3.0
    target = offset * np.expm1(np.linspace(0.0, np.log1p(theta_deg[-1] / offset), n_angles))
    index = np.unique(np.abs(theta_deg[:, None] - target[None, :]).argmin(axis=0))
    return np.union1d(index, [0, len(theta_deg) - 1])


def git_describe(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--elscata", required=True, type=Path, help="ELSEPA's elscata executable")
    parser.add_argument(
        "--elsepa-data",
        type=Path,
        default=os.environ.get("ELSEPA_DATA"),
        help="ELSEPA's data directory (z_nnn.den, z_nnn.dfs); default $ELSEPA_DATA",
    )
    parser.add_argument("--elements", nargs="+", default=DEFAULT_ELEMENTS)
    parser.add_argument(
        "--angles",
        default="128",
        help="number of ELSEPA angles kept per energy, or 'all' (default 128)",
    )
    parser.add_argument("--energies", type=int, default=len(DEFAULT_ENERGIES_EV))
    parser.add_argument("--energy-min-ev", type=float, default=DEFAULT_ENERGIES_EV[0])
    parser.add_argument("--energy-max-ev", type=float, default=DEFAULT_ENERGIES_EV[-1])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--work", type=Path, default=None, help="keep ELSEPA's output here")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args(argv)
    if args.elsepa_data is None:
        parser.error("set --elsepa-data or ELSEPA_DATA to ELSEPA's data directory")

    energies = np.geomspace(args.energy_min_ev, args.energy_max_ev, args.energies)
    energies = np.array([float(f"{e:.4E}") for e in energies])  # as ELSEPA reads them
    numbers = sorted({ELEMENT_SYMBOLS.index(s) + 1 for s in args.elements})
    work = Path(args.work or tempfile.mkdtemp(prefix="elsepa_"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            z: pool.submit(run_element, z, energies, args.elscata, args.elsepa_data, work)
            for z in numbers
        }
        runs = {z: f.result() for z, f in futures.items()}

    theta = runs[numbers[0]][0]["theta_deg"]
    for z, per_energy in runs.items():
        for r in per_energy:
            if not np.array_equal(r["theta_deg"], theta):
                raise RuntimeError(f"Z={z}: ELSEPA's angular grid changed")
    keep = select_angles(theta, len(theta) if args.angles == "all" else int(args.angles))

    log_dcs = np.array([[np.log(r["dcs_cm2_sr"][keep]) for r in runs[z]] for z in numbers])
    # ln(DCS) spans about 25 per curve; 16 bits over each curve's own range keep
    # it to 2e-4 (0.02 % in the cross section), half the size of float32
    low = log_dcs.min(axis=-1)
    high = log_dcs.max(axis=-1)
    span = np.where(high > low, high - low, 1.0)
    quantized = np.round((log_dcs - low[..., None]) / span[..., None] * 65535).astype(np.uint16)
    total = np.array([[r["total_cm2"] for r in runs[z]] for z in numbers])
    transport = np.array([[r["transport_cm2"] for r in runs[z]] for z in numbers])
    metadata = {
        "program": "ELSEPA (elscata)",
        "citation": (
            "F. Salvat, A. Jablonski and C. J. Powell, ELSEPA - Dirac partial-wave "
            "calculation of elastic scattering of electrons and positrons by atoms, "
            "positive ions and molecules, Comput. Phys. Commun. 165 (2005) 157-190, "
            "doi:10.1016/j.cpc.2004.09.006"
        ),
        "source": "https://github.com/eScatter/elsepa (Apache-2.0)",
        "settings": {key: value for key, (value, _) in ELSEPA_SETTINGS.items()},
        "settings_meaning": {key: meaning for key, (_, meaning) in ELSEPA_SETTINGS.items()},
        "units": {
            "energy_ev": "eV",
            "theta_deg": "degree",
            "log_dcs": (
                "ln(cm^2/sr) per atom = log_dcs_low + log_dcs_u16 / 65535 * "
                "(log_dcs_high - log_dcs_low)"
            ),
            "total_cm2": "cm^2 per atom",
            "transport_cm2": "cm^2 per atom (first transport cross section)",
        },
        "angles": f"{len(keep)} of ELSEPA's {len(theta)}",
        "generator": "scripts/generate_mott_tables.py",
        "generator_commit": git_describe(REPO),
        "generated": datetime.date.today().isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        z=np.array(numbers, dtype=np.int16),
        energy_ev=energies,
        theta_deg=theta[keep],
        log_dcs_u16=quantized,
        log_dcs_low=low,
        log_dcs_high=high,
        total_cm2=total,
        transport_cm2=transport,
        metadata=np.array(json.dumps(metadata, indent=1)),
    )
    size = args.out.stat().st_size
    print(f"{args.out}: {len(numbers)} elements, {len(energies)} energies, ", end="")
    print(f"{len(keep)} angles, {size / 1024:.0f} KiB (ELSEPA output in {work})")


if __name__ == "__main__":
    main()
