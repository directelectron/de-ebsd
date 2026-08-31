"""Physical constants and derived diffraction prefactors.

Values are transcribed verbatim from EMsoft (EMsoftOO) ``mod_global.f90`` so that
the structure-factor and wavelength results are bit-for-bit comparable with
EMsoft's Fortran implementation.
"""
from __future__ import annotations

import math

# --- fundamental constants (EMsoft mod_global.f90, exact CODATA-2018 set) ----
C_PI = 3.14159265358979323846
C_LIGHT = 299792458.0  # speed of light [m/s]
C_PLANCK = 6.62607015e-34  # Planck constant [J s]
C_BOLTZMANN = 1.380649e-23  # Boltzmann constant [J/K]
C_CHARGE = 1.602176634e-19  # elementary charge [C]
C_RESTMASS = 9.1093837090e-31  # electron rest mass [kg]
C_AVOGADRO = 6.02214076e23

# electron rest energy in keV (EMsoft hard-codes 511 / 1022 inside the WK code)
ELECTRON_REST_KEV = 511.0

# derived diffraction prefactors (EMsoft mod_diffraction.f90)
# preg converts a potential Fourier coefficient V_g [volts] into the scaled
# coefficient U_g [nm^-2]:  U_g = preg * V_g.
# preg = 2 * m0 * e / h^2 * 1e-18
PREG = 0.664840340614319

# Weickenmeier-Kohl unit-scaling constants (mod_diffraction.f90 WK branch).
# swk scales g [nm^-1] into WK reciprocal units (Angstrom * 2*pi).
SWK = 0.1 * 2.0 * C_PI  # = 2*pi/10 = 0.6283185307...
# dwwk converts a Debye-Waller B factor [nm^2] into <u^2> [Angstrom^2]:
# ul = sqrt(B * dwwk).
DWWK = 100.0 / (8.0 * C_PI**2)  # = 1.2665147955...

# Structure-factor prefactor coefficient. The full pref used by EMsoft is
#   pref = PREF_COEF / V_cell[Angstrom^3] / (4*pi)
# and the potential-to-U scaling is pre = pref * PREG.
PREF_COEF = 0.04787801

# Doyle-Turner summation prefactor for kinematical X-ray form factors (XR mode).
XR_PREF = 0.4178214


def electron_wavelength_nm(voltage_kv: float, v0mod: float = 0.0) -> float:
    """Relativistic electron wavelength in nm (EMsoft ``CalcWaveLength``).

    Parameters
    ----------
    voltage_kv
        Accelerating voltage in kV.
    v0mod
        Mean inner potential (the modulus of V_g for g = 0), in volts, added as
        a refraction correction. Pass 0 to omit the correction.
    """
    temp1 = 1.0e9 * C_PLANCK / math.sqrt(2.0 * C_RESTMASS * C_CHARGE)
    temp2 = C_CHARGE * 0.5 * voltage_kv * 1000.0 / C_RESTMASS / (C_LIGHT**2)
    psihat = voltage_kv * (1.0 + temp2) * 1000.0 + v0mod
    return temp1 / math.sqrt(psihat)


def relativistic_correction(voltage_kv: float) -> float:
    """Relativistic correction factor gamma = 1 + e V / (m0 c^2)."""
    temp2 = C_CHARGE * 0.5 * voltage_kv * 1000.0 / C_RESTMASS / (C_LIGHT**2)
    return 1.0 + 2.0 * temp2


def interaction_constant(voltage_kv: float, v0mod: float = 0.0) -> float:
    """Interaction constant sigma in (V nm)^-1 (EMsoft ``Sigma``)."""
    lam = electron_wavelength_nm(voltage_kv, v0mod)
    gamma = relativistic_correction(voltage_kv)
    sigma = 2.0 * C_PI * C_RESTMASS * gamma * C_CHARGE * lam
    return 1.0e-18 * sigma / C_PLANCK**2
