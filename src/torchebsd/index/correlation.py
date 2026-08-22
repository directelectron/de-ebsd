"""SO(3) cross-correlation (I2): the core of spherical indexing.

Given the spherical-harmonic spectra of two functions on the sphere, their
cross-correlation over all orientations ``C(g) = <E(n), M(g n)>`` is a
band-limited function on SO(3) whose Wigner-D coefficients are the outer product
of the two spectra. One inverse SO(3) FFT (SOFT) yields ``C(g)`` on the whole
grid -- O(L^3 log L), not a brute-force sweep. See ``INDEXING_DESIGN.md`` (I2).
"""

from __future__ import annotations


def so3_cross_correlation(master_coeffs, pattern_coeffs, bandwidth):
    """Cross-correlation of two SH spectra over SO(3).

    Returns ``C(g)`` sampled on the SOFT grid; the peak is the best orientation.
    ``pattern_coeffs`` may carry a leading batch (many patterns at once).
    """
    raise NotImplementedError("I2: SO(3) cross-correlation via the Wigner-D FFT")
