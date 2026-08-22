"""Peak detection on the SO(3) correlation grid (I5).

Finds the best orientation(s) from ``C(g)``: the discrete argmax plus a
sub-pixel/sub-grid refinement (local quadratic or Wigner-D upsampling) so the
result is not limited to the SOFT grid spacing. Returns orix ``Rotation``s.
See ``INDEXING_DESIGN.md`` (milestone I5).
"""

from __future__ import annotations


def peak_orientation(correlation, n_peaks=1):
    """Top ``n_peaks`` orientations from an SO(3) correlation grid, sub-grid refined.

    Returns orix ``Rotation``(s) with the correlation value(s) as a quality metric.
    """
    raise NotImplementedError("I5: SO(3) peak detection + sub-grid refinement")
