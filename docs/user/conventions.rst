=====================
Units and conventions
=====================

These are load-bearing. A master pattern is only comparable to EMsoft's or
kikuchipy's because every one of them matches, so read this page before
interpreting a number or feeding a de-ebsd array to another library.
The design notes under :doc:`../dev/index` explain the reasoning behind them.

Units
=====

Lengths are in nanometres and reciprocal lengths in nm\ :sup:`-1`. Energies are
in **eV at the API surface** (``Microscope(energy=20e3)``) but keV internally
and in the transport API, which is why :class:`~de_ebsd.sim.Microscope` exposes
:attr:`~de_ebsd.sim.Microscope.energy_kev`. Fourier coefficients ``U_g`` are in
nm\ :sup:`-2` and ``V_g`` in volts.

Cartesian crystal frame
=======================

EMsoft's ``mod_crystallography.f90`` convention: **x parallel to a**,
**z parallel to c***. The direct structure matrix ``dsm`` and the reciprocal
``rsm = inv(dsm)^T`` are element-for-element EMsoft's, verified against the
source. Every beam direction and g-vector in the dynamical computation lives in
this frame.

Square-Lambert storage
======================

The master pattern is stored as ``[row, col]`` with **row = Lambert Y** and
**col = Lambert X**, both spanning ``[-sqrt(pi/2), +sqrt(pi/2)]``. The
hemisphere index is **0 = north** (``z >= 0``) and **1 = south**. Interpolation
is bilinear :func:`torch.nn.functional.grid_sample` with ``align_corners=True``.

These are exactly kikuchipy's conventions, so a de-ebsd master pattern feeds an
external projector unchanged -- and vice versa, which is what makes
``examples/simulation/compare_emsoft.py`` a meaningful comparison rather than a
re-derivation.

Orientations and detector geometry
==================================

Quaternions are in **orix order** ``(a, b, c, d)``, scalar first. Projection
centers use the **Bruker convention**, and detector geometry follows kikuchipy
(``tilt``, ``azimuthal``, ``twist``, gnomonic bounds per Britton et al. 2016);
:meth:`~de_ebsd.sim.EBSDDetector.direction_cosines` is tested equal to
kikuchipy's to 1e-10.

The orientation returned by an indexer is **the sample orientation passed to**
:meth:`~de_ebsd.sim.MasterPattern.project` -- no inverse, no transpose. A
convention error here does not degrade gracefully; it puts the answer 90 to 180
degrees away, which is exactly what the round-trip tests are built to catch.

Space groups and origin choice
==============================

Space-group operators come from the diffpy tables, which use **origin choice
1**. For the 24 two-origin groups, :class:`~de_ebsd.crystal.Crystal` defaults to
**origin choice 2** by conjugating the operators onto the nearest inversion
center.

.. warning::

    This is verified for the cubic groups. Other two-origin groups should be
    checked against the International Tables before use -- pinning the full ITA
    shift table is roadmap item M3. Getting this wrong moves atoms, so it
    changes structure factors and therefore band intensities.
