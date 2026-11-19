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

Euler angles are **Bunge ZXZ, in radians**, and an orientation matrix ``g`` is
orix's ``Rotation.to_matrix()``: it takes a sample-frame direction to the
crystal frame, ``v_crystal = g @ v_sample``
(:func:`~de_ebsd.core.rotation.euler_to_matrix`,
:func:`~de_ebsd.core.rotation.matrix_to_euler`).

The sample frame: OIM's
=======================

**OIM (EDAX TSL) is the reference.** Every orientation de-ebsd reports -- an
indexer's, a refine's, a dictionary entry's -- and every tensor beside it, such
as a strain, is in **OIM's sample frame**. It is the frame patterns are
simulated in (kikuchipy's, :meth:`~de_ebsd.sim.MasterPattern.project`), so an
orientation goes into an ``.ang``, an orix ``CrystalMap``, OIM, MTEX or
kikuchipy exactly as de-ebsd returns it: there is nothing to convert.

The evidence is a real scan. A 200 x 200 nickel map recorded on a Direct
Electron camera was indexed by de-ebsd's dynamical indexer and re-indexed by
OIM from the same patterns, at the same pattern center. Written to an ``.ang``
unturned, de-ebsd's Euler angles agree with OIM's to a median of 0.24 degrees,
95.7% of points within 2 degrees, and the IPF-X, IPF-Y and IPF-Z colors of the two
maps agree at as many points. Turned 90 degrees about the surface normal, no point
was within 5 degrees.

OIM refers its Euler angles to sample axes of its own, not to the axes an
``.ang`` writes the scan's positions along (x along a row, y down the columns):
the two are a quarter-turn apart about the surface normal. So an orientation is
**not** turned to match the positions -- the positions stay as scanned, and the
orientations stay in OIM's frame.

A turn about the normal is easy to get wrong unseen: an **IPF-Z map cannot show
it** (a turn about z leaves z where it was), so a map turned 90 degrees looks
right in IPF-Z while every orientation in it is wrong. IPF-X and IPF-Y maps, a
misorientation to a reference given in another frame, and an ``.ang`` opened in
OIM are where it shows. de-ebsd once applied such a turn, fitted to a
simulator's ground truth rather than to an instrument;
:func:`~de_ebsd.core.frames.scan_euler`,
:func:`~de_ebsd.core.frames.to_scan_frame` and
:func:`~de_ebsd.core.frames.scan_tensor` remain only as deprecated no-ops.

A misorientation between two orientations does not depend on the frame, as long
as both are in the same one: a turn of the sample frame cancels in
``g @ reference.T`` (:func:`~de_ebsd.core.rotation.misorientation_deg`).

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
