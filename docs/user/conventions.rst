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

The sample frame and the scan's frame
=====================================

Every orientation de-ebsd reports is in **kikuchipy's sample frame**, the frame
patterns are simulated in. That is the frame orix, kikuchipy and EMsoft read, so
an indexer's answer goes to them unchanged.

A scan has a frame of its own: **x along a row of the scan, y down its columns,
z into the sample**. It is kikuchipy's sample frame turned **90 degrees about
the surface normal**: a direction ``v`` in kikuchipy's frame is
``SCAN_TURN @ v`` in the scan's, with

.. math::

    \mathrm{SCAN\_TURN} = \begin{pmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix},

so the scan's x is kikuchipy's -y. An orientation becomes
``g @ SCAN_TURN.T`` and a tensor such as a strain
``SCAN_TURN @ e @ SCAN_TURN.T``. :mod:`de_ebsd.core.frames` does each:

* :func:`~de_ebsd.core.frames.scan_euler` -- orientation matrices to Euler
  angles in the scan's frame;
* :func:`~de_ebsd.core.frames.to_scan_frame` -- Euler angles to Euler angles;
* :func:`~de_ebsd.core.frames.scan_tensor` -- a tensor field, turned as the
  orientations are, so a strain map shares the axes of the orientation map
  beside it.

The conversion is **explicit, and the caller applies it at the edge** -- when a
map is drawn against the scan's axes or saved for a program that reads them.
Nothing in de-ebsd applies it for you, because a turn buried inside one path is
one the other paths silently lack. It is easy to miss: an IPF-Z map cannot show
it (a turn about z leaves z where it was), so a map drawn without the turn looks
right while every orientation in it is 90 degrees off. Measured against a
duplex-steel scan's ground truth, no pattern was within 5 degrees until it was
turned. IPF-X and IPF-Y maps, a misorientation to a reference given in the
scan's frame, and saved ``.ang`` files are where it shows.

A misorientation between two orientations does not depend on the frame, as long
as both are in the same one: the turn cancels in ``g @ reference.T``
(:func:`~de_ebsd.core.rotation.misorientation_deg`).

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
