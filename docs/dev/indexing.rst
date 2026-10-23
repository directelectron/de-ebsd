=======================
Indexing and refinement
=======================

:mod:`de_ebsd.index` is the inverse of :mod:`de_ebsd.sim`: simulation generates
patterns from orientations, indexing recovers orientations from patterns. Both
are GPU-native and differentiable. It depends only on orix and on de-ebsd's own
detector geometry, never on kikuchipy, and emits orix ``Rotation`` and
``CrystalMap`` objects, so other tools can call it as a backend.

Dictionary indexing
===================

:class:`~de_ebsd.index.DictionaryIndexer` projects the master over a
fundamental-zone sample of orientations and scores every pattern against every
entry by one normalized cross-correlation matrix product. The grid is coarse by
design: it seeds a refinement.

Spherical indexing
==================

:class:`~de_ebsd.index.SphericalIndexer` follows EMSphInx (Lenthe, Singh and De
Graef, 2019). The master pattern is a function ``M(n)`` on the unit sphere.
Back-projecting an experimental pattern gives ``E(n)``, and the cross-correlation
as a function of the trial orientation ``g``,

.. math::

    C(g) = \langle E(n),\, M(g \cdot n) \rangle ,

is a function on SO(3). Expanding both in spherical harmonics up to a bandwidth
``L`` makes ``C(g)`` band-limited, with Wigner-D coefficients equal to the outer
product of the two harmonic spectra. One inverse SO(3) transform (SOFT) yields
``C(g)`` on a whole grid of orientations at once, in ``O(L^3 log L)``, instead
of a brute-force sweep. The peak is the orientation.

The pipeline is:

1. back-project each pattern to the sphere (``index.backproject``);
2. transform the master once and the patterns in batches to spherical
   harmonics (``index.spherical.harmonics``);
3. correlate over SO(3) with the Wigner-D transform
   (``index.spherical.wigner``, ``index.spherical.correlation``);
4. take the peak (``index.spherical.indexer``); multi-candidate peak extraction
   is not implemented;
5. refine, then assemble an orix ``CrystalMap`` (``index.io``).

The returned orientation is the sample orientation passed to
``MasterPattern.project``, with no inverse and no transpose.

Because the whole path is differentiable, a coarse index can be sharpened by
gradient ascent on the correlation (``index.refine``), the pattern center can be
refined jointly by autograd through the back-projection, and a master that is
itself differentiable in crystal parameters opens forward-model fitting of a
structure to a scan.

Refinement
==========

- ``refine_orientation`` maximizes a masked spherical correlation by Adam (which
  also refines the pattern center) or by a batched Levenberg-Marquardt step with
  an analytic Jacobian (orientation only).
- ``refine_hr`` samples a few thousand band-edge pixels per pattern and runs
  Levenberg-Marquardt on those, jointly in orientation and, optionally, pattern
  center.
- ``determine_pattern_center`` calibrates one pattern center shared by many
  patterns, alternating per-pattern orientation refinement with a global step,
  over a schedule of progressively less blurred masters.
- ``hrebsd`` measures lattice strain and rotation by cross-correlating regions of
  a pattern against a reference, solving the shift equation for the eight
  trace-free components of the displacement gradient; a second pass remaps the
  reference through the gnomonic geometry.

Live indexing
=============

:class:`~de_ebsd.index.LiveIndexer` indexes a camera's frames where they already
are, on the GPU, in a fixed number of batched steps. Frames are divided by the
scan's running static background and have their own dynamic background removed;
one matrix product against every phase's stacked dictionary gives the phase and
a seed orientation; a local dictionary of seven projections per level, with a
parabola through each axis's scores, refines the seed. The same machinery
calibrates the pattern center from a buffer of frames and fits each pattern's
strain together with its orientation. Fused CUDA kernels, compiled at run time
when CuPy is installed, do the projection and scoring in one pass.
