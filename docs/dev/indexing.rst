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

It is the one multi-phase indexer, for either physics. Each phase's master may be
a dynamical :class:`~de_ebsd.sim.MasterPattern` or a kinematic
:class:`~de_ebsd.sim.KinematicMaster`, sampled onto the same Lambert grid
(:meth:`~de_ebsd.sim.KinematicMaster.to_master_pattern`), so a CPU without the
minutes a dynamical master takes indexes several phases the same way a GPU does
(:func:`~de_ebsd.index.live.dynamical_indexing_available` says which to use).
Frames may be NumPy, torch or CuPy arrays; a GPU array is read in place through
DLPack. Orientations and strains come back in kikuchipy's sample frame; turning
them into a scan's is the caller's, with :mod:`de_ebsd.core.frames`.

The design goal is a cost per buffer that is fixed and known in advance: every
step below is a batched operation over the whole buffer, and nothing loops over
patterns or phases. Only each pattern's phase, orientation and score -- and its
strain, when asked for -- leave the device.

Preparing the patterns
----------------------

:meth:`~de_ebsd.index.LiveIndexer.prepare` turns a buffer of raw frames into the
rows that are scored, in three steps:

1. Each frame is averaged over the whole sensor down to the refine shape
   (``refine_shape``, 128 x 128 by default). Averaging the whole sensor, rather
   than cropping, keeps every band and every count.
2. It is divided by the scan's **static background**, the running mean of the
   frames the scan has seen (:meth:`~de_ebsd.index.LiveIndexer.observe`). A live
   scan's frames come from a few grains at a time, so the mean waits for 32
   frames before it is used; :meth:`~de_ebsd.index.LiveIndexer.restart_static`
   starts it from frames chosen from all over the specimen (a survey's), which
   average their bands away in fewer. On a single crystal every frame carries
   the same bands, so the mean would keep them: there it is blurred.
3. The pattern's own slowly varying **dynamic background** is removed, the rows
   are normalized to zero mean and unit length, and a copy is averaged again to
   the search shape (``search_shape``, 32 x 32 by default).

Whatever of the grains' bands the static background still holds -- the
**ghost** -- is tracked from the orientations already fitted, so the strain fit
below can take it out of patterns and simulations alike.

Searching the dictionary
------------------------

Each phase's master is projected at the search shape over a sample of its
fundamental zone, at the dictionary's angular step: about 30,000 entries for a
cubic phase at 3 degrees. Every phase's entries are stacked into one matrix, so
:meth:`~de_ebsd.index.LiveIndexer.search` is one matrix product of the buffer
against that matrix: the best entry gives the phase and a seed orientation
together, and the score is the normalized cross-correlation. A low-symmetry
phase's dictionary is scored 65,536 entries at a time, keeping each pattern's
best so far, so the scores never outgrow one chunk.

The search's copy of the masters is blurred by 2 degrees on the sphere. A sharp
dynamical pattern's score falls so steeply away from its orientation that at a
3-degree step no entry may be near enough, and a distant wrong one wins; on a
twin scan 6-9 % of the patterns landed 40-60 degrees off unblurred, none at
2 degrees. The search only has to land in the right basin -- the refine does the
rest.

Refining the orientation
------------------------

:meth:`~de_ebsd.index.LiveIndexer.refine` builds a local dictionary around each
seed, projected from the (unblurred) master on the fly at the refine shape: the
seed and the six rotations of one step about each axis. A parabola through each
axis's three scores moves the orientation, and the step halves each level, from
2 degrees down to 0.12. Every level is the same seven projections per pattern,
so its cost is fixed. With CuPy on a CUDA device the levels run as one fused
kernel launch, one block per pattern, and the levels of a degree and up are
scored at half the refine resolution: a step that size moves the bands by
several pixels, which a quarter of them resolves as well.

With ``fit=True``, :meth:`~de_ebsd.index.LiveIndexer.index` fits the three
rotations on from the refine's result by Gauss-Newton on the normalized
pattern, with the Jacobian taken from the master's gradient. The stepped levels
stop a little short on a noisy pattern; the fit takes them the rest of the way.
An indexer that will fit is built with
:func:`~de_ebsd.index.live.fitted_refine_steps_deg`, which stops the stepped
levels at 1 degree where the fit is fused: the fit does what the finer levels
did, for less.

Calibrating the pattern center
------------------------------

:meth:`~de_ebsd.index.LiveIndexer.refine_pattern_center` refines the center
shared by a buffer the same way the orientation is refined: seven candidate
centers per level -- the current one and one step either way in x, y and z --
each scored by re-refining every pattern's orientation there, with a parabola
per coordinate and the step halving from 0.008 to 0.001 of the detector. It
uses the buffer's best-scored patterns, and refines their orientations far finer
than indexing does (down to 0.03 degrees): a center 0.004 off moves each
pattern's best orientation by a few tenths of a degree, which coarser steps
follow only in jumps, leaving the mean score flat and the center stuck short.

Live strain
===========

A strained lattice changes the pattern in ways an orientation cannot: the bands
move relative to one another. There are two ways to measure it live: fitting
each pattern against a dynamical master, or correlating it against a real
reference pattern.

Fitted against the master
-------------------------

:meth:`~de_ebsd.index.LiveIndexer.refine_deformation` (and
:meth:`~de_ebsd.index.LiveIndexer.index_with_strain`, which indexes and then
calls it) fits each pattern's orientation and strain together. A crystal
deformed by ``F`` shows the master's pattern at ``g F^-1 r`` for a ray ``r``, so
a candidate is the matrix ``g (I - beta)``, with ``beta`` eight trace-free
numbers: three rotations and five deviatoric strains. The fused kernels project
that matrix like any orientation. Each Gauss-Newton iteration projects the
current fit and its 16 central differences and solves the 8 x 8 normal
equations, for every pattern at once.

- **The hydrostatic strain is not measured.** A gnomonic projection cannot see a
  uniform dilation, so the ninth component is left out; the strain returned is
  the symmetric, trace-free Biot strain in the sample frame, with the lattice
  rotation split off by a polar decomposition.
- **The rotations are fitted with the strains** because a strain left out of the
  model is partly fitted as a turn: fitting both is also what sharpens the
  orientations.
- **Each step is bounded.** A step solves a local linear model, and on a poor
  pattern -- a grain boundary, two patterns overlapping -- that model can point
  far outside the elastic range. Without a limit on each step, 1-3 % of a real
  scan's fits ran off to strains of 0.1 and more. A fit that ends scoring below
  its seed is refused: its strain is NaN.
- **A line search** tries multiples of each Gauss-Newton step, because a plain
  step closes only about half of what is left (the simulation is sharper than
  the pattern). Three iterations with it did what seven plain ones did; on a
  15 kV silicon indent the four strongest strains reached 0.94-1.05 of
  CrossCourt's HR-EBSD values.

On CUDA with CuPy the fit instead runs a fixed schedule of passes in one launch,
one block per pattern, keeping the best-scored point it visits: about 9 ms per
256 patterns on a Titan X, against about 42 ms for the torch fit.

**The pattern center must come from the geometry.** On a flat screen, moving the
center is exactly a linear map of the ray directions, so within one pattern a
center error is indistinguishable from a strain. Each pattern may be given its
own center (:meth:`~de_ebsd.index.LiveIndexer.center_maps`), but that center has
to come from where the beam was on the specimen and the detector geometry -- a
center fitted to each pattern on its own would absorb the very strain being
measured.

Against a reference pattern
---------------------------

:meth:`~de_ebsd.index.LiveIndexer.strain_from_reference` is HR-EBSD: it
cross-correlates regions of each pattern against one real reference frame
(``hrebsd``) and solves for the strain and lattice rotation relative to it. No
master enters, so how well the simulation matches the camera does not limit it.
In exchange it measures only differences from the reference, which should
itself be strain-free, and only within the reference's grain: the remap holds
for lattice rotations of a few degrees.

On a twin scan with a random 3e-3 strain at a 128 x 128 refine, the error
relative to a reference in each grain was 1.7e-3, against 7.3e-4 absolute from
the fit against the master; correlating regions against simulations instead was
6e-3. The fit against the master measures absolute strain everywhere it indexes;
a reference is the independent check where a strain-free region is known.
