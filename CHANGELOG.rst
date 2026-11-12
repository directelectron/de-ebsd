=========
Changelog
=========

All notable changes to de-ebsd are recorded here. Entries are written per pull
request as fragment files under ``upcoming_changes/`` and assembled at release
time by `towncrier <https://towncrier.readthedocs.io/>`_ — see
``upcoming_changes/README.rst``.

.. towncrier release notes start

0.5.0 (2026-11-12)
==================

Bug Fixes
---------

- DynamicalIndexer now holds a lock of its own around every call into the live indexer, or the one passed as *lock*. Two threads indexing with one indexer at once, such as a live display and a scan, overwrote each other's rows in the shared device workspace and scored far outside a correlation (up to 1e12, or -inf), corrupting the orientations and scores they returned.


0.4.0 (2026-11-08)
==================

API and Behaviour Changes
-------------------------

- :func:`~de_ebsd.index.phase_from_crystal` gave its orix phase a lattice in Angstrom, the unit orix and diffsims compute d-spacings in, where it used to pass the crystal's nanometres through, and kept a hexagonal or trigonal crystal's atoms at their own fractional coordinates, where orix had moved them off their sites.
- Atoms were drawn in Jmol's element colors, from one table (:data:`~de_ebsd.viz.ELEMENT_COLORS`, 37 elements), so the 14 elements de-ebsd colored before changed color, and :func:`~de_ebsd.viz.patterns.crystal_structure_figure` drew a cell's atoms on its faces, edges and corners as well as the ones inside it.


New Features
------------

- :class:`de_ebsd.crystal.Phase` is a phase as an application lists it -- a name, its CIF and the orix structure read through :func:`~de_ebsd.index.io.phase_from_cif` -- and :mod:`de_ebsd.crystal.cod` searches the Crystallography Open Database by elements (one row per structure, the most-studied first) and fetches an entry's CIF, with the standard library only. :data:`de_ebsd.viz.PHASE_COLORS` and :func:`~de_ebsd.viz.phase_color` are the phase palette SpyDE and Autopilot share.
- :class:`~de_ebsd.Setup` described a microscope set-up once -- a :class:`~de_ebsd.Microscope` with an :class:`~de_ebsd.EBSDDetector` or the new :class:`~de_ebsd.TEMDetector` (camera length, screen width, stage tilts) -- with :data:`~de_ebsd.sim.microscope.FIELDS` giving its editable numbers' labels, units and ranges for a user interface, ``detector_to_sample`` on both detectors, and :mod:`de_ebsd.io.hyperspy` reading and writing it in a HyperSpy signal's metadata; :meth:`~de_ebsd.diffraction.kinematic.KinematicPatternModel.spots_on` placed kinematic spots on a TEM screen in pixels, and :func:`~de_ebsd.viz.unit_cell` gave a crystal's cell with its face, edge and corner atoms.
- :class:`~de_ebsd.index.live.LiveIndexer` is the one multi-phase indexer for either physics: a phase's master may now be a :class:`~de_ebsd.sim.KinematicMaster`, sampled onto the Lambert grid by the new :meth:`~de_ebsd.sim.KinematicMaster.to_master_pattern`; frames may be CuPy (any DLPack) arrays, read in place, or one ``(H, W)`` frame; ``search_shape`` and ``refine_shape`` may be one number, the long side in the detector's aspect; :meth:`~de_ebsd.index.live.LiveIndexer.phase_entries` and :meth:`~de_ebsd.index.live.LiveIndexer.phase_rotations` count dictionary entries within a phase. :func:`~de_ebsd.index.live.fused_fit_available`, :func:`~de_ebsd.index.live.fitted_refine_steps_deg` and :func:`~de_ebsd.index.live.dynamical_indexing_available` say what the device can do, without reaching into private modules.
- :func:`~de_ebsd.index.search_pattern_center` found a scan's pattern center from its own patterns when it was not known to within the reach of :func:`~de_ebsd.index.determine_pattern_center`: a coarse-to-fine grid of candidate centers, each scored by how far each pattern's best dictionary match stands above the rest, then fitted from dictionary seeds by :func:`~de_ebsd.index.solve_pattern_center`, searched first around a given ``start=``, and against a blurred static background as well when the patterns may be one crystal's; :func:`~de_ebsd.index.looks_like_kikuchi` tells Kikuchi patterns from other frames before a search is spent on them, :func:`~de_ebsd.index.dictionary_scores` scores patterns against a dictionary too large to keep, and :func:`~de_ebsd.index.bin_pattern` bins patterns to a size.
- Add ``de_ebsd.pipeline``: the EBSD pipeline Autopilot runs (indexing, pattern-center calibration, refinement, crystal maps, the set-up, spots, unit cells, phases and the accelerator lock), moved here verbatim from Autopilot. It overlaps parts of the rest of de_ebsd; those are reconciled one behavior at a time.
- Kinematic Kikuchi bands are a second master beside the dynamical one: :class:`~de_ebsd.sim.KinematicMaster` draws :class:`~de_ebsd.sim.kinematic_bands.Reflectors` (:func:`~de_ebsd.sim.kinematic_bands.cubic_reflectors`, or :func:`~de_ebsd.sim.kinematic_bands.reflectors_from_phase` from a Crystal or an orix phase) and projects, builds a :class:`~de_ebsd.index.DictionaryIndexer` and is refined by :func:`~de_ebsd.index.refine_orientation` through the same geometry as a :class:`~de_ebsd.sim.MasterPattern`, with no Bloch-wave simulation; :func:`~de_ebsd.sim.kinematic_bands.band_lines` and :func:`~de_ebsd.sim.kinematic_bands.zone_axis_points` give its band overlay in detector pixels. :class:`~de_ebsd.index.DictionaryIndexer` gains :meth:`~de_ebsd.index.DictionaryIndexer.match` (the ``keep`` best entries per pattern), :meth:`~de_ebsd.index.DictionaryIndexer.from_patterns` (a dictionary already simulated), ``rotations=`` (any orientation set, such as :func:`~de_ebsd.index.sample_orientations`), ``dynamic_background=`` and ``progress=``; :func:`~de_ebsd.index.refine_orientation` takes ``progress=``; and :func:`~de_ebsd.index.orientation_similarity_map`, :func:`~de_ebsd.index.merge_phases` and :func:`~de_ebsd.index.average_dot_product_map` read an indexed scan. :func:`~de_ebsd.index.to_crystal_map` takes Bunge Euler angles, and its quality is optional.
- de-ebsd took over what SpyDE and Autopilot each carried: Bunge Euler conversions and a point-group-aware :func:`~de_ebsd.core.rotation.misorientation_deg`, the explicit turn into a scan's frame (:mod:`de_ebsd.core.frames`), :func:`~de_ebsd.core.device.default_device`, a ``lock=`` on every indexer and refinement for an application sharing one accelerator between threads, :func:`~de_ebsd.index.phase_from_cif`, and a public :func:`~de_ebsd.viz.ipf_colors`.


Bug Fixes
---------

- :meth:`MasterPattern.project <de_ebsd.sim.MasterPattern.project>` given a quaternion tensor that requires a gradient detached it, so a projection was not differentiable in the orientation; it is now.


0.3.0 (2026-11-05)
==================

New Features
------------

- ``de_ebsd.index.live.common.pool_into`` averages frames into a buffer the caller owns, at the refine shape; ``observe``, ``prepare`` and ``index`` take such a stack in place of the frames, so each frame is read at full resolution once. (`#21 <https://github.com/directelectron/de-ebsd/pull/21>`_)


Bug Fixes
---------

- A pattern the strain fit could not move kept its seed orientation instead of stopping its whole batch with a singular-matrix error. (`#18 <https://github.com/directelectron/de-ebsd/pull/18>`_)


Performance
-----------

- ``LiveIndexer.index`` allocates nothing on the device once it has run: the indexer holds its working buffers from when it is built, and ``index`` takes an ``out=`` for its results. A 1,024-frame ``index(fit=True)`` peaked at 32 MiB rather than 260 MiB on a Titan X, with the same results to the bit. (`#21 <https://github.com/directelectron/de-ebsd/pull/21>`_)


0.2.1 (2026-10-26)
==================

Bug Fixes
---------

- :meth:`~de_ebsd.crystal.Crystal.from_cif` counted each atom of a hexagonal crystal once, where hcp Ti used to come out with 6 or 12 atoms per cell and three or six times its density, and it read a rhombohedral structure given on rhombohedral axes, which it used to refuse. (`#16 <https://github.com/directelectron/de-ebsd/pull/16>`_)


0.2.0 (2026-10-23)
==================

API and Behaviour Changes
-------------------------

- Public arguments are named in full and the same way everywhere: ``pc``, ``pc0`` and ``pc_ref`` are ``pattern_center`` (and ``reference_pattern_center``, for ``hrebsd`` and ``remap_reference``) including ``EBSDDetector(pattern_center=...)``; ``lm_lambda`` and ``lam0`` are ``damping``; ``lr`` and ``pc_lr`` are ``learning_rate`` and ``pattern_center_learning_rate``; ``refine_pc`` is ``refine_pattern_center``; ``n_iters`` is ``n_iterations``; ``sf`` is ``structure_factors``; and the blur width of ``remove_dynamic_background`` is ``sigma``. ``refine_hr`` now leaves the pattern center alone unless ``refine_pattern_center=True``, like the other refiners, and ``partition_strong_weak`` no longer takes ``u_mod``.
- The package was renamed from ``torchebsd`` to ``de_ebsd`` (distribution ``de-ebsd``); there is no legacy import alias, so ``import torchebsd`` must become ``import de_ebsd``.
- ``de_ebsd.index.live`` is a package (``indexer``, ``background``, ``deformation``, ``calibration``); ``LiveIndexer`` keeps its methods, and the scan's static background is reached as ``indexer.background``. ``direction_cosines_from_pc`` is ``direction_cosines_from_pattern_center``.


New Features
------------

- :meth:`~de_ebsd.Crystal.explore` opens a STEM orientation explorer linking a band-line sphere, the crystal structure, and a projection of geometric Kikuchi lines and kinematic spots, with no master pattern needed. :func:`~de_ebsd.index.hrebsd` measures small orientation and strain shifts by cross-correlating patterns against a reference, and :func:`~de_ebsd.index.remap_reference` rotates a reference onto a neighboring pattern first. (`#3 <https://github.com/directelectron/de-ebsd/pull/3>`_)
- :class:`~de_ebsd.index.LiveIndexer` indexes EBSD patterns as a scan streams them, at over 8000 patterns a second on a GPU: a dictionary search against dynamical master patterns, then a refine of every orientation, with the static and dynamic backgrounds taken from the frames themselves. :meth:`~de_ebsd.index.LiveIndexer.refine_pattern_center` fits the pattern center shared by a batch to about 0.001, and :meth:`~de_ebsd.index.LiveIndexer.set_pattern_center` moves to a new center without rebuilding the dictionary. (`#4 <https://github.com/directelectron/de-ebsd/pull/4>`_)
- :meth:`~de_ebsd.MasterPatternSimulator.compute` gained a ``progress(done, total)`` callback, called as each batch of beam directions is simulated, so an application can show how far a long simulation has got. (`#7 <https://github.com/directelectron/de-ebsd/pull/7>`_)
- :meth:`~de_ebsd.index.LiveIndexer.index_with_strain` fits each live-indexed pattern's deviatoric strain together with its orientation, against the dynamical master, and :meth:`~de_ebsd.index.LiveIndexer.strain_from_reference` measures strain relative to a reference pattern. Frames are now divided by the static background rather than having it subtracted, and :meth:`~de_ebsd.index.LiveIndexer.restart_static` can start the static background from a search's frames and blur it for a single crystal.


Bug Fixes
---------

- :func:`~de_ebsd.index.determine_pattern_center` recovered the pattern center from a start 0.03 off on a real dynamical master, where it used to see no slope and stay put, and no longer failed with a singular matrix when a sampled direction landed on a pole; a new ``blur`` argument sets its coarse-to-fine schedule (``blur=None`` restores the old fit). (`#1 <https://github.com/directelectron/de-ebsd/pull/1>`_)
- A simulated master pattern recorded ``repr(crystal)`` in ``metadata["crystal"]`` where an EMsoft-imported one recorded the :class:`~de_ebsd.crystal.Crystal`, so :class:`~de_ebsd.index.DictionaryIndexer` could not derive the point group from a master de-ebsd had just produced.


Performance
-----------

- :class:`~de_ebsd.index.LiveIndexer` builds its search dictionary in place and scores it a chunk of entries at a time, so a low-symmetry phase needs about the dictionary's own size on the GPU rather than three times it (a monoclinic phase's 3° dictionary: 4.5 GB → 1.55 GB). A new ``dtype`` argument stores the dictionary in float16, the default on GPUs with tensor cores, which halves it again.


Removals
--------

- Removed ``SetupWidget`` (use ``MicroscopeWidget``, now also exported from ``de_ebsd``), ``StructureFactors.as_torch_lut``, ``KinematicPatternModel.band_hkl`` and ``Microscope.vacuum_wavelength``, none of which was used; ``orientation_from_zone_axis`` is no longer re-exported from ``de_ebsd.viz.explorer``.
- The geometry-only ``KinematicalBackend`` (``backend="kinematical"``) is removed, and ``MasterPatternSimulator`` now defaults to ``backend="bloch_prism"``, the accurate batched Bloch-wave solver; pass ``backend="bloch"`` for the one-direction-at-a-time float64 reference.


Documentation
-------------

- Dictionary and spherical indexing examples were added, each a closed round trip on a synthetic two-grain scan with known orientations, and the gallery was regrouped into ``examples/simulation`` and ``examples/indexing``.
- The documentation is built and published to GitHub Pages by a new Docs workflow, and the build no longer warns on a clean checkout.


Maintenance
-----------

- One implementation each of the rotation-vector conversion, the master-pattern lookup, the Levenberg-Marquardt damping schedule, device selection and the numerical floors; the version is read from ``de_ebsd.__version__``; the design notes moved into ``docs/dev``; NOTICE credits EMsoft, kikuchipy, orix and EMSphInx; the repository is formatted with ``ruff format``, checked in CI.
- Shared helpers (skew matrix, rotation-vector conversion, detector rays, row normalizer, test band master) now live in one place each instead of being copied between modules, and source docstrings no longer point at the planning notes.
- The viz widgets' embedded JavaScript moved into standalone ``.js`` files shipped as package data.
- de-ebsd was set up for release on PyPI, with a license file, a towncrier changelog and Prepare Release / Release workflows.


0.1.0
=====

Initial development version (not published).
