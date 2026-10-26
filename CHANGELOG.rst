=========
Changelog
=========

All notable changes to de-ebsd are recorded here. Entries are written per pull
request as fragment files under ``upcoming_changes/`` and assembled at release
time by `towncrier <https://towncrier.readthedocs.io/>`_ — see
``upcoming_changes/README.rst``.

.. towncrier release notes start

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
