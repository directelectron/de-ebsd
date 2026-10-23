============
Architecture
============

The simulation chain composes small objects, in the manner of abTEM's
``Atoms -> Potential -> PlaneWave -> multislice -> Detector``:

==================  ===================================  ==========================================
abTEM               de-ebsd                              Responsibility
==================  ===================================  ==========================================
``Atoms``           ``Crystal``                          lattice, space group, asymmetric unit,
                                                         metric tensors, orbits
``Potential``       ``StructureFactors``                 complex ``U_g`` with absorption, ``V_0``,
                                                         wavelength
``PlaneWave``       ``Microscope`` and ``KVectorGrid``   beam energy and geometry; incident
                                                         directions on the Lambert grid
``multislice()``    ``MasterPatternBackend``             intensity per beam direction
measurement         ``MasterPattern``                    Lambert-stored sphere: ``show``,
                                                         ``sample``, ``project``, ``save``
``Detector``        ``EBSDDetector``                     pixel grid, pattern center, tilts,
                                                         direction cosines
==================  ===================================  ==========================================

``MasterPatternSimulator`` is the composition root. Its stages are inspectable
(``structure_factors()``, ``reflections(k)``, ``dynamical_matrix(k)``), and
``compute(device)`` returns the :class:`~de_ebsd.sim.MasterPattern`.

Layers
======

Each layer depends only on those above it, and :mod:`de_ebsd.core` imports
nothing from de-ebsd:

.. code-block:: text

    core/          frames, constants, coordinate maps
    crystal/       unit cells, scattering, structure factors
    diffraction/   reflection lists, Bethe partition, Bloch-wave backends
    sim/           master-pattern generation, detector geometry, Monte Carlo
    index/         orientation indexing and refinement (the inverse problem)
    viz/           the anyplotlib display layer

Design rules
============

- Stages are plain data objects; the simulator owns caching and wiring.
- Setup-time math (structure factors, symmetry) is numpy float64; the
  per-direction hot path is torch and batched. The boundary is explicit:
  backends copy the structure-factor table to torch once, in ``prepare``.
- ``refine()`` returns a new simulator rather than mutating, and carries over
  every cache that is still consistent: raising ``npx`` or changing the Bethe
  thresholds never recomputes ``U_g``; changing ``dmin`` or the energy does.
- All plotting lives in :mod:`de_ebsd.viz`; anyplotlib is imported lazily so
  headless computation never needs it.

Backends
========

Every intensity model sits behind one seam,
:class:`~de_ebsd.diffraction.backends.MasterPatternBackend`: ``prepare`` once,
then ``intensities(directions)`` in chunks. The simulator owns gridding,
chunking and assembly; backends own the physics.

``bloch_prism`` (the default)
    Replaces the eigendecomposition with the physically equivalent scattering
    matrix ``exp(2 pi i A z / (2 k_n))`` (batched ``torch.linalg.matrix_exp``,
    pure matrix products) followed by a depth recurrence. Everything is batched:
    excitation errors, a candidate-restricted Bethe partition, the dynamical
    matrix with weak-beam folding, the propagation and the ``S_gh`` reduction,
    bucketed by strong-beam count so each batch is a uniform ``[B, S, S]``. It
    reproduces ``bloch`` to about 1e-13 in float64 and 1e-4 in float32.

``bloch``
    The float64 reference, one direction at a time. A non-Hermitian
    eigendecomposition neither batches nor accelerates on a GPU (cuSOLVER has no
    batched general-eigenvalue kernel), so this backend is CPU-bound and exists
    to validate ``bloch_prism``, not for production.

Symmetry reduction is independent of the backend: only the point group's
fundamental sector is sampled and the result is replicated, which is exact (the
folded grid points are bit-identical representatives) and about 16 times fewer
evaluations for a cubic crystal. The fold is cached per ``(npx, point group)``.

On a Pascal TITAN X, a 500-pixel silicon master (about two million directions)
takes about 10 s in float32 and 16 s in float64, against about 180 s with the
fold disabled.

Display
=======

Every stage object exposes ``.show(...)`` returning an
:class:`anyplotlib.Figure`, which is interactive in Jupyter through
``_repr_html_`` and exportable with ``to_html``. ``Crystal.overview`` returns a
host-neutral ``PhaseOverview``: JSON-compatible metadata plus one native
two-panel figure. ``Microscope.show`` returns an editable anywidget whose values
round-trip through ``to_microscope()`` and ``to_detector()``. The widgets use
Jupyter and host CSS variables with standalone fallbacks, so a host application
owns layout and figure lifetime while de-ebsd supplies figures and state.

Differentiability
=================

Gradients flow end to end from crystal parameters to intensity. The
``bloch_prism`` path is torch-native (``matrix_exp``, a depth recurrence and
``S_gh``), so gradients pass through the non-Hermitian complex solve back to
``U_g``. ``differentiable_ucg_lut`` assembles ``U_g`` in torch, and
``scattering_torch.wk_scattering_factor_torch`` ports the full
Weickenmeier-Kohl amplitude, so the master pattern is differentiable in atom
positions, occupancies and the Debye-Waller factor. Every stage is forward
bit-exact against the numpy path and every gradient is checked against finite
differences. Lattice-parameter gradients would additionally need the metric and
``g`` in torch.
