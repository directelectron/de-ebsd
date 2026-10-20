==========
User guide
==========

de-ebsd is layered, and each layer is usable on its own. The forward chain runs
:mod:`de_ebsd.core` (frames, constants) -> :mod:`de_ebsd.crystal` (unit cells,
structure factors) -> :mod:`de_ebsd.diffraction` (Bloch waves) ->
:mod:`de_ebsd.sim` (master patterns). :mod:`de_ebsd.index` runs the inverse
problem, matching measured patterns back to orientations, and
:mod:`de_ebsd.viz` draws every object through a single display layer.

Every intermediate stage is a real object you can inspect -- the structure
factors, the reflection list, the dynamical matrix at one beam direction, the
depth distribution -- rather than a hidden step inside a monolithic simulate
call. That is the point of the design, and the
:doc:`examples <../auto_examples/index>` are written to show it.

.. toctree::
    :maxdepth: 2

    installation
    conventions
    related_projects
    bibliography
