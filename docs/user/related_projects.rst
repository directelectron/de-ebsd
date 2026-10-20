================
Related projects
================

de-ebsd is a simulation and indexing library, not an analysis suite. It is
built to interoperate rather than replace.

`EMsoft <https://github.com/EMsoft-org/EMsoftOO>`_
    The Fortran reference implementation of dynamical EBSD simulation
    :cite:`callahan2013dynamical`. de-ebsd's physics is ported from it, and it
    remains the validation oracle -- but de-ebsd never calls it at runtime.

`kikuchipy <https://kikuchipy.org>`_
    Processing, simulation and analysis of EBSD patterns, built on HyperSpy.
    de-ebsd shares its square-Lambert and detector-geometry conventions, so
    master patterns pass between the two unchanged. kikuchipy's bundled nickel
    master pattern is the reference in
    ``examples/simulation/compare_emsoft.py``.

`orix <https://orix.readthedocs.io>`_
    Crystal orientations, symmetry and crystal maps. Every de-ebsd indexer
    takes and returns orix objects: :class:`~orix.quaternion.Rotation` in,
    :class:`~orix.crystal_map.CrystalMap` out.

`pyxem <https://pyxem.readthedocs.io>`_
    Multi-dimensional diffraction microscopy, and the wider ecosystem these
    conventions come from.

`abTEM <https://github.com/abTEM/abTEM>`_
    Not an EBSD package, but the source of de-ebsd's API philosophy: composable
    objects whose intermediate stages are inspectable, rather than one opaque
    simulate call.
