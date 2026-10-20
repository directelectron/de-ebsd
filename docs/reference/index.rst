=============
API reference
=============

The public API of de-ebsd, generated from the docstrings. Each page links to
the source line on GitHub.

de-ebsd is layered, and the reference follows that layering: :mod:`de_ebsd.core`
(frames, constants) -> :mod:`de_ebsd.crystal` (unit cells, structure factors) ->
:mod:`de_ebsd.diffraction` (Bloch waves) -> :mod:`de_ebsd.sim` (master patterns),
with :mod:`de_ebsd.index` matching patterns back to orientations and
:mod:`de_ebsd.viz` drawing everything.

The names most workflows need are re-exported at the top level:

.. code-block:: python

    >>> import de_ebsd as de
    >>> de.Crystal, de.Microscope, de.MasterPatternSimulator, de.EBSDDetector
    (...)

.. currentmodule:: de_ebsd

.. rubric:: Modules

.. autosummary::
    :toctree: generated
    :template: custom-module-template.rst
    :recursive:

    core
    crystal
    diffraction
    sim
    index
    viz
