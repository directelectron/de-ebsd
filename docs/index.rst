=======
de-ebsd
=======

Differentiable, GPU-accelerated dynamical EBSD master-pattern simulation in
PyTorch, with a composable object-oriented API inspired by `abTEM
<https://github.com/abTEM/abTEM>`_.

A master pattern is the backscattered-electron intensity as a function of
crystal direction, stored on the square-Lambert grid. de-ebsd builds it from
first principles -- crystal, structure factors with absorption, reflections,
Bloch-wave dynamical matrix, depth-integrated dynamical solve -- with every
intermediate stage inspectable and the whole chain refinable. The physics is
ported from `EMsoft <https://github.com/EMsoft-org/EMsoftOO>`_
:cite:`callahan2013dynamical`, which remains the validation oracle; de-ebsd does
not call EMsoft at runtime. :mod:`de_ebsd.index` then matches measured patterns
back to orientations, by dictionary or by spherical cross-correlation
:cite:`lenthe2019spherical`.

.. grid:: 2
    :gutter: 2

    .. grid-item-card::
        :link: user/index
        :link-type: doc

        User guide :octicon:`book;1em;sd-text-info`
        ^^^

        Install de-ebsd, learn how the layers compose, and find the
        conventions the whole library depends on.

    .. grid-item-card::
        :link: auto_examples/index
        :link-type: doc

        Examples :octicon:`zap;1em;sd-text-info`
        ^^^

        Executable end-to-end workflows: simulate a master pattern from a CIF,
        and index a scan back to orientations.

    .. grid-item-card::
        :link: reference/index
        :link-type: doc

        API reference :octicon:`code;1em;sd-text-info`
        ^^^

        Every public module, class and function, generated from the
        docstrings, with links to the source.

    .. grid-item-card::
        :link: dev/index
        :link-type: doc

        Contributing :octicon:`people;1em;sd-text-info`
        ^^^

        Run the tests, add an example, and understand what validation a
        physics change has to pass.

Install
=======

Python 3.11 or newer. torch comes from the CUDA 12.4 index configured in
``pyproject.toml``:

.. code-block:: shell

    git clone https://github.com/directelectron/de-ebsd.git
    cd de-ebsd
    uv sync

See :doc:`user/installation` for CPU-only installs and the optional groups.

.. code-block:: python

    import de_ebsd as de

    crystal = de.Crystal.from_parameters(
        a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))]
    )
    microscope = de.Microscope(energy=20e3, sample_tilt=70.0)
    sim = de.MasterPatternSimulator(crystal, microscope, npx=500, dmin=0.05)
    master = sim.compute(device="cuda")

.. toctree::
    :hidden:
    :caption: User guide

    user/index

.. toctree::
    :hidden:
    :caption: Learning resources

    auto_examples/index

.. toctree::
    :hidden:
    :caption: Reference

    reference/index
    changelog

.. toctree::
    :hidden:
    :caption: Development

    dev/index
