============
Contributing
============

Running the checks
==================

.. code-block:: shell

    uv sync --group dev
    uv run pytest -q
    uv run ruff check .

The suite runs on CPU in a few minutes. Tests that need an optional display or
GPU dependency skip themselves rather than fail, so a clean CPU run is a real
pass.

What a physics change has to prove
==================================

de-ebsd's claim is parity with EMsoft, so a change to the simulation chain is
not done when it runs -- it is done when it still reproduces the oracle:

.. code-block:: shell

    uv run python scripts/validate_master_pattern.py --material ni --backend bloch_prism

Nickel uses kikuchipy's bundled master pattern and needs no external files.
Silicon needs a local EMsoft ``Si_MC.h5`` run, which is not redistributed here;
point ``--mc`` or ``DE_EBSD_SI_MC`` at your copy. :doc:`validation` records the
numbers a change is expected to hold.

Round-trip tests are the other half. Anything touching orientation conventions,
Lambert storage or detector geometry must keep the closed round trip green:
project a master at a known orientation, index it back, and land within the
grid cell. A convention error shows up there as a 90-to-180-degree miss, not as
a small regression.

Adding an example
=================

Examples live in ``examples/<layer>/`` and are executed during the docs build
with ``abort_on_example_error``, so a broken example breaks the build.

1. Drop a ``.py`` file in ``examples/simulation/`` or ``examples/indexing/``,
   written in the Sphinx-Gallery format: a module docstring with the title,
   then ``# %%`` cells with reStructuredText comment blocks.
2. Add its filename to ``EXAMPLE_ORDER`` in ``docs/_ext/gallery_order.py``.
   Filenames carry no numeric prefix, so that list is what fixes the order;
   anything unlisted sorts alphabetically at the end.
3. Keep it CPU-practical -- a few tens of seconds. Small grids and coarse
   ``dmin`` are teaching settings, and every example says so.
4. End each cell that should render a figure with the figure as its final
   expression; the gallery captures it through ``_repr_html_``.

Build the site to check it:

.. code-block:: shell

    uv sync --group docs
    uv run python -m sphinx -b html -W --keep-going docs docs/_build/html

Design notes
============

The reasoning that does not belong in docstrings lives here. Read the relevant
page before changing a layer.

.. toctree::
    :maxdepth: 1

    architecture
    algorithm
    monte_carlo
    indexing
    validation
