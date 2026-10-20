============
Installation
============

de-ebsd requires Python 3.11 or newer. The distribution is named ``de-ebsd``;
the importable package is ``de_ebsd``.

With uv
=======

``pyproject.toml`` pins torch to the CUDA 12.4 index, so a plain sync gets a
GPU-capable build on Windows and Linux:

.. code-block:: shell

    git clone https://github.com/CSSFrancis/de-ebsd.git
    cd de-ebsd
    uv sync

Optional groups
===============

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Group
      - What it adds
    * - ``--group dev``
      - ``pytest`` and ``ruff``, for running the test suite and the linter.
    * - ``--group docs``
      - Sphinx, the pydata theme and Sphinx-Gallery, to build this site.
    * - ``--group notebook``
      - JupyterLab and a kernel, to run ``examples/*.ipynb`` and the
        interactive ``.show()`` widgets.
    * - ``--extra validate``
      - ``kikuchipy``, needed only to load reference master patterns in the
        validation scripts.

For example, to build the documentation:

.. code-block:: shell

    uv sync --group docs --group notebook
    uv run python -m sphinx -b html -W --keep-going docs docs/_build/html

Verifying the install
=====================

.. code-block:: shell

    uv run python -c "import de_ebsd; print(de_ebsd.__version__)"
    uv run pytest -q

The suite runs on CPU. Tests that need an optional display or GPU dependency
skip themselves rather than fail.

Upgrading from ``torchebsd``
============================

The package was renamed from ``torchebsd`` to ``de_ebsd``. Update imports
(``import de_ebsd as de``) and run ``uv sync`` again. The validation scripts
read ``DE_EBSD_SI_MC`` and ``DE_EBSD_NI_MC`` instead of the ``TORCHEBSD_*``
environment variables. Restart any running Jupyter kernels. Saved
master-pattern HDF5 files are unchanged; Python pickles referring to the old
module path need migrating in the old environment. No legacy import alias is
installed.
