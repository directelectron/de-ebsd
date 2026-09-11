Full simulation workflows
=========================

These independent examples follow the Sphinx-Gallery format: a leading
reStructuredText description, executable Python, and commented narrative
sections. They run on a CPU without reference-data downloads or EMsoft.
Each example explicitly selects a dynamical backend.

Start with the CIF-to-detector workflow, then explore energy-resolved Monte
Carlo statistics and numerical convergence. The small direction grids and
electron counts keep the examples practical to execute; they are teaching
settings, not evidence of converged experimental accuracy.

``plot_04_compare_emsoft.py`` compares against kikuchipy's bundled nickel
reference using its stored crystal, Bethe settings, reflection cutoff and
Monte Carlo depth profile. It reports correlations and normalized residuals
for both hemispheres and a detector projection, and displays both 3D spheres.
The reference is 8-bit, so the example does not claim bitwise intensity parity.
An unquantized local EMsoft file can be selected by editing ``reference_path``.

Run a script from the repository root::

    uv run python examples/plot_01_cif_to_detector.py

Build the gallery (including downloadable notebooks)::

    uv sync --group docs --group notebook
    uv run python -m sphinx -b html -W --keep-going docs docs/_build/html

Open ``docs/_build/html/index.html`` to view the results. In a plain Python
process, figures are constructed but no browser is opened automatically.
Use the generated notebook or gallery to interact with the plots, or call
``figure.save_html("my_figure.html")`` on a figure returned by ``.show()``.

Sphinx-Gallery captures the final expression in each code block through the
figure's ``_repr_html_`` method. In Jupyter, that same expression displays the
native anyplotlib widget. Camera controls work in exported HTML; Python
callbacks and editable setup widgets require a running notebook kernel.

The existing ``cif_to_master_pattern.ipynb`` remains available as a separate
notebook. Download ``Si.cif`` beside a downloaded CIF example notebook, or run
it from the repository root. CIF lengths are in angstroms; torchebsd converts
them to nanometres.
