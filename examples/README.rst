Examples
========

Independent, executable examples in the Sphinx-Gallery format: a leading
reStructuredText description, executable Python, and commented narrative
sections. They run on a CPU without reference-data downloads or EMsoft, and are
grouped by the de-ebsd layer they exercise -- **simulation** builds master
patterns from a crystal, **indexing** matches measured patterns back to
orientations.

The small direction grids, electron counts and scans keep the examples practical
to execute; they are teaching settings, not evidence of converged experimental
accuracy. Each example selects its backend explicitly.

Run a script from the repository root::

    uv run python examples/simulation/cif_to_detector.py

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

The standalone ``cif_to_master_pattern.ipynb`` notebook covers the same ground
as the CIF example in notebook form. Download ``Si.cif`` beside a downloaded CIF
example notebook, or run it from the repository root. CIF lengths are in
angstroms; de-ebsd converts them to nanometres.
