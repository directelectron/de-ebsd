"""Explicit gallery ordering, in an importable module so Sphinx can pickle it.

Sphinx pickles its environment (``sphinx_gallery_conf`` included) between runs,
and a function defined inside ``conf.py`` has no importable qualified name, so
the pickle fails. Keeping the sort key here gives it one.
"""

# Teaching order within each gallery subsection. Example filenames carry no
# numeric prefix, so this list is what fixes the order; anything unlisted sorts
# alphabetically after the named files.
EXAMPLE_ORDER = (
    # simulation
    "cif_to_detector.py",
    "energy_resolved_simulation.py",
    "resolution_and_backends.py",
    "compare_emsoft.py",
    # indexing
    "dictionary_indexing.py",
    "spherical_indexing.py",
)


def example_sort_key(filename: str) -> tuple[int, str]:
    """Sort key placing ``filename`` at its declared position, else at the end."""
    rank = EXAMPLE_ORDER.index(filename) if filename in EXAMPLE_ORDER else len(EXAMPLE_ORDER)
    return (rank, filename)
