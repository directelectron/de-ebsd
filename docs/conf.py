"""Build the executable gallery with native anyplotlib HTML output.

The gallery is split into one subsection per de-ebsd layer (``examples/<layer>``,
each with its own ``README.rst`` heading). Example filenames carry no ordering
prefix, so both the subsection order and the order within each are declared here.
"""

import sys
from pathlib import Path

from sphinx_gallery.sorting import ExplicitOrder, FunctionSortKey

sys.path.insert(0, str(Path(__file__).parent / "_ext"))
from gallery_order import example_sort_key  # noqa: E402

project = "de-ebsd"
author = "Carter Francis"
extensions = ["sphinx_gallery.gen_gallery"]
root_doc = "index"
html_theme = "alabaster"
exclude_patterns = ["_build", "_ext"]

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r".*\.py$",
    "subsection_order": ExplicitOrder(["../examples/simulation", "../examples/indexing"]),
    "within_subsection_order": FunctionSortKey(example_sort_key),
    # anyplotlib Figure supplies _repr_html_; no custom renderer is needed.
    "capture_repr": ("_repr_html_", "__repr__"),
    "image_scrapers": (),
    "copyfile_regex": r".*\.cif$",
    "download_all_examples": True,
    "abort_on_example_error": True,
    "remove_config_comments": True,
}
