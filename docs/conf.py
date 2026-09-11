"""Sphinx configuration, following the pyxem / kikuchipy documentation model.

Same toolchain as those projects: the pydata theme, a recursive-autosummary API
reference built from the numpydoc docstrings, sphinx-design cards on the landing
page, and an executable Sphinx-Gallery.

The gallery is split into one subsection per de-ebsd layer (``examples/<layer>``,
each with its own ``README.rst`` heading). Example filenames carry no ordering
prefix, so both the subsection order and the order within each are declared here.
"""

import inspect
import subprocess
import sys
from pathlib import Path

from sphinx_gallery.sorting import ExplicitOrder, FunctionSortKey

import de_ebsd

sys.path.insert(0, str(Path(__file__).parent / "_ext"))
from gallery_order import example_sort_key  # noqa: E402

project = "de-ebsd"
author = "Carter Francis"
copyright = "2026, Carter Francis"  # noqa: A001 - Sphinx requires this name
version = release = de_ebsd.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx_codeautolink",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
]

root_doc = "index"
exclude_patterns = ["_build", "_ext"]
templates_path = ["_templates"]

# -- HTML ---------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = f"de-ebsd {version}"
html_theme_options = {
    "github_url": "https://github.com/CSSFrancis/de-ebsd",
    "header_links_before_dropdown": 6,
    "navigation_with_keys": False,
    "show_prev_next": False,
    "show_toc_level": 2,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/de-ebsd",
            "icon": "fa-solid fa-box",
        },
    ],
}
html_context = {"github_user": "CSSFrancis", "github_repo": "de-ebsd", "github_version": "main"}

# The gallery embeds large interactive anyplotlib figures; copying the page
# sources duplicates every one of them (~18 MB), and linkcode already points at
# the real source on GitHub.
html_copy_source = False
html_show_sourcelink = False

# -- API reference ------------------------------------------------------------

autosummary_generate = True
autosummary_ignore_module_all = False  # __all__ is the curated public surface
autodoc_typehints = "none"  # numpydoc renders the types from the docstrings
autodoc_default_options = {"show-inheritance": True}

numpydoc_show_class_members = False  # the class template lists them explicitly
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True
numpydoc_validation_checks = set()  # report-only linting is a separate concern

# wigner_d and wigner_D differ only in case, so their generated stub filenames
# collide on case-insensitive filesystems (Windows, macOS). Disambiguate both
# the re-exported and the submodule-qualified names.
autosummary_filename_map = {
    "de_ebsd.index.spherical.wigner_d": "de_ebsd.index.spherical.wigner_d_small",
    "de_ebsd.index.spherical.wigner_D": "de_ebsd.index.spherical.wigner_D_full",
    "de_ebsd.index.spherical.wigner.wigner_d": "de_ebsd.index.spherical.wigner.wigner_d_small",
    "de_ebsd.index.spherical.wigner.wigner_D": "de_ebsd.index.spherical.wigner.wigner_D_full",
}

# -- Cross-references ---------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "orix": ("https://orix.readthedocs.io/en/stable", None),
    "kikuchipy": ("https://kikuchipy.org/en/stable", None),
    "diffpy.structure": ("https://www.diffpy.org/diffpy.structure", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
}

bibtex_bibfiles = ["user/bibliography.bib"]
bibtex_reference_style = "author_year"

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

codeautolink_autodoc_inject = False
codeautolink_concat_default = True

# -- Gallery ------------------------------------------------------------------

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
    "doc_module": ("de_ebsd",),
    "backreferences_dir": "reference/generated/backreferences",
    "reference_url": {"de_ebsd": None},
}

# -- linkcode -----------------------------------------------------------------

_SRC = Path(__file__).resolve().parent.parent / "src"
try:
    _REVISION = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
except Exception:  # noqa: BLE001 - a tarball build has no git; fall back to the branch
    _REVISION = "main"


def linkcode_resolve(domain, info):
    """Point every documented object at its source line on GitHub."""
    if domain != "py" or not info["module"]:
        return None
    obj = sys.modules.get(info["module"])
    if obj is None:
        return None
    for part in info["fullname"].split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    obj = inspect.unwrap(obj)
    try:
        path = Path(inspect.getsourcefile(obj)).resolve().relative_to(_SRC)
        line = inspect.getsourcelines(obj)[1]
    except (TypeError, OSError, ValueError):
        return None
    return (
        f"https://github.com/CSSFrancis/de-ebsd/blob/{_REVISION}/src/"
        f"{path.as_posix()}#L{line}"
    )
