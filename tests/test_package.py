"""Check distribution metadata and every import path after the package rename."""

import importlib
import importlib.metadata
import pkgutil

import de_ebsd


def test_distribution_identity():
    distribution = importlib.metadata.distribution("de-ebsd")
    assert distribution.metadata["Name"] == "de-ebsd"
    assert distribution.version == de_ebsd.__version__


def test_all_submodules_import():
    modules = list(pkgutil.walk_packages(de_ebsd.__path__, prefix="de_ebsd."))
    assert modules
    for module in modules:
        imported = importlib.import_module(module.name)
        assert imported.__name__ == module.name


def test_lazy_public_exports():
    for package_name in ("de_ebsd", "de_ebsd.index", "de_ebsd.viz"):
        package = importlib.import_module(package_name)
        for name in package.__all__:
            assert getattr(package, name) is not None
