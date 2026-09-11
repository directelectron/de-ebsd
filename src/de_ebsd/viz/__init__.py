"""The anyplotlib display layer: every ``.show()`` in de_ebsd renders through here.

anyplotlib and the widget stack are imported lazily, so headless compute jobs
never pay for (or require) them.
"""

__all__ = [
    "DepthWidget",
    "MicroscopeWidget",
    "PhaseOverview",
    "SetupWidget",
    "bar_figure",
    "crystal_structure_figure",
    "hemispheres_figure",
    "image_figure",
    "phase_overview_figure",
    "sampling_sphere_figure",
    "show_correlation",
    "show_map",
    "show_match",
    "show_refinement",
    "sphere_figure",
    "to_uint8_rgb",
]

_MODULES = {
    "figures": ("bar_figure", "image_figure", "to_uint8_rgb"),
    "patterns": (
        "crystal_structure_figure",
        "hemispheres_figure",
        "phase_overview_figure",
        "sampling_sphere_figure",
        "sphere_figure",
    ),
    "maps": ("show_correlation", "show_map", "show_match", "show_refinement"),
    "views": ("PhaseOverview",),
    "widgets": ("DepthWidget", "MicroscopeWidget", "SetupWidget"),
}
_ORIGIN = {name: module for module, names in _MODULES.items() for name in names}


def __getattr__(name: str):
    module = _ORIGIN.get(name)
    if module is None:
        raise AttributeError(f"module 'de_ebsd.viz' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f"de_ebsd.viz.{module}"), name)


def __dir__() -> list[str]:
    return sorted(__all__)
