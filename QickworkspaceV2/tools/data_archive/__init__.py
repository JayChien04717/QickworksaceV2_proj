"""Browse, selectively read, and plot a Qickworkspace experiment archive."""

from .archive import ExperimentArchive
from .catalog import ArchiveScanSummary, CatalogManager
from .reader import ExperimentReader, LabeledArray
from .record import ExperimentRecord

_PLOT_EXPORTS = {"PlotRegistry", "default_plot_registry"}


def __getattr__(name):
    if name in _PLOT_EXPORTS:
        from . import plotting

        value = getattr(plotting, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ArchiveScanSummary",
    "CatalogManager",
    "ExperimentArchive",
    "ExperimentReader",
    "ExperimentRecord",
    "LabeledArray",
    "PlotRegistry",
    "default_plot_registry",
]
