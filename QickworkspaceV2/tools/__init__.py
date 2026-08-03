"""Tools package."""
from .fitting import *  # noqa

_SYSTEM_TOOL_EXPORTS = {
    "get_next_filename_labber",
    "hdf5_generator",
    "config_to_yaml",
    "auto_unit",
}

_HDF5_EXPORTS = {
    "ExperimentReference",
    "ValidationReport",
    "generate_experiment_id",
    "validate_experiment_id",
    "save_result",
    "load_result",
    "inspect_file",
    "validate_file",
    "find_experiments",
    "rebuild_catalog",
    "convert_labber_file",
}

_ARCHIVE_EXPORTS = {
    "ArchiveScanSummary",
    "CatalogManager",
    "ExperimentArchive",
    "ExperimentReader",
    "ExperimentRecord",
    "LabeledArray",
    "PlotRegistry",
    "default_plot_registry",
}


def __getattr__(name):
    if name in _SYSTEM_TOOL_EXPORTS:
        from .system_tool import get_next_filename_labber, hdf5_generator, config_to_yaml, auto_unit

        exports = {
            "get_next_filename_labber": get_next_filename_labber,
            "hdf5_generator": hdf5_generator,
            "config_to_yaml": config_to_yaml,
            "auto_unit": auto_unit,
        }
        globals().update(exports)
        return exports[name]
    if name in _HDF5_EXPORTS:
        from . import hdf5_store

        value = getattr(hdf5_store, name)
        globals()[name] = value
        return value
    if name in _ARCHIVE_EXPORTS:
        from . import data_archive

        value = getattr(data_archive, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_SYSTEM_TOOL_EXPORTS | _HDF5_EXPORTS | _ARCHIVE_EXPORTS)
