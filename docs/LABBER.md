# Labber export

V2 can export a completed or partial result without acquiring again. The HDF5 layout is adapted from the supplied `tools/Labber_saver.py`; it uses h5py directly and does not require the Labber Python package in the V2 environment.

```python
# Export to an independent Labber directory.
LABBER_PATH = "labber_data"  # Or an absolute path on your preferred drive.
labber_path = lab.save_labber(directory=LABBER_PATH)

# Or export a specific result to an explicit filename.
labber_path = result.save_labber("labber_data/resonator_Q1.hdf5")

# Multi-target data: select which target Labber displays.
labber_path = result.save_labber("labber_data/Q2.hdf5", target="Q2")

# Display individual shots in Labber instead of averaged IQ.
labber_path = result.save_labber("labber_data/Q1_shots.hdf5", data="shots")
```

`lab.save_labber(result, path=...)` accepts the same options. Optional `comment`, `tags` and `save_plot=False` control annotations and plot embedding. Explicit paths must end in `.hdf5`; existing files are never overwritten. With neither path nor directory, the filename is `<experiment>_<target>_<iq|shots>.hdf5` beside the native result. Choose another filename for another export.

Every acquisition cell in `ChipCalibration.ipynb`, `QickworkspaceV2.ipynb` and `notebooks/advanced_measurements.ipynb` now calls the saver immediately after acquisition returns. `NotebookLab.save_labber(results)` handles procedure lists/dictionaries, readout-grid run IDs, scan children and all targets; without an explicit path it returns a mapping from `run_id/target` to exported file. A disabled procedure returning `None` is skipped. Direct Measurement/Session notebooks use `save_labber_results(result, load=lab.load, directory=LABBER_PATH)` (or `session.load`) for the same behavior. No filename loops or HDF5 formatting logic are needed in notebook cells.

Labber displays one complex signal channel, named `<target>_iq` or `<target>_shots`. The inner sweep is the first Labber step; other sweep dimensions and readout events remain separate steps. Shot export uses shot index as the inner step. Frequencies retain their recorded units and exact rounded/nonuniform coordinates. Categorical readout labels are stored as Labber combo definitions.

The complete V2 record is embedded under `/metagroup`, including **all targets**, raw complex IQ, captured shots, named dimensions/coordinates, resolved configuration, metadata and fit results. The final analysis PNG is saved at `/metagroup/plots/analysis.png`; if plotting fails, raw export still succeeds and `/metagroup` records `plot_error`. Labber's `Completed` flag means the export file is complete; inspect embedded acquisition status to distinguish interrupted measurements.

Export does not replace the native acquisition, alter `result.path`, update calibration or add a measurement. Loading and analysis continue to use the original V2 result/store. The embedded group uses the same schema as `ExperimentData.save()`; it is not a V1 result adapter.

Notebook experiment cells pass the independently editable LABBER_PATH to directory=. Directory exports use YYYY/MM/Data_MMDD/<experiment>_<target>_<kind>_<run_id>.hdf5. Native acquisition stays in data. An explicit filename uses path= instead of directory=. Existing results retained across notebook source reloads can be exported without reacquisition.

## File search catalog

Labber exports now register automatically in `LABBER_PATH/catalog.sqlite`. Use `labtools.catalog.find_experiments(data_root=LABBER_PATH, qubit="Q1")` and `rebuild_catalog(LABBER_PATH)`. Native HDF5 can use `result.save(path, catalog_root=...)`. See [labtools](../labtools/README.md) for the independent file APIs, schema, error handling and query filters.

```python
from labtools.catalog import find_experiments, rebuild_catalog

rows = find_experiments(data_root=LABBER_PATH, qubit="Q1", experiment="t1_ge",
                        start="2026-09-01", acquisition_status="completed")
rebuild = rebuild_catalog(LABBER_PATH)
print(rebuild["indexed"], rebuild["errors"])
```

The file catalog is separate from the worker run journal and Labber GUI database. It stores metadata, not IQ/shots. Registration failures warn after preserving the HDF5. Rebuild reports unreadable/unsupported files without modifying them. `catalog=False` disables Labber registration; `catalog_root=...` overrides the index root for explicit file paths. Native records and embedded Labber records can both be read through `labtools.hdf5.read_hdf5(path)` or `ExperimentData.load(path)`.
