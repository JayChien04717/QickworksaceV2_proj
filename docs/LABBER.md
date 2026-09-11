# Labber export

V2 can export a completed or partial result without acquiring again. The HDF5 layout is adapted from the supplied `tools/Labber_saver.py`; it uses h5py directly and does not require the Labber Python package in the V2 environment.

```python
# Export the latest NotebookLab result beside its native acquisition file.
labber_path = lab.save_labber()

# Or export a specific result to an explicit filename.
labber_path = result.save_labber("data/resonator_Q1.hdf5")

# Multi-target data: select which target Labber displays.
labber_path = result.save_labber("data/Q2.hdf5", target="Q2")

# Display individual shots in Labber instead of averaged IQ.
labber_path = result.save_labber("data/Q1_shots.hdf5", data="shots")
```

`lab.save_labber(result, path=...)` accepts the same options. Optional `comment`, `tags` and `save_plot=False` control annotations and plot embedding. Explicit paths must end in `.hdf5`; existing files are never overwritten. With no path, the filename is `<experiment>_<target>_<iq|shots>.hdf5` beside the native result. Choose another filename for another export.

Every acquisition cell in `ChipCalibration.ipynb`, `QickworkspaceV2.ipynb` and `notebooks/advanced_measurements.ipynb` now calls the saver immediately after acquisition returns. `NotebookLab.save_labber(results)` handles procedure lists/dictionaries, readout-grid run IDs, scan children and all targets; without an explicit path it returns a mapping from `run_id/target` to exported file. A disabled procedure returning `None` is skipped. Direct Measurement/Session notebooks use `save_labber_results(result, load=lab.load)` (or `session.load`) for the same behavior. No filename loops or HDF5 formatting logic are needed in notebook cells.

Labber displays one complex signal channel, named `<target>_iq` or `<target>_shots`. The inner sweep is the first Labber step; other sweep dimensions and readout events remain separate steps. Shot export uses shot index as the inner step. Frequencies retain their recorded units and exact rounded/nonuniform coordinates. Categorical readout labels are stored as Labber combo definitions.

The complete V2 record is embedded under `/metagroup`, including **all targets**, raw complex IQ, captured shots, named dimensions/coordinates, resolved configuration, metadata and fit results. The final analysis PNG is saved at `/metagroup/plots/analysis.png`; if plotting fails, raw export still succeeds and `/metagroup` records `plot_error`. Labber's `Completed` flag means the export file is complete; inspect embedded acquisition status to distinguish interrupted measurements.

Export does not replace the native acquisition, alter `result.path`, update calibration or add a measurement. Loading and analysis continue to use the original V2 result/store. The embedded group uses the same schema as `ExperimentData.save()`; it is not a V1 result adapter.
