# Lab tools

This package has no imports of QICK or `QickworkspaceV2`. It is installed with
the workspace, and can also be used directly by other scientific scripts.

```text
labtools/
  fitting/functions.py   Equations and individual fit functions, defaults and bounds
  fitting/solver.py      Callable-based numerical optimizer and diagnostics
  fitting/result.py      Fit result value object
  fitting/fit_n_res.py   Complex-IQ broadband resonator detection
  hdf5/                 Named-array HDF5 reader/writer and atomic replacement
  labber/               Labber schema and saver
  catalog.py            Rebuildable SQLite file index
  serialization.py      Portable JSON metadata encoding
```

Experiment-specific trace selection, final plots and calibration update rules
remain in the SDK. Each equation stays beside its fit function. The solver does
not select models or own physical starting values and bounds.

## Save and search

```python
from labtools.labber import save_labber
from labtools.catalog import find_experiments, rebuild_catalog

LABBER_PATH = r"D:\Labber_Data\chip"
path = save_labber(result, directory=LABBER_PATH)
rows = find_experiments(data_root=LABBER_PATH, qubit="Q1", experiment="t1_ge",
                        start="2026-09-01", tags=["diagnostic"])
report = rebuild_catalog(LABBER_PATH)
print(report["indexed"], report["errors"])
```

`NotebookLab.save_labber` calls this same saver. Successful Labber exports are
automatically indexed in `LABBER_PATH/catalog.sqlite`. If an explicit file path
is used, the default catalog root is its parent; `catalog_root=...` overrides
that choice. `catalog=False` disables registration. All indexed files must be
inside their catalog root. Native files can be indexed with
`result.save(path, catalog_root=...)`, or the standalone
`save_hdf5(path, record, traces, catalog_root=...)` function.

The SQLite `files` table stores relative path, run ID, experiment, UTC timestamp,
targets, tags, acquisition status, analysis status, quality, format, data kind
and comment. Paths identify exports: IQ, shots and different targets from one
run coexist. IQ and shots remain in HDF5; SQLite contains no scientific arrays.
The worker's `runs.sqlite3` execution journal remains separate.

Rebuilding reads HDF5 metadata without loading arrays. Unreadable or unsupported
files are reported; scientific files are never deleted or rewritten. Hidden
staging files and linked directories are skipped. A failed index update warns
with the saved file path and does not discard the export. Fitting failure does
not change acquisition status. Unknown acquisition status stays unknown.

Labber publication inherits the destination folder's Windows permissions and
refuses to overwrite an existing export. Native saving supports atomic
checkpoint replacement. `read_hdf5(path)` reads standalone records and native
records embedded in Labber files without requiring the SDK.
