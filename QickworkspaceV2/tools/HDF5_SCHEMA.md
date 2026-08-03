# Native HDF5 schema v1

Each file contains exactly one experiment. The HDF5 file is authoritative;
`catalog.sqlite` is a disposable search index that can be rebuilt from root
attributes with `rebuild_catalog()`.

```text
/
├── attrs: schema_name, schema_version, write_complete, experiment_id,
│          experiment_type, timestamp_utc, timestamp_local, data_kind,
│          analysis_id, plot_id, quality, interrupted
├── meta/
│   ├── comment
│   ├── tags
│   ├── config_json
│   ├── metadata_json
│   ├── provenance_json
│   └── lineage_json
├── axes/<name>/values       # label, unit, description, scale attrs
├── raw/<name>               # raw IQ and acquisition datasets
├── analysis/<name>          # fit curves, residuals, metrics, matrices
└── results/
    ├── fit_result_json
    ├── fit_params
    ├── fit_errors
    └── summary_json
```

Numeric arrays keep their original dtype. Arrays with at least 64 elements use
chunked gzip level-4 compression and shuffle. Complex IQ is stored directly as
a complex HDF5 dataset; magnitude and phase are calculated by readers.

Raw and analysis datasets may carry a JSON `dims` attribute. Every dimension
name must correspond to a group under `/axes`, in the same order as the dataset
shape. `validate_file()` checks these relationships.

Files are first written as `<name>.h5.partial` with `write_complete=false`.
After all content is flushed, the flag becomes true and the file is atomically
renamed. Existing completed paths are never overwritten.

Experiment IDs use UTC plus 64 random bits:

```text
20260714T073045123456Z-6Y2K9M8Q4R7WP
```

The timestamp portion makes IDs sortable. Experiment type, qubit, and tags are
not part of the ID and may be corrected without changing identity.

## API

```python
from QickworkspaceV2.tools.hdf5_store import (
    find_experiments,
    generate_experiment_id,
    inspect_file,
    load_result,
    rebuild_catalog,
    save_result,
    validate_file,
)
```

`ExperimentData.save()` and `ExperimentData.load()` delegate to these APIs.
Legacy `saveLabber()` methods remain available; `convert_labber_file()` is the
only native-storage function that imports the optional Labber SDK.

## Archive reader

`ExperimentArchive(data_root)` scans completed file metadata and maintains the
rebuildable catalog. Its `ExperimentRecord` results are lightweight. Use
`record.load_raw(name, selection=...)` or `record.load_analysis(name)` for a
single dataset, `record.load()` for the complete `ExperimentData`, and
`record.plot()` to dispatch through the stable `plot_id` registry.

Selective reads return `LabeledArray`, which carries `values`, `dims`, `axes`,
dataset attributes, and the original HDF5 dataset path.
