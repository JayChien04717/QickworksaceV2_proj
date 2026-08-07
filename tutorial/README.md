# QickworkspaceV2 tutorial notebooks

The experiment notebooks require a live QICK session. Before hardware cells,
set both the Pyro4 address and the required data directory:

```python
from QickworkspaceV2 import BaseExperiment

BaseExperiment.connect_pyro4(
    ns_host="192.168.10.82",
    ns_port=8888,
    proxy_name="myqick",
    data_path=r"D:\Labber_Data\Jay\test",
)
```

Configuration examples should construct `ExperimentConfig` from actual config
data:

```python
from QickworkspaceV2 import ExperimentConfig
from QickworkspaceV2.config.system_cfg import config_list

cfg_all = ExperimentConfig(config_list)
cfg = cfg_all.get_qubit("Q1")
```

There is no `QICKBackend` or `SimulatedBackend` in the current package. Pure
analysis, data loading, configuration, and calibration-store code can be used
offline; experiment construction and acquisition require the shared
`BaseExperiment` hardware session.

## Sequence

| Notebook | Topic |
| --- | --- |
| [00_quickstart.ipynb](00_quickstart.ipynb) | Session, config, and first experiment |
| [01_config_and_store.ipynb](01_config_and_store.ipynb) | `ExperimentConfig` and `CalibrationStore` |
| [02_running_experiments.ipynb](02_running_experiments.ipynb) | Resonator, qubit, Rabi, T1, and Ramsey |
| [03_batch_pipeline.ipynb](03_batch_pipeline.ipynb) | Sequential batches; threaded parallel API and hardware-safety warning |
| [04_auto_calibrate.ipynb](04_auto_calibrate.ipynb) | Seven-step `AutoCalibrate` pipeline |
| [05_custom_experiment.ipynb](05_custom_experiment.ipynb) | Custom `BaseProgram` and `BaseExperiment` |
| [06_real_hardware.ipynb](06_real_hardware.ipynb) | QICK connection and saving |
| [07_data_management.ipynb](07_data_management.ipynb) | Native HDF5 save/load, catalog search, validation, and viewer workflow |
| [08_instrument_manager.ipynb](08_instrument_manager.ipynb) | Instrument manager, Yoko flux, and RF sources |

The connection examples in these notebooks use the same required `data_path`
contract.

## Current data-management module

`ExperimentData.save()` now writes the native HDF5 v1 format through
`QickworkspaceV2.tools.hdf5_store`. Use these public helpers for offline data
work:

```python
from QickworkspaceV2.tools.hdf5_store import (
    find_experiments,
    inspect_file,
    load_result,
    rebuild_catalog,
    validate_file,
)
```

The generated `catalog.sqlite` is a disposable search index. HDF5 files remain
the source of truth and can rebuild the catalog with `rebuild_catalog(data_root)`.

The repository also includes a local viewer:

```bash
python hdf5_viewer/hdf5_viewer_server.py
```

Open `http://127.0.0.1:8765`, choose a data folder, and use **Rebuild** if the
catalog is missing or stale. On Windows, `hdf5_viewer/start_hdf5_viewer.bat` launches the
same server.

## Additional Notes

| Note | Topic |
| --- | --- |
| [T1_nonuniform_sweep.md](T1_nonuniform_sweep.md) | QICK Macro, DMEM/register concepts, and a non-uniform T1 time sweep implementation |

Current hardware notebooks import `config_list` from `QickworkspaceV2.config.system_cfg` and add run-specific `QickSweep1D` values on a copied qubit config.
