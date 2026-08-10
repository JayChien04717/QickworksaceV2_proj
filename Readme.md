# QickworkspaceV2

Self-contained QICK-based quantum calibration framework for single-qubit,
multiplexed-readout, characterization, instrument-control, and automated
calibration workflows.

This package replaces the older `qick_workspace` codebase. Code inside
`QickworkspaceV2/` must not import from `qick_workspace`; shared implementations
live in this repository.

## Current capabilities

- Unified experiment lifecycle through `BaseExperiment` and `BaseProgram`
- Complex-IQ acquisition, live plotting, fitting, and quality analysis
- Typed experiment output through `ExperimentData`, including HDF5 save/load
- Resonator, ge/ef qubit, coherence, single-shot, tomography, AllXY, and RB
  experiments
- Multiplexed one-tone, two-tone, punchout, Rabi, Ramsey, T1, RB, tomography,
  and single-shot experiments in `experiments_mux/`
- Timestamped `CalibrationStore`, dependency graph, drift monitor, and the
  seven-step `AutoCalibrate` pipeline
- PyVISA drivers and a safety-limit-aware instrument manager for Yokogawa
  GS200, R&S SGS100A, and Anritsu MG3692 sources
- An experimental FastAPI service and a stable experiment registry for clients

The former `QICKBackend` and `SimulatedBackend` APIs are no longer part of this
project. Hardware sessions are configured directly on `BaseExperiment`.

## Repository layout

```text
QickworkspaceV2/
  analysis/          Fit and quality-analysis helpers
  calibration/       Store, dependency graph, monitor, and pipeline
  config/            Example system configuration
  core/              Experiment lifecycle, programs, results, composites
  experiments/       Single-qubit experiment implementations
  experiments_mux/   Multiplexed-readout experiments
  instruments/       PyVISA drivers and instrument manager
  plotter/           Live plotting and plot utilities
  tools/             Configuration, fitting, data, scoring, and RF helpers

example/             Example data and viewer scripts
tutorial/            Notebook tutorials
```

## Installation

This repository currently has no `pyproject.toml` or installable package
metadata. Run it from the repository root after installing its dependencies:

```bash
python -m pip install -r requirements.txt
python -c "import QickworkspaceV2; print(QickworkspaceV2.__version__)"
```

Real hardware runs also require a working QICK/Pyro4 setup and the VISA backend
used by the lab computer. Labber is optional and is not installed by
`requirements.txt`.

## Hardware session

Connect once near the beginning of a notebook. `data_path` is required because
legacy experiment saving uses the shared session output directory.

```python
from QickworkspaceV2 import BaseExperiment

soc, soccfg = BaseExperiment.connect_pyro4(
    ns_host="192.168.10.82",
    ns_port=8888,
    proxy_name="myqick",
    data_path=r"D:\Labber_Data\Jay\test",
)
```

If another tool has already created the QICK objects:

```python
BaseExperiment.setup(
    soc,
    soccfg,
    data_path=r"D:\Labber_Data\Jay\test",
)
```

There is currently no simulated backend. Offline work can test pure analysis,
configuration, serialization, calibration-store, and registry code, but an
experiment instance requires an initialized QICK session.

## Configuration

`ExperimentConfig` is normally constructed from a list of nested qubit configurations.
The repository provides `config_list` as an editable example:

```python
from QickworkspaceV2 import ExperimentConfig
from QickworkspaceV2.config.system_cfg import config_list

cfg_all = ExperimentConfig(config_list)
cfg = cfg_all.get_qubit("Q1")

# get_qubit() returns a flat, independent copy for a single run.
cfg["reps"] = 1000
cfg["steps"] = 101

# Persist a change in the multi-qubit config manager.
cfg_all.update("qb_freq_ge", 2872.65, q_index="Q1")
cfg_all.update("res.res_gain_ge", 0.2, q_index="Q1")
```

Common flattened keys include `ro_ch`, `res_ch`, `qb_ch`, `reps`,
`relax_delay`, `steps`, `res_freq_ge`, `qb_freq_ge`, `pi_gain_ge`, and
`qb_mixer`.

For multiplexed experiments, build a slot-stable configuration with
`cfg_all.muxconfig(["Q1", "Q3"])`; inactive resonator gains are set to zero
while tone/readout slots retain their physical indices.

## Running an experiment

```python
from qick.asm_v2 import QickSweep1D
from QickworkspaceV2 import ExperimentConfig
from QickworkspaceV2.config.system_cfg import config_list
from QickworkspaceV2.experiments.resonator import ResonatorSpec

cfg_all = ExperimentConfig(config_list)
cfg = cfg_all.get_qubit("Q1")
cfg.update({
    "steps": 101,
    "res_freq_ge": QickSweep1D("freqloop", 6707, 6727),
})

expt = ResonatorSpec(cfg)
result = expt.run(py_avg=5)

print(result.quality)
print(result.fit_result)
```

`BaseExperiment.run()` defaults to `iq_process="all"`. For a 1D sweep the live
view shows amplitude, phase, I, and Q. Choose one display channel when needed:

```python
result = expt.run(py_avg=5, iq_process="abs")
result = expt.run(py_avg=5, iq_process="real")
result = expt.run(py_avg=5, iq_process="imag")
result = expt.run(py_avg=5, iq_process="phase")
```

Two-dimensional plots fall back from `"all"` to amplitude. The acquired data
remains complex in `result.raw_iq`; `iq_process` selects plotting/analysis
presentation and is recorded in `result.metadata`.

Multi-readout experiments keep `result.raw_iq` as the primary analysis trace
and store the complete readout matrix in `result.raw_data["readouts"]`. Read a
specific trace by index or label:

```python
pre_reset = result.get_readout("pre_reset")
post_reset = result.get_readout("post_reset")
```

`result.y_axis` is reserved for a second sweep coordinate and never contains
processed IQ.

Analysis runs automatically when an experiment binds an `Analysis` class, but
the analysis figure is opt-in:

```python
result = expt.run(py_avg=5, plot_analysis=True)
# Or reuse the latest acquisition without touching hardware:
expt.plot()
```

`ExperimentData` also retains old notebook conveniences:

```python
fit_params, fit_errors = result
frequency = float(result)       # only when a scalar result is available
```

## Native HDF5 storage and catalog

`ExperimentData.save()` writes the native `qickworkspace.experiment` v1 format
with raw complex IQ, dimension-aware axes, fit/analysis results, comments, and
tags. Configuration remains managed by `ExperimentConfig`; use
`cfg_all.to_yaml(q_id=...)` when a formatted configuration is needed. When no
filename is supplied it creates a UTC-based unique
experiment ID, a date-organized filename, and a rebuildable SQLite catalog:

```python
from QickworkspaceV2 import ExperimentData
from QickworkspaceV2.tools.hdf5_store import find_experiments

path = result.save(
    data_root="data",
    comment="attenuator 調整後重新量測",
    tags=["cooldown-202607", "final"],
)
loaded = ExperimentData.load(path)

# SQLite is only a search index; no SQL knowledge is required.
runs = find_experiments(
    experiment_type="s008_T1_ge",
    qubit="Q1",
    tags=["final"],
    start="2026-07-01",
    data_root="data",
)
latest_t1 = runs[0].load()
```

To browse a Data path, query the SQLite catalog and load only the selected
experiment:

```python
from QickworkspaceV2.tools import find_experiments, load_result

runs = find_experiments(
    experiment_type="s008_T1_ge",
    qubit="Q1",
    data_root=r"D:\Labber_Data\Jay\test",
)
result = load_result(runs[0].path)
```

The offline examples in [`test/hdf5_reader_demo`](test/hdf5_reader_demo)
generate T1, Rabi, single-shot, RB, tomography, and SSH optimization files and
exercise indexing, selective raw reads, comments, tags, and plotting.

HDF5 remains the source of truth. If `catalog.sqlite` is deleted or data is
moved, rebuild it with `rebuild_catalog("data")`. Explicit paths remain
supported, but existing files are never overwritten. See
[`QickworkspaceV2/tools/HDF5_SCHEMA.md`](QickworkspaceV2/tools/HDF5_SCHEMA.md)
for the complete schema and the `inspect_file`, `validate_file`, and Labber
conversion APIs.

The legacy Labber-style saver remains available for existing notebooks:

```python
expt.saveLabber(qb_idx=1)
```

## Calibration

`CalibrationStore` persists values and update timestamps in JSON:

```python
from QickworkspaceV2 import CalibrationStore

store = CalibrationStore("data/calibrations.json")
store.set("Q1", "qb_freq_ge", 2872.65)
store.get("Q1", "qb_freq_ge")
store.is_stale("Q1", "qb_freq_ge", max_age_hours=24)
```

`AutoCalibrate` updates both the live `ExperimentConfig` and the optional
store. Its current default order is:

1. Resonator spectroscopy
2. Qubit spectroscopy
3. Power Rabi
4. T1
5. Ramsey frequency correction
6. Spin echo
7. Single-shot optimization

```python
from QickworkspaceV2 import AutoCalibrate, CalibrationStore, ExperimentConfig
from QickworkspaceV2.config.system_cfg import config_list

cfg_all = ExperimentConfig(config_list)
store = CalibrationStore("data/calibrations.json")

auto = AutoCalibrate(cfg_all, "Q1", cal_store=store)
auto.run(skip=("ss_opt",))
auto.summary()
```

The calibration package also exports `CalibrationGraph`, `CalibrationNode`, and
`CalibrationMonitor` for stale-only or scheduled workflows.

## Sequential and parallel composition

`BatchExperiment` runs experiments sequentially and can stop on a bad quality
flag. `ParallelExperiment` uses threads, so it must only be used when the
experiments truly have independent hardware resources; it does not arbitrate
access to a shared QICK board.

```python
from QickworkspaceV2 import BatchExperiment

batch = BatchExperiment(
    [("res", ResonatorSpec(cfg_res)), ("t1", T1(cfg_t1))],
    stop_on_bad=True,
)
results = batch.run(py_avg=5)
```

## Instrument manager

Use `BaseInstrumentManager` for named instruments, lab safety limits, and Yoko
ramps used by liveplot sweeps:

```python
from QickworkspaceV2.instruments import BaseInstrumentManager

inst = BaseInstrumentManager()
inst.add_yoko(
    "q1_flux",
    "GPIB0::1::INSTR",
    limits={"current": (-3e-3, 3e-3)},
)

result = expt.run(
    py_avg=5,
    instrument_manager=inst,
    yoko_name="q1_flux",
    yoko_value=flux_values,
    yoko_mode="current",
)
```

See [`QickworkspaceV2/instruments/README.md`](QickworkspaceV2/instruments/README.md)
for driver and manager details.

## Development and validation

- Never import `qick_workspace` from inside `QickworkspaceV2/`.
- Hardware-independent data, analysis, persistence, and experiment-contract
  tests run with `python -m unittest discover -s test -v`; program validation
  remains notebook- and hardware-in-the-loop driven.
- The experiment registry in `core/experiment_registry.py` is the stable list
  for generic clients. Experimental classes may exist without registry entries.
- Keep hardware-dependent imports lazy where possible so analysis and data
  utilities remain usable without QICK installed.

Useful imports:

```python
from QickworkspaceV2 import (
    AutoCalibrate,
    BaseExperiment,
    BatchExperiment,
    CalibrationStore,
    ExperimentConfig,
    ExperimentData,
    ParallelExperiment,
    QualityFlag,
)

from QickworkspaceV2.experiments.resonator import ResonatorSpec, Punchout
from QickworkspaceV2.experiments.qubit_ge import QubitSpec, PowerRabi
from QickworkspaceV2.experiments.coherence import T1, Ramsey, SpinEcho
from QickworkspaceV2.experiments_mux import MuxOneTone, MuxPowerRabi, MuxT1
```
