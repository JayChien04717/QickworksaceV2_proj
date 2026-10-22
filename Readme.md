# QickworkspaceV2

A self-contained QICK-based framework for superconducting-qubit
characterization and automated calibration: one experiment lifecycle, live
plotting, fitting, typed results, native HDF5 storage, and a calibration
pipeline that writes results back into the configuration.

QickworkspaceV2 replaces the older `qick_workspace` codebase and does not
depend on it.

## Features

- **One experiment lifecycle.** Every experiment is a `BaseExperiment` +
  `BaseProgram` pair: compile, live plot, acquire complex IQ, fit, analyse.
- **Experiment library.** Resonator, ge/ef qubit spectroscopy, Rabi,
  coherence (T1, Ramsey, spin echo), single-shot, AllXY, tomography,
  randomized benchmarking, cryoscope, cooling, and TWPA characterization.
- **Multiplexed readout.** One-tone, two-tone, punchout, Rabi, Ramsey, T1,
  single-shot, AllXY, RB, and tomography in `experiments_mux/`.
- **Typed results.** `ExperimentData` carries raw IQ, axes, fit results, and a
  quality flag, and still unpacks like the old `(fit_params, errors)` tuple.
- **Native HDF5 storage** with a searchable SQLite catalog and a local browser
  viewer.
- **Calibration.** Timestamped `CalibrationStore`, dependency graph, drift
  monitor, and a seven-step `AutoCalibrate` pipeline.
- **Instruments.** PyVISA drivers for Yokogawa GS200, R&S SGS100A, and Anritsu
  MG3692 behind a safety-limit-aware `BaseInstrumentManager`.

## Repository layout

```text
QickworkspaceV2/
  analysis/          Fit and quality analysis
  calibration/       Store, dependency graph, monitor, AutoCalibrate
  config/            Example system configuration (config_list)
  core/              Experiment lifecycle, programs, results, registry
  experiments/       Single-qubit experiments, one package per family
  experiments_mux/   Multiplexed-readout experiments
  instruments/       PyVISA drivers and instrument manager
  plotter/           Live plotting
  service/           Experimental FastAPI service
  tools/             Configuration, fitting, HDF5 store, RF helpers
test/                Hardware-independent unit tests
tutorial/            Step-by-step notebooks (start here)
hdf5_viewer/         Local web viewer for saved data
```

## Installation

There is no installable package yet; run from the repository root.

```bash
python -m pip install -r requirements.txt
python -c "import QickworkspaceV2; print(QickworkspaceV2.__version__)"
```

Hardware runs additionally need a reachable QICK board served over Pyro4 and,
for instruments, a VISA backend. Labber is optional and installed separately.

## Quick start

```python
from qick.asm_v2 import QickSweep1D

from QickworkspaceV2 import BaseExperiment, ExperimentConfig
from QickworkspaceV2.config.system_cfg import config_list
from QickworkspaceV2.experiments.resonator import ResonatorSpec

# 1. Connect once per notebook. data_path is required.
BaseExperiment.connect_pyro4(
    ns_host="192.168.10.82",
    ns_port=8888,
    proxy_name="myqick",
    data_path=r"D:\Labber_Data\Jay\test",
)

# 2. Get a flat, independent config copy for one qubit.
cfg_all = ExperimentConfig(config_list)
cfg = cfg_all.get_qubit("Q1")
cfg.update({
    "steps": 101,
    "res_freq_ge": QickSweep1D("freqloop", 6707, 6727),
})

# 3. Run.
result = ResonatorSpec(cfg).run(py_avg=5)
print(result.quality, result.fit_result)
```

If the QICK objects already exist, use
`BaseExperiment.setup(soc, soccfg, data_path=...)` instead of `connect_pyro4`.

There is no simulated backend. Analysis, configuration, storage, and
calibration-store code work offline; constructing and running an experiment
requires a live QICK session.

## Configuration

`ExperimentConfig` takes a list of nested per-qubit dicts and flattens them.

```python
cfg_all = ExperimentConfig(config_list)
cfg = cfg_all.get_qubit("Q1")              # independent copy, safe to edit
cfg["reps"] = 1000

cfg_all.update("qb_freq_ge", 2872.65, q_index="Q1")     # persistent change
cfg_all.update("res.res_gain_ge", 0.2, q_index="Q1")

print(cfg_all.to_yaml(q_id="Q1"))
```

Common keys: `ro_ch`, `res_ch`, `qb_ch`, `reps`, `relax_delay`, `steps`,
`res_freq_ge`, `qb_freq_ge`, `pi_gain_ge`, `qb_mixer`.

For multiplexed experiments, `cfg_all.muxconfig(["Q1", "Q3"])` builds a
slot-stable configuration: unused resonator gains are zeroed while tones and
readout channels keep their physical indices.

## Working with results

```python
result = expt.run(py_avg=5)                  # live view: amp, phase, I, Q
result = expt.run(py_avg=5, iq_process="abs")   # or "real", "imag", "phase"
result = expt.run(py_avg=5, plot_analysis=True) # also draw the fit figure
expt.plot()                                  # re-plot without new acquisition

result.raw_iq          # always complex
result.y_axis          # second sweep axis (2D), never processed IQ
result.get_readout("post_reset")   # one trace of a multi-readout program

fit_params, fit_errors = result      # legacy tuple unpacking
frequency = float(result)            # when a scalar result exists
```

`iq_process` only changes what is displayed and analysed; it is recorded in
`result.metadata`. Two-dimensional plots fall back to amplitude.

## Saving and finding data

`result.save()` writes one HDF5 file per experiment (raw IQ, axes, fit and
analysis results, comments, tags) and indexes it in `catalog.sqlite`. Existing
files are never overwritten.

```python
from QickworkspaceV2 import ExperimentData
from QickworkspaceV2.tools import find_experiments, load_result

path = result.save(
    data_root="data",
    comment="re-measured after attenuator change",
    tags=["cooldown-202607", "final"],
)
loaded = ExperimentData.load(path)

runs = find_experiments(
    experiment_type="s008_T1_ge",
    qubit="Q1",
    tags=["final"],
    start="2026-07-01",
    data_root="data",
)
latest_t1 = load_result(runs[0].path)
```

HDF5 is the source of truth; if the catalog is lost or data is moved, run
`rebuild_catalog("data")`. The full schema and the `inspect_file`,
`validate_file`, and Labber-conversion helpers are documented in
[`QickworkspaceV2/tools/HDF5_SCHEMA.md`](QickworkspaceV2/tools/HDF5_SCHEMA.md).
Legacy notebooks can keep using `expt.saveLabber(qb_idx=1)`.

**Viewer.** Double-click `hdf5_viewer/start_hdf5_viewer.bat` to browse, plot,
and compare saved files at <http://127.0.0.1:8765>. It needs `fastapi`,
`uvicorn`, `h5py`, `numpy`, and `watchfiles`.

**Sample data.** `tutorial/read_tool/` generates an offline dataset (T1, Rabi,
single-shot, RB, tomography) for trying the catalog and viewer without
hardware.

## Calibration

```python
from QickworkspaceV2 import AutoCalibrate, CalibrationStore

store = CalibrationStore("data/calibrations.json")
store.set("Q1", "qb_freq_ge", 2872.65)
store.is_stale("Q1", "qb_freq_ge", max_age_hours=24)

auto = AutoCalibrate(cfg_all, "Q1", cal_store=store)
auto.run(skip=("ss_opt",))
auto.summary()
```

`AutoCalibrate` runs resonator spectroscopy -> qubit spectroscopy -> power
Rabi -> T1 -> Ramsey -> spin echo -> single-shot optimization, updating both
`cfg_all` and the store. `CalibrationGraph`, `CalibrationNode`, and
`CalibrationMonitor` support stale-only and scheduled recalibration.

## Batches

```python
from QickworkspaceV2 import run_batch

results = run_batch(
    [("res", ResonatorSpec(cfg_res)), ("t1", T1(cfg_t1))],
    py_avg=5,
    stop_on_bad=True,
)
```

`run_parallel()` uses threads and does not arbitrate access to a shared QICK
board; only use it for experiments on independent hardware.

## Instruments

```python
from QickworkspaceV2.instruments import BaseInstrumentManager

inst = BaseInstrumentManager()
inst.add_yoko("q1_flux", "GPIB0::1::INSTR", limits={"current": (-3e-3, 3e-3)})

result = expt.run(
    py_avg=5,
    instrument_manager=inst,
    yoko_name="q1_flux",
    yoko_value=flux_values,
    yoko_mode="current",
)
```

Details: [`QickworkspaceV2/instruments/README.md`](QickworkspaceV2/instruments/README.md).

## REST service (experimental)

```bash
uvicorn QickworkspaceV2.service.api:app --host 0.0.0.0 --port 8000
```

Generic clients should use the stable list in
`QickworkspaceV2/core/experiment_registry.py`.

## Tutorials

| Notebook | Topic |
| --- | --- |
| [00_quickstart](tutorial/00_quickstart.ipynb) | Session, config, first experiment |
| [01_config_and_store](tutorial/01_config_and_store.ipynb) | `ExperimentConfig` and `CalibrationStore` |
| [02_running_experiments](tutorial/02_running_experiments.ipynb) | Resonator, qubit, Rabi, T1, Ramsey |
| [03_batch_pipeline](tutorial/03_batch_pipeline.ipynb) | Sequential and parallel batches |
| [04_auto_calibrate](tutorial/04_auto_calibrate.ipynb) | `AutoCalibrate` pipeline |
| [05_custom_experiment](tutorial/05_custom_experiment.ipynb) | Writing your own experiment |
| [06_real_hardware](tutorial/06_real_hardware.ipynb) | QICK connection and saving |
| [07_data_management](tutorial/07_data_management.ipynb) | HDF5, catalog search, viewer |
| [08_instrument_manager](tutorial/08_instrument_manager.ipynb) | Yoko flux and RF sources |

## Writing a new experiment

1. Create `QickworkspaceV2/experiments/<family>/<name>.py`.
2. Subclass `BaseProgram` and implement `_initialize` and `_body`.
3. Subclass `BaseExperiment`; set `PROGRAM`, `X_AXIS` (and `Y_AXIS` for 2D),
   `EXPT_NAME`, and `TAG`; optionally bind an `Analysis` class.
4. Export it from the family's `__init__.py`.

See [`tutorial/05_custom_experiment.ipynb`](tutorial/05_custom_experiment.ipynb)
and [`AGENTS.md`](AGENTS.md) for the full contract.

## Development

```bash
python -m unittest discover -s test -v      # hardware-independent tests
python -m compileall -q QickworkspaceV2 test
ruff check QickworkspaceV2                  # PEP 8 lint (settings in ruff.toml)
ruff format QickworkspaceV2
```

- Never import `qick_workspace` inside `QickworkspaceV2/`.
- Keep hardware-dependent imports lazy so analysis and data tools work without
  QICK.
- During QICK-only hardware tests, keep every gain and gain-sweep endpoint
  below `0.1` and leave external instruments untouched (see `AGENTS.md`).
