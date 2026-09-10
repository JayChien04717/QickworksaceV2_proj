# QickworkspaceV2

Self-contained QICK-based quantum calibration workspace for running resonator,
qubit, coherence, single-shot, TWPA, and calibration pipeline experiments.

This repository is intended to replace the older `qick_workspace` codebase. All
shared experiment, fitting, plotting, instrument, and calibration utilities live
inside this package.

## Features

- Unified experiment wrapper built around `BaseExperiment`
- QICK program helpers through `BaseProgram`
- Notebook-friendly experiment results through `ExperimentData`
- Resonator, qubit, coherence, setup, characterization, and TWPA experiments
- Calibration store and automated ge-transition calibration pipeline
- HDF5/Labber-style data saving helpers
- FastAPI service for async experiment and calibration jobs
- Instrument drivers for Yokogawa GS200, R&S SGS100A, and Anritsu MG3692
- `BaseInstrumentManager` for registering instruments, checking ranges, and
  printing live status/help in notebooks

## Repository Layout

```text
QickworkspaceV2/
  analysis/          Fit and quality-analysis helpers
  calibration/       Calibration store, graph, monitor, and pipeline
  config/            System and qubit configuration
  core/              BaseExperiment, BaseProgram, ExperimentData
  experiments/       Experiment implementations
  instruments/       PyVISA drivers and instrument manager
  plotter/           Live plotting and plot utilities
  service/           FastAPI REST service
  tools/             Fitting, data, scoring, and RF helper utilities

example/             Example HDF5 data and viewer scripts
tutorial/            Tutorial notebooks, including instrument manager tests
```

## Installation

Create an environment with the dependencies in `requirements.txt`:

```bash
pip install -r requirements.txt
```

For real hardware runs, the environment also needs QICK, Pyro4, and the proper
VISA backend for your lab computer.

## Quick Import Check

```bash
python -c "import QickworkspaceV2; print(QickworkspaceV2.__version__)"
```

## QICK Session Setup

In a notebook, connect once at the beginning of the session:

```python
from QickworkspaceV2 import BaseExperiment

soc, soccfg = BaseExperiment.connect_pyro4(
    ns_host="192.168.10.82",
    ns_port=8888,
    proxy_name="myqick",
    data_path=r"D:\Labber_Data\Jay\test",
)
```

If you already have `soc` and `soccfg`, register them directly:

```python
BaseExperiment.setup(soc, soccfg, data_path=r"D:\Labber_Data\Jay\test")
```

## Configuration

Use `ExperimentConfig` to get a mutable copy of one qubit's flattened config:

```python
from QickworkspaceV2 import ExperimentConfig

cfg_all = ExperimentConfig()
cfg = cfg_all.get_qubit("Q1")

cfg["reps"] = 1000
cfg["steps"] = 101
```

Important common keys include:

- `ro_ch`
- `res_ch`
- `qb_ch`
- `reps`
- `relax_delay`
- `steps`
- `res_freq_ge`
- `qb_freq_ge`
- `pi_gain_ge`
- `qb_mixer`

## Running Experiments

```python
from QickworkspaceV2 import ExperimentConfig
from QickworkspaceV2.experiments.resonator import ResonatorSpec

cfg = ExperimentConfig().get_qubit("Q1")
expt = ResonatorSpec(cfg)

result = expt.run(py_avg=5)
print(result.fit_result)
```

Live plotting shows all IQ views by default (`Abs`, `Phase`, `I`, and `Q`) for
1D sweeps. To force a single displayed channel, pass `iq_process`:

```python
result = expt.run(py_avg=5)                    # default: iq_process="all"
result = expt.run(py_avg=5, iq_process="abs")  # options: "abs", "phase", "real", "imag"
```

For 2D heatmaps and Yoko sweeps, `iq_process="all"` falls back to the amplitude
channel so the plot remains a single color map.

`ExperimentData` stores raw IQ data, x/y axes, fit parameters, quality flags,
and config metadata. It also keeps old notebook compatibility:

```python
fit_params, fit_errors = result
freq = float(result)
```

## Instrument Manager

Instrument drivers and the notebook-friendly `BaseInstrumentManager` live in
`QickworkspaceV2/instruments`. See `QickworkspaceV2/instruments/README.md` for
usage details, multi-instrument naming, safety limits, and the test notebook
workflow.

## Calibration Store

`CalibrationStore` persists calibrated values as timestamped JSON:

```python
from QickworkspaceV2 import CalibrationStore

store = CalibrationStore("cal_Q1.json")
store.set("Q1", "qb_freq_ge", 4500.0)
store.get("Q1", "qb_freq_ge")
store.is_stale("Q1", "qb_freq_ge", max_age_hours=24)
```

## Auto Calibration

`AutoCalibrate` runs the standard ge-transition pipeline and writes results back
to both `ExperimentConfig` and `CalibrationStore`:

```python
from QickworkspaceV2 import ExperimentConfig, CalibrationStore, AutoCalibrate

cfg_all = ExperimentConfig()
store = CalibrationStore("cal_Q1.json")

auto = AutoCalibrate(cfg_all, "Q1", cal_store=store)
auto.run()
auto.summary()
```

The default pipeline includes:

1. Resonator spectroscopy
2. Qubit spectroscopy
3. Power Rabi
4. Ramsey
5. Spin echo
6. T1
7. Single-shot optimization

## Data Saving

Experiments can save results through the modern `ExperimentData.save(...)` path
or the backward-compatible `saveLabber(...)` path:

```python
result.save("data/resonator_q1.h5")
expt.saveLabber(qb_idx=1)
```

## REST Service

Run the FastAPI service:

```bash
uvicorn QickworkspaceV2.service.api:app --host 0.0.0.0 --port 8000
```

Common endpoints:

- `POST /experiments/run`
- `GET /experiments/{id}/result`
- `POST /calibrate/{qubit}/run`

## Development Notes

- Do not import from the old `qick_workspace` package inside `QickworkspaceV2`.
- Keep shared code inside `QickworkspaceV2/tools`, `QickworkspaceV2/plotter`,
  and `QickworkspaceV2/instruments`.
- There is no full automated test suite yet. Validation is mostly notebook-based
  and hardware-in-the-loop.
- Use `SimulatedBackend` or offline smoke tests when hardware is unavailable.

## Import Examples

```python
from QickworkspaceV2 import (
    BaseExperiment,
    ExperimentConfig,
    CalibrationStore,
    AutoCalibrate,
)

from QickworkspaceV2.experiments.resonator import ResonatorSpec, Punchout
from QickworkspaceV2.experiments.qubit_ge import QubitSpec, PowerRabi
from QickworkspaceV2.experiments.coherence import T1, Ramsey, SpinEcho
from QickworkspaceV2.instruments import BaseInstrumentManager
```
