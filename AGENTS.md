# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What This Repo Is

`QickworkspaceV2/` is a fully self-contained automated quantum calibration framework built on QICK (Quantum Instruction Control Kit). It replaces the original `qick_workspace` codebase. **No file inside `QickworkspaceV2/` may import from `qick_workspace`** — all shared code is maintained as standalone copies inside `QickworkspaceV2/tools/`, `QickworkspaceV2/plotter/`, and `QickworkspaceV2/instruments/`.

## Verifying the Package

```bash
python -c "import QickworkspaceV2; print(QickworkspaceV2.__version__)"
python -c "from QickworkspaceV2 import BaseExperiment; print(BaseExperiment)"
```

There is no lint or type-check command. The hardware-independent suite runs in
the lab QICK environment:

```powershell
C:\Users\cluster\anaconda3\envs\qick_gui\python.exe -m unittest discover -s test -v
C:\Users\cluster\anaconda3\envs\qick_gui\python.exe -m compileall -q QickworkspaceV2 test
```

The default system `python` may not contain NumPy or QICK. Experiment validation
remains notebook-driven plus hardware-in-the-loop. `QICKBackend` and
`SimulatedBackend` have been removed; offline validation covers
hardware-independent analysis, configuration, persistence, data contracts, and
registry code.

## Architecture

The framework is layered:

```
experiments/ → core/ → QICK/Pyro4 hardware
                ↓
           analysis/ + tools/ + plotter/
                ↓
          calibration/ (store, graph, pipeline)
```

### Core Run/Fit/Save Flow

Every experiment inherits from `BaseExperiment` (`core/base_experiment.py`). The subclass contract:

- Set class attributes: `EXPT_NAME`, `TAG`, `X_LABEL`, `TITLE_PREFIX`, `SWEEP_KEYS_TO_REMOVE`
- Optionally bind `Analysis = SomeAnalysis` (a `BaseAnalysis` subclass)
- Override `_create_program()` → return a `BaseProgram` instance
- Override `_extract_sweep_axis(prog)` → return the x-axis numpy array
- Optionally override `_post_fit(x_vals)` → run fitting, populate `self.fit_params` / `self.fit_errors`

`run(py_avg, iq_process="all")` compiles the QICK program, streams live plots, acquires IQ data, calls `_post_fit`, runs `Analysis`, and returns `ExperimentData`. Analysis figures are opt-in through `plot_analysis=True` or a later `expt.plot()` call.

`ExperimentData` supports backward-compatible tuple unpacking (`fit_params, err = result`) and scalar coercion (`float(result)`), so old notebook code still works unchanged.

### QICK Programs

`BaseProgram` (`core/base_program.py`) wraps `AveragerProgramV2`. Key helper methods:
- `setup_resonator(cfg, prefix="ge")` / `setup_qubit_gen(cfg, prefix)`
- `setup_qb_pulse(cfg, prefix, ..., name, gain_key)` — declares a transition-aware named pulse
- `setup_standard_gates(cfg, prefix)` — registers `x180_ge`, `y180_ge`, `x90_ge`, etc.
- `apply_cool(cfg)` + `cooling_body()` — active-reset cooling
- `measure(cfg, pins=...)` — fire readout and collect ADC with optional marker pins
- `setup_active_reset(cfg)` + `activate_reset(cfg)` — configure and execute tProc feedback reset

In `_body()`, call `self.pulse(ch=..., name=..., t=0)` then `self.delay_auto(dt)` then `self.measure(cfg)`.

### Configuration

`ExperimentConfig` is defined in `tools/system_tool.py`; `config/system_cfg.py` contains the example `config_list`. Construct it as `ExperimentConfig(config_list)`. It flattens nested sub-dicts (`ch`, `res`, `qb`, `cooling`) into a single flat dict per qubit. Always call `.get_qubit("Q1")` to get a **copy** — mutation is safe and expected.

Critical flat config keys: `ro_ch`, `res_ch`, `qb_ch`, `reps`, `relax_delay`, `steps`, `res_freq_ge`, `qb_freq_ge`, `pi_gain_ge`, `qb_mixer`.

### Calibration Store

`CalibrationStore` (`calibration/store.py`) persists parameters as timestamped JSON. It is the live replacement for editing config dicts by hand:

```python
store = CalibrationStore("cal_Q1.json")
store.set("Q1", "qb_freq_ge", 4500.0)          # auto-saves
store.get("Q1", "qb_freq_ge")                   # → 4500.0
store.is_stale("Q1", "qb_freq_ge", max_age_hours=24)
store.update_from_dict("Q1", {...})
```

`AutoCalibrate` (`calibration/pipeline.py`) runs a 7-step ge-transition pipeline (res_spec → qubit_spec → power_rabi → t1 → ramsey → spin_echo → ss_opt) and writes results back into both `ExperimentConfig` and `CalibrationStore` automatically.

### IQ Data Convention

Raw hardware data remains complex in `ExperimentData.raw_iq`. `iq_process` controls the live/analysis display channel and is recorded in `ExperimentData.metadata`; `"all"` is the 1D default, while 2D views fall back to amplitude. `ExperimentData.y_axis` is the optional second sweep axis, not processed IQ data.

For programs with more than one readout per point, `raw_iq` remains the primary
analysis trace for backward compatibility. The complete matrix is stored in
`raw_data["readouts"]` with dimensions beginning in `"readout"`; use
`result.get_readout(index_or_label)` to retrieve one readout. Never place
processed IQ in `y_axis`.

### Hardware Test Safety

- Hardware tests may connect only to QICK/Pyro unless the user explicitly adds
  an instrument to the scope.
- Do not instantiate, read, or control Yoko, SGS100A, MG3692, or other RF-source
  drivers during QICK-only validation.
- Before acquisition, inspect every config key containing `gain`; every scalar
  gain and every sweep endpoint must be strictly below `0.1`.
- Keep cooling disabled unless it is the feature under test. A fitting failure
  without a connected chip is expected; validate program compilation and data
  shape instead.
- Connect through `BaseExperiment.connect_pyro4(...)`, using `ns_port` and
  `proxy_name`; do not bypass the framework session with a direct Pyro helper.

### Native HDF5 Storage

`ExperimentData.save()` delegates to `tools/hdf5_store.py`. The native v1 file is one experiment per HDF5 and preserves raw complex IQ, named axes/dimensions, analysis arrays, fit results, comments, tags, and lineage. New experiment runs do not copy `self.cfg` into `ExperimentData.config`; use `ExperimentConfig.to_yaml()` when configuration display is needed. Automatic saves use a UTC timestamp + random experiment ID and update a rebuildable `catalog.sqlite`. Existing paths are never overwritten. Keep `saveLabber()` for legacy notebooks; new storage code must not depend on Labber.

### Import Paths Inside `QickworkspaceV2/`

Depth-relative dot-count from package root:
- `experiments/<family>/` → 3 levels deep → `from ...core.base_program import BaseProgram`
- `analysis/` or `plotter/` → 2 levels → `from ..tools.fitting import fitlor`
- `calibration/` → 2 levels → `from ..config.system_cfg import ExperimentConfig`

Never use `from qick_workspace...` anywhere in `QickworkspaceV2/`. The fallback for `abcd_rf_fit` is `from ...tools.abcd_rf_fit.abcd_rf_fit import analyze`, not the qick_workspace path.

### Adding a New Experiment

1. Create `QickworkspaceV2/experiments/<family>/<name>.py`
2. Define `class MyProgram(BaseProgram)` with `_initialize` + `_body`
3. Define `class MyExperiment(BaseExperiment)` with the class attributes above
4. Export from `QickworkspaceV2/experiments/<family>/__init__.py`
5. Optionally add an `Analysis` subclass in `QickworkspaceV2/analysis/`
