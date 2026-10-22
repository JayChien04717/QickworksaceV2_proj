# AGENTS.md

Operating instructions for coding agents (Claude Code, Codex, Cursor, Copilot,
and other tools that read the [AGENTS.md](https://agents.md) standard) working
in this repository. Read it before every task.

**Working code only. Finish the job. Plausibility is not correctness.**

---

## 1. Non-negotiables

These override everything else in this file when in conflict.

1. **Never fabricate.** No invented file paths, API names, config keys, or
   test results. If unsure, read the file, run the command, or say so.
2. **Disagree when you disagree.** If the user's premise is wrong, say so
   before doing the work.
3. **Stop when confused.** If a request has two plausible interpretations that
   materially change the result, ask instead of picking silently.
4. **Touch only what you must.** Every changed line must trace to the request.
   No drive-by refactors or reformatting of unrelated code.
5. **Hardware safety first.** Follow section 6 for any run that talks to QICK.

---

## 2. What this repo is

`QickworkspaceV2/` is a self-contained automated quantum calibration framework
built on QICK (Quantum Instruction Control Kit). It replaces the old
`qick_workspace` codebase.

**No file inside `QickworkspaceV2/` may import from `qick_workspace`.** Shared
code lives as standalone copies in `QickworkspaceV2/tools/`,
`QickworkspaceV2/plotter/`, and `QickworkspaceV2/instruments/`.

### Stack

- Python 3.12 in the lab `qick_gui` Conda environment
- QICK tProc v2, Pyro4, NumPy/SciPy, Matplotlib, h5py
- Dependencies: `requirements.txt` (Labber is optional and installed separately)
- Runtime: Windows notebook client connected to QICK hardware

### Layout

```text
QickworkspaceV2/
  analysis/          BaseAnalysis subclasses (fitting + quality)
  calibration/       CalibrationStore, graph, monitor, AutoCalibrate
  config/            Example config_list
  core/              BaseExperiment, BaseProgram, ExperimentData, registry
  experiments/       Single-qubit experiments, one package per family
  experiments_mux/   Multiplexed-readout experiments
  instruments/       PyVISA drivers and BaseInstrumentManager
  plotter/           Live plotting
  service/           Experimental FastAPI service
  tools/             Config, fitting, HDF5 store, Labber saver, RF helpers
test/                Hardware-independent unittest suite
tutorial/            Notebook tutorials
hdf5_viewer/         Local browser viewer for saved HDF5 data
```

Do not modify notebook outputs or legacy Labber compatibility without an
explicit need.

---

## 3. Commands

Use the `qick_gui` interpreter; the system `python` may lack NumPy or QICK.

```powershell
$PY = "C:\Users\cluster\anaconda3\envs\qick_gui\python.exe"

& $PY -m pip install -r requirements.txt                          # install
& $PY -c "import QickworkspaceV2; print(QickworkspaceV2.__version__)"  # package check
& $PY -m compileall -q QickworkspaceV2 test                       # compile check
& $PY -m unittest discover -s test -v                             # all tests
& $PY -m unittest discover -s test -p "test_name.py" -v           # one file
ruff check QickworkspaceV2                                        # PEP 8 lint
ruff format QickworkspaceV2                                       # PEP 8 format
```

- Ruff settings live in `ruff.toml` (format at 79 columns, lint limit 99,
  notebooks excluded, `__init__.py` import order preserved). There is no type
  checker.
- Prefer single-file test runs while iterating; run the full suite at the end.
- Experiment validation remains notebook-driven plus hardware-in-the-loop.
  `QICKBackend` and `SimulatedBackend` have been removed; offline tests cover
  analysis, configuration, persistence, data contracts, and the registry.

---

## 4. Architecture

```
experiments/ -> core/ -> QICK/Pyro4 hardware
                 |
        analysis/ + tools/ + plotter/
                 |
        calibration/ (store, graph, pipeline)
```

### Experiment contract

Every experiment inherits from `BaseExperiment` (`core/base_experiment.py`):

- Class attributes: `EXPT_NAME`, `TAG`, `X_LABEL`, `TITLE_PREFIX`,
  `SWEEP_KEYS_TO_REMOVE`
- `PROGRAM = MyProgram`
- `X_AXIS = SweepAxis.pulse(name, parameter)` or `SweepAxis.time(tag)`; add
  `Y_AXIS` for a 2D QICK sweep. Override the axis hooks only for computed or
  external axes.
- Optional `Analysis = SomeAnalysis` (a `BaseAnalysis` subclass)
- Optional `_post_fit(x_vals)` that fills `self.fit_params` / `self.fit_errors`

`run(py_avg, iq_process="all")` compiles the program, streams live plots,
acquires IQ, calls `_post_fit`, runs `Analysis`, and returns `ExperimentData`.
Analysis figures are opt-in (`plot_analysis=True` or `expt.plot()`).

`ExperimentData` keeps tuple unpacking (`fit_params, err = result`) and scalar
coercion (`float(result)`) so old notebooks keep working.

### QICK programs

`BaseProgram` (`core/base_program.py`) wraps `AveragerProgramV2`:

- `setup_resonator(cfg, prefix="ge")`, `setup_qubit_gen(cfg, prefix)`
- `setup_qb_pulse(cfg, prefix, ..., name, gain_key)` for transition-aware pulses
- `setup_standard_gates(cfg, prefix)` registers `x180_ge`, `y180_ge`, `x90_ge`, ...
- `apply_cool(cfg)` + `cooling_body()` for active-reset cooling
- `setup_active_reset(cfg)` + `activate_reset(cfg)` for tProc feedback reset
- `measure(cfg, pins=...)` fires readout and collects ADC

In `_body()`: `self.pulse(ch=..., name=..., t=0)`, then `self.delay_auto(dt)`,
then `self.measure(cfg)`.

### Configuration

`ExperimentConfig` lives in `tools/system_tool.py`; `config/system_cfg.py` has
the example `config_list`. `ExperimentConfig(config_list)` flattens the nested
`ch`, `res`, `qb`, `cooling` dicts per qubit. `.get_qubit("Q1")` returns a
**copy**, so mutating it is safe and expected.

Critical flat keys: `ro_ch`, `res_ch`, `qb_ch`, `reps`, `relax_delay`, `steps`,
`res_freq_ge`, `qb_freq_ge`, `pi_gain_ge`, `qb_mixer`.

### Calibration

`CalibrationStore` (`calibration/store.py`) persists values as timestamped
JSON (`set`, `get`, `is_stale`, `update_from_dict`; `set` auto-saves).
`AutoCalibrate` (`calibration/pipeline.py`) runs res_spec -> qubit_spec ->
power_rabi -> t1 -> ramsey -> spin_echo -> ss_opt and writes results into both
`ExperimentConfig` and the store.

### IQ data convention

- `ExperimentData.raw_iq` stays complex. `iq_process` only selects the
  display/analysis channel and is recorded in `metadata`; `"all"` is the 1D
  default, 2D falls back to amplitude.
- Multi-readout programs keep the primary trace in `raw_iq` and the full matrix
  in `raw_data["readouts"]` (first dimension `"readout"`); use
  `result.get_readout(index_or_label)`.
- `ExperimentData.y_axis` is the second sweep axis. Never put processed IQ there.

### HDF5 storage

`ExperimentData.save()` delegates to `tools/hdf5_store.py`: one experiment per
file, raw complex IQ, named axes, analysis arrays, fit results, comments, tags,
lineage. Auto-saves use a UTC timestamp + random ID and update a rebuildable
`catalog.sqlite`; existing paths are never overwritten. New runs do not copy
`self.cfg` into `ExperimentData.config` (use `ExperimentConfig.to_yaml()`).
Keep `saveLabber()` for legacy notebooks, but new storage code must not depend
on Labber. Full schema: `QickworkspaceV2/tools/HDF5_SCHEMA.md`.

---

## 5. Conventions

- **Imports:** package-relative, by depth from the package root:
  - `experiments/<family>/` -> `from ...core.base_program import BaseProgram`
  - `analysis/`, `plotter/` -> `from ..tools.fitting import fitlor`
  - `calibration/` -> `from ..tools.system_tool import ExperimentConfig`
  - `abcd_rf_fit` fallback: `from ...tools.abcd_rf_fit.abcd_rf_fit import analyze`
- **Style:** PEP 8 via ruff. Do not rename existing public names (e.g.
  `SingleShot_gef`, `Labber_saver`) to satisfy naming rules; notebooks depend
  on them.
- **Errors:** hardware capability errors are explicit; fitting failures yield
  `QualityFlag.BAD` with a message.
- **Tests:** `unittest`, mocks for QICK-independent acquisition, real HDF5
  round trips for storage.
- **Lazy imports:** keep hardware-dependent imports lazy so analysis and data
  utilities work without QICK.
- **Registry:** `core/experiment_registry.py` is the stable list for generic
  clients; experimental classes may exist without entries.

### Adding a new experiment

1. Create `QickworkspaceV2/experiments/<family>/<name>.py`.
2. Define `class MyProgram(BaseProgram)` with `_initialize` and `_body`.
3. Define `class MyExperiment(BaseExperiment)` with `PROGRAM` and `X_AXIS`
   (plus `Y_AXIS` for 2D).
4. Export it from `QickworkspaceV2/experiments/<family>/__init__.py`.
5. Optionally add an `Analysis` subclass in `QickworkspaceV2/analysis/`.

---

## 6. Hardware test safety

- Connect only to QICK/Pyro unless the user explicitly adds an instrument.
- Do not instantiate, read, or control Yoko, SGS100A, MG3692, or other RF
  sources during QICK-only validation.
- Before acquisition, inspect every config key containing `gain`: every scalar
  gain and every sweep endpoint must be strictly below `0.1`.
- Keep cooling disabled unless it is the feature under test.
- Without a connected chip, fitting failure is expected; validate program
  compilation and acquired data shape instead.
- Connect through `BaseExperiment.connect_pyro4(ns_host, ns_port=8888,
  proxy_name="myqick", data_path=None)`; never bypass the framework session
  with a direct Pyro helper.

---

## 7. How to work

### Before writing code

- State the plan in one or two sentences; for non-trivial work, a numbered
  list with a verification check per step.
- Read the files you will touch and their callers. Match existing patterns.
- Surface assumptions out loud instead of burying them in the diff.

### Writing code

- Minimum code that solves the stated problem. No speculative features,
  configurability, or abstractions for single-use code.
- Handle failures that can actually happen, not impossible ones.
- Clean up orphans your own change creates; mention (don't delete)
  pre-existing dead code.

### Verifying

- Turn vague asks into checkable goals ("fix the bug" -> a failing test that
  then passes; "refactor" -> the suite passes before and after, no public API
  change).
- Run the tests/linters that exist. Read the whole traceback. Fix the cause,
  not the test. Never report "done" from a plausible-looking diff alone.

### When to ask

Ask when the request is ambiguous in a way that matters, touches something
load-bearing or versioned, needs credentials or production resources, or the
stated goal conflicts with the literal request. Proceed without asking for
trivial, reversible changes or ambiguity a file read can resolve.

### Communication

Direct and concise. No flattery or filler. Give a clear answer when one
exists; otherwise say so and give your best read on the tradeoffs. After two
failed corrections on the same issue, stop, summarize what you learned, and ask
for a sharper prompt.

---

## 8. Project learnings

Append a one-line, concrete rule here when the user corrects an approach.
Tighten an existing line instead of duplicating it; remove lines that no
longer apply.

- Use the `qick_gui` interpreter for verification; the system Python may not
  have NumPy or QICK.
- The test directory is singular: `test/`.
- The current lab board reports QICK 0.2.381 while `qick_gui` has 0.2.405;
  treat configuration `KeyError` failures as a possible version mismatch.
