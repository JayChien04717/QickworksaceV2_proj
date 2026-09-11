# AGENTS.md

## Project contract

QICK Workspace is a native QICK tProc v2 measurement SDK and editable laboratory workspace. Package: `QickworkspaceV2`; distribution: `qickworkspace`. Never import `qick_workspace` or `archive` from the SDK. Preserve original files in `archive/v1`.

The user explicitly removed simulation. Do not add a simulated backend, fake measurement mode, recorder-based compiler replacement, or notebook MODE switch. Numerical test fixtures and mocked transport in tests are permitted; they are not measured data.

## User-facing workflow

The chip-calibration entry is root `ChipCalibration.ipynb`. Preserve the earlier `QickworkspaceV2.ipynb` and its recorded measurements. Other entries are `measure.py` and `lab/config.py`.
Keep maintained notebooks in English, including Markdown, code comments, docstrings and user-facing messages. Preserve historical notebooks in archive/v1. The detailed Chinese architecture/API reference is docs/REFACTOR_GUIDE.md.
Use `qb = lab.config["Q1"]`, `qb.update(res_gain_ge=0.15, res_sigma=0.01)`, then `run_cfg = qb.for_run(steps=101, ...)`. `qb` is a live working view. `for_run` creates an independent editable dict, preserving native QickSweep1D objects. `lab.config.update_all(...)` explicitly applies shared settings. Select fixed qubit IDs and use the native `res_sigma`, `wait_us` and `detuning_mhz` keys; do not add V1 configuration adapters or key aliases.

`Measurement.run(Program, run_cfg, py_avg=...)` directly compiles the edited dictionary, acquires, saves and analyzes. Daily changes must not require a schema/build function. All experiments show LivePlot during acquisition. Clear its own display after completion, interruption or failure, leaving final analysis figures and logs intact. Per-round checkpointing and cancellation recovery remain independent of plotting.

`NotebookLab` owns Notebook final presentation, analysis, cancellation recovery and host procedures; maintained ChipCalibration cells only import SDK APIs, edit parameters and call them. Arbitrary experiment-specific keys are allowed. Wiring profiles remain validated initial settings; they do not overwrite the user's edited run config.

Keep native `_initialize(cfg)` / `_body(cfg)`, QICK pulse/delay/loop APIs and calibrated `qb.x()` / `qb.halfx()` helpers. Parallel layers must be explicit. Do not introduce a custom quantum language.

## Ownership

- `lab/project.yaml` and `lab/profiles/*`: connection, wiring, initial GE/EF pulses and readout groups.
- `lab/config.py`, root notebook/script: daily editing. Preserve notebook user edits; there is no generator to overwrite them.
- `lab/procedures`: separate DC/TWPA workflows. Keep custom sequence examples inline in the main notebook; do not add redundant demo modules.
- `device`: strict wiring models plus editable dictionary API.
- `experiments`: one independently maintained package per experiment and transition. Each owns `program.py` (native Program and build), `parameters.py` (node schema/defaults), `analysis.py` (analysis, final/scan plots and calibration update rules), and `__init__.py` (public exports, ExperimentSpec and explicit Program.EXPERIMENT binding). Keep analyze/plot/updates as separate functions in that local file; they return results without displaying or writing calibration. Do not merge T1 GE/EF into coherence.py or a transition-switched experiment class.
- `programs`, `backends`: native QICK helpers, compiler and acquisition mapping.
- `data`, `analysis`, `plotting`: named complex IQ/shots, immutable acquisition and separate analysis revisions.
- `runtime`: direct Measurement, schema-based Session for automation, hardware lease, instrument lifecycle and worker.
- `calibration`: explicit revision-checked proposals and graph.
- Experiment `updates(result, target)` returns `CalibrationUpdates`: native working values plus optional strict device paths. Notebook `apply_fit(result)` and Session proposals consume that hook; do not restore experiment-specific update switches in Session or fit-to-key dictionaries in Notebook cells. Derive pulse/readout update context from the saved resolved configuration. Diagnostics declare why they have no automatic updates.
- `notebook.py`: NotebookLab orchestration, shared result presentation, accepted-fit checks, explicit working-file updates and host-scan summaries. Keep these out of Notebook cells; pass dependencies explicitly. Experiment analysis and specialized final plots belong to its experiment package. In `labtools/fitting/functions.py`, maintain each equation beside its independent `fit_<model>()`, which owns starting values, bounds, derived metrics and physical quality checks. `fit_curve()` is only a callable-based numerical solver; never add model-name dispatch or model-specific branches there. Experiment analyzers call the specific fitter; `FIT_FUNCTIONS` only selects configured overrides. Avoid duplicate formulas.
- `runtime/catalog.py`: the canonical worker/CLI automation catalog, including custom project modules.
- Broadband is a separate `broadband_resonator_spectrum` experiment with local analysis, plotting and parameters. It shares only the native one-tone pulse sequence with the narrow resonator program. Preserve the original complex-IQ detector in `labtools/fitting/fit_n_res.py`; do not replace it with Lorentzian fitting or put a broadband mode switch into the narrow experiment. Convert MHz/Hz explicitly at the analysis boundary.
- `data/transport.py`: result arrays, labels and PNG serialization. The companion owns the sole HTTP client in `qick_agent/client.py`; do not add wrappers or a second SDK client.

## Verification

Use `.venv/Scripts/python.exe` (Python 3.12, QICK 0.2.422 in this workspace):

```powershell
.venv/Scripts/python -m pytest -q
.venv/Scripts/python tests/verify_notebooks.py
.venv/Scripts/ruff check QickworkspaceV2 labtools tests lab measure.py
```

Notebook verification validates syntax and compiles explicitly tagged configuration/program cells against the real QICK compiler fixture. It skips all connection/acquisition/calibration cells. Do not execute the hardware notebook or measure.py merely to test software. Physical acquisition is separate from compiler validation.

Frequencies MHz, time us, gain normalized. Preserve fixed qubit IDs, digital ADC membership, MUX tone slots, actual rounded axes, multiple readout events and shots. Do not infer identity from ordering or silently squeeze dimensions.

Calibration commit remains explicit; fitting failure must preserve acquisition. Never guess a Ramsey correction sign or mark a CZ calibrated from a chevron image. Resource locks protect a single board; cancellation occurs between returned acquisition rounds/host points.

Clean disposable test/build/cache artifacts after maintenance. Do not delete `.venv`, archived originals or actual measurement data. Verify resolved deletion paths remain inside the intended workspace.

Declare qubit drive generators explicitly in each program using `setup_qubit_gen`; use `setup_readout` for resonator generators, ADCs and readout pulses together. Declare standard gates separately when needed; do not restore an all-in-one device setup helper.

Keep local parameter files readable: document each field's physical meaning and units, distinguish absolute frequencies from offsets, and explain defaults and inclusive/exclusive bounds beside the native Pydantic keywords. Do not introduce custom validation wrappers just to rename those keywords.

Readout punch-out uses native nested FPGA gainloop/freqloop sweeps in its own experiment. Preserve gain-by-frequency dimensions and compiled coordinates; do not replace either axis with a host scan.

General numerical fitting, HDF5/Labber I/O and file catalogs belong to the sibling `labtools` package. It must not import QICK or `QickworkspaceV2`. The SDK owns trace selection and experiment orchestration. Do not restore old utility import paths or duplicate implementations.
