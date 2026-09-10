# AGENTS.md

## Project contract

QICK Workspace is a native QICK tProc v2 measurement SDK and editable laboratory workspace. Package: `QickworkspaceV2`; distribution: `qickworkspace`. Never import `qick_workspace` or `archive` from the SDK. Preserve original files in `archive/v1`.

The user explicitly removed simulation. Do not add a simulated backend, fake measurement mode, recorder-based compiler replacement, or notebook MODE switch. Numerical test fixtures and mocked transport in tests are permitted; they are not measured data.

## User-facing workflow

The primary entries are root `QickworkspaceV2.ipynb`, `measure.py`, and `lab/config.py`.
Keep maintained notebooks in English, including Markdown, code comments, docstrings and user-facing messages. Preserve historical notebooks in archive/v1. The detailed Chinese architecture/API reference is docs/REFACTOR_GUIDE.md.
Use `qb = config_all["Q1"]`, `qb.update(res_gain_ge=0.15, res_sigma=0.01)`, then `run_cfg = qb.for_run(steps=101, ...)`. `qb` is a live working view. `for_run` creates an independent editable dict, preserving native QickSweep1D objects. `config_all.update_all(...)` explicitly applies shared settings. Select fixed qubit IDs and use the native `res_sigma`, `wait_us` and `detuning_mhz` keys; do not add V1 configuration adapters or key aliases.

`Measurement.run(Program, run_cfg, py_avg=...)` directly compiles the edited dictionary, acquires, saves and analyzes. Daily changes must not require a schema/build function. Arbitrary experiment-specific keys are allowed. Wiring profiles remain validated initial settings; they do not overwrite the user's edited run config.

Keep native `_initialize(cfg)` / `_body(cfg)`, QICK pulse/delay/loop APIs and calibrated `qb.x()` / `qb.halfx()` helpers. Parallel layers must be explicit. Do not introduce a custom quantum language.

## Ownership

- `lab/project.yaml` and `lab/profiles/*`: connection, wiring, initial GE/EF pulses and readout groups.
- `lab/config.py`, root notebook/script: daily editing. Preserve notebook user edits; there is no generator to overwrite them.
- `lab/procedures`: separate DC/TWPA workflows. Keep custom sequence examples inline in the main notebook; do not add redundant demo modules.
- `device`: strict wiring models plus editable dictionary API.
- `experiments`: one independently maintained module per experiment and transition. Do not merge T1 GE/EF into coherence.py or a transition-switched experiment class.
- `programs`, `backends`: native QICK helpers, compiler and acquisition mapping.
- `data`, `analysis`, `plotting`: named complex IQ/shots, immutable acquisition and separate analysis revisions.
- `runtime`: direct Measurement, schema-based Session for automation, hardware lease, instrument lifecycle and worker.
- `calibration`, `integrations`: explicit revision-checked proposals, graph, NVIDIA wrappers and worker client.

## Verification

Use `.venv/Scripts/python.exe` (Python 3.12, QICK 0.2.422 in this workspace):

```powershell
.venv/Scripts/python -m pytest -q
.venv/Scripts/python tests/verify_notebooks.py
.venv/Scripts/ruff check QickworkspaceV2 tests lab measure.py
.venv/Scripts/python tests/verify_blueprint.py PATH_TO_BLUEPRINT/core
```

Notebook verification validates syntax and compiles explicitly tagged configuration/program cells against the real QICK compiler fixture. It skips all connection/acquisition/calibration cells. Do not execute the hardware notebook or measure.py merely to test software. Physical acquisition is separate from compiler validation.

Frequencies MHz, time us, gain normalized. Preserve fixed qubit IDs, digital ADC membership, MUX tone slots, actual rounded axes, multiple readout events and shots. Do not infer identity from ordering or silently squeeze dimensions.

Calibration commit remains explicit; fitting failure must preserve acquisition. Never guess a Ramsey correction sign or mark a CZ calibrated from a chevron image. Resource locks protect a single board; cancellation occurs between returned acquisition rounds/host points.

Clean disposable test/build/cache artifacts after maintenance. Do not delete `.venv`, archived originals or actual measurement data. Verify resolved deletion paths remain inside the intended workspace.
