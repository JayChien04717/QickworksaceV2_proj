# V2 worker and Agent

The worker owns the active `Session`, experiment registry, board connection and persistent jobs. The companion uses the single generic client in `qick_agent/client.py`; it does not import QICK or execute experiment scripts.

## Catalog

`GET /catalog` returns `{ "experiments": [...], "revision": "..." }`. Each entry contains `id`, `version`, `description` and a full JSON Schema in `parameters`. Native experiment parameters come from `ExperimentSpec.parameters`; `run_options` and `request_id` describe worker submission controls. The catalog includes modules configured in the selected project. The revision hashes the sorted declarations, including schemas and experiment versions.

UI, companion CLI and Agent consume this response without AST parsing, generated wrappers or a local production snapshot. `qickworkspace catalog` uses the same SDK catalog builder for inspecting a project without starting a worker. A worker outage is reported as unavailable; it does not activate a second catalog source.

## Start and use

From the repository root, start the worker with the SDK interpreter:

```powershell
& ./QickworksaceV2/.venv/Scripts/python.exe -m qick_server --project QickworksaceV2/lab/project.yaml --port 8001
```

Then start the companion backend in `qick_gui` and the UI as described in the [root README](../../README.md). `QICK_WORKER_URL` defaults to `http://127.0.0.1:8001`.

Restart the worker and companion backend after changing registered experiment modules. No export step is required. Backend startup reads the catalog to declare Agent tools.

## API

| Method and path | Purpose |
|---|---|
| `GET /health`, `/device`, `/config` | Worker availability and selected configuration |
| `GET /catalog` | Current canonical catalog and complete schemas |
| `POST /experiments/check` | Validate and compile; no acquisition |
| `POST /experiments/run` | Submit one job; returns HTTP 202 and job ID |
| `GET /experiments/{job_id}` | Status, progress and run ID |
| `GET /requests/lookup?request_id=...` | Read-only lookup when the submission reply was lost |
| `GET /experiments/{job_id}/result` | Acquired arrays, fit quality, metadata and PNG |
| `POST /experiments/{job_id}/cancel` | Cooperative cancellation between acquisition rounds |
| `GET /runs` | Saved worker runs |
| `GET /runs/{run_id}/artifacts` | Full acquisition descriptor, byte count and SHA-256 |
| `GET /runs/{run_id}/artifacts/acquisition.h5` | Download immutable source acquisition |

Protocol version 2 submissions use `experiment`, `parameters`, `run_options`, `request_id` and `catalog_revision`. A new submission with a missing/stale revision returns HTTP 409 (`catalog_changed`). Queued jobs check the revision again before execution. Reusing an ID with the same experiment, parameters and options reconnects to the same job, even after the catalog changes; changed arguments with that ID are rejected. A caller timeout does not stop the worker or authorize a new measurement.

Agent tools retain their declared schema and revision. A mismatch requires restarting the backend to reload declarations; validation does not silently switch to a different schema. The generic client obtains a revision automatically unless the caller supplies its declared revision explicitly.

## Recovery and reuse by other Agents

`qick_agent/execution.py` provides `WorkerExecution` and `RequestJournal` without companion models, storage, HTML or QICK imports. The journal commits the exact submission to SQLite before POST and scopes request IDs to the worker URL. Transport/5xx errors trigger a GET lookup. If lookup cannot resolve the request, the caller receives `SubmissionUncertain` with its request ID. Repeating a locally saved request performs lookup only. Keep the journal file to retain unresolved requests across restarts.

The companion adapter in `provider.py` owns history import and presentation metadata. Its journal is `data/worker_requests.sqlite3`. Other applications can use `WorkerExecution` with their own journal path, or use `WorkerClient` directly and manage persistence themselves.

Recovery is handled by the Agent tools and execution service. There is no manual request-ID recovery page in the UI.

`data/transport.py` serializes results. Acquisition success, analysis status and quality remain separate; no-chip fits cannot establish calibration. Full source HDF5 is retained, including shots and categorical labels. See the [result contract](../../docs/QICK_RESULT_CONTRACT.md).
