"""Single-worker local API. Notebook and agent clients share this hardware owner."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from threading import Event, RLock
from uuid import uuid4
import json
import hashlib

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from QickworkspaceV2.backends import AcquisitionCancelled
from labtools.serialization import digest, jsonable
from QickworkspaceV2.data.store import sqlite


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment: str
    parameters: dict = Field(default_factory=dict)
    run_options: dict = Field(default_factory=dict)
    request_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1, max_length=128)
    catalog_revision: str | None = None


def create_app(session):
    executor, lock, cancellations = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="qick-worker"),
        RLock(),
        {},
    )
    db_path = session.store.root / "jobs.sqlite3"
    with sqlite(db_path) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, request_id TEXT UNIQUE, digest TEXT, request TEXT, status TEXT, run_id TEXT, error TEXT)"
        )
        if "progress" not in {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}:
            db.execute("ALTER TABLE jobs ADD COLUMN progress TEXT NOT NULL DEFAULT '{}'")

    @asynccontextmanager
    async def lifespan(app):
        from QickworkspaceV2.runtime.session import _hardware_lease

        with _hardware_lease("worker-store:" + str(session.store.root)):
            with sqlite(db_path) as db:
                pending = db.execute(
                    "SELECT id, progress FROM jobs WHERE status IN ('queued', 'running', 'cancel_requested')"
                ).fetchall()
                for job in pending:
                    progress = {**json.loads(job["progress"]), "phase": "interrupted"}
                    db.execute(
                        "UPDATE jobs SET status='interrupted', error=?, progress=? WHERE id=?",
                        (
                            "Worker restarted; inspect stored runs before retrying",
                            json.dumps(progress), job["id"],
                        ),
                    )
            try:
                yield
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="QICK Workspace", version="2.0.0", lifespan=lifespan)
    app.state.session = session

    def get_job(job_id):
        with sqlite(db_path) as db:
            row = db.execute(
                "SELECT id, request_id, status, run_id, error, progress FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        if not row:
            raise HTTPException(404, "Unknown job")
        job = dict(row)
        job["progress"] = json.loads(job["progress"])
        return job

    def update(job_id, status, run_id=None, error=""):
        with sqlite(db_path) as db:
            progress = json.loads(db.execute("SELECT progress FROM jobs WHERE id=?", (job_id,)).fetchone()[0])
            progress["phase"] = status
            db.execute(
                "UPDATE jobs SET status=?, run_id=COALESCE(?,run_id), error=?, progress=? WHERE id=?",
                (status, run_id, error, json.dumps(progress), job_id),
            )

    def execute(job_id, request, cancel):
        if cancel.is_set():
            update(job_id, "cancelled")
            with lock:
                cancellations.pop(job_id, None)
            return
        update(job_id, "running")
        try:
            require_catalog_revision(request.catalog_revision)

            def progress(event):
                result = event.get("result")
                run_id = result.run_id if result is not None else event.get("run_id")
                with sqlite(db_path) as db:
                    state = json.loads(
                        db.execute("SELECT progress FROM jobs WHERE id=?", (job_id,)).fetchone()[0]
                    )
                    state["phase"] = event["state"]
                    for key in ("completed", "total"):
                        if key in event:
                            state[key] = int(event[key])
                    db.execute(
                        "UPDATE jobs SET run_id=COALESCE(?,run_id), progress=? WHERE id=?",
                        (run_id, json.dumps(state), job_id),
                    )

            result = session.run(
                request.experiment,
                **request.parameters,
                **request.run_options,
                cancel=cancel,
                on_progress=progress,
            )
            update(job_id, "completed", result.run_id)
        except AcquisitionCancelled as exc:
            update(job_id, "cancelled", error=str(exc))
        except Exception as exc:
            update(job_id, "failed", error=str(exc))
        finally:
            with lock:
                cancellations.pop(job_id, None)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "protocol_version": 2,
            "backend": "qick",
            "hardware_verified": False,
            "connection_state": "proxy_configured" if session.backend.soc is not None else "unconfigured",
        }

    @app.get("/config")
    def config_snapshot():
        from QickworkspaceV2.device.models import ProjectConfig, read_yaml

        revision, values = session.calibration.snapshot()
        device = session.base_device.with_updates(values)
        connection = None
        if session.project_path is not None:
            connection = ProjectConfig.model_validate(read_yaml(session.project_path)).connection.model_dump()
        try:
            capabilities = {"status": "available", "data": session.backend.capabilities()}
        except Exception as exc:
            capabilities = {"status": "error", "message": str(exc)}
        return jsonable(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "qick_worker",
                "project_path": str(session.project_path) if session.project_path is not None else None,
                "hardware": device.hardware.model_dump(),
                "device": device.config.model_dump(),
                "defaults": session.defaults.model_dump(),
                "calibration_revision": revision,
                "connection": connection,
                "capabilities": {**capabilities, "hardware_verified": False},
            }
        )

    @app.get("/device")
    def describe_device():
        return jsonable({"device": session.device.summary(), "capabilities": session.backend.capabilities()})

    @app.get("/catalog")
    def catalog():
        from QickworkspaceV2.runtime.catalog import catalog_payload

        return catalog_payload(session.registry)

    def require_catalog_revision(expected):
        current = catalog()["revision"]
        if expected != current:
            raise HTTPException(409, {
                "code": "catalog_changed", "expected_revision": expected,
                "current_revision": current, "message": "Reload the catalog and tool declarations before submitting.",
            })

    @app.post("/experiments/check")
    def check(request: RunRequest):
        if request.catalog_revision is not None:
            require_catalog_revision(request.catalog_revision)
        try:
            validate_run_options(request)
            spec, parsed, plan, device, defaults, revision = session._prepare(
                request.experiment, request.parameters, request.run_options
            )
            program = session.backend.compile(plan)
            session._check_readouts(program, plan)
            return {
                "ready": True,
                "compiled": True,
                "hardware_verified": False,
                "backend": "qick",
                "targets": list(plan.cfg["targets"]),
                "calibration_revision": revision,
                "readout_channels": list(program.ro_chs),
                "host_sweep": plan.metadata.get("host_sweep"),
                "run_options": defaults.model_dump(),
                "message": "Compiled; no acquisition performed",
            }
        except Exception as exc:
            return {"ready": False, "message": str(exc), "error_type": type(exc).__name__}

    def validate_run_options(request):
        allowed = {"reps", "soft_avgs", "relax_delay_us", "iq_process"}
        if set(request.run_options) - allowed or set(request.parameters) & allowed:
            raise ValueError("Run options must use the run_options mapping")

    @app.post("/experiments/run", status_code=202)
    def submit(request: RunRequest):
        try:
            validate_run_options(request)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        fingerprint = digest(
            {
                "experiment": request.experiment,
                "parameters": request.parameters,
                "run_options": request.run_options,
            }
        )
        with lock:
            with sqlite(db_path) as db:
                old = db.execute(
                    "SELECT id, digest FROM jobs WHERE request_id=?", (request.request_id,)
                ).fetchone()
                if old:
                    if old["digest"] != fingerprint:
                        raise HTTPException(409, "request_id was already used with different parameters")
                    return get_job(old["id"])
            try:
                session._prepare(request.experiment, request.parameters, request.run_options)
            except (ValueError, KeyError, TypeError) as exc:
                raise HTTPException(422, str(exc)) from exc
            require_catalog_revision(request.catalog_revision)
            job_id, cancel = uuid4().hex, Event()
            with sqlite(db_path) as db:
                db.execute(
                    "INSERT INTO jobs (id, request_id, digest, request, status, run_id, error, progress) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        job_id, request.request_id, fingerprint, request.model_dump_json(), "queued", None, "",
                        json.dumps({"phase": "queued", "completed": 0, "total": None}),
                    ),
                )
            cancellations[job_id] = cancel
            executor.submit(execute, job_id, request, cancel)
        return get_job(job_id)

    @app.get("/requests/lookup")
    def lookup_request(request_id: str = Query(min_length=1, max_length=128)):
        """Read-only recovery, including requests whose submission reply was lost."""
        with sqlite(db_path) as db:
            row = db.execute("SELECT id FROM jobs WHERE request_id=?", (request_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Request not found; this lookup never submits a measurement")
        return get_job(row["id"])

    @app.get("/experiments/{job_id}")
    def status(job_id: str):
        return get_job(job_id)

    @app.post("/experiments/{job_id}/cancel")
    def cancel_job(job_id: str):
        with lock:
            job = get_job(job_id)
            if job["status"] in ("queued", "running", "cancel_requested"):
                if job_id in cancellations:
                    cancellations[job_id].set()
                    update(job_id, "cancel_requested")
        return {
            **get_job(job_id),
            "message": "Cancellation is cooperative between acquisition rounds; it is not an FPGA emergency stop",
        }

    @app.get("/experiments/{job_id}/result")
    def result(job_id: str):
        job = get_job(job_id)
        if job["status"] != "completed":
            raise HTTPException(409, job)
        from QickworkspaceV2.data.transport import to_worker_result

        record = session.load(job["run_id"])
        payload = to_worker_result(record, artifact_dir=session.store.directory(job["run_id"]))
        # Reaching this endpoint requires the durable job to be completed.
        # Acquisition state belongs to the worker journal, not analysis metadata.
        payload.update(status="success", acquisition_status="completed")
        payload.pop("error", None)
        payload["metadata"].update(
            acquisition_status="completed",
            quality=record.quality.value,
            created_at=record.created_at,
            parameters=record.metadata.get("parameters", {}),
            **{
                key: record.metadata[key]
                for key in ("resolved_config", "device_snapshot", "run_options", "calibration_revision")
                if key in record.metadata
            },
        )
        payload["artifacts"] = artifacts(job["run_id"])
        return jsonable(payload)

    def acquisition_file(run_id):
        try:
            path = session.store.directory(run_id) / "acquisition.h5"
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(session.store.root) or resolved != path:
                raise ValueError("Artifact must be stored directly inside the run store")
            if not resolved.is_file():
                raise FileNotFoundError()
            return resolved
        except (ValueError, OSError) as exc:
            raise HTTPException(404, "Unknown acquisition artifact") from exc

    @app.get("/runs/{run_id}/artifacts")
    def artifacts(run_id: str):
        path = acquisition_file(run_id)
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        return [{"name": "acquisition.h5", "run_id": run_id, "size_bytes": path.stat().st_size,
                 "sha256": checksum, "media_type": "application/x-hdf5",
                 "url": f"/runs/{run_id}/artifacts/acquisition.h5"}]

    @app.get("/runs/{run_id}/artifacts/acquisition.h5")
    def download_acquisition(run_id: str):
        return FileResponse(acquisition_file(run_id), media_type="application/x-hdf5",
                            filename=f"{run_id}-acquisition.h5")

    @app.get("/runs")
    def runs():
        return session.store.list()

    return app
