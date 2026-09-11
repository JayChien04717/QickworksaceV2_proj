"""Durable run journal, immutable acquisitions and append-only analysis revisions."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
import json
import os
import sqlite3
import tempfile

from QickworkspaceV2.data.models import ExperimentData, FitResult
from QickworkspaceV2.data.serialization import dumps
from QickworkspaceV2.data.atomic import replace_file


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".json-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(dumps(data))
            stream.flush()
            os.fsync(stream.fileno())
        replace_file(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextmanager
def sqlite(path):
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


class RunStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "runs.sqlite3"
        with sqlite(self.db_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, experiment TEXT, created_at TEXT, status TEXT, request TEXT, error TEXT)")

    def directory(self, run_id):
        if not run_id or not all(c in "0123456789abcdef" for c in run_id) or len(run_id) != 32:
            raise ValueError("Invalid run ID")
        return self.root / run_id

    def begin(self, run_id, experiment, request):
        directory = self.directory(run_id)
        directory.mkdir(exist_ok=False)
        atomic_json(directory / "request.json", request)
        with sqlite(self.db_path) as db:
            db.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                       (run_id, experiment, datetime.now(timezone.utc).isoformat(), "running", dumps(request), ""))

    def status(self, run_id, status, error=""):
        with sqlite(self.db_path) as db:
            db.execute("UPDATE runs SET status=?, error=? WHERE id=?", (status, error, run_id))

    def save_acquisition(self, result):
        path = self.directory(result.run_id) / "acquisition.h5"
        if path.exists():
            raise FileExistsError("Acquisitions are immutable")
        result.save(path)
        self.status(result.run_id, "acquired")

    def save_analysis(self, result, *, update_status=True):
        directory = self.directory(result.run_id) / "analyses"
        directory.mkdir(exist_ok=True)
        # The session is serialized; exclusive file creation also rejects accidental competing revisions.
        revision = len(list(directory.glob("*.json"))) + 1
        path = directory / f"{revision:04d}.json"
        record = {"revision": revision, "created_at": datetime.now(timezone.utc).isoformat(),
                  "fits": result.fits, "status": result.analysis_status, "message": result.analysis_message,
                  "metadata": result.metadata,
                  "trace_metadata": {q: t.metadata for q, t in result.traces.items()}}
        with path.open("x", encoding="utf-8") as stream:
            stream.write(dumps(record))
        if update_status:
            self.status(result.run_id, "completed" if result.analysis_status != "failed" else "analysis_failed", result.analysis_message)
        return revision

    def load(self, run_id, revision=None, *, partial=False):
        result = ExperimentData.load(self.directory(run_id) / ("partial.h5" if partial else "acquisition.h5"))
        analyses = sorted((self.directory(run_id) / "analyses").glob("*.json"))
        if analyses and revision != 0:
            path = analyses[-1] if revision is None else self.directory(run_id) / "analyses" / f"{revision:04d}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            result.fits = {q: FitResult(**f) for q, f in record["fits"].items()}
            result.analysis_status, result.analysis_message = record["status"], record["message"]
            result.metadata = record.get("metadata", result.metadata)
            for q, metadata in record["trace_metadata"].items():
                result[q].metadata = metadata
        return result

    def list(self, limit=100):
        with sqlite(self.db_path) as db:
            return [dict(row) for row in db.execute("SELECT id, experiment, created_at, status, error FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))]
