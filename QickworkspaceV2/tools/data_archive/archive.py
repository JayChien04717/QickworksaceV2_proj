"""High-level archive facade for a Qickworkspace data directory."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .catalog import CatalogManager
from .record import ExperimentRecord


class ExperimentArchive:
    """Index and browse all completed experiments below one data path."""

    def __init__(self, data_root, *, auto_refresh: bool = True):
        self.data_root = Path(data_root).expanduser().resolve()
        self.catalog = CatalogManager(self.data_root)
        if auto_refresh:
            # A folder may have received copied/moved HDF5 files without going
            # through save_result(), so scan metadata when it is handed to us.
            self.catalog.refresh()
        else:
            self.catalog.ensure()

    @staticmethod
    def _record(reference) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id=reference.experiment_id,
            timestamp_utc=reference.timestamp_utc,
            timestamp_local=reference.timestamp_local,
            experiment_type=reference.experiment_type,
            qubits=reference.qubits,
            tags=reference.tags,
            quality=reference.quality,
            session_id=reference.session_id,
            comment_preview=reference.comment_preview,
            path=reference.path,
        )

    def refresh(self):
        return self.catalog.refresh()

    def rebuild(self):
        return self.catalog.rebuild()

    def query(self, **filters) -> list[ExperimentRecord]:
        return [self._record(item) for item in self.catalog.query(**filters)]

    def all(self, *, limit=None) -> list[ExperimentRecord]:
        return self.query(limit=limit)

    def get(self, experiment_id: str) -> ExperimentRecord:
        for record in self.query():
            if record.experiment_id == experiment_id:
                return record
        raise KeyError(f"Experiment not found: {experiment_id}")

    def latest(self, experiment_type=None, *, qubit=None, tags=None) -> ExperimentRecord:
        records = self.query(
            experiment_type=experiment_type,
            qubit=qubit,
            tags=tags,
            limit=1,
        )
        if not records:
            raise LookupError("No experiment matched the requested filters")
        return records[0]

    def stats(self) -> dict:
        records = self.all()
        return {
            "total": len(records),
            "experiment_types": dict(Counter(item.experiment_type for item in records)),
            "qubits": dict(Counter(q for item in records for q in item.qubits)),
            "qualities": dict(Counter(item.quality for item in records)),
        }
