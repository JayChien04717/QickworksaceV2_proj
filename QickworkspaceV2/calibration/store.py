"""Transactional, device-scoped calibration proposals with optimistic revisions."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
import json
import re

from QickworkspaceV2.data.serialization import dumps
from QickworkspaceV2.data.store import sqlite


@dataclass(frozen=True)
class CalibrationProposal:
    updates: dict
    expected_revision: int
    source_run_id: str
    scope: str
    quality: str = "accepted"
    source: str = "measured"
    id: str = field(default_factory=lambda: uuid4().hex)


def allowed_path(path):
    return bool(
        re.fullmatch(
            r"qubits\.[A-Za-z][A-Za-z0-9_]*\.transitions\.[a-z]+\.(frequency_mhz|pulse\.(gain|pi_gain|pi2_gain|length_us|sigma_us|phase_deg|drag_alpha|drag_delta_mhz))",
            path,
        )
        or re.fullmatch(
            r"readout_groups\.[A-Za-z][A-Za-z0-9_]*\.members\.[A-Za-z][A-Za-z0-9_]*\.(frequency_mhz|gain|phase_deg|rotation_deg|threshold)",
            path,
        )
        or re.fullmatch(
            r"couplers\.[A-Za-z][A-Za-z0-9_]*\.(calibrated|frequency_mhz|pulse\.(gain|length_us|sigma_us|phase_deg)|phase_corrections_deg\.[A-Za-z][A-Za-z0-9_]*)",
            path,
        )
    )


class CalibrationStore:
    def __init__(self, path, *, scope):
        self.path, self.scope = Path(path), scope
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS calibration_revisions (scope TEXT, revision INTEGER, proposal_id TEXT UNIQUE, source_run_id TEXT, source TEXT, created_at TEXT, updates TEXT, PRIMARY KEY(scope, revision))"
            )

    def snapshot(self):
        values, revision = {}, 0
        with sqlite(self.path) as db:
            for row in db.execute(
                "SELECT revision, updates FROM calibration_revisions WHERE scope=? ORDER BY revision",
                (self.scope,),
            ):
                revision = row["revision"]
                values.update(json.loads(row["updates"]))
        return revision, values

    def commit(self, proposal, *, validate=None):
        if proposal.scope != self.scope:
            raise ValueError("Proposal belongs to a different device, wiring or backend scope")
        if proposal.quality != "accepted" or not proposal.updates:
            raise ValueError("Only nonempty accepted proposals may update calibration")
        if proposal.source not in ("manual", "measured"):
            raise ValueError("Unknown calibration source")
        for path in proposal.updates:
            if not allowed_path(path):
                raise ValueError(f"Not a calibration parameter: {path}")
        with sqlite(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT * FROM calibration_revisions WHERE proposal_id=?", (proposal.id,)
            ).fetchone()
            if existing:
                if existing["scope"] != self.scope or existing["updates"] != dumps(proposal.updates):
                    raise ValueError("Proposal ID was reused with a different payload")
                return existing["revision"]
            rows = db.execute(
                "SELECT revision, updates FROM calibration_revisions WHERE scope=? ORDER BY revision",
                (self.scope,),
            ).fetchall()
            revision = rows[-1]["revision"] if rows else 0
            if revision != proposal.expected_revision:
                raise RuntimeError(
                    f"Calibration revision changed: expected {proposal.expected_revision}, actual {revision}; rerun or review"
                )
            combined = {}
            for row in rows:
                combined.update(json.loads(row["updates"]))
            combined.update(proposal.updates)
            if validate:
                validate(combined)
            db.execute(
                "INSERT INTO calibration_revisions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    self.scope,
                    revision + 1,
                    proposal.id,
                    proposal.source_run_id,
                    proposal.source,
                    datetime.now(timezone.utc).isoformat(),
                    dumps(proposal.updates),
                ),
            )
        return revision + 1

    def history(self):
        with sqlite(self.path) as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM calibration_revisions WHERE scope=? ORDER BY revision", (self.scope,)
                )
            ]

    def is_stale(self, parameter, max_age_hours=24):
        for row in reversed(self.history()):
            if parameter in json.loads(row["updates"]):
                age = datetime.now(timezone.utc) - datetime.fromisoformat(row["created_at"])
                return age.total_seconds() > max_age_hours * 3600
        return True
