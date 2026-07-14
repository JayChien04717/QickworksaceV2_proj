"""Lightweight experiment references returned by an ExperimentArchive."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .reader import ExperimentReader


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    timestamp_utc: str
    timestamp_local: str
    experiment_type: str
    qubits: tuple[str, ...]
    tags: tuple[str, ...]
    quality: str
    session_id: Optional[str]
    comment_preview: str
    path: Path

    def reader(self) -> ExperimentReader:
        return ExperimentReader(self.path)

    def inspect(self) -> dict:
        return self.reader().inspect()

    def load(self):
        return self.reader().load()

    def axes(self):
        return self.reader().axes()

    def raw_keys(self) -> list[str]:
        return self.reader().raw_keys()

    def analysis_keys(self) -> list[str]:
        return self.reader().analysis_keys()

    def load_raw(self, name: str = "iq", *, selection=None):
        return self.reader().raw(name, selection=selection)

    def load_analysis(self, name: str, *, selection=None):
        return self.reader().analysis(name, selection=selection)

    def plot(self, *, kind: str | None = None, registry=None, **kwargs):
        if registry is None:
            from .plotting import default_plot_registry

            registry = default_plot_registry
        return registry.plot(self, kind=kind, **kwargs)
