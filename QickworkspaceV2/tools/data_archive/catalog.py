"""Catalog coordination kept separate from experiment reading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..hdf5_store import CATALOG_FILENAME, find_experiments, rebuild_catalog


@dataclass(frozen=True)
class ArchiveScanSummary:
    indexed: int
    catalog_path: Path


class CatalogManager:
    def __init__(self, data_root):
        self.data_root = Path(data_root).expanduser().resolve()
        self.catalog_path = self.data_root / CATALOG_FILENAME

    def ensure(self) -> ArchiveScanSummary:
        self.data_root.mkdir(parents=True, exist_ok=True)
        if not self.catalog_path.exists():
            return self.rebuild()
        return ArchiveScanSummary(len(find_experiments(data_root=self.data_root)), self.catalog_path)

    def rebuild(self) -> ArchiveScanSummary:
        count = rebuild_catalog(self.data_root)
        return ArchiveScanSummary(count, self.catalog_path)

    def refresh(self) -> ArchiveScanSummary:
        """Rescan safely; HDF5 remains the source of truth."""
        return self.rebuild()

    def query(self, **filters):
        return find_experiments(data_root=self.data_root, **filters)
