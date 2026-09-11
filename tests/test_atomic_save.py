from pathlib import Path

import pytest

from labtools.hdf5 import atomic


def windows_locked():
    error = PermissionError("Windows destination temporarily locked")
    error.winerror = 5
    return error


def test_hdf5_checkpoint_retries_transient_windows_lock(accepted_record, tmp_path, monkeypatch):
    from QickworkspaceV2 import ExperimentData

    path = tmp_path / "partial.h5"
    accepted_record.save(path)
    before = path.read_bytes()
    original = atomic.os.replace
    calls = []

    def replace(source, destination):
        calls.append((source, destination))
        if len(calls) < 3:
            assert Path(destination).read_bytes() == before
            raise windows_locked()
        original(source, destination)

    monkeypatch.setattr(atomic.os, "replace", replace)
    monkeypatch.setattr(atomic.time, "sleep", lambda seconds: None)
    accepted_record.metadata["new_round"] = 2
    accepted_record.save(path)
    assert len(calls) == 3
    assert ExperimentData.load(path).metadata["new_round"] == 2
    assert not list(tmp_path.glob(".run-*"))


def test_permanent_lock_stays_bounded_and_preserves_previous_file(tmp_path, monkeypatch):
    source, destination = tmp_path / "new", tmp_path / "old"
    source.write_text("new")
    destination.write_text("old")
    delays = []
    monkeypatch.setattr(atomic.os, "replace", lambda *args: (_ for _ in ()).throw(windows_locked()))
    monkeypatch.setattr(atomic.time, "sleep", delays.append)
    with pytest.raises(PermissionError):
        atomic.replace_file(source, destination)
    assert len(delays) == 7 and sum(delays) < 1.3
    assert source.read_text() == "new" and destination.read_text() == "old"
