from pathlib import Path
from copy import deepcopy
import json
import os
import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture
def device():
    from QickworkspaceV2 import Device

    return Device.from_files(
        ROOT / "lab/profiles/direct/hardware.yaml", ROOT / "lab/profiles/direct/device.yaml"
    )


@pytest.fixture
def limited_device(device):
    """Explicit validation limits independent of editable laboratory profiles."""
    from QickworkspaceV2 import Device

    hardware = device.hardware.model_dump()
    for port in hardware["generators"].values():
        port["max_gain"] = 0.8
    for group in device.config.readout_groups.values():
        hardware["generators"][group.generator]["max_gain"] = 0.5
    return Device(hardware, device.config.model_dump())


@pytest.fixture
def limited_session(limited_device, session, tmp_path):
    from QickworkspaceV2 import Session

    return Session(limited_device, session.backend, data_dir=tmp_path / "limited_runs")


@pytest.fixture
def session(device, tmp_path, native_config, monkeypatch):
    """Real compiler; replace only the board transport with constant shape fixtures.

    These arrays are deliberately uninformative and must fail physical fit checks.
    No runtime emulation backend is installed or exposed by the package.
    """
    from QickworkspaceV2 import Session, QICKBackend, BaseProgram
    from qick import QickConfig
    import numpy as np

    def fixed_iq(program, soc, **kwargs):
        shape = tuple(count for loop, count, *_ in program.loops if loop != "reps")
        return [np.ones((ro["trigs"], *shape, 2)) for ro in program.ro_chs.values()]

    monkeypatch.setattr(BaseProgram, "acquire", fixed_iq)
    return Session(
        device,
        QICKBackend(object(), QickConfig(native_config), resource_id="test-fixture"),
        data_dir=tmp_path / "runs",
    )


@pytest.fixture
def accepted_record():
    """An explicit numerical data fixture for analysis/storage/protocol tests."""
    import numpy as np
    from QickworkspaceV2 import ExperimentData, TraceData
    from QickworkspaceV2.experiments.t1_ge import analyze

    x = np.linspace(0, 100, 81)
    trace = TraceData(
        (0.2 + np.exp(-x / 25))[:, None],
        ("delay", "readout"),
        {"delay": x, "readout": ["measurement"]},
        {"delay": "us"},
    )
    result = ExperimentData(
        "t1_ge", {"Q1": trace}, metadata={"targets": ["Q1"], "iq_process": "abs", "test_fixture": True,
                                        "acquisition_status": "completed"}
    )
    result.fits = analyze(result)
    result.analysis_status = "completed"
    return result


@pytest.fixture
def native_config():
    """Logical multi-channel compiler fixture, not a physical board description."""
    qick = pytest.importorskip("qick")
    cfg = json.loads((ROOT / "tests/fixtures/qick_testbench.json").read_text())
    gen, ro = cfg["gens"][0], cfg["readouts"][0]
    cfg["gens"] = [dict(deepcopy(gen), tproc_ch=i) for i in range(7)]
    cfg["readouts"] = [dict(deepcopy(ro), tproc_ctrl=8 + i, trigger_bit=i, tproc_ch=i) for i in range(3)]
    cfg["sw_version"] = qick.__version__
    return cfg
