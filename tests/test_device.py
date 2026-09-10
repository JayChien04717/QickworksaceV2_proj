from copy import deepcopy
import pytest
from pydantic import ValidationError
from QickworkspaceV2 import Device, Session, QICKBackend
from qick import QickConfig


def test_reordering_never_changes_mapping(device):
    cfg = device.config.model_dump()
    cfg["qubits"] = dict(reversed(list(cfg["qubits"].items())))
    reordered = Device(device.hardware, cfg)
    assert reordered.flat_qubit("Q1") == device.flat_qubit("Q1")
    assert reordered.neighbors("Q1") == ["Q2"]


def test_unknown_fields_rejected(device):
    cfg = device.config.model_dump()
    cfg["qubits"]["Q1"]["transitions"]["ge"]["pulse"]["simga_us"] = 1
    with pytest.raises(ValidationError, match="simga_us"):
        Device(device.hardware, cfg)


def test_shared_active_drive_rejected_but_single_target_works(device, tmp_path, native_config):
    cfg = device.config.model_dump()
    cfg["qubits"]["Q2"]["drive"] = "drive_q1"
    s = Session(Device(device.hardware, cfg), QICKBackend(None, QickConfig(native_config)), data_dir=tmp_path)
    assert s.check("t1_ge", target="Q1")["ready"]
    assert not s.check("t1_ge", target="Q1,Q2")["ready"]


def test_three_qubits_same_execution_path(device, tmp_path, native_config):
    hw, cfg = device.hardware.model_dump(), device.config.model_dump()
    hw["generators"]["drive_q3"] = {"channel": 5, "nqz": 1, "mixer_mhz": 0.0, "max_gain": 0.8}
    hw["generators"]["readout_q3"] = {"channel": 6, "nqz": 2, "mixer_mhz": 0.0, "max_gain": 0.5}
    hw["readouts"]["adc_q3"] = 2
    cfg["qubits"]["Q3"] = deepcopy(cfg["qubits"]["Q1"])
    cfg["qubits"]["Q3"]["drive"] = "drive_q3"
    cfg["readout_groups"]["R3"] = deepcopy(cfg["readout_groups"]["R1"])
    cfg["readout_groups"]["R3"]["generator"] = "readout_q3"
    tone = cfg["readout_groups"]["R3"]["members"].pop("Q1")
    tone["adc"] = "adc_q3"
    cfg["readout_groups"]["R3"]["members"]["Q3"] = tone
    native = Session(
        Device(hw, cfg), QICKBackend(None, QickConfig(native_config)), data_dir=tmp_path / "native"
    )
    program = native.compile("t1_ge", target="Q3,Q1,Q2")
    assert set(program.ro_chs) == {0, 1, 2}
    assert all(ro["trigs"] == 1 for ro in program.ro_chs.values())


def test_mux_subset_keeps_slot(tmp_path, native_config):
    d = Device.from_files("lab/profiles/mux/hardware.yaml", "lab/profiles/mux/device.yaml")
    native_config["gens"][2].update(
        type="axis_sg_mux8_v1", n_tones=8, has_gain=True, has_phase=True, has_mixer=True
    )
    s = Session(d, QICKBackend(None, QickConfig(native_config)), data_dir=tmp_path)
    p = s.compile("t1_ge", target="Q2")
    assert list(p.ro_chs) == [1]
    assert p.gen_chs[2]["mux_tones"][1]["freq_rounded"] == pytest.approx(6740, abs=0.001)


def test_run_gain_limits(limited_session):
    session = limited_session
    assert session.check("power_rabi_ge", stop=0.8)["ready"]
    assert not session.check("power_rabi_ge", stop=0.99)["ready"]
    assert session.check("resonator_spec", gain_scale=2.5)["ready"]
    assert not session.check("resonator_spec", gain_scale=4)["ready"]
