from copy import deepcopy
import numpy as np
import pytest
from QickworkspaceV2 import Session, Device, QICKBackend


@pytest.mark.parametrize("target", ["Q1,Q2", "Q2,Q1", "Q2"])
def test_native_mux_and_pynq_readout(target, native_config, tmp_path):
    from qick import QickConfig

    cfg = deepcopy(native_config)
    cfg["gens"][2].update(type="axis_sg_mux8_v1", n_tones=8, has_gain=True, has_phase=True, has_mixer=True)
    # PYNQ-controlled readout validates the separate static readout path.
    for ro in cfg["readouts"]:
        ro.pop("tproc_ctrl")
    device = Device.from_files("lab/profiles/mux/hardware.yaml", "lab/profiles/mux/device.yaml")
    s = Session(device, QICKBackend(None, QickConfig(cfg)), data_dir=tmp_path)
    for name in (
        "t1_ge",
        "resonator_spec",
        "qubit_spec_ge",
        "ramsey_ge",
        "power_rabi_ge",
        "single_shot",
        "coupler_chevron",
    ):
        if name == "coupler_chevron" and "," not in target:
            continue
        program = s.compile(name, target=target)
        assert len(program.binprog["pmem"]) > 0
        assert program.gen_chs[2]["mux_tones"][1]["freq_rounded"] == pytest.approx(6740, abs=0.001)
        assert list(program.ro_chs) == [int(q[-1]) - 1 for q in target.split(",")]


def test_hardware_host_sweep_keeps_quantized_frequency(native_config, tmp_path):
    from qick import QickConfig

    class FixtureAcquisition(QICKBackend):
        def acquire(self, program, plan, **kwargs):
            from QickworkspaceV2.backends.qick import unpack_iq

            return unpack_iq(program, plan, [np.ones((1, 2)) for ch in program.ro_chs])

    device = Device.from_files("lab/profiles/direct/hardware.yaml", "lab/profiles/direct/device.yaml")
    s = Session(
        device, FixtureAcquisition(None, QickConfig(native_config), resource_id="fixture"), data_dir=tmp_path
    )
    r = s.run("resonator_spec", target="Q1", points=11)
    assert len(r["Q1"].coords["frequency"]) == 11
    assert not r.is_good()  # Flat data is retained, not accepted as a resonance.
