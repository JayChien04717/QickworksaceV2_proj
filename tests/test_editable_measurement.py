from threading import Event
import json
import numpy as np
import pytest
from qick import QickConfig
from qick.asm_v2 import QickSweep1D
from QickworkspaceV2 import ExperimentConfig, Measurement, BaseProgram, ExperimentData
from QickworkspaceV2.device.editable import prepare_program_config
from QickworkspaceV2.runtime.measurement import infer_axes
from QickworkspaceV2.backends.qick import extract_coords, AcquisitionCancelled
from QickworkspaceV2.experiments.base import ProgramPlan
from QickworkspaceV2.experiments.resonator_spec import ResonatorSpecProgram
from QickworkspaceV2.experiments.t1_ge import T1GEProgram
from QickworkspaceV2.experiments.t1_ef import T1EFProgram
from QickworkspaceV2.experiments.spin_echo_ge import SpinEchoGEProgram
from QickworkspaceV2.experiments.single_shot import SingleShotProgram


@pytest.fixture
def config_all(device):
    return ExperimentConfig(device)


@pytest.fixture
def measurement(native_config, tmp_path):
    return Measurement(None, QickConfig(native_config), data_path=tmp_path, resource_id="native-test")


def test_live_bulk_edits_and_isolated_run(config_all):
    q1 = config_all["Q1"]
    q1.update(res_gain_ge=0.15, res_sigma=0.01, custom_pulse_count=7)
    sweep = QickSweep1D("freqloop", 6710, 6725)
    run_cfg = q1.for_run(steps=61, res_freq_ge=sweep, relax_delay=0)
    assert run_cfg["res_freq_ge"] is sweep
    run_cfg["res_gain_ge"] = 0.22
    run_cfg["custom_pulse_count"] = 9
    assert q1["res_gain_ge"] == 0.15
    assert q1["custom_pulse_count"] == 7
    assert not hasattr(q1["res_freq_ge"], "spans")
    assert config_all["Q2"]["res_gain_ge"] != 0.15
    config_all.update_all(res_sigma=0.03, reps=55)
    assert all(config_all[q]["res_sigma"] == 0.03 for q in config_all.qubits)
    assert run_cfg["res_sigma"] == 0.01


def test_native_dictionary_updates_and_fixed_qubit_selection(config_all):
    config_all["Q1"].update(res_gain_ge=0.15, res_sigma=0.01)
    cfg = config_all["Q1"].for_run()
    sweep = QickSweep1D("waitloop", 0, 100)
    cfg.update([("steps", 31), ("wait_us", sweep), ("detuning_mhz", 0.3)])
    assert cfg["wait_us"] is sweep
    assert cfg["detuning_mhz"] == 0.3
    assert cfg["res_sigma"] == 0.01
    with pytest.raises(KeyError, match="Unknown qubit"):
        config_all[0]
    with pytest.raises(ValueError, match="unique qubit IDs"):
        config_all.for_run("Q1", "Q1")
    with pytest.raises(TypeError, match="mapping keyed by qubit ID"):
        ExperimentConfig([{"name": "Q1"}])


def test_editable_settings_roundtrip(config_all, tmp_path):
    config_all["Q1"].update(qb_freq_ge=2880.1, my_extra="pulse revision")
    path = config_all.save(tmp_path / "working.yaml")
    restored = ExperimentConfig.load(path)
    assert restored["Q1"]["qb_freq_ge"] == 2880.1
    assert restored["Q1"]["my_extra"] == "pulse revision"
    assert restored.for_run("Q1", "Q2")["couplers"]["C12"]["port"]["channel"] == 4


def test_multi_target_working_values_remain_independent(config_all):
    cfg = config_all.for_run("Q2", "Q1", steps=41)
    cfg["qubits"]["Q2"]["res_gain_ge"] = 0.12
    compiled = prepare_program_config(cfg)
    assert compiled["targets"] == ("Q2", "Q1")
    assert compiled["qubits"]["Q2"]["res_gain_ge"] == 0.12
    assert config_all["Q2"]["res_gain_ge"] != 0.12
    assert compiled["readout_groups"]["R2"]["members"]["Q2"]["gain"] == 0.12


def test_frequency_gain_and_sigma_edits_reach_compiled_waveforms(config_all, measurement):
    q1 = config_all["Q1"]
    cfg = q1.for_run(
        steps=51,
        res_freq_ge=QickSweep1D("freqloop", 6710, 6725),
        res_gain_ge=0.15,
        res_pulse_type="flat_top",
        res_sigma=0.01,
    )
    p = measurement.compile(ResonatorSpecProgram, cfg)
    np.testing.assert_allclose(
        p.get_pulse_param("Q1__res_pulse", "freq", as_array=True), np.linspace(6710, 6725, 51), atol=0.001
    )
    assert p.get_pulse_param("Q1__res_pulse", "gain") == pytest.approx(0.15, abs=1e-4)
    first = p.envelopes[2]["envs"]["Q1__res_envelope"]["data"].shape
    cfg["res_sigma"] = 0.02
    p2 = measurement.compile(ResonatorSpecProgram, cfg)
    assert p2.envelopes[2]["envs"]["Q1__res_envelope"]["data"].shape[0] > first[0]


@pytest.mark.parametrize("program", [T1GEProgram, T1EFProgram, SpinEchoGEProgram])
def test_wait_loop_name_and_transition(program, config_all, measurement):
    cfg = config_all["Q1"].for_run(steps=41, wait_us=QickSweep1D("my_wait", 0, 100))
    p = measurement.compile(program, cfg)
    assert ("my_wait", 41) in [(name, count) for name, count, *_ in p.loops]
    assert p.cfg["transition"] == program.TRANSITION
    if program.TRANSITION == "ef":
        assert "Q1__x_ge" in p.pulses and "Q1__x_ef" in p.pulses
    axes = infer_axes(p)
    coords = extract_coords(p, ProgramPlan(program, p.cfg, axes), "Q1")["delay"] * getattr(
        program, "AXIS_SCALE", 1
    )
    scale = getattr(program, "AXIS_SCALE", 1)
    np.testing.assert_array_equal(coords, p.get_time_param("evolution", "t", as_array=True) * scale)
    # Native integer step rounding accumulates over the loop. Bound it by one
    # timing tick per half-delay per step, rather than treating requested values as actual values.
    tick = 1 / measurement.soccfg["tprocs"][0]["f_time"]
    assert abs(coords[-1] - 100) <= 41 * tick * scale


def test_mux_subset_uses_updated_inactive_tone_context(native_config, tmp_path):
    from QickworkspaceV2 import Device

    config_all = ExperimentConfig(
        Device.from_files("lab/profiles/mux/hardware.yaml", "lab/profiles/mux/device.yaml")
    )
    native_config["gens"][2].update(
        type="axis_sg_mux8_v1", n_tones=8, has_gain=True, has_phase=True, has_mixer=True
    )
    lab = Measurement(None, QickConfig(native_config), data_path=tmp_path)
    q2 = config_all["Q2"]
    config_all["Q1"].update(res_freq_ge=6718.5)
    p = lab.compile(T1GEProgram, q2.for_run(steps=41, wait_us=QickSweep1D("waitloop", 0, 100)))
    assert list(p.ro_chs) == [1]
    assert p.gen_chs[2]["mux_tones"][0]["freq_rounded"] == pytest.approx(6718.5, abs=0.001)
    assert p.gen_chs[2]["mux_tones"][1]["freq_rounded"] == pytest.approx(6740, abs=0.001)


@pytest.mark.parametrize("key,value", [("res_gain_ge", 0.9), ("qb_ch", -1), ("ro_ch", True), ("reps", 0)])
def test_invalid_native_values_rejected(key, value, limited_device, measurement):
    config_all = ExperimentConfig(limited_device)
    cfg = config_all["Q1"].for_run(steps=41, wait_us=QickSweep1D("waitloop", 0, 100))
    cfg[key] = value
    with pytest.raises(ValueError):
        measurement.compile(T1GEProgram, cfg)


def test_no_connection_cannot_acquire(measurement, config_all):
    with pytest.raises(RuntimeError, match="No QICK connection"):
        measurement.run(T1GEProgram, config_all["Q1"].for_run())


def test_direct_acquisition_saves_averages_and_failed_analysis(measurement, config_all, monkeypatch):
    measurement.backend.soc = object()
    calls = []

    def acquire(program, soc, **kwargs):
        calls.append(kwargs)
        return [np.full((1, 41, 2), len(calls) + channel) for channel in program.ro_chs]

    monkeypatch.setattr(BaseProgram, "acquire", acquire)
    cfg = config_all.for_run("Q2", "Q1", steps=41, wait_us=QickSweep1D("waitloop", 0, 100))

    def broken(result):
        raise RuntimeError("fit failure fixture")

    r = measurement.run(T1GEProgram, cfg, py_avg=2, analyze=broken)
    assert r.targets == ("Q2", "Q1")
    np.testing.assert_allclose(r["Q2"].iq, 2.5 + 2.5j)
    np.testing.assert_allclose(r["Q1"].iq, 1.5 + 1.5j)
    assert r.analysis_status == "failed"
    assert len(calls) == 2
    raw = measurement.load(r.run_id, revision=0)
    assert raw.analysis_status == "not_analyzed"
    np.testing.assert_array_equal(raw["Q2"].iq, r["Q2"].iq)
    compiled = json.loads(
        (measurement.store.directory(r.run_id) / "compiled.json").read_text(encoding="utf-8")
    )
    assert "asm" in compiled and compiled["config"]["steps"] == 41


def test_shots_rounds_and_normalization(measurement, config_all, monkeypatch):
    measurement.backend.soc = object()
    count = [0]

    def acquire(program, soc, **kwargs):
        count[0] += 1
        return [np.full((2, 2), count[0])]

    def get_raw(program):
        return [np.full((40, 2, 2), count[0] * program.ro_chs[0]["length"])]

    monkeypatch.setattr(BaseProgram, "acquire", acquire)
    monkeypatch.setattr(BaseProgram, "get_raw", get_raw)
    r = measurement.run(SingleShotProgram, config_all["Q1"].for_run(reps=40), py_avg=2)
    assert r["Q1"].shots.shape == (80, 2)
    np.testing.assert_allclose(r["Q1"].shots[:40], 1 + 1j)
    np.testing.assert_allclose(r["Q1"].shots[40:], 2 + 2j)
    assert list(r["Q1"].coords["readout"]) == ["ground", "excited"]


def test_cancel_keeps_partial(measurement, config_all, monkeypatch):
    measurement.backend.soc = object()
    monkeypatch.setattr(BaseProgram, "acquire", lambda p, soc, **kw: [np.ones((1, 41, 2))])
    cancel = Event()

    def progress(event):
        cancel.set()

    cfg = config_all["Q1"].for_run(steps=41, wait_us=QickSweep1D("waitloop", 0, 100))
    with pytest.raises(AcquisitionCancelled):
        measurement.run(T1GEProgram, cfg, py_avg=2, cancel=cancel, on_progress=progress)
    run = measurement.store.list()[0]
    assert run["status"] == "cancelled"
    assert ExperimentData.load(measurement.store.directory(run["id"]) / "partial.h5")["Q1"].iq.shape == (
        41,
        1,
    )


@pytest.mark.parametrize("sigma", [0.013, 0.02, 0.037])
def test_edited_envelopes_do_not_overlap_ge_ef_or_gate_layers(sigma, config_all, measurement, caplog):
    config_all["Q1"].update(sigma_ge=sigma, sigma_ef=sigma)
    cfg = config_all["Q1"].for_run(steps=41, wait_us=QickSweep1D("waitloop", 0, 100))
    measurement.compile(T1EFProgram, cfg)

    class GatePair(BaseProgram):
        def _initialize(self, cfg):
            self.setup_device(cfg, gates=True)

        def _body(self, cfg):
            self.qubit("Q1").halfx()
            self.qubit("Q1").x()
            self.measure(cfg)

    measurement.compile(GatePair, config_all["Q1"].for_run())
    assert not any("conflict" in record.message for record in caplog.records)
