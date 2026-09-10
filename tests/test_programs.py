import numpy as np
import pytest
from QickworkspaceV2 import Session, QICKBackend, BaseProgram, Parameters, ProgramPlan, ExperimentSpec, Sweep
from QickworkspaceV2.backends.qick import extract_coords, unpack_iq


@pytest.mark.parametrize(
    "name",
    [
        "time_of_flight",
        "resonator_spec",
        "qubit_spec_ge",
        "power_rabi_ge",
        "time_rabi_ge",
        "t1_ge",
        "ramsey_ge",
        "spin_echo_ge",
        "single_shot",
        "conditional_ramsey",
        "coupler_chevron",
        "allxy",
        "randomized_benchmarking",
    ],
)
def test_real_qick_compiler(name, device, native_config, tmp_path):
    from qick import QickConfig

    s = Session(device, QICKBackend(None, QickConfig(native_config)), data_dir=tmp_path)
    _, _, plan, _, _, _ = s._prepare(name, {"target": "Q1,Q2"})
    program = s.backend.compile(plan)
    assert len(program.binprog["pmem"]) > 0
    s._check_readouts(program, plan)
    for q in plan.cfg["targets"]:
        coords = extract_coords(program, plan, q)
        for sweep in plan.sweeps:
            assert len(coords[sweep.name]) == sweep.points


@pytest.mark.parametrize(
    "name", ["qubit_spec_ge", "power_rabi_ge", "time_rabi_ge", "t1_ge", "ramsey_ge", "spin_echo_ge"]
)
def test_real_qick_ef(name, device, native_config, tmp_path):
    from qick import QickConfig

    s = Session(device, QICKBackend(None, QickConfig(native_config)), data_dir=tmp_path)
    p = s.compile(name.replace("_ge", "_ef"), target="Q1,Q2")
    assert "Q1__x_ge" in p.pulses
    assert list(p.ro_chs) == [0, 1]


def test_gate_timing_is_explicit(session, monkeypatch):
    calls = []
    original_pulse, original_delay = BaseProgram.pulse, BaseProgram.delay_auto

    def pulse(self, *args, **kwargs):
        calls.append(("pulse", kwargs["name"]))
        return original_pulse(self, *args, **kwargs)

    def delay(self, *args, **kwargs):
        calls.append(("delay", None))
        return original_delay(self, *args, **kwargs)

    monkeypatch.setattr(BaseProgram, "pulse", pulse)
    monkeypatch.setattr(BaseProgram, "delay_auto", delay)
    session.compile("ramsey_ge", target="Q1,Q2")
    start = calls.index(("pulse", "Q1__halfx_ge"))
    assert calls[start : start + 3] == [("pulse", "Q1__halfx_ge"), ("pulse", "Q2__halfx_ge"), ("delay", None)]


def test_raw_mapping_and_multidimensional_shape(device, native_config, tmp_path):
    from qick import QickConfig

    s = Session(device, QICKBackend(None, QickConfig(native_config)), data_dir=tmp_path)
    _, _, plan, _, _, _ = s._prepare(
        "coupler_chevron", {"target": "Q2,Q1", "gain_points": 3, "length_points": 8}, {"reps": 2}
    )
    p = s.backend.compile(plan)
    average = [np.full((1, 3, 8, 2), i + 1.0) for i in range(2)]
    raw = [np.full((2, 3, 8, 1, 2), i + 1.0) for i in range(2)]
    traces = unpack_iq(p, plan, average, raw)
    assert traces["Q1"].dims == ("gain", "length", "readout")
    assert traces["Q1"].shots.shape == (3, 8, 2, 1)
    q1_index = list(p.ro_chs).index(0)
    assert np.all(traces["Q1"].iq == (q1_index + 1) * (1 + 1j))
    with pytest.raises(ValueError, match="shape"):
        unpack_iq(p, plan, [np.zeros((3, 8, 2))] * 2)


def test_native_flat_authoring(session, device, native_config, tmp_path):
    class NativeSpec(BaseProgram):
        def _initialize(self, cfg):
            self.setup_resonator(cfg)
            self.setup_qubit_gen(cfg, "ge")
            self.add_loop("freqloop", cfg["steps"])
            self.setup_qb_pulse(cfg, "ge", name="qb_pulse", pulse_type="flat_top")

        def _body(self, cfg):
            self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
            self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
            self.delay_auto(0.05)
            self.measure(cfg)

    def build(ctx, params):
        cfg = ctx.native_config()
        sweep = Sweep("frequency", 2800.0, 2900.0, 81, "MHz", "freqloop", "freq", "qb_pulse")
        cfg.update(steps=81, qb_freq_ge=sweep.qick())
        return ProgramPlan(NativeSpec, cfg, (sweep,))

    spec = ExperimentSpec("native", Parameters, build)
    result = session.run(spec)
    assert result["Q1"].iq.shape == (81, 1)
    from qick import QickConfig

    hardware = Session(device, QICKBackend(None, QickConfig(native_config)), data_dir=tmp_path / "native")
    program = hardware.compile(spec)
    assert len(program.binprog["pmem"]) > 0
    assert len(program.get_pulse_param("qb_pulse", "freq", as_array=True)) == 81


def test_clifford_inverses():
    from QickworkspaceV2.programs.cliffords import cliffords, randomized_sequence

    assert len(cliffords()) == 24
    eye = np.eye(2, dtype=complex)
    gates = {
        "halfx": (eye - 1j * np.array([[0, 1], [1, 0]])) / np.sqrt(2),
        "halfy": (eye - 1j * np.array([[0, -1j], [1j, 0]])) / np.sqrt(2),
    }
    for seed in range(10):
        sequence = randomized_sequence(40, np.random.default_rng(seed))
        u = eye
        for gate in sequence:
            u = gates[gate] @ u
        assert abs(np.trace(u)) == pytest.approx(2)


def test_conditional_ramsey_reuses_waveforms_with_small_generator_memory(device, native_config, tmp_path):
    from qick import QickConfig

    native_config["gens"][1]["maxlen"] = 16384
    session = Session(device, QICKBackend(None, QickConfig(native_config)), data_dir=tmp_path)
    program = session.compile("conditional_ramsey", target="Q1,Q2")
    for channel in (0, 1):
        assert len(program.envelopes[channel]["envs"]) == 1
        assert program.envelopes[channel]["next_addr"] <= native_config["gens"][channel]["maxlen"]
    assert program.get_pulse_param("Q2__x_ge", "gain") == pytest.approx(0.1, abs=1e-4)
    assert program.get_pulse_param("Q2__halfx_ge", "gain") == pytest.approx(0.05, abs=1e-4)
    assert program.get_pulse_param("Q2__y_ge", "phase") == pytest.approx(90)
    assert len(program.get_pulse_param("Q2__analysis90", "phase", as_array=True)) == 81


def test_qubit_envelope_cache_preserves_distinct_shapes(device, native_config, tmp_path):
    from qick import QickConfig
    from QickworkspaceV2 import ExperimentConfig, Measurement

    class Waveforms(BaseProgram):
        def _initialize(self, cfg):
            self.setup_device(cfg)
            qc = cfg["qubits"]["Q1"]
            self.setup_qb_pulse(qc, name="gauss")
            self.setup_qb_pulse(qc, name="same", shape="gaussian", phase=90, gain_override=0.025)
            self.setup_qb_pulse({**qc, "sigma_ge": 0.04}, name="shorter_sigma")
            self.setup_qb_pulse(qc, name="longer", length_mult=6)
            self.setup_qb_pulse(qc, name="cosine", shape="cosine")
            for name, alpha, delta in (("drag", 0.2, -200), ("alpha", 0.4, -200), ("delta", 0.2, -150)):
                self.setup_qb_pulse(
                    {**qc, "drag_alpha_ge": alpha, "drag_delta_ge": delta}, name=name, shape="drag"
                )

        def _body(self, cfg):
            self.measure(cfg)

    lab = Measurement(None, QickConfig(native_config), data_path=tmp_path)
    program = lab.compile(Waveforms, ExperimentConfig(device)["Q1"].for_run())
    envelopes = program.envelopes[0]["envs"]
    assert len(envelopes) == 7
    assert "Q1__same__env" not in envelopes
    assert not np.array_equal(envelopes["Q1__drag__env"]["data"], envelopes["Q1__alpha__env"]["data"])
    assert not np.array_equal(envelopes["Q1__drag__env"]["data"], envelopes["Q1__delta__env"]["data"])


def test_native_envelope_memory_overflow_rejected_before_acquisition(device, native_config, tmp_path):
    from qick import QickConfig
    from QickworkspaceV2 import ExperimentConfig, Measurement

    class NativeEnvelopes(BaseProgram):
        def _initialize(self, cfg):
            self.setup_device(cfg)
            for name in ("first", "second"):
                self.add_gauss(ch=1, name=name, sigma=0.2, length=1.0, even_length=True)

        def _body(self, cfg):
            self.measure(cfg)

    native_config["gens"][1]["maxlen"] = 16384
    lab = Measurement(None, QickConfig(native_config), data_path=tmp_path)
    with pytest.raises(ValueError, match="Generator 1 needs .* envelope samples; firmware supports 16384"):
        lab.compile(NativeEnvelopes, ExperimentConfig(device)["Q2"].for_run())
    assert lab.store.list() == []
