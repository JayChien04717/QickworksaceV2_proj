import numpy as np
import pytest
from qick import QickConfig
from qick.asm_v2 import QickSweep1D

from QickworkspaceV2 import BaseProgram, ExperimentConfig, ExperimentData, Measurement, TraceData
from QickworkspaceV2.analysis.diagnostics import summarize_traces
from QickworkspaceV2.analysis.readout import analyze_single_shot
from QickworkspaceV2.experiments.time_of_flight import TimeOfFlightProgram, TimeOfFlightParameters, analyze as analyze_tof
from QickworkspaceV2.experiments.active_reset_rabi import ActiveResetRabiProgram
from QickworkspaceV2.experiments.single_shot_gef import SingleShotGEFProgram, analyze as analyze_gef
from QickworkspaceV2.experiments.qubit_temperature import QubitTemperatureProgram
from QickworkspaceV2.experiments.resonator_spec import ResonatorSpecProgram
from QickworkspaceV2.experiments.t1_ge import T1GEProgram


@pytest.mark.parametrize("interleaved", [None, "x", "y", "halfx", "mhalfx", "halfy", "mhalfy"])
def test_rb_recovery_inverts_interleaved_gate_sequence(interleaved):
    from QickworkspaceV2.programs.cliffords import randomized_sequence
    eye = np.eye(2, dtype=complex)
    x, y = np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]])
    gates = {"x": -1j*x, "y": -1j*y, "halfx": (eye-1j*x)/np.sqrt(2),
             "halfy": (eye-1j*y)/np.sqrt(2), "mhalfx": (eye+1j*x)/np.sqrt(2),
             "mhalfy": (eye+1j*y)/np.sqrt(2)}
    rng = np.random.default_rng(812)
    for depth in (0, 1, 7, 31):
        total = eye.copy()
        for gate in randomized_sequence(depth, rng, interleaved):
            total = gates[gate] @ total
        assert abs(np.trace(total)) == pytest.approx(2)


def test_working_update_rejects_partial_mismatch_and_invalid_fit(notebook_lab, tmp_path):
    from QickworkspaceV2 import FitResult
    from QickworkspaceV2.notebook import apply_working
    lab, config = notebook_lab
    result = ExperimentData("test", {"Q1": TraceData(np.array([1j]), ("readout",),
                                                    {"readout": np.array(["g"])}, {})},
                            metadata={"acquisition_status": "partial", "targets": ["Q1"]})
    result.path = tmp_path / "acquisition.h5"
    result.analysis_status = "completed"
    result.fits = {"Q1": FitResult("test", True, {"center": 123.0})}
    path = tmp_path / "working.yaml"
    previous = dict(config["Q1"])
    with pytest.raises(ValueError, match="Partial"):
        apply_working(result, config, path, target="Q1", res_freq_ge=123)
    assert dict(config["Q1"]) == previous and not path.exists()
    result.metadata["acquisition_status"] = "completed"
    with pytest.raises(ValueError):
        apply_working(result, config, path, target="Q2", res_freq_ge=123)
    result.fits["Q1"].success = False
    with pytest.raises(ValueError, match="Fit rejected"):
        apply_working(result, config, path, target="Q1", res_freq_ge=123)
    result.fits["Q1"].success = True
    with pytest.raises(ValueError, match="Nonfinite"):
        apply_working(result, config, path, target="Q1", res_freq_ge=float("nan"))
    apply_working(result, config, path, target="Q1", res_freq_ge=123)
    assert config["Q1"]["res_freq_ge"] == 123
    assert ExperimentConfig.load(path)["Q1"]["res_freq_ge"] == 123


def test_feedback_plot_uses_population_and_only_overlays_fitted_event():
    from QickworkspaceV2 import FitResult
    from QickworkspaceV2.plotting import plot_result
    shots = np.tile(np.array([[-1, 1], [1, 1]], complex), (3, 1, 1))
    trace = TraceData(np.zeros((3, 2)), ("gain", "readout"),
                      {"gain": np.arange(3), "readout": np.array(["pre", "post"])},
                      shots=shots, shot_dims=("gain", "shot", "readout"),
                      metadata={"threshold": 0, "rotation_deg": 0, "fit_signal": "population", "fit_event": 0})
    result = ExperimentData("active_reset", {"Q1": trace}, metadata={"iq_process": "abs"})
    result.fits["Q1"] = FitResult("rabi", True, x_fit=[0, 1, 2], y_fit=[.5, .5, .5], residuals=[0, 0, 0])
    figure = plot_result(result)
    np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), [.5, .5, .5])
    assert "population" in figure.axes[0].get_ylabel()
    assert len(figure.axes[0].lines) == 3
    assert len(plot_result(result, event=1).axes[0].lines) == 1
    assert len(plot_result(result, signal="abs").axes[0].lines) == 1


@pytest.fixture
def notebook_lab(device, native_config, tmp_path):
    return (Measurement(None, QickConfig(native_config), data_path=tmp_path, resource_id="chip-tests"),
            ExperimentConfig(device))


@pytest.mark.parametrize("state", ["g", "e", "f"])
def test_tof_state_preparation_compiles(notebook_lab, state):
    lab, config = notebook_lab
    cfg = config["Q1"].for_run(reps=1, check_e=state == "e", check_f=state == "f",
                               transition="ef" if state == "f" else "ge")
    p = lab.compile(TimeOfFlightProgram, cfg)
    assert p.ro_chs[0]["trigs"] == 1
    assert ("Q1__x_ge" in p.pulses) == (state != "g")
    assert ("Q1__x_ef" in p.pulses) == (state == "f")
    with pytest.raises(ValueError):
        TimeOfFlightParameters(check_e=True, check_f=True)


@pytest.mark.parametrize("mode", ["never", "always", "conditional"])
def test_feedback_rabi_native_compile(notebook_lab, mode):
    lab, config = notebook_lab
    cfg = config["Q1"].for_run(steps=9, qb_gain_ge=QickSweep1D("gainloop", 0, .1),
                               ro_threshold=.02, ro_phase=0, reset_mode=mode)
    p = lab.compile(ActiveResetRabiProgram, cfg)
    assert p.ro_chs[0]["trigs"] == 2
    assert p.feedback_threshold == round(.02 * p.ro_chs[0]["length"])
    assert ("skip_reset" in str(p)) == (mode == "conditional")
    cfg["ro_phase"] = 30
    with pytest.raises(ValueError, match="Hardware feedback"):
        lab.compile(ActiveResetRabiProgram, cfg)


def test_gef_and_temperature_compile(notebook_lab):
    lab, config = notebook_lab
    assert lab.compile(SingleShotGEFProgram, config["Q1"].for_run(reps=40)).ro_chs[0]["trigs"] == 3
    cfg = config["Q1"].for_run(steps=9, qb_gain_ef=QickSweep1D("gainloop", 0, .1))
    assert lab.compile(QubitTemperatureProgram, cfg).ro_chs[0]["trigs"] == 2


def test_keyboard_interrupt_keeps_current_result_and_analysis_status(notebook_lab, monkeypatch):
    lab, config = notebook_lab
    lab.backend.soc = object()
    monkeypatch.setattr(BaseProgram, "acquire", lambda p, soc, **kw: [np.ones((1, 9, 2))])
    cfg = config["Q1"].for_run(steps=9, wait_us=QickSweep1D("waitloop", 0, 20))
    previous = lab.run(T1GEProgram, cfg, py_avg=1, analyze=None)

    def interrupt(event):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        lab.run(T1GEProgram, cfg, py_avg=1000, on_progress=interrupt)
    partial = lab.last_result
    assert partial.run_id != previous.run_id
    assert partial.path.name == "partial.h5"
    assert partial.metadata["completed_averages"] == 1
    assert partial.metadata["interrupted"] is True
    analyzed = lab.analyze(partial, summarize_traces)
    assert analyzed.analysis_status == "completed"
    assert lab.store.list()[0]["status"] == "cancelled"
    assert not (partial.path.parent / "acquisition.h5").exists()
    loaded = lab.load(partial.run_id, partial=True)
    assert loaded.metrics == analyzed.metrics
    # An interrupt before the next first round must never expose the old result.
    monkeypatch.setattr(BaseProgram, "acquire", lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        lab.run(T1GEProgram, cfg, py_avg=2)
    assert lab.last_result is None


def test_direct_host_scan_preserves_rounded_axis_and_children(notebook_lab, monkeypatch):
    lab, config = notebook_lab
    lab.backend.soc = object()
    monkeypatch.setattr(BaseProgram, "acquire", lambda p, soc, **kw: [np.ones((1, 2))])
    cfg = config["Q1"].for_run()
    result = lab.scan(ResonatorSpecProgram, cfg, "res_freq_ge", [6712, 6713, 6714],
                      axis_name="frequency", unit="MHz", py_avg=1, analyze=None,
                      )
    assert result["Q1"].iq.shape == (3, 1)
    assert len(result.metadata["child_runs"]) == 3
    assert result.metadata["host_coordinate_source"] == "compiled"
    assert len(lab.store.list()) == 4
    np.testing.assert_allclose(result["Q1"].coords["frequency"], [6712, 6713, 6714], atol=.001)


def test_population_thresholds_each_shot_and_preserves_iq():
    shots = np.array([[-1], [3]], complex)
    trace = TraceData(np.array([1]), ("readout",), {"readout": ["measurement"]},
                      shots=shots, shot_dims=("shot", "readout"), metadata={"threshold": 0})
    np.testing.assert_array_equal(trace.signal("population"), [.5])
    np.testing.assert_array_equal(trace.iq, [1])
    np.testing.assert_array_equal(trace.shots, shots)


def test_notebook_sdk_clears_live_plot_after_interrupt_before_final_plot(notebook_lab, monkeypatch, tmp_path):
    from QickworkspaceV2 import NotebookLab
    raw, config = notebook_lab
    raw.backend.soc = object()
    facade = NotebookLab(working_config=tmp_path / "working.yaml")
    facade._measurement, facade.config = raw, config
    displayed = []
    class Handle:
        def update(self, value):
            displayed[0] = value

    def display(value, **kwargs):
        displayed.append(value)
        return Handle() if kwargs.get("display_id") else None

    monkeypatch.setattr("IPython.display.display", display)
    calls = []

    def acquire(p, soc, **kw):
        calls.append(True)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return [np.ones((1, 9, 2))]

    monkeypatch.setattr(BaseProgram, "acquire", acquire)

    result = facade.run(T1GEProgram, config["Q1"].for_run(
        steps=9, wait_us=QickSweep1D("waitloop", 0, 20)), py_avg=1000)
    assert len(calls) == 2 and result.metadata["interrupted"]
    assert result.analysis_status == "completed" and not result.is_good()
    assert result.path.name == "partial.h5"
    assert (result.path.parent / "t1_ge.png").exists() and len(displayed) == 2
    assert displayed[0].data == ""  # Live PNG was removed without clearing the final Figure.
    assert raw.store.list()[0]["status"] == "cancelled"
    assert not facade.apply_fit(result)
    assert not facade.working_config.exists()


def test_host_scan_interruption_keeps_accumulated_parent(notebook_lab, monkeypatch):
    raw, config = notebook_lab
    raw.backend.soc = object()
    monkeypatch.setattr(BaseProgram, "acquire", lambda p, soc, **kw: [np.ones((1, 2))])
    seen = []

    def interrupt(event):
        seen.append(event["result"])
        if event["completed"] == 2:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        raw.scan(ResonatorSpecProgram, config["Q1"].for_run(), "res_freq_ge", [6712, 6713, 6714],
                 axis_name="frequency", py_avg=1, on_progress=interrupt)
    assert seen[0].run_id == seen[1].run_id == raw.last_result.run_id
    result = raw.load(raw.last_run_id, partial=True)
    assert result["Q1"].iq.shape == (2, 1)
    assert result.metadata["completed_points"] == 2 and result.metadata["requested_points"] == 3
    assert result.metadata["interrupted"]
    assert len(result.metadata["child_runs"]) == 2
    assert all(raw.load(child).metadata["acquisition_status"] == "completed"
               for child in result.metadata["child_runs"])



def test_fixed_classifier_and_gef_labels():
    rng = np.random.default_rng(32)
    shots = .02 * (rng.normal(size=(100, 3)) + 1j*rng.normal(size=(100, 3))) + np.array([-1, 1, 2j])
    trace = TraceData(shots.mean(axis=0), ("readout",),
                      {"readout": ["ground", "excited", "second_excited"]},
                      shots=shots, shot_dims=("shot", "readout"))
    assert analyze_gef(ExperimentData("single_shot_gef", {"Q1": trace}))["Q1"].parameters["fidelity"] == 1
    ge = TraceData(shots[:, :2].mean(axis=0), ("readout",), {"readout": ["ground", "excited"]},
                   shots=shots[:, :2], shot_dims=("shot", "readout"))
    result = ExperimentData("single_shot", {"Q1": ge})
    assert analyze_single_shot(result, threshold=0)["Q1"].parameters["fidelity"] == 1
    assert analyze_single_shot(result, threshold=2)["Q1"].parameters["fidelity"] == .5


def test_tof_analysis_respects_selected_threshold():
    trace = TraceData(np.array([0, 1, 3, 1], complex)[:, None], ("time", "readout"),
                      {"time": [0, 1, 2, 3], "readout": ["measurement"]})
    result = ExperimentData("time_of_flight", {"Q1": trace}, metadata={"parameters": {"tof_threshold": 2}})
    fit = analyze_tof(result)["Q1"]
    assert fit.parameters["first_crossing_us"] == 2
    assert not fit.success  # A diagnostic is not a trigger calibration.
