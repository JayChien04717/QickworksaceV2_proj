"""Real QICK compilation with numerical transport fixtures; no board acquisition."""

import numpy as np
import pytest
from qick import QickConfig

from QickworkspaceV2 import BaseProgram, ExperimentConfig, Measurement, NotebookLab, QickSweep1D
from QickworkspaceV2.backends.qick import extract_coords
from QickworkspaceV2.experiments.resonator_punchout import PunchoutProgram, plot
from QickworkspaceV2.runtime.measurement import infer_axes


def run_config(device):
    qb = ExperimentConfig(device)["Q1"]
    return qb.for_run(
        f_steps=8, g_steps=3,
        res_freq_ge=QickSweep1D("freqloop", qb["res_freq_ge"] - 10, qb["res_freq_ge"] + 10),
        res_gain_ge=QickSweep1D("gainloop", 0.02, 0.08),
    )


def test_punchout_compiles_two_fpga_loops_and_target_specific_coordinates(session):
    _, _, plan, _, _, _ = session._prepare(
        "resonator_punchout", {"target": "Q2,Q1", "g_steps": 3, "f_steps": 8, "gain_stop": 0.08}
    )
    assert "host_sweep" not in plan.metadata
    program = session.backend.compile(plan)
    assert [(name, count) for name, count, *_ in program.loops if name != "reps"] == [
        ("gainloop", 3), ("freqloop", 8)
    ]
    assert [axis.name for axis in infer_axes(program)] == ["gain", "frequency"]
    assert set(program.gen_chs) == {2, 3}  # No qubit drive is needed.
    session._check_readouts(program, plan)
    for target in ("Q2", "Q1"):
        coords = extract_coords(program, plan, target)
        assert coords["gain"].shape == (3,) and coords["frequency"].shape == (8,)
        np.testing.assert_array_equal(
            coords["frequency"], program.get_pulse_param(target + "__res_pulse", "freq", as_array=True)
        )
        np.testing.assert_array_equal(
            coords["gain"], program.get_pulse_param(target + "__res_pulse", "gain", as_array=True)
        )
        center = session.device.flat_qubit(target)["res_freq_ge"]
        np.testing.assert_allclose(coords["frequency"][[0, -1]], [center - 10, center + 10], atol=0.001)


def test_notebook_and_catalog_compile_the_same_native_punchout(session, device, native_config, tmp_path):
    direct = Measurement(None, QickConfig(native_config), data_path=tmp_path / "direct")
    program = direct.compile(PunchoutProgram, run_config(device))
    catalog = session.compile("resonator_punchout", g_steps=3, f_steps=8, gain_stop=0.08)
    np.testing.assert_array_equal(program.binprog["pmem"], catalog.binprog["pmem"])


def test_one_fpga_map_per_average_preserves_iq_axes_and_final_plot(session, monkeypatch):
    calls = []
    expected = np.arange(24).reshape(3, 8) + 1j * np.arange(24, 48).reshape(3, 8)

    def acquire(program, soc, **kwargs):
        calls.append(program)
        return [np.stack((expected.real, expected.imag), axis=-1)[None, ...]]

    monkeypatch.setattr(BaseProgram, "acquire", acquire)
    result = session.run("resonator_punchout", g_steps=3, f_steps=8, gain_stop=0.08, soft_avgs=2)
    assert len(calls) == 2 and calls[0] is calls[1]  # No Python loop over gain/frequency.
    assert result["Q1"].dims == ("gain", "frequency", "readout")
    np.testing.assert_array_equal(result["Q1"].iq[..., 0], expected)
    assert "child_runs" not in result.metadata and len(session.store.list()) == 1
    loaded = session.store.load(result.run_id)
    np.testing.assert_array_equal(loaded["Q1"].iq, result["Q1"].iq)
    for axis in ("gain", "frequency"):
        np.testing.assert_array_equal(loaded["Q1"].coords[axis], result["Q1"].coords[axis])
    assert result.analysis_status == "completed" and not result.fits["Q1"].success
    figure = plot(result)
    assert figure.axes[0].get_xlabel() == "frequency (MHz)"
    assert figure.axes[0].get_ylabel() == "gain (normalized gain)"
    np.testing.assert_allclose(figure.axes[0].collections[0].get_array().reshape(3, 8), abs(expected))


def test_interrupted_punchout_clears_live_map_and_shows_partial_final_map(
    session, device, native_config, tmp_path, monkeypatch
):
    lab = NotebookLab(working_config=tmp_path / "working.yaml")
    lab._measurement = Measurement(object(), QickConfig(native_config), data_path=tmp_path / "partial")
    displayed, updates, calls = [], [], []

    class Handle:
        def update(self, value):
            updates.append(value)

    def display(value, **kwargs):
        displayed.append(value)
        return Handle() if kwargs.get("display_id") else None

    def acquire(program, soc, **kwargs):
        calls.append(program)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return [np.ones((1, 3, 8, 2))]

    monkeypatch.setattr("IPython.display.display", display)
    monkeypatch.setattr(BaseProgram, "acquire", acquire)
    result = lab.run(PunchoutProgram, run_config(device), py_avg=10)
    assert result.metadata["interrupted"] and result.path.name == "partial.h5"
    assert result["Q1"].iq.shape == (3, 8, 1)
    assert len(displayed) == 2 and updates[-1].data == ""
    assert displayed[-1].axes[0].get_xlabel() == "frequency (MHz)"
    assert (result.path.parent / "resonator_punchout.png").exists()


def test_punchout_checks_gain_sweep_endpoint_against_port_limit(limited_session):
    with pytest.raises(ValueError, match="gain exceeds configured port limit"):
        limited_session.compile("resonator_punchout", gain_stop=0.6)


@pytest.mark.parametrize("unsupported", ["mux", "static_readout"])
def test_punchout_rejects_unsupported_readout_instead_of_using_host_loop(
    device, native_config, tmp_path, unsupported
):
    cfg = run_config(device)
    if unsupported == "mux":
        cfg["readout_mode"] = "mux"
    else:
        del native_config["readouts"][cfg["ro_ch"]]["tproc_ctrl"]
    lab = Measurement(None, QickConfig(native_config), data_path=tmp_path)
    with pytest.raises(ValueError, match="FPGA punch-out requires"):
        lab.compile(PunchoutProgram, cfg)
