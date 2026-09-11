"""Numerical records verify update ownership; no hardware acquisition here."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pytest

from QickworkspaceV2 import ExperimentConfig, ExperimentData, FitResult, NotebookLab, TraceData
from QickworkspaceV2.experiments import default_registry
from QickworkspaceV2.calibration.updates import CalibrationUpdates


def record(name, tmp_path, *, values=None, cfg=None, target="Q1"):
    result = ExperimentData(
        name,
        {target: TraceData(np.array([1j]), ("readout",), {"readout": ["measurement"]})},
        metadata={"acquisition_status": "completed", "resolved_config": {"qubits": {target: cfg or {}}}},
    )
    result.fits = {target: FitResult("numerical_fixture", True, values or {"tau": 25.0})}
    result.analysis_status = "completed"
    result.path = tmp_path / "acquisition.h5"
    return result


def facade(device, tmp_path):
    lab = NotebookLab(working_config=tmp_path / "working.yaml", calibration_enabled=True)
    lab.config = ExperimentConfig(device)
    return lab


@pytest.mark.parametrize("transition", ["ge", "ef"])
@pytest.mark.parametrize("name,key", [("t1", "t1"), ("ramsey", "ramsey"), ("spin_echo", "echo")])
def test_coherence_updates_only_its_local_metric(name, key, transition, device, tmp_path):
    lab = facade(device, tmp_path)
    original = dict(lab.config["Q1"])
    result = record(name + "_" + transition, tmp_path)
    assert lab.apply_fit(result)
    expected = {**original, f"{key}_{transition}_us": 25.0}
    assert dict(lab.config["Q1"]) == expected
    assert dict(ExperimentConfig.load(lab.working_config)["Q1"]) == expected
    assert default_registry().get(result.experiment).updates(result, "Q1").for_device() == {}


@pytest.mark.parametrize("transition", ["ge", "ef"])
def test_rabi_uses_measured_sigma_and_never_touches_other_transition(transition, device, tmp_path):
    lab = facade(device, tmp_path)
    cfg = {f"sigma_{transition}": 0.035, f"pulse_type_{transition}": "arb"}
    result = record("power_rabi_" + transition, tmp_path, values={"pi_gain": 0.08, "pi2_gain": 0.04}, cfg=cfg)
    original = dict(lab.config["Q1"])
    assert lab.apply_fit(result)
    expected = {**original, **cfg, f"pi_gain_{transition}": 0.08, f"pi2_gain_{transition}": 0.04}
    assert dict(lab.config["Q1"]) == expected
    plan = default_registry().get(result.experiment).updates(result, "Q1")
    assert plan.for_device()[f"qubits.Q1.transitions.{transition}.pulse.sigma_us"] == 0.035
    assert all(f".transitions.{transition}." in path for path in plan.for_device())


@pytest.mark.parametrize(
    "rejection", ["partial", "failed_fit", "wrong_target", "nonfinite", "disabled", "diagnostic"]
)
def test_rejected_updates_leave_disk_and_live_settings_unchanged(rejection, device, tmp_path):
    lab = facade(device, tmp_path)
    result = record("t1_ge", tmp_path)
    if rejection == "partial":
        result.metadata["acquisition_status"] = "partial"
    elif rejection == "failed_fit":
        result.fits["Q1"].success = False
    elif rejection == "wrong_target":
        lab.qubit = "Q2"
    elif rejection == "nonfinite":
        result.fits["Q1"].parameters["tau"] = float("nan")
    elif rejection == "disabled":
        lab.calibration_enabled = False
    else:
        result.experiment = "time_of_flight"
    original = deepcopy(dict(lab.config["Q1"]))
    assert not lab.apply_fit(result)
    assert dict(lab.config["Q1"]) == original
    assert not lab.working_config.exists()
    assert not (tmp_path / "working_update.json").exists()


def test_session_calls_registered_update_hook_without_experiment_switch(session):
    result = session.run("qubit_spec_ge")  # Existing transport fixture, real compiler.
    result.fits = {"Q1": FitResult("fixture", True, {"center": 2810})}
    spec = session.registry.get(result.experiment)
    session.registry.register(
        replace(
            spec,
            updates=lambda r, q: CalibrationUpdates(
                {"qb_freq_ge": 2820}, {"qb_freq_ge": f"qubits.{q}.transitions.ge.frequency_mhz"}
            ),
        ),
        replace=True,
    )
    assert session.propose(result).updates == {"qubits.Q1.transitions.ge.frequency_mhz": 2820}
    result.metadata["acquisition_status"] = "partial"
    with pytest.raises(ValueError, match="Partial"):
        session.propose(result)


def test_best_readout_uses_saved_settings_and_excludes_partial(device, tmp_path, monkeypatch):
    lab = facade(device, tmp_path)
    measured = {"readout_group": "R1", "res_gain_ge": 0.075, "ro_length": 2.0, "res_length": 2.3}
    result = record("single_shot", tmp_path, cfg=measured, values={"threshold": 0.02, "rotation_deg": 90})
    monkeypatch.setattr(
        lab, "load", lambda run_id: result if run_id == result.run_id else pytest.fail("Selected partial")
    )
    rows = [
        {"passed": True, "complete": True, "fidelity": 0.95, "run_id": result.run_id},
        {"passed": True, "complete": False, "fidelity": 1.0, "run_id": "partial"},
    ]
    assert lab.apply_best_readout(rows)
    for key in ("res_gain_ge", "ro_length", "res_length"):
        assert lab.config["Q1"][key] == measured[key]
    assert lab.config["Q1"]["ro_phase"] == 90


@pytest.mark.parametrize("transition", ["ge", "ef"])
def test_best_drag_uses_measured_alpha_delta_and_rejects_mixed_transition(transition, device, tmp_path):
    lab = facade(device, tmp_path)
    best = record(
        "drag_" + transition,
        tmp_path,
        values={"return_error_signal": 0.01},
        cfg={f"drag_alpha_{transition}": 0.3, f"drag_delta_{transition}": -205.0},
    )
    worse = deepcopy(best)
    worse.fits["Q1"].parameters["return_error_signal"] = 0.2
    # Display rows are not the source of calibration values.
    runs = [{"alpha": 999, "result": best}, {"alpha": -999, "result": worse}]
    assert lab.apply_best_drag(runs)
    assert lab.config["Q1"][f"drag_alpha_{transition}"] == 0.3
    assert lab.config["Q1"][f"drag_delta_{transition}"] == -205
    worse.experiment = "drag_ef" if transition == "ge" else "drag_ge"
    before = lab.working_config.read_bytes()
    assert not lab.apply_best_drag(runs)
    assert lab.working_config.read_bytes() == before


def test_ramsey_pair_derives_signed_correction_from_measured_drives(device, tmp_path):
    lab = facade(device, tmp_path)
    minus = record("ramsey_ge", tmp_path, values={"frequency": 0.25}, cfg={"qb_freq_ge": 2999.8})
    plus = record("ramsey_ge", tmp_path, values={"frequency": 0.15}, cfg={"qb_freq_ge": 3000.2})
    runs = {"minus": minus, "plus": plus, "center": 9999, "offset": 9999}
    assert lab.apply_ramsey_pair(runs)
    assert lab.config["Q1"]["qb_freq_ge"] == pytest.approx(3000.05)
    before = lab.working_config.read_bytes()
    minus.metadata["acquisition_status"] = "partial"
    assert not lab.apply_ramsey_pair(runs)
    assert lab.working_config.read_bytes() == before


def test_grid_plot_retains_unmeasured_cells_and_fidelity_values():
    from QickworkspaceV2.experiments.single_shot.analysis import plot_grid

    rows = [
        {"length_us": 1.0, "gain": 0.05, "fidelity": 0.91},
        {"length_us": 1.0, "gain": 0.08, "fidelity": 0.97},
    ]
    figure = plot_grid(rows, [0.05, 0.08], [1.0, 2.0])
    data = figure.axes[0].collections[0].get_array().reshape(2, 2)
    np.testing.assert_allclose(data[0], [0.91, 0.97])
    assert data.mask[1].all()


def test_notebook_update_cells_call_sdk_without_fit_parameter_mappings():
    nb = json.loads((Path(__file__).parents[1] / "ChipCalibration.ipynb").read_text(encoding="utf-8"))
    import ast

    calls = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("apply_fit", "apply_best_readout", "apply_best_drag"):
                    calls.append(node)
                    assert len(node.args) == 1 and not node.keywords
    assert calls
