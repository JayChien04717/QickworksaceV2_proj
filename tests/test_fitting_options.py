"""Numerical fixtures verify editable fit settings, never hardware calibration."""

import numpy as np
import pytest

from QickworkspaceV2.analysis.fitting import (
    FIT_OPTIONS,
    fit_exponential,
    fit_curve,
    fit_resonator,
    notch_response,
)
from QickworkspaceV2.experiments.t1_ge import analyze
from QickworkspaceV2.analysis import fitting


def test_native_run_fit_settings_are_used_without_mutating_iq(accepted_record):
    raw = accepted_record["Q1"].iq.copy()
    accepted_record.metadata["parameters"] = {
        "fit": {"p0": {"tau": 12}, "bounds": {"tau": (5, 15)}, "maxfev": 2000}
    }
    constrained = analyze(accepted_record)["Q1"]
    assert constrained.parameters["tau"] == pytest.approx(15, abs=1e-4)
    accepted_record.metadata["parameters"]["fit"]["bounds"]["tau"] = (5, 50)
    assert analyze(accepted_record)["Q1"].parameters["tau"] == pytest.approx(25)
    np.testing.assert_array_equal(raw, accepted_record["Q1"].iq)


def test_editing_central_defaults_changes_experiment_analysis(accepted_record, monkeypatch):
    monkeypatch.setitem(FIT_OPTIONS["exponential"], "bounds", {"tau": (5, 15)})
    assert analyze(accepted_record)["Q1"].parameters["tau"] == pytest.approx(15, abs=1e-4)


@pytest.mark.parametrize(
    "settings",
    [
        {"p0": {"typo": 2}},
        {"bounds": {"typo": (0, 1)}},
        {"bounds": {"tau": (20, 10)}},
        {"p0": {"tau": -1}},
        {"p0": np.array([0, 1, -1])},
        {"p0": {"tau": float("nan")}},
    ],
)
def test_invalid_named_settings_fail_visibly(settings):
    x = np.linspace(0, 100, 81)
    with pytest.raises(ValueError):
        fit_exponential(x, 0.2 + np.exp(-x / 25), **settings)


def test_custom_formula_accepts_named_parameters():
    def linear(x, slope, intercept):
        return slope * x + intercept

    x = np.linspace(0, 5, 30)
    fit = fit_curve(linear, x, 2 * x + 3, p0={"slope": 1, "intercept": 0}, bounds={"slope": (0, 4)})
    assert fit.success
    assert fit.parameters == pytest.approx({"slope": 2, "intercept": 3})


def test_complex_notch_named_options_use_physical_units():
    f = np.linspace(6700, 6710, 101)
    z = notch_response(f, 6705, 0.8, 0.6, 0.05, 0.9, 0.1, 0.02)
    fit = fit_resonator(
        f,
        z,
        p0={"center": 6704.8, "linewidth": 1, "delay_us": 0.01},
        bounds={"center": (6704, 6706), "linewidth": (0.1, 2)},
    )
    assert fit.success, fit.message
    assert fit.parameters["center"] == pytest.approx(6705, abs=1e-5)


@pytest.mark.parametrize("fitter", list(fitting.FIT_FUNCTIONS.values()))
def test_each_fit_function_rejects_missing_data(fitter):
    assert not fitter([], []).success


@pytest.mark.parametrize(
    "model,formula,parameters",
    [
        ("exponential", fitting.exponential, {"offset": 0.2, "amplitude": 1.0, "tau": 25.0}),
        ("lorentzian", fitting.lorentzian, {"offset": 0.2, "amplitude": 1.0, "center": 50.0, "width": 5.0}),
        (
            "asymmetric_lorentzian",
            fitting.asymmetric_lorentzian,
            {"offset": 0.2, "amplitude": 1.0, "center": 50.0, "width": 5.0, "asymmetry": 0.1},
        ),
        ("rabi", fitting.oscillation, {"offset": 0.2, "amplitude": 1.0, "frequency": 0.1, "phase": 0.0}),
        (
            "ramsey",
            fitting.damped_oscillation,
            {"offset": 0.2, "amplitude": 1.0, "tau": 25.0, "frequency": 0.1, "phase": 0.1},
        ),
        (
            "ramsey_slope",
            fitting.sloped_damped_oscillation,
            {"offset": 0.2, "amplitude": 1.0, "tau": 25.0, "frequency": 0.1, "phase": 0.1, "slope": 0.001},
        ),
        (
            "ramsey_two_frequency",
            fitting.two_frequency_damped_oscillation,
            {
                "offset": 0.2,
                "amplitude": 1.0,
                "tau": 25.0,
                "frequency": 0.1,
                "phase": 0.1,
                "amplitude2": 0.4,
                "frequency2": 0.17,
                "phase2": 0.2,
            },
        ),
        ("gaussian", fitting.gaussian, {"offset": 0.2, "amplitude": 1.0, "center": 50.0, "sigma": 5.0}),
        (
            "double_gaussian",
            fitting.double_gaussian,
            {
                "amplitude_g": 1.0,
                "center_g": 25.0,
                "sigma_g": 5.0,
                "amplitude_e": 0.8,
                "center_e": 70.0,
                "sigma_e": 7.0,
            },
        ),
        ("rb", fitting.rb_decay, {"offset": 0.2, "amplitude": 0.8, "probability": 0.98}),
        ("rotation_x", fitting.rotation_x, {"offset": 0.5, "error_rad": 0.02}),
        ("rotation_x_half", fitting.rotation_x_half, {"offset": 0.5, "error_rad": 0.02}),
        (
            "rotation_x_half_decay",
            fitting.rotation_x_half_decay,
            {"offset": 0.5, "error_rad": 0.02, "tau": 25.0},
        ),
        ("poisson", fitting.poisson, {"mean": 5.0}),
    ],
)
def test_independent_fit_functions_recover_known_numeric_data(model, formula, parameters):
    x = np.arange(101, dtype=float)
    y = formula(x, **parameters)
    # Reverse order also verifies that every direct entry handles unsorted axes.
    fit = fitting.FIT_FUNCTIONS[model](x[::-1], y[::-1], p0=parameters)
    assert fit.success, fit.message
    expected = {
        ("decay_probability" if key == "probability" else key): value for key, value in parameters.items()
    }
    assert {key: fit.parameters[key] for key in expected} == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_generic_solver_does_not_apply_exponential_calibration_rules():
    x = np.linspace(0, 100, 81)
    parameters = {"offset": 0.2, "amplitude": 1.0, "tau": 1000.0}
    y = fitting.exponential(x, **parameters)
    numerical = fit_curve(fitting.exponential, x, y, p0=parameters)
    physical = fit_exponential(x, y, p0=parameters)
    assert numerical.success and not physical.success
    assert "too short to resolve" in physical.message
    with pytest.raises(TypeError, match="formula callable"):
        fit_curve("exponential", x, y)


def test_native_model_selection_calls_independent_fitter(accepted_record):
    trace = accepted_record["Q1"]
    x = trace.coords["delay"]
    trace.iq = fitting.gaussian(x, 0.2, 1, 50, 5)[:, None].astype(complex)
    original = trace.iq.copy()
    accepted_record.metadata["parameters"] = {"fit": {"model": "gaussian", "p0": {"sigma": 7}}}
    fit = analyze(accepted_record)["Q1"]
    assert fit.model == "gaussian" and fit.success
    assert fit.parameters["sigma"] == pytest.approx(5)
    np.testing.assert_array_equal(trace.iq, original)
