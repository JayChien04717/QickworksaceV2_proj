import numpy as np
import pytest
from labtools.fitting import fit_resonator, notch_response
from QickworkspaceV2.analysis.transmission import normalize_transmission


def test_complex_notch_recovers_loss_and_delay():
    f = np.linspace(6695, 6705, 301)
    expected_q = 6700 / 0.8
    rng = np.random.default_rng(2)
    z = notch_response(f, 6700, 0.8, 0.6, 0.12, 1.4, 0.7, 0.18)
    z += 0.002 * (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f)))
    fit = fit_resonator(f, z)
    assert fit.success, fit.message
    assert fit.parameters["center"] == pytest.approx(6700, abs=0.005)
    assert fit.parameters["Q_loaded"] == pytest.approx(expected_q, rel=0.015)
    assert fit.parameters["Q_internal"] == pytest.approx(expected_q / (1 - 0.6 * np.cos(0.12)), rel=0.02)
    assert fit.parameters["delay_us"] == pytest.approx(0.18, abs=0.001)


def test_complex_notch_rejects_flat_data():
    assert not fit_resonator(np.arange(20), np.ones(20)).success


def test_twpa_reference_coordinates_are_checked(session):
    reference = session.run("twpa_probe", points=11)
    measured = session.run("twpa_probe", points=11)
    measured["Q1"].iq *= 10
    gain = normalize_transmission(measured, reference)
    np.testing.assert_allclose(gain["Q1"].signal("db"), 20)
    reference["Q1"].coords["frequency"] += 1
    with pytest.raises(ValueError, match="coordinates"):
        normalize_transmission(measured, reference)
