from dataclasses import replace
import numpy as np
import pytest
from QickworkspaceV2 import TraceData, ExperimentData
from QickworkspaceV2.analysis import fit_curve, train_classifier, joint_probabilities


def test_accepted_data_roundtrip(accepted_record, tmp_path):
    path = accepted_record.save(tmp_path / "record.h5")
    loaded = ExperimentData.load(path)
    assert loaded.metrics == accepted_record.metrics
    np.testing.assert_array_equal(loaded["Q1"].iq, accepted_record["Q1"].iq)
    assert loaded.metrics["Q1"]["tau"] == pytest.approx(25, rel=0.001)


def test_degenerate_data_not_calibrated():
    x = np.linspace(0, 100, 100)
    assert not fit_curve("exponential", x, np.ones(100)).success
    y = np.exp(-x / 25)
    y[3] = np.nan
    assert not fit_curve("exponential", x, y).success
    assert not fit_curve("ramsey", [0, 1, 2], [0, 1, 0]).success


def test_analysis_failure_keeps_raw(session):
    def broken(data):
        raise RuntimeError("analysis intentionally failed")

    session.register(replace(session.registry.get("t1_ge"), id="broken", analyze=broken))
    r = session.run("broken")
    assert r.analysis_status == "failed"
    assert session.load(r.run_id, revision=0)["Q1"].iq.shape == (81, 1)
    with pytest.raises(ValueError):
        session.propose(r)


def test_multidimensional_plot_and_persistence(session, tmp_path):
    result = session.run("coupler_chevron", gain_points=5, length_points=11)
    assert result["Q1"].iq.shape == (5, 11, 1)
    result.plot().savefig(tmp_path / "chevron.png")
    assert (tmp_path / "chevron.png").stat().st_size > 1000
    assert session.load(result.run_id)["Q2"].dims == ("gain", "length", "readout")


def test_classifier_and_joint_probabilities():
    rng = np.random.default_rng(12)
    ground = 0.05 * (rng.normal(size=1000) + 1j * rng.normal(size=1000))
    excited = ground + 1 + 0.5j
    classifier = train_classifier(ground, excited)
    probabilities = joint_probabilities({"Q1": excited, "Q2": excited}, {"Q1": classifier, "Q2": classifier})
    assert probabilities["11"] > 0.98
    assert sum(probabilities.values()) == pytest.approx(1)


def test_tomography_requires_classifier(session):
    assert not session.check("state_tomography", target="Q1,Q2")["ready"]


def test_outer_readout_scan(session):
    r = session.scan("resonator_spec", "gain_scale", [0.5, 1.0], points=11)
    assert r["Q1"].dims == ("gain_scale", "frequency", "readout")
    assert r["Q1"].iq.shape == (2, 11, 1)
    assert not r.is_good()


def test_no_silent_dimension_guessing():
    with pytest.raises(ValueError):
        TraceData(np.ones((3, 2)), ("delay",), {"delay": np.arange(3)})
