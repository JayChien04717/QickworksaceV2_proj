from dataclasses import replace
import pytest
from QickworkspaceV2 import CalibrationProposal, CalibrationGraph, CalibrationNode


def test_revision_conflict_idempotency_and_scope(session):
    r = session.run("power_rabi_ge")
    from QickworkspaceV2 import FitResult

    r.fits = {"Q1": FitResult("rabi", True, {"pi_gain": 0.2, "pi2_gain": 0.1})}
    proposal = session.propose(r)
    assert session.commit(proposal) == 1
    assert session.commit(proposal) == 1
    with pytest.raises(RuntimeError, match="revision changed"):
        session.commit(replace(proposal, id="new"))
    with pytest.raises(ValueError, match="scope"):
        session.commit(replace(proposal, scope="another chip"))
    assert session.device.config.qubits["Q1"].transitions["ge"].pulse.pi_gain == r.metrics["Q1"]["pi_gain"]


def test_calibration_cannot_rewire(session):
    p = CalibrationProposal({"qubits.Q1.drive": "drive_q2"}, 0, "manual", session.scope, source="manual")
    with pytest.raises(ValueError, match="Not a calibration"):
        session.commit(p)


def test_transaction_rolls_back_invalid_gain(limited_session):
    session = limited_session
    p = CalibrationProposal(
        {"qubits.Q1.transitions.ge.pulse.pi_gain": 0.99}, 0, "manual", session.scope, source="manual"
    )
    with pytest.raises(ValueError):
        session.commit(p)
    assert session.calibration.snapshot() == (0, {})


def test_graph_cycle_and_quality_gate(session):
    with pytest.raises(ValueError, match="cycle"):
        CalibrationGraph(
            [
                CalibrationNode("a", "t1_ge", depends_on=("b",)),
                CalibrationNode("b", "t1_ge", depends_on=("a",)),
            ]
        )
    graph = CalibrationGraph(
        [
            CalibrationNode("rabi", "power_rabi_ge", commit=True),
            CalibrationNode("t1_ge", "t1_ge", depends_on=("rabi",)),
        ]
    )
    report = graph.run(session, target="Q1")
    assert report["rabi"]["status"] == "failed"
    assert report["t1_ge"]["status"] == "blocked"
    assert session.calibration.snapshot() == (0, {})
