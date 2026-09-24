"""Hardware-independent scheduling and calibration-boundary tests."""
from copy import deepcopy
import json
import threading
import unittest

from QickworkspaceV2_agent.nodes.contracts import RunRequest
from QickworkspaceV2_agent.nodes.workflow import Workflow, WorkflowNode


class FakeResult:
    def __init__(self, name, *, status="completed", quality="good", metrics=None, suggested_updates=None):
        self.run_id = name
        self.status = status
        self.quality = quality
        self.metrics = metrics or {}
        self.suggested_updates = suggested_updates or {}

    def to_dict(self):
        return deepcopy(vars(self))


class FakeWorkspace:
    def __init__(self, outcomes=None):
        self.outcomes = outcomes or {}
        self.calls = []
        self.commits = []
        self.records = []
        self.calibration = {}
        self.cancel_event = threading.Event()

    def run(self, request, *, parent_id):
        self.calls.append((request, parent_id, deepcopy(self.calibration)))
        outcome = self.outcomes.get(request.experiment, FakeResult(request.experiment))
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return outcome(self)
        return outcome

    def commit(self, result):
        self.commits.append(result.run_id)
        self.calibration.update(result.suggested_updates)

    def save_workflow(self, report):
        self.records.append(deepcopy(report))


def node(name, *, depends_on=(), commit=False, require_good=True, config=None):
    return WorkflowNode(
        name, RunRequest(experiment=name, qubit="Q1", config=config or {}, persist=False),
        depends_on=depends_on, commit=commit, require_good=require_good,
    )


class WorkflowTests(unittest.TestCase):
    def test_invalid_graphs_never_call_workspace(self):
        invalid_graphs = [
            [],
            [node("a"), node("a")],
            [node("a", depends_on=("missing",))],
            [node("a", depends_on=("a",))],
            [node("a", depends_on=("b",)), node("b", depends_on=("a",))],
        ]
        for nodes in invalid_graphs:
            with self.subTest(nodes=nodes):
                workspace = FakeWorkspace()
                with self.assertRaises(ValueError):
                    Workflow(nodes).run(workspace)
                self.assertEqual(workspace.calls, [])
                self.assertEqual(workspace.records, [])

    def test_json_roundtrip_keeps_definition_and_order(self):
        workflow = Workflow([node("b", depends_on=("a",)), node("a", commit=True)], id="calibration_Q1")
        restored = Workflow.from_dict(json.loads(json.dumps(workflow.to_dict())))
        self.assertEqual(restored.to_dict(), workflow.to_dict())
        self.assertEqual(restored.validate(), ["a", "b"])
        with self.assertRaises(ValueError):
            Workflow.from_dict({**workflow.to_dict(), "unexpected": 1})
        with self.assertRaises(ValueError):
            WorkflowNode.from_dict({**node("a").to_dict(), "retry": 10})

    def test_commits_before_downstream_reads_fresh_calibration(self):
        workspace = FakeWorkspace({"a": FakeResult("run_a", suggested_updates={"pi_gain_ge": 0.04})})
        report = Workflow([node("b", depends_on=("a",)), node("a", commit=True)], id="flow").run(workspace)
        self.assertEqual(report.status, "completed")
        self.assertEqual([call[0].experiment for call in workspace.calls], ["a", "b"])
        self.assertEqual(workspace.calls[1][2]["pi_gain_ge"], 0.04)
        self.assertTrue(all(call[1] == "flow" for call in workspace.calls))
        self.assertEqual(workspace.commits, ["run_a"])
        self.assertEqual(report.nodes["a"]["commit_status"], "committed")
        self.assertEqual(workspace.records[-1], report.to_dict())
        self.assertEqual(workspace.records[0]["nodes"]["a"]["status"], "pending")
        self.assertTrue(any(record["nodes"]["a"]["result"] and record["nodes"]["a"]["commit_status"] == "pending" for record in workspace.records))

    def test_failed_dependency_blocks_descendants_but_independent_branch_runs(self):
        workspace = FakeWorkspace({"a": FakeResult("a", status="failed", quality="bad")})
        report = Workflow([
            node("a"), node("b", depends_on=("a",), require_good=False), node("c"),
            node("d", depends_on=("b",)),
        ]).run(workspace, stop_on_failure=False)
        self.assertEqual(report.status, "failed")
        self.assertEqual([call[0].experiment for call in workspace.calls], ["a", "c"])
        self.assertEqual(report.nodes["b"]["status"], "blocked")
        self.assertEqual(report.nodes["d"]["status"], "blocked")

    def test_default_stop_on_failure_blocks_independent_work(self):
        workspace = FakeWorkspace({"a": RuntimeError("board unavailable")})
        report = Workflow([node("a"), node("b")]).run(workspace)
        self.assertEqual(len(workspace.calls), 1)
        self.assertIn("board unavailable", report.nodes["a"]["error"])
        self.assertEqual(report.nodes["b"]["status"], "blocked")

    def test_bad_quality_is_preserved_but_cannot_commit_or_feed_default_dependency(self):
        for quality in ["warning", "bad", "no_information"]:
            with self.subTest(quality=quality):
                workspace = FakeWorkspace({"a": FakeResult("a", quality=quality)})
                report = Workflow([node("a", commit=True), node("b", depends_on=("a",))]).run(workspace)
                self.assertEqual(report.status, "blocked")
                self.assertEqual(report.nodes["a"]["status"], "completed")
                self.assertEqual(report.nodes["a"]["commit_status"], "rejected_quality")
                self.assertEqual(report.nodes["b"]["status"], "blocked")
                self.assertEqual(workspace.commits, [])

    def test_quality_gate_can_be_explicitly_relaxed_for_acquisition_only(self):
        workspace = FakeWorkspace({"a": FakeResult("a", quality="no_information")})
        report = Workflow([node("a"), node("b", depends_on=("a",), require_good=False)]).run(workspace)
        self.assertEqual(report.status, "completed")
        self.assertEqual(len(workspace.calls), 2)
        self.assertEqual(workspace.commits, [])

    def test_references_resolve_typed_values_without_mutating_definition(self):
        request_config = {
            "frequency": {"$ref": "a.metrics.frequency"},
            "nested": [{"$ref": "a.metrics.points.1"}],
            "gain": {"$ref": "a.suggested_updates.pi_gain_ge"},
            "literal": "a.metrics.frequency",
        }
        workspace = FakeWorkspace({"a": FakeResult("a", metrics={"frequency": 4501.2, "points": [1, 2]}, suggested_updates={"pi_gain_ge": 0.03})})
        workflow = Workflow([node("b", depends_on=("a",), config=request_config), node("a")])
        before = deepcopy(workflow.to_dict())
        report = workflow.run(workspace)
        self.assertEqual(report.status, "completed")
        self.assertEqual(workspace.calls[1][0].config, {"frequency": 4501.2, "nested": [2], "gain": 0.03, "literal": "a.metrics.frequency"})
        self.assertEqual(workflow.to_dict(), before)

    def test_undeclared_and_malformed_references_fail_before_execution(self):
        for reference in [
            {"$ref": "a.metrics.x"},
            {"$ref": "a.__class__.__name__"},
            {"$ref": "a.metrics.x", "extra": 1},
            {"$ref": 123},
        ]:
            with self.subTest(reference=reference):
                workspace = FakeWorkspace()
                with self.assertRaises(ValueError):
                    Workflow([node("a"), node("b", config={"x": reference})]).run(workspace)
                self.assertEqual(workspace.calls, [])

    def test_missing_result_reference_blocks_only_after_source_runs(self):
        workspace = FakeWorkspace()
        report = Workflow([node("a"), node("b", depends_on=("a",), config={"x": {"$ref": "a.metrics.absent"}})]).run(workspace)
        self.assertEqual(len(workspace.calls), 1)
        self.assertEqual(report.nodes["b"]["status"], "failed")
        self.assertIn("Missing value", report.nodes["b"]["error"])

    def test_reference_resolution_in_run_and_constructor_parameters(self):
        workspace = FakeWorkspace({"a": FakeResult("a", metrics={"averages": 3, "frequencies": [10.0, 20.0]})})
        dependent = WorkflowNode("b", RunRequest(
            "b", "Q1", run={"py_avg": {"$ref": "a.metrics.averages"}},
            init={"pump_freqs": {"$ref": "a.metrics.frequencies"}},
        ), depends_on=("a",))
        report = Workflow([node("a"), dependent]).run(workspace)
        self.assertEqual(report.status, "completed")
        self.assertEqual(workspace.calls[1][0].run, {"py_avg": 3})
        self.assertEqual(workspace.calls[1][0].init, {"pump_freqs": [10.0, 20.0]})

    def test_cancellation_before_start_prevents_acquisition(self):
        workspace = FakeWorkspace()
        workspace.cancel_event.set()
        report = Workflow([node("a", commit=True), node("b")]).run(workspace)
        self.assertEqual(report.status, "interrupted")
        self.assertEqual(workspace.calls, [])
        self.assertEqual(workspace.commits, [])

    def test_cancellation_between_acquisition_and_commit_preserves_result(self):
        def acquire(workspace):
            workspace.cancel_event.set()
            return FakeResult("a")
        workspace = FakeWorkspace({"a": acquire})
        report = Workflow([node("a", commit=True), node("b")]).run(workspace)
        self.assertEqual(report.status, "interrupted")
        self.assertEqual(len(workspace.calls), 1)
        self.assertEqual(workspace.commits, [])
        self.assertEqual(report.nodes["a"]["result"]["status"], "completed")

    def test_keyboard_interrupt_is_recorded_and_stops_independent_nodes(self):
        workspace = FakeWorkspace({"a": KeyboardInterrupt()})
        report = Workflow([node("a"), node("b")]).run(workspace, stop_on_failure=False)
        self.assertEqual(report.status, "interrupted")
        self.assertEqual(len(workspace.calls), 1)
        self.assertIsNotNone(workspace.records[-1]["finished_at"])

    def test_commit_failure_preserves_acquisition_and_blocks_descendants(self):
        class BrokenCommit(FakeWorkspace):
            def commit(self, result):
                raise OSError("store unavailable")
        workspace = BrokenCommit()
        report = Workflow([node("a", commit=True), node("b", depends_on=("a",))]).run(workspace)
        self.assertEqual(report.status, "failed")
        self.assertEqual(report.nodes["a"]["result"]["status"], "completed")
        self.assertEqual(report.nodes["a"]["commit_status"], "failed")
        self.assertEqual(report.nodes["b"]["status"], "blocked")

    def test_persistence_failure_prevents_acquisition(self):
        class BrokenPersistence(FakeWorkspace):
            def save_workflow(self, report):
                raise OSError("disk unavailable")
        workspace = BrokenPersistence()
        with self.assertRaises(OSError):
            Workflow([node("a")]).run(workspace)
        self.assertEqual(workspace.calls, [])


if __name__ == "__main__":
    unittest.main()
