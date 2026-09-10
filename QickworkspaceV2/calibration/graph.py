"""Dependency-ordered calibration. Failed fits block descendants, not raw data."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CalibrationNode:
    id: str
    experiment: str
    parameters: dict = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    commit: bool = False


class CalibrationGraph:
    def __init__(self, nodes):
        self.nodes = {n.id: n for n in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError("Duplicate calibration node ID")
        self.order()

    def order(self):
        ordered, visiting, visited = [], set(), set()

        def visit(name):
            if name in visiting:
                raise ValueError(f"Calibration dependency cycle at {name}")
            if name in visited:
                return
            if name not in self.nodes:
                raise ValueError(f"Unknown calibration dependency {name}")
            visiting.add(name)
            for dependency in self.nodes[name].depends_on:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)
            ordered.append(name)

        for name in self.nodes:
            visit(name)
        return ordered

    def run(self, session, *, target, on_progress=None):
        report = {}
        for name in self.order():
            node = self.nodes[name]
            if any(report[d]["status"] != "completed" for d in node.depends_on):
                report[name] = {"status": "blocked", "reason": "Dependency failed quality checks"}
                continue
            try:
                result = session.run(node.experiment, target=target, **node.parameters)
                if not result.is_good():
                    report[name] = {
                        "status": "failed",
                        "run_id": result.run_id,
                        "reason": result.analysis_message or "Fit quality failed",
                    }
                else:
                    if node.commit:
                        session.commit(session.propose(result))
                    report[name] = {"status": "completed", "run_id": result.run_id, "metrics": result.metrics}
            except Exception as exc:
                report[name] = {"status": "failed", "reason": str(exc)}
            if on_progress:
                on_progress(name, report[name])
        return report
