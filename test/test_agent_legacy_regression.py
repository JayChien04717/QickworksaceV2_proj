"""Run the existing hardware-independent contracts against the new namespace.

The original tests remain the pulse/data regression baseline while both
packages coexist. Reuse their assertions in memory instead of maintaining a
second edited copy of every test. JobServer's independent tests run normally.
"""
from pathlib import Path
import types
import unittest


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        if path.stem.startswith("test_agent_") or path.stem == "test_qick_job_server":
            continue
        source = path.read_text(encoding="utf-8-sig")
        if "QickworkspaceV2" not in source:
            continue
        module = types.ModuleType("agent_regression_" + path.stem)
        module.__file__ = str(path)
        exec(compile(source.replace("QickworkspaceV2", "QickworkspaceV2_agent"), str(path), "exec"), module.__dict__)
        suite.addTests(loader.loadTestsFromModule(module))
    return suite
