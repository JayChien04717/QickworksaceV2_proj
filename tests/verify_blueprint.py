"""Run the actual Blueprint discovery/runner/storage contract against our generated wrappers.

Usage: python tests/verify_blueprint.py PATH_TO_BLUEPRINT_CORE
The repository is read-only; integration output goes into .test-data.
"""

from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import hashlib
import importlib.util
import json
import sys
import types

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
source = Path(sys.argv[1]).resolve()
package = types.ModuleType("_qca_contract")
package.__path__ = [str(source)]
sys.modules[package.__name__] = package
modules = {}
for name in ("models", "discovery", "runner", "storage"):
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.{name}", source / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    modules[name] = module

import numpy as np
from QickworkspaceV2 import ExperimentData, TraceData
from QickworkspaceV2.experiments import default_registry
from QickworkspaceV2.experiments.t1_ge import analyze
from QickworkspaceV2.integrations.nvidia import export_wrappers, to_blueprint

output = root / ".test-data/blueprint_contract" / uuid4().hex
paths = export_wrappers(default_registry(), output / "scripts")
schemas = modules["discovery"].discover_experiments(output / "scripts")
assert len(schemas) == len(paths)
for path in paths:
    validation = modules["discovery"].validate_script(path)
    assert validation["valid"], validation
# Explicit numerical record fixture: test serialization, never impersonate board acquisition.
x = np.linspace(0, 100, 81)
result = ExperimentData(
    "t1_ge",
    {
        q: TraceData(
            (0.2 + np.exp(-x / 25))[:, None],
            ("delay", "readout"),
            {"delay": x, "readout": ["measurement"]},
            {"delay": "us"},
            shots=np.arange(81 * 4).reshape(81, 4, 1) * (1 + 2j),
            shot_dims=("delay", "shot", "readout"),
        )
        for q in ("Q1", "Q2")
    },
    metadata={"targets": ["Q1", "Q2"], "iq_process": "abs", "test_fixture": True},
)
result.fits = analyze(result)
result.analysis_status = "completed"
payload = to_blueprint(result)
parsed = modules["runner"]._parse_and_validate_result(json.dumps(payload))
assert len(parsed["arrays"]) >= 10
record = modules["models"].ExperimentResult(
    id=result.run_id,
    type="t1_ge",
    timestamp=result.created_at,
    status=parsed["status"],
    target="Q1,Q2",
    params={"target": "Q1,Q2"},
    results=parsed["results"],
    arrays=parsed["arrays"],
    plots=parsed["plots"],
    metadata=parsed["metadata"],
)
modules["storage"].save_experiment(record, output / "blueprint")
loaded = modules["storage"].load_experiment(result.run_id, output / "blueprint")
assert loaded.arrays["Q1_I"] == payload["data"]["Q1_I"]["value"]
assert loaded.plots[0]["data"] == payload["plots"][0]["data"]
np.testing.assert_array_equal(loaded.arrays["Q1_shots_I"], result["Q1"].shots.real)
np.testing.assert_array_equal(loaded.arrays["Q1_shots_Q"], result["Q1"].shots.imag)
assert loaded.metadata["array_metadata"]["Q1_shots_I"]["dims"] == ["delay", "shot", "readout"]
assert loaded.metadata["array_metadata"]["Q1_shots_I"]["shape"] == [81, 4, 1]
assert loaded.metadata["array_metadata"]["Q1_readout"]["labels"] == ["measurement"]
report = {
    "status": "passed",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "wrappers_discovered": len(schemas),
    "tests": [
        "official AST discovery",
        "official script validation",
        "official runner JSON parsing and array extraction",
        "official HDF5/SQLite storage roundtrip including PNG",
        "per-shot I/Q, named dimensions, singleton readout axis and categorical labels preserved",
    ],
    "source_sha256": {
        name: hashlib.sha256((source / f"{name}.py").read_bytes()).hexdigest() for name in modules
    },
    "scope": "Local protocol compatibility, no LLM provider or physical QICK acquisition",
}
path = root / "docs/verification/blueprint.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
