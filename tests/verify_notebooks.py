"""Notebook maintenance check: validate syntax and compile run cells without acquisition.

Only cells tagged imports/configuration/definition/prepare run here. Acquisition,
connection, display, calibration updates, and all IPython magics are skipped.
Uses QICK's real compiler with a logical testbench configuration.
"""

from pathlib import Path
from copy import deepcopy
from tempfile import TemporaryDirectory
import ast
import json
import os
import sys
import nbformat
import qick
from qick import QickConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from QickworkspaceV2 import Measurement
from QickworkspaceV2.runtime.measurement import infer_axes


def verify():
    config = json.loads((ROOT / "tests/fixtures/qick_testbench.json").read_text(encoding="utf-8"))
    gen, ro = config["gens"][0], config["readouts"][0]
    config["gens"] = [dict(deepcopy(gen), tproc_ch=i) for i in range(7)]
    config["readouts"] = [dict(deepcopy(ro), tproc_ctrl=8 + i, trigger_bit=i, tproc_ch=i) for i in range(3)]
    config["sw_version"] = qick.__version__
    report = {"hardware_acquired": False, "notebooks": {}, "script_compiled": False}
    with TemporaryDirectory(prefix="qick-compile-") as directory:
        compiler = Measurement(None, QickConfig(config), data_path=directory)
        for path in [ROOT / "QickworkspaceV2.ipynb", ROOT / "ChipCalibration.ipynb", *sorted((ROOT / "notebooks").glob("*.ipynb"))]:
            book = nbformat.read(path, as_version=4)
            nbformat.validate(book)
            namespace = {"__name__": "__notebook__"}
            compiled = []
            for index, cell in enumerate(book.cells):
                if cell.cell_type != "code":
                    continue
                role = cell.metadata.get("qick_role", "display")
                source = cell.source
                if role == "magic":
                    continue
                ast.parse(source, filename=f"{path.name}:cell{index + 1}")
                if role in ("imports", "configuration", "definition", "prepare"):
                    if cell.metadata.get("requires_readout_calibration"):
                        # Compiler fixture only; no measured calibration or file write.
                        namespace["qb"].update(ro_threshold=0.01, ro_phase=0)
                    exec(compile(source, str(path), "exec"), namespace)
                if role == "prepare":
                    program = compiler.compile(namespace["Program"], namespace["run_cfg"])
                    infer_axes(program)
                    compiled.append(namespace["Program"].__name__)
            report["notebooks"][path.name] = {"cells": len(book.cells), "compiled_programs": compiled}
        from measure import Program, make_run_config

        program = compiler.compile(Program, make_run_config())
        infer_axes(program)
        report["script_compiled"] = True
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    verify()
