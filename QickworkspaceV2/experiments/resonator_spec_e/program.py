"""resonator_spec_e: program."""

from __future__ import annotations
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import ProgramPlan


class ResonatorSpecEProgram(BaseProgram):
    TRANSITION = "ge"

    def _initialize(self, cfg):
        # Mux tone registers are static: the recipe uses the common host-sweep runner.
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, cfg["transition"])
            self.setup_standard_gates(qc, cfg["transition"])
        self.setup_readout(cfg)
        frequency = cfg["qubits"][cfg["targets"][0]]["res_freq_ge"]
        if getattr(frequency, "spans", {}):
            self.add_sweep_loop(cfg, frequency, "freqloop")

    def _body(self, cfg):
        with self.parallel():
            for qb in self.qubits.values():
                qb.x()
        self.measure(cfg)


def build_resonator(ctx, p):
    cfg = ctx.config()
    for qc in cfg["qubits"].values():
        qc["res_gain_ge"] *= p.gain_scale
    for group in cfg["readout_groups"].values():
        for q, member in group["members"].items():
            if q in cfg["targets"]:
                member["gain"] *= p.gain_scale
    # Host sweep works on mux/PYNQ readouts as well as tProc-controlled readouts.
    return ProgramPlan(
        ResonatorSpecEProgram,
        cfg,
        metadata={
            "host_sweep": {
                "kind": "readout_frequency",
                "name": "frequency",
                "unit": "MHz",
                "start": p.start,
                "stop": p.stop,
                "points": p.points,
            }
        },
    )
