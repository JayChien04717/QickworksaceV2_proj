"""resonator_spec_f: program."""

from __future__ import annotations
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import ProgramPlan


class ResonatorSpecFProgram(BaseProgram):
    TRANSITION = "ef"

    def _initialize(self, cfg):
        # Mux tone registers are static: the recipe uses the common host-sweep runner.
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, cfg["transition"])
            self.setup_standard_gates(qc, cfg["transition"])
        self.setup_readout(cfg)
        frequency = cfg["qubits"][cfg["targets"][0]]["res_freq_ge"]
        if getattr(frequency, "spans", {}):
            self.add_sweep_loop(cfg, frequency, "freqloop")
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, "ge")
            self.setup_standard_gates(qc, "ge")

    def _body(self, cfg):
        for qc in cfg["qubits"].values():
            self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, "x_ge"), t=0)
        self.wait_pulses()
        with self.parallel():
            for qb in self.qubits.values():
                qb.x()
        self.measure(cfg)


def build_resonator(ctx, p):
    cfg = ctx.config("ef")
    for qc in cfg["qubits"].values():
        qc["res_gain_ge"] *= p.gain_scale
    for group in cfg["readout_groups"].values():
        for q, member in group["members"].items():
            if q in cfg["targets"]:
                member["gain"] *= p.gain_scale
    # Host sweep works on mux/PYNQ readouts as well as tProc-controlled readouts.
    return ProgramPlan(
        ResonatorSpecFProgram,
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
