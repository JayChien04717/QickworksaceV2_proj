"""time_of_flight: program."""

from __future__ import annotations
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import ProgramPlan
from .parameters import TimeOfFlightParameters


class TimeOfFlightProgram(BaseProgram):
    ACQUISITION_MODE = "decimated"
    TRANSITION = None

    @staticmethod
    def transition_for(cfg):
        return "ef" if cfg.get("check_f", False) else "ge"

    def _initialize(self, cfg):
        options = TimeOfFlightParameters.model_validate(
            {
                name: cfg[name]
                for name in ("check_e", "check_f", "prepare_delay_us", "tof_threshold")
                if name in cfg
            }
        )
        expected = "ef" if options.check_f else "ge"
        if cfg["transition"] != expected:
            raise ValueError(f"TOF state preparation requires transition={expected!r}")
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, cfg["transition"])
            if options.check_e or options.check_f:
                self.setup_standard_gates(qc, cfg["transition"])
        self.setup_readout(cfg)
        if options.check_f:
            for qc in cfg["qubits"].values():
                self.setup_qubit_gen(qc, "ge")
                self.setup_standard_gates(qc, "ge")

    def _body(self, cfg):
        if cfg.get("check_f", False):
            for qc in cfg["qubits"].values():
                self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, "x_ge"), t=0)
            self.wait_pulses()
        if cfg.get("check_e", False) or cfg.get("check_f", False):
            with self.parallel():
                for qb in self.qubits.values():
                    qb.x()
            self.delay_auto(cfg.get("prepare_delay_us", 0.05))
        self.measure(cfg)


def build_tof(ctx, p):
    cfg = ctx.config("ef" if p.check_f else "ge")
    cfg.update(p.model_dump(exclude={"target"}))
    cfg["reps"] = 1
    return ProgramPlan(
        TimeOfFlightProgram,
        cfg,
        metadata={
            "acquisition_mode": "decimated",
            "note": "One hardware rep per software average to fit the decimated buffer",
        },
    )
