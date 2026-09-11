"""drag_ge: program."""

from QickworkspaceV2 import BaseProgram, ProgramPlan


class DragGEProgram(BaseProgram):
    TRANSITION = "ge"
    READOUT_EVENTS = ("ground", "excited", "return")

    def _initialize(self, cfg):
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, cfg["transition"])
            self.setup_standard_gates(qc, cfg["transition"])
        self.setup_readout(cfg)
        for qc in cfg["qubits"].values():
            self.setup_qb_pulse(
                qc, "ge", name="drag_x", shape="drag", pulse_type="arb", gain_key="pi_gain_ge", phase=0
            )
            self.setup_qb_pulse(
                qc, "ge", name="drag_mx", shape="drag", pulse_type="arb", gain_key="pi_gain_ge", phase=180
            )

    def _body(self, cfg):
        self.delay_auto(cfg["reset_wait_us"])
        self.measure(cfg)  # Ground reference.
        self.delay_auto(cfg["reset_wait_us"])
        with self.parallel():
            for qb in self.qubits.values():
                qb.x()
        self.measure(cfg)  # Excited reference.
        self.delay_auto(cfg["reset_wait_us"])
        for _ in range(cfg["iterations"]):
            for name in ("drag_x", "drag_mx"):
                for qc in cfg["qubits"].values():
                    self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, name), t=0)
                self.delay_auto(0.01)
        self.measure(cfg)


def build(ctx, p):
    cfg = ctx.config("ge")
    cfg.update(alpha=p.alpha, iterations=p.iterations, reset_wait_us=p.reset_wait_us)
    for qc in cfg["qubits"].values():
        qc.update(drag_alpha_ge=p.alpha, drag_delta_ge=p.delta_mhz)
    return ProgramPlan(DragGEProgram, cfg, readout_events=("ground", "excited", "return"))
