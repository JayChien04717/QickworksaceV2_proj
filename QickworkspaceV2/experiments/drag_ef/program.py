"""drag_ef: program."""

from QickworkspaceV2 import BaseProgram, ProgramPlan


class DragEFProgram(BaseProgram):
    TRANSITION = "ef"
    READOUT_EVENTS = ("ground", "excited", "return")

    def _initialize(self, cfg):
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, cfg["transition"])
            self.setup_standard_gates(qc, cfg["transition"])
        self.setup_readout(cfg)
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, "ge")
            self.setup_standard_gates(qc, "ge")
        for qc in cfg["qubits"].values():
            self.setup_qb_pulse(
                qc, "ef", name="drag_x", shape="drag", pulse_type="arb", gain_key="pi_gain_ef", phase=0
            )
            self.setup_qb_pulse(
                qc, "ef", name="drag_mx", shape="drag", pulse_type="arb", gain_key="pi_gain_ef", phase=180
            )

    def _body(self, cfg):
        self.delay_auto(cfg["reset_wait_us"])
        for qc in cfg["qubits"].values():
            self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, "x_ge"), t=0)
        self.wait_pulses()
        self.measure(cfg)  # Prepared |e> reference.
        self.delay_auto(cfg["reset_wait_us"])
        for qc in cfg["qubits"].values():
            self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, "x_ge"), t=0)
        self.wait_pulses()
        with self.parallel():
            for qb in self.qubits.values():
                qb.x()
        self.measure(cfg)  # Prepared |f> reference.
        self.delay_auto(cfg["reset_wait_us"])
        for qc in cfg["qubits"].values():
            self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, "x_ge"), t=0)
        self.wait_pulses()
        for _ in range(cfg["iterations"]):
            for name in ("drag_x", "drag_mx"):
                for qc in cfg["qubits"].values():
                    self.pulse(ch=qc["qb_ch_ef"], name=self.pulse_name(qc, name), t=0)
                self.delay_auto(0.01)
        self.measure(cfg)


def build(ctx, p):
    cfg = ctx.config("ef")
    cfg.update(alpha=p.alpha, iterations=p.iterations, reset_wait_us=p.reset_wait_us)
    for qc in cfg["qubits"].values():
        qc.update(drag_alpha_ef=p.alpha, drag_delta_ef=p.delta_mhz)
    return ProgramPlan(DragEFProgram, cfg, readout_events=("e_reference", "f_reference", "return"))
