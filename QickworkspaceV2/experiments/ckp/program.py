"""ckp: program."""

from QickworkspaceV2 import BaseProgram, ProgramPlan, Sweep


class CKPProgram(BaseProgram):
    TRANSITION = "ge"

    def _initialize(self, cfg):
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, "ge")
        self.setup_readout(cfg)
        self.add_loop("resfreqloop", cfg["res_steps"])
        self.add_loop("qbfreqloop", cfg["qb_steps"])
        for qc in cfg["qubits"].values():
            self.setup_qb_pulse(qc, "ge", name="qb_pulse", pulse_type="flat_top")

    def _body(self, cfg):
        for qc in cfg["qubits"].values():
            self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, "qb_pulse"), t=0)
        self.measure(cfg)


def build(ctx, p):
    cfg = ctx.config()
    cfg.update(res_steps=p.readout_points, qb_steps=p.qubit_points)
    for qc in cfg["qubits"].values():
        fr, fq = qc["res_freq_ge"], qc["qb_freq_ge"]
        axes = (
            Sweep(
                "readout_frequency",
                fr + p.readout_start,
                fr + p.readout_stop,
                p.readout_points,
                "MHz",
                "resfreqloop",
                "freq",
                "{target}__res_pulse",
            ),
            Sweep(
                "qubit_frequency",
                fq + p.qubit_start,
                fq + p.qubit_stop,
                p.qubit_points,
                "MHz",
                "qbfreqloop",
                "freq",
                "{target}__qb_pulse",
            ),
        )
        qc.update(
            res_freq_ge=axes[0].qick(),
            qb_freq_ge=axes[1].qick(),
            qb_gain_ge=p.drive_gain,
            qb_flat_top_length_ge=p.drive_length_us,
        )
    return ProgramPlan(CKPProgram, cfg, axes)
