"""dispersive: program."""

from QickworkspaceV2 import ProgramPlan, Sweep
from ..single_shot import SingleShotProgram


class DispersiveProgram(SingleShotProgram):
    def _initialize(self, cfg):
        super()._initialize(cfg)
        self.add_sweep_loop(cfg, cfg["qubits"][cfg["targets"][0]]["res_freq_ge"], "freqloop")


def build(ctx, p):
    cfg = ctx.config()
    cfg.update(steps=p.points, reset_wait_us=p.reset_wait_us)
    for qc in cfg["qubits"].values():
        center = qc["res_freq_ge"]
        axis = Sweep(
            "frequency",
            center + p.start,
            center + p.stop,
            p.points,
            "MHz",
            "freqloop",
            "freq",
            "{target}__res_pulse",
        )
        qc["res_freq_ge"] = axis.qick()
    return ProgramPlan(DispersiveProgram, cfg, (axis,), SingleShotProgram.READOUT_EVENTS, True)
