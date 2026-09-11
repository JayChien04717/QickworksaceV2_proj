"""twpa_probe: program."""

from QickworkspaceV2 import BaseProgram, ProgramPlan


class TWPAProbeProgram(BaseProgram):
    TRANSITION = "ge"

    def _initialize(self, cfg):
        self.setup_readout(cfg)

    def _body(self, cfg):
        self.measure(cfg)


def build(ctx, p):
    if len(ctx.targets) != 1:
        raise ValueError("A TWPA probe uses one explicitly selected signal/readout path")
    cfg = ctx.config()
    qc = cfg["qubits"][ctx.targets[0]]
    center = qc["res_freq_ge"]
    qc["res_gain_ge"] *= p.gain_scale
    cfg["readout_groups"][qc["readout_group"]]["members"][ctx.targets[0]]["gain"] *= p.gain_scale
    return ProgramPlan(
        TWPAProbeProgram,
        cfg,
        metadata={
            "host_sweep": {
                "kind": "readout_frequency",
                "name": "frequency",
                "unit": "MHz",
                "start": p.start_mhz - center,
                "stop": p.stop_mhz - center,
                "points": p.points,
            }
        },
    )
