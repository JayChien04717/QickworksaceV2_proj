"""Broadband experiment identity with the same native one-tone pulse sequence."""

from QickworkspaceV2.experiments.base import ProgramPlan
from QickworkspaceV2.experiments.resonator_spec.program import ResonatorSpecProgram


class BroadbandResonatorSpecProgram(ResonatorSpecProgram):
    """Reuse only the hardware sequence; parameters, analysis and catalog are independent."""


def build(ctx, p):
    cfg = ctx.config()
    cfg.update(count=p.count, detection_options=p.detection_options.model_dump(), y_mode=p.y_mode)
    for qc in cfg["qubits"].values():
        qc["res_gain_ge"] *= p.gain_scale
    for group in cfg["readout_groups"].values():
        for q, member in group["members"].items():
            if q in cfg["targets"]:
                member["gain"] *= p.gain_scale
    return ProgramPlan(
        BroadbandResonatorSpecProgram,
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
