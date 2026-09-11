"""Resonator punch-out: FPGA gain outer loop and frequency inner loop."""

from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.programs.sweeps import Sweep
from QickworkspaceV2.experiments.base import ProgramPlan


class PunchoutProgram(BaseProgram):
    TRANSITION = "ge"

    def _initialize(self, cfg):
        for target in cfg["targets"]:
            qc = cfg["qubits"][target]
            if qc["readout_mode"] != "direct":
                raise ValueError("FPGA punch-out requires direct readout; MUX tones are static")
            if "tproc_ctrl" not in self.soccfg["readouts"][qc["ro_ch"]]:
                raise ValueError("FPGA punch-out requires a tProc-controlled readout frequency")
            for parameter, loop in (("res_gain_ge", "gainloop"), ("res_freq_ge", "freqloop")):
                if set(getattr(qc[parameter], "spans", {})) != {loop}:
                    raise ValueError(f"{parameter} must use QickSweep1D({loop!r}, start, stop)")
        for name in ("g_steps", "f_steps"):
            if type(cfg[name]) is not int or cfg[name] < 2:
                raise ValueError(f"{name} must be an integer of at least 2")

        # Same nesting as the original punch-out: gain rows, frequency columns.
        self.setup_readout(cfg)
        self.add_loop("gainloop", cfg["g_steps"])
        self.add_loop("freqloop", cfg["f_steps"])

    def _body(self, cfg):
        self.measure(cfg)


def build_punchout(ctx, p):
    """Map catalog inputs to two native sweeps; no host-loop metadata."""
    cfg = ctx.config()
    cfg.update(g_steps=p.g_steps, f_steps=p.f_steps)
    gain = Sweep("gain", p.gain_start, p.gain_stop, p.g_steps,
                 "normalized gain", "gainloop", "gain", "{target}__res_pulse")
    frequency = None
    for target, qc in cfg["qubits"].items():
        center = qc["res_freq_ge"]
        axis = Sweep("frequency", center + p.freq_start, center + p.freq_stop, p.f_steps,
                     "MHz", "freqloop", "freq", "{target}__res_pulse")
        if frequency is None:
            frequency = axis
        qc["res_gain_ge"] = gain.qick()
        qc["res_freq_ge"] = axis.qick()
        member = cfg["readout_groups"][qc["readout_group"]]["members"][target]
        member["gain"] = qc["res_gain_ge"]
        member["frequency_mhz"] = qc["res_freq_ge"]
    return ProgramPlan(PunchoutProgram, cfg, (gain, frequency))
