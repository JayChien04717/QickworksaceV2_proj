"""active_reset_rabi: program."""

import numpy as np
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.programs.sweeps import Sweep
from QickworkspaceV2.experiments.base import ProgramPlan
from .parameters import ActiveResetParameters


class ActiveResetRabiProgram(BaseProgram):
    TRANSITION = "ge"
    READOUT_EVENTS = ("pre_reset", "post_reset")
    CAPTURE_SHOTS = True

    def _initialize(self, cfg):
        if len(cfg["targets"]) != 1:
            raise ValueError("Feedback Rabi requires one explicitly selected target")
        options = ActiveResetParameters.model_validate(
            {k: cfg[k] for k in ActiveResetParameters.model_fields if k in cfg and k != "target"}
        )
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, cfg["transition"])
            self.setup_standard_gates(qc, cfg["transition"])
        self.setup_readout(cfg)
        qc = cfg["qubits"][cfg["targets"][0]]
        self.add_sweep_loop(cfg, qc["qb_gain_ge"])
        self.setup_qb_pulse(qc, "ge", name="qb_pulse")
        if qc.get("ro_threshold") is None:
            raise ValueError("Calibrate ro_threshold on the selected unrotated I/Q feedback component")
        required_rotation = 0 if options.reset_component == "I" else 90
        if abs((qc["ro_phase"] - required_rotation + 180) % 360 - 180) > 1e-6:
            raise ValueError("Hardware feedback reads raw I/Q; ro_phase must be 0 for I or 90 for Q")
        threshold = float(qc["ro_threshold"])
        if not np.isfinite(threshold):
            raise ValueError("ro_threshold must be finite")
        self.feedback_threshold = int(round(threshold * self.ro_chs[qc["ro_ch"]]["length"]))
        if not -(2**23) <= self.feedback_threshold < 2**23:
            raise ValueError("Integrated feedback threshold exceeds the signed 24-bit compare operand")
        if "tproc_ch" not in self.soccfg["readouts"][qc["ro_ch"]]:
            raise ValueError("Selected ADC has no tProc feedback input")

    def _body(self, cfg):
        qc = cfg["qubits"][cfg["targets"][0]]
        self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, "qb_pulse"), t=0)
        self.wait_pulses()
        self.measure(cfg)
        self.wait_auto(cfg.get("read_wait_us", 0.2), ros=True, gens=True)
        mode = cfg.get("reset_mode", "conditional")
        if mode == "conditional":
            skip = "<" if cfg.get("reset_excited_if", ">=") == ">=" else ">="
            self.read_and_jump(
                qc["ro_ch"], cfg.get("reset_component", "I"), self.feedback_threshold, skip, "skip_reset"
            )
        if mode != "never":
            self.resync(cfg.get("feedback_slack_us", 0.05))
            self.qubit(cfg["targets"][0]).x()
        if mode == "conditional":
            self.label("skip_reset")
        self.resync(cfg.get("feedback_slack_us", 0.05))
        self.delay_auto(cfg.get("reset_post_delay_us", 0.05))
        self.measure(cfg)


def build(ctx, p):
    cfg = ctx.config()
    cfg.update(p.model_dump(exclude={"target", "start", "stop", "points"}), steps=p.points)
    axis = Sweep("gain", p.start, p.stop, p.points, "normalized gain", "sweep", "gain", "{target}__qb_pulse")
    for qc in cfg["qubits"].values():
        qc["qb_gain_ge"] = axis.qick()
    return ProgramPlan(ActiveResetRabiProgram, cfg, (axis,), ("pre_reset", "post_reset"), True)
