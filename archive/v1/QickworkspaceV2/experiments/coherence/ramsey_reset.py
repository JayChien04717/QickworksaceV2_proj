"""Ramsey experiment with measurement-based active reset on tProc v2."""

from __future__ import annotations

from ...analysis.qubit import RamseyAnalysis
from ..qubit_ge.rabi_reset import ActiveResetRabi
from ..qubit_ge.rabi_reset import ActiveResetRabiProgram
from .ramsey import Ramsey


class ActiveResetRamseyProgram(ActiveResetRabiProgram):
    """Ramsey sequence followed by conditional pi reset and verification."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg, prefix="ge")
        self.setup_qubit_gen(cfg, prefix="ge")

        ro_ch = cfg["ro_ch"]
        tproc_ch = self.soccfg["readouts"][ro_ch].get("tproc_ch")
        if tproc_ch is None or int(tproc_ch) < 0:
            raise RuntimeError(
                f"readout channel {ro_ch} has no tProc feedback input "
                "('tproc_ch'); measurement-based active reset is unavailable"
            )

        self.add_loop("waitloop", cfg["steps"])
        self.setup_qb_pulse(
            cfg, "ge", name="qb_pulse1", gain_key="pi2_gain_ge"
        )
        ramsey_phase = (
            cfg.get("qb_phase", 0)
            + cfg["wait_time"] * 360 * cfg["virtual_detune"]
        )
        self.setup_qb_pulse(
            cfg,
            "ge",
            name="qb_pulse2",
            gain_key="pi2_gain_ge",
            phase=ramsey_phase,
        )
        self.setup_qb_pulse(
            cfg, "ge", name="reset_pi", gain_key="pi_gain_ge"
        )

        threshold_raw = int(round(cfg["threshold"] * cfg["ro_length"]))
        self.reset_threshold_normalized = float(cfg["threshold"])
        self.reset_threshold_raw = threshold_raw
        self.reset_readout_length = float(cfg["ro_length"])
        self.reset_component = "I"
        self.ground_test = "<"
        self.add_reg("reset_threshold")
        self.write_reg("reset_threshold", threshold_raw)

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)

        self.pulse(ch=cfg["qb_ch"], name="qb_pulse1", t=0)
        self.delay_auto(cfg["wait_time"] + 0.01, tag="wait")
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse2", t=0)
        self.delay_auto(cfg.get("ramsey_readout_delay", 0.05))

        # Ramsey result is also the feedback input. Only non-ground shots
        # receive reset_pi.
        self.activate_reset(cfg)
        self.delay_auto(cfg.get("reset_post_delay", 0.05))

        # Verification readout.
        self._readout(cfg)


class ActiveResetRamsey(Ramsey):
    """Ramsey fit from pre-reset data with post-reset verification exposed."""

    EXPT_NAME = "s006b_Ramsey_active_reset_ge"
    TITLE_PREFIX = "Qubit Ramsey ge (Active Reset)"
    Analysis = RamseyAnalysis

    def __init__(self, config):
        super().__init__(config)
        self.pre_reset_population = None
        self.post_reset_population = None
        self.reset_verification_iq = None

    def _create_program(self):
        return ActiveResetRamseyProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    _get_readout_threshold = ActiveResetRabi._get_readout_threshold
    _acquire = ActiveResetRabi._acquire


__all__ = ["ActiveResetRamseyProgram", "ActiveResetRamsey"]