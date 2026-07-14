"""Power Rabi with measurement-based active reset on tProc v2.

Each sweep point executes two readouts.  The first readout is both the
Power-Rabi result and the input consumed by the tProc feedback path.  If it
classifies the qubit as excited, a calibrated pi pulse is applied.  The second
readout verifies the post-reset state.
"""

from __future__ import annotations

import numpy as np

from qick.asm_v2 import QickSweep1D

from ...analysis.qubit import PowerRabiAnalysis
from ...core.base_experiment import BaseExperiment
from ...core.base_program import BaseProgram
from ...core.experiment_components import AcquisitionResult


class ActiveResetRabiProgram(BaseProgram):
    """tProc-v2 Power-Rabi program with measurement-based active reset."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg, prefix="ge")
        self.setup_qubit_gen(cfg, prefix="ge")

        ro_ch = cfg["ro_ch"]
        if "tproc_ch" not in self.soccfg["readouts"][ro_ch]:
            raise RuntimeError(
                f"readout channel {ro_ch} has no tProc feedback input "
                "('tproc_ch'); measurement-based active reset is unavailable "
                "with this firmware/readout channel"
            )

        self.add_loop("gainloop", cfg["steps"])

        # Current QickworkspaceV2 convention is to supply qb_gain_ge as a
        # QickSweep1D.  Accept v1-style start/step as a convenience as well.
        gain = cfg.get("qb_gain_ge")
        if not hasattr(gain, "is_sweep") and "start" in cfg and "step" in cfg:
            stop = cfg["start"] + cfg["step"] * (cfg["steps"] - 1)
            cfg["qb_gain_ge"] = QickSweep1D("gainloop", cfg["start"], stop)

        self.setup_qb_pulse(cfg, "ge", name="rabi_pulse")
        self.setup_qb_pulse(
            cfg,
            "ge",
            name="reset_pi",
            gain_key="pi_gain_ge",
        )

        component = str(cfg.get("reset_component", "I")).upper()
        if component not in {"I", "Q"}:
            raise ValueError("reset_component must be 'I' or 'Q'")
        self.reset_component = component

        excited_if = cfg.get("reset_excited_if", ">=")
        if excited_if not in {">=", "<"}:
            raise ValueError("reset_excited_if must be '>=' or '<'")
        # We jump over the reset pulse when the measurement is classified as
        # ground, so this is the logical complement of reset_excited_if.
        self.ground_test = "<" if excited_if == ">=" else ">="

        if "reset_threshold_raw" in cfg:
            threshold_raw = int(cfg["reset_threshold_raw"])
        else:
            threshold = cfg.get("reset_threshold", cfg.get("threshold"))
            if threshold is None:
                raise KeyError(
                    "active reset requires reset_threshold (normalized I/Q "
                    "units) or reset_threshold_raw (accumulator units)"
                )
            # acquire() reports length-normalized I/Q, while read_input()
            # exposes the raw accumulated integer from the readout buffer.
            threshold_raw = int(
                round(float(threshold) * self.ro_chs[ro_ch]["length"])
            )

        # A register avoids the 24-bit immediate limit of cond_jump().
        if not -(2**31) <= threshold_raw < 2**31:
            raise ValueError("active-reset threshold does not fit in int32")
        self.reset_threshold_raw = threshold_raw
        self.add_reg("reset_threshold")
        self.write_reg("reset_threshold", threshold_raw)

    def _readout(self, cfg):
        """Play the readout tone and trigger the configured ADC."""
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(
            ros=[cfg["ro_ch"]],
            pins=cfg.get("reset_trigger_pins", [0]),
            t=cfg["trig_time"],
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)

        # Prepare the state swept by the Power-Rabi experiment.
        self.pulse(ch=cfg["qb_ch"], name="rabi_pulse", t=0)
        self.delay_auto(cfg.get("rabi_readout_delay", 0.05))

        # First readout: Power-Rabi result and feedback input.
        self._readout(cfg)

        # wait_auto() blocks instruction execution until the ADC result is
        # ready.  It does not move the reference time, so delay_auto() is also
        # required to keep the conditional pulse scheduled in the future.
        read_wait = float(cfg.get("read_wait", 0.0))
        feedback_slack = float(cfg.get("extra_delay", 0.10))
        self.wait_auto(read_wait, gens=False, ros=True)
        self.delay_auto(read_wait + feedback_slack, gens=True, ros=True)

        # Skip the pi pulse when the first readout says the qubit is in |g>.
        self.read_and_jump(
            ro_ch=cfg["ro_ch"],
            component=self.reset_component,
            threshold="reset_threshold",
            test=self.ground_test,
            label="AFTER_ACTIVE_RESET",
        )
        self.pulse(ch=cfg["qb_ch"], name="reset_pi", t=0)
        self.label("AFTER_ACTIVE_RESET")

        # The compile-time timeline includes reset_pi.  Both branches therefore
        # receive an equal reset slot; the ground branch simply idles through it.
        self.delay_auto(cfg.get("reset_post_delay", 0.05))

        # Second readout: verify the state after conditional reset.
        self._readout(cfg)


class ActiveResetRabi(BaseExperiment):
    """Power Rabi with feedback reset and a second verification readout.

    Required active-reset configuration keys
    -----------------------------------------
    reset_threshold : float
        Length-normalized I or Q threshold, in the same units returned by a
        non-thresholded QICK acquisition.  Alternatively provide
        ``reset_threshold_raw`` in raw accumulator units.

    Optional keys are ``reset_component`` (``"I"`` or ``"Q"``),
    ``reset_excited_if`` (``">="`` or ``"<"``), ``read_wait``,
    ``extra_delay``, and ``reset_post_delay``.
    """

    EXPT_NAME = "s005c_power_rabi_active_reset_ge"
    TAG = "Rabi"
    X_LABEL = "Dac Gain (a.u)"
    TITLE_PREFIX = "Qubit Power Rabi ge (Active Reset)"
    SWEEP_KEYS_TO_REMOVE = ["qb_gain_ge"]
    X_SAVE_NAME = "Gain"
    X_SAVE_UNIT = "DAC unit"
    X_SAVE_SCALE = 1.0
    LivePlot = False

    Analysis = PowerRabiAnalysis

    def __init__(self, config):
        super().__init__(config)
        self.reset_verification_iq = None

    def _create_program(self):
        return ActiveResetRabiProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("rabi_pulse", "gain", as_array=True)

    def _get_readout_threshold(self):
        # ``threshold`` is used by the on-FPGA feedback logic here.  Do not let
        # BaseExperiment reinterpret it as software population discrimination.
        return None

    def _acquire(self, prog, axes, ctx):
        acquired = prog.acquire(
            self.soc,
            rounds=ctx.py_avg,
            progress=True,
        )
        channel_data = np.asarray(acquired[0])
        if channel_data.ndim < 3 or channel_data.shape[0] < 2:
            raise RuntimeError(
                "active-reset Rabi expected two readouts per shot, got "
                f"shape {channel_data.shape}"
            )

        rabi_iq = channel_data[0].dot([1, 1j])
        self.reset_verification_iq = channel_data[1].dot([1, 1j])
        self.iqdata = rabi_iq
        return AcquisitionResult(
            raw_iq=rabi_iq,
            avg_count=ctx.py_avg,
            metadata={
                "active_reset": True,
                "rabi_read_index": 0,
                "feedback_read_index": 0,
                "reset_verification_read_index": 1,
                "reset_threshold_raw": prog.reset_threshold_raw,
                "reset_component": prog.reset_component,
            },
        )

    def _finalize_result(self, acq, axes, ctx):
        result = super()._finalize_result(acq, axes, ctx)
        if self.reset_verification_iq is not None:
            result.analysis_data["reset_verification_iq"] = {
                "values": np.asarray(self.reset_verification_iq),
                "dims": ["x"],
            }
        return result


__all__ = ["ActiveResetRabiProgram", "ActiveResetRabi"]
