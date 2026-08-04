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
        """Initialize pulse and acquisition resources.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.

        Raises
        ------
        KeyError
            If the operation cannot be completed.
        RuntimeError
            If the operation cannot be completed.
        ValueError
            If the operation cannot be completed.
        """
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

        threshold = cfg.get("threshold", cfg.get("reset_threshold"))
        self.reset_threshold_normalized = (
            None if threshold is None else float(threshold)
        )
        if "reset_threshold_raw" in cfg:
            threshold_raw = int(cfg["reset_threshold_raw"])
        else:
            if threshold is None:
                raise KeyError(
                    "active reset requires threshold (normalized I/Q "
                    "units) or reset_threshold_raw (accumulator units)"
                )
            # acquire() reports length-normalized I/Q, while read_input()
            # exposes the raw accumulated integer from the readout buffer. Add
            # back the readout offset that acquire(remove_offset=True) removes.
            iq_offset = np.asarray(
                self.soccfg["readouts"][ro_ch].get("iq_offset", [0.0, 0.0]),
                dtype=float,
            ).reshape(-1)
            offset_index = 0 if component == "I" else 1
            component_offset = (
                float(iq_offset[offset_index])
                if iq_offset.size > offset_index
                else 0.0
            )
            threshold_raw = int(
                round(
                    (float(threshold) + component_offset)
                    * self.ro_chs[ro_ch]["length"]
                )
            )

        # A register avoids the 24-bit immediate limit of cond_jump().
        if not -(2**31) <= threshold_raw < 2**31:
            raise ValueError("active-reset threshold does not fit in int32")
        self.reset_threshold_raw = threshold_raw
        self.add_reg("reset_threshold")
        self.write_reg("reset_threshold", threshold_raw)

    def _readout(self, cfg):
        """Play the readout tone and trigger the configured ADC.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(
            ros=[cfg["ro_ch"]],
            pins=cfg.get("reset_trigger_pins", [0]),
            t=cfg["trig_time"],
        )

    def _body(self, cfg):
        """Execute one iteration of the pulse sequence.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
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
    threshold : float
        Length-normalized I or Q threshold, in the same units returned by a
        non-thresholded QICK acquisition.  Alternatively provide
        ``reset_threshold_raw`` in raw accumulator units.

    Optional keys are ``reset_component`` (``"I"`` or ``"Q"``),
    ``reset_excited_if`` (``">="`` or ``"<"``), ``read_wait``,
    ``extra_delay``, and ``reset_post_delay``.
    """

    EXPT_NAME = "s005c_power_rabi_active_reset_ge"
    TAG = "PowerRabi"
    X_LABEL = "Dac Gain (a.u)"
    TITLE_PREFIX = "Qubit Power Rabi ge (Active Reset)"
    SWEEP_KEYS_TO_REMOVE = ["qb_gain_ge"]
    X_SAVE_NAME = "Gain"
    X_SAVE_UNIT = "DAC unit"
    X_SAVE_SCALE = 1.0
    LivePlot = False

    Analysis = PowerRabiAnalysis

    def __init__(self, config):
        """Initialize the ActiveResetRabi instance.

        Parameters
        ----------
        config : Any
            Experiment configuration.
        """
        super().__init__(config)
        self.pre_reset_population = None
        self.post_reset_population = None
        self.reset_verification_iq = None

    def _create_program(self):
        """Create the QICK program for this experiment.

        Returns
        -------
        Any
            Result of the operation.
        """
        return ActiveResetRabiProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        """Extract the primary sweep axis from the program.

        Parameters
        ----------
        prog : Any
            Value for ``prog``.

        Returns
        -------
        Any
            Result of the operation.
        """
        return prog.get_pulse_param("rabi_pulse", "gain", as_array=True)

    def _get_readout_threshold(self):
        # ``threshold`` is used by the on-FPGA feedback logic here.  Do not let
        # BaseExperiment reinterpret it as software population discrimination.
        """Return readout threshold.

        Returns
        -------
        Any
            Result of the operation.
        """
        return None

    def _acquire(self, prog, axes, ctx):
        """Acquire experiment data.

        Parameters
        ----------
        prog : Any
            Value for ``prog``.
        axes : Any
            Value for ``axes``.
        ctx : Any
            Value for ``ctx``.

        Returns
        -------
        Any
            Result of the operation.

        Raises
        ------
        RuntimeError
            If the operation cannot be completed.
        """
        self.pre_reset_population = None
        self.post_reset_population = None
        self.reset_verification_iq = None

        threshold = prog.reset_threshold_normalized
        angle = 0.0 if prog.reset_component == "I" else np.pi / 2
        acquired = prog.acquire(
            self.soc,
            rounds=ctx.py_avg,
            threshold=threshold,
            angle=angle,
            progress=True,
        )
        channel_data = np.asarray(acquired[0])
        if channel_data.shape[0] < 2:
            raise RuntimeError(
                "active-reset Rabi expected two readouts per shot, got "
                f"shape {channel_data.shape}"
            )

        if threshold is not None:
            pre_reset = np.asarray(channel_data[0], dtype=float)
            post_reset = np.asarray(channel_data[1], dtype=float)
            # QICK threshold acquisition reports P(component >= threshold).
            # Convert it when the configured excited cloud is below threshold.
            if prog.ground_test == ">=":
                pre_reset = 1.0 - pre_reset
                post_reset = 1.0 - post_reset
            self.pre_reset_population = pre_reset
            self.post_reset_population = post_reset
            self.reset_verification_iq = post_reset
        else:
            if channel_data.ndim < 3 or channel_data.shape[-1] != 2:
                raise RuntimeError(
                    "non-thresholded active-reset data must end in an I/Q axis, "
                    f"got shape {channel_data.shape}"
                )
            pre_reset = channel_data[0].dot([1, 1j])
            post_reset = channel_data[1].dot([1, 1j])
            self.reset_verification_iq = post_reset

        self.iqdata = pre_reset
        if threshold is not None:
            analysis_data = {
                "pre_reset_population": {
                    "values": pre_reset,
                    "dims": ["x"],
                },
                "post_reset_population": {
                    "values": post_reset,
                    "dims": ["x"],
                },
            }
        else:
            analysis_data = {
                "reset_verification_iq": {
                    "values": np.asarray(self.reset_verification_iq),
                    "dims": ["x"],
                }
            }

        return AcquisitionResult(
            raw_iq=pre_reset,
            analysis_data=analysis_data,
            avg_count=ctx.py_avg,
            metadata={
                "active_reset": True,
                "threshold_discrimination": threshold is not None,
                "threshold": threshold,
                "rabi_read_index": 0,
                "feedback_read_index": 0,
                "reset_verification_read_index": 1,
                "reset_threshold_raw": prog.reset_threshold_raw,
                "reset_component": prog.reset_component,
            },
        )


__all__ = ["ActiveResetRabiProgram", "ActiveResetRabi"]
