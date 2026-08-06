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

        tproc_ch = self.soccfg["readouts"][ro_ch].get("tproc_ch")
        if tproc_ch is None or int(tproc_ch) < 0:
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
        # Reserve the same waveform slot in the no-reset control while
        # applying zero drive.
        self.setup_qb_pulse(
            cfg,
            "ge",
            name="reset_idle",
            gain_override=0,
        )

        reset_mode = str(cfg.get("reset_mode", "conditional")).lower()
        if reset_mode not in {"conditional", "always", "never"}:
            raise ValueError(
                "reset_mode must be 'conditional', 'always', or 'never'"
            )
        self.reset_mode = reset_mode

        # Feedback reads the raw accumulator, so store the configured
        # normalized threshold in accumulator units.
        threshold_raw = int(round(cfg["threshold"] * cfg["ro_length"]))
        self.reset_threshold_normalized = float(cfg["threshold"])
        self.reset_threshold_raw = threshold_raw
        self.reset_readout_length = float(cfg["ro_length"])
        self.reset_component = "I"
        self.ground_test = "<"
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

        if self.reset_mode == "conditional":
            self.activate_reset(cfg)
        else:
            # A/B controls keep the same readout and timing slot.
            self._readout(cfg)
            self.wait_auto(
                float(cfg.get("read_wait", 0.15)), gens=True, ros=True
            )
            self.resync(0.05)
            pulse_name = (
                "reset_pi" if self.reset_mode == "always" else "reset_idle"
            )
            self.pulse(ch=cfg["qb_ch"], name=pulse_name, t=0)

        # Every mode includes an equal-duration reset pulse slot.
        self.delay_auto(cfg.get("reset_post_delay", 0.05))

        # Second readout: verify the state after conditional reset.
        self._readout(cfg)


class ActiveResetRabi(BaseExperiment):
    """Power Rabi with feedback reset and a second verification readout.

    Required active-reset configuration keys
    -----------------------------------------
    threshold : float
        Length-normalized I or Q threshold, in the same units returned by a
        non-thresholded QICK acquisition.

    Optional keys are ``reset_mode`` (``"conditional"``, ``"always"``, or
    ``"never"``), ``read_wait``, and ``reset_post_delay``. Feedback uses I;
    values below threshold are ground and skip the reset pulse.
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
            if channel_data.shape[-1] != 2:
                raise RuntimeError(
                    "thresholded active-reset data must end in the QICK "
                    f"[population, Q-placeholder] axis, got {channel_data.shape}"
                )
            # QICK retains a final two-component axis after software
            # thresholding. Only component 0 contains the population;
            # component 1 is a zero placeholder and must not reach plotting.
            threshold_data = np.asarray(channel_data[..., 0], dtype=float)
            pre_reset = threshold_data[0]
            post_reset = threshold_data[1]
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
                "reset_threshold_normalized": threshold,
                "reset_readout_length": prog.reset_readout_length,
                "reset_component": prog.reset_component,
                "reset_mode": prog.reset_mode,
            },
        )


__all__ = ["ActiveResetRabiProgram", "ActiveResetRabi"]
