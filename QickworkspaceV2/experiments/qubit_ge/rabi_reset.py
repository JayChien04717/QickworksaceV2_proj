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
from ...core.acquisition import decode_readouts
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

        self.add_loop("gainloop", cfg["steps"])

        # Current QickworkspaceV2 convention is to supply qb_gain_ge as a
        # QickSweep1D.  Accept v1-style start/step as a convenience as well.
        gain = cfg.get("qb_gain_ge")
        if not hasattr(gain, "is_sweep") and "start" in cfg and "step" in cfg:
            stop = cfg["start"] + cfg["step"] * (cfg["steps"] - 1)
            cfg["qb_gain_ge"] = QickSweep1D("gainloop", cfg["start"], stop)

        self.setup_qb_pulse(cfg, "ge", name="rabi_pulse")
        self.setup_active_reset(cfg)

    def _readout(self, cfg):
        """Trigger a readout using the active-reset marker pins."""
        self.measure(
            cfg,
            pins=cfg.get("reset_trigger_pins", [0]),
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
    ``"never"``), ``reset_component`` (I or Q), ``reset_excited_if``,
    ``read_wait``, ``feedback_slack``, and ``reset_post_delay``.
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
        readouts = decode_readouts(acquired, threshold=threshold is not None)
        if readouts.shape[0] < 2:
            raise RuntimeError(
                "active-reset Rabi expected two readouts per shot, got "
                f"shape {readouts.shape}"
            )

        if threshold is not None:
            pre_reset = np.asarray(readouts[0], dtype=float)
            post_reset = np.asarray(readouts[1], dtype=float)
            # QICK threshold acquisition reports P(component >= threshold).
            # Convert it when the configured excited cloud is below threshold.
            if prog.ground_test == ">=":
                pre_reset = 1.0 - pre_reset
                post_reset = 1.0 - post_reset
            self.pre_reset_population = pre_reset
            self.post_reset_population = post_reset
            self.reset_verification_iq = post_reset
        else:
            pre_reset = readouts[0]
            post_reset = readouts[1]
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
            axes={
                "readout": {
                    "values": ["pre_reset", "post_reset"],
                    "label": "Readout",
                }
            },
            raw_data={
                "readouts": np.stack((pre_reset, post_reset)),
            },
            analysis_data=analysis_data,
            dataset_dims={"readouts": ["readout", "x"]},
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
