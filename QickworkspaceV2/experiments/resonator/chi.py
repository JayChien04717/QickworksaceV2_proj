"""Dispersive-shift measurement from ground/excited resonator spectra."""

from __future__ import annotations

import numpy as np

from ...analysis.resonator import DispersiveShiftAnalysis
from ...core.base_experiment import BaseExperiment
from ...core.base_program import BaseProgram
from ...core.experiment_data import ExperimentData, QualityFlag
from .res_spec import ResonatorSpec


class ExcitedResonatorSpecProgram(BaseProgram):
    """Prepare |e> with the calibrated ge pi pulse, then sweep readout."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg, prefix="ge")
        self.setup_qubit_gen(cfg, prefix="ge")
        self.setup_qb_pulse(
            cfg, prefix="ge", name="qb_pi_pulse", gain_key="pi_gain_ge"
        )
        self.add_loop("freqloop", cfg["steps"])

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
        self.delay_auto(0.02)
        self.measure(cfg)


class _ExcitedResonatorSpec(ResonatorSpec):
    """Internal excited-state spectrum used by :class:`DispersiveShift`."""

    EXPT_NAME = "resonator_spec_e"
    TITLE_PREFIX = "Resonator Spectroscopy |e>"

    def _create_program(self):
        return ExcitedResonatorSpecProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )


class DispersiveShift(BaseExperiment):
    """Measure resonator spectra in |g> and |e> and extract chi."""

    EXPT_NAME = "resonator_chi_ge"
    TAG = "Chi"
    X_LABEL = "Frequency (MHz)"
    Y_LABEL = "Qubit state"
    TITLE_PREFIX = "Dispersive Shift"
    SWEEP_KEYS_TO_REMOVE = ["res_freq_ge"]
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6
    Y_SAVE_NAME = "Qubit State"
    Y_SAVE_UNIT = "0:g, 1:e"
    Analysis = DispersiveShiftAnalysis

    def _create_program(self):
        raise NotImplementedError("DispersiveShift runs separate |g> and |e> programs")

    def _extract_sweep_axis(self, prog):
        raise NotImplementedError("Sweep axes are supplied by the two spectra")

    def run(self, py_avg: int, solve_type: str = "hm", **kwargs) -> ExperimentData:
        """Acquire |g> and |e> spectra using the calibrated ge pi pulse."""
        required = ("qb_freq_ge", "pi_gain_ge", "sigma_ge")
        missing = [key for key in required if key not in self.cfg]
        if missing:
            raise KeyError(
                "DispersiveShift requires completed ge Rabi calibration; "
                f"missing config keys: {missing}"
            )

        run_options = dict(kwargs)
        run_options.pop("plot_analysis", None)
        ground = ResonatorSpec(dict(self.cfg)).run(
            py_avg, solve_type=solve_type, plot_analysis=False, **run_options
        )
        excited = _ExcitedResonatorSpec(dict(self.cfg)).run(
            py_avg, solve_type=solve_type, plot_analysis=False, **run_options
        )

        if ground.x_axis is None or excited.x_axis is None:
            raise RuntimeError("Ground or excited resonator sweep has no frequency axis")
        if not np.allclose(ground.x_axis, excited.x_axis):
            raise RuntimeError("Ground and excited resonator frequency axes do not match")

        self._sweep_vals_x = np.asarray(ground.x_axis)
        self._sweep_vals_y = np.array([0.0, 1.0])
        self.iqdata = np.stack(
            [np.squeeze(ground.raw_iq), np.squeeze(excited.raw_iq)]
        )
        config_snapshot = self._snapshot_config()
        config_snapshot["_solve_type"] = solve_type
        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=self.iqdata,
            x_axis=self._sweep_vals_x,
            y_axis=self._sweep_vals_y,
            config=config_snapshot,
            metadata={
                "states": ["g", "e"],
                "solve_type": solve_type,
                "iq_process": kwargs.get("iq_process", self.IQ_PROCESS),
            },
            avg_count=py_avg,
            quality=QualityFlag.NO_INFORMATION,
            x_name=self.X_SAVE_NAME,
            x_unit=self.X_SAVE_UNIT,
            x_scale=self.X_SAVE_SCALE,
            y_name=self.Y_SAVE_NAME,
            y_unit=self.Y_SAVE_UNIT,
        )
        self.result = self.Analysis().run(result)
        return self.result


# Concise notebook alias.
Chi = DispersiveShift

__all__ = ["Chi", "DispersiveShift", "ExcitedResonatorSpecProgram"]
