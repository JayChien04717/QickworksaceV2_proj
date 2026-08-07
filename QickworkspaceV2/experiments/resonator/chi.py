"""Dispersive-shift measurement from ground/excited resonator spectra."""

from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm

from ...analysis.resonator import DispersiveShiftAnalysis
from ...core.base_experiment import BaseExperiment
from ...core.base_program import BaseProgram
from ...core.experiment_data import ExperimentData, QualityFlag
from .res_spec import ResonatorSpec


class ExcitedResonatorSpecProgram(BaseProgram):
    """Prepare |e> with the calibrated ge pi pulse, then sweep readout."""

    def _initialize(self, cfg):
        """Initialize pulse and acquisition resources.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.setup_resonator(cfg, prefix="ge")
        self.setup_qubit_gen(cfg, prefix="ge")
        self.setup_qb_pulse(
            cfg, prefix="ge", name="qb_pi_pulse", gain_key="pi_gain_ge"
        )
        self.add_loop("freqloop", cfg["steps"])

    def _body(self, cfg):
        """Execute one iteration of the pulse sequence.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
        self.delay_auto(0.02)
        self.measure(cfg)


class ReadoutFrequencySNRProgram(BaseProgram):
    """Collect prepared g/e shots across a readout-frequency sweep."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg, prefix="ge")
        self.setup_qubit_gen(cfg, prefix="ge")
        self.setup_qb_pulse(
            cfg, prefix="ge", name="qb_pi_pulse", gain_key="pi_gain_ge"
        )
        self.add_loop("shotloop", cfg["snr_shots"])
        self.add_loop("freqloop", cfg["steps"])

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.measure(cfg)
        self.delay_auto(cfg["relax_delay"])
        self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
        self.delay_auto(0.02)
        self.measure(cfg)


def _decode_frequency_shots(acquired, shots, frequencies):
    """Return complex IQ with shape (state, shot, frequency)."""
    values = np.asarray(acquired)
    if values.shape[-1] != 2:
        raise RuntimeError(f"Expected trailing IQ axis, received shape {values.shape}")
    values = np.squeeze(values.dot([1.0, 1.0j]))
    if values.ndim != 3 or values.shape[0] != 2:
        raise RuntimeError(
            "Expected two readouts with shot/frequency loops; "
            f"received shape {values.shape}"
        )
    if values.shape[1:] == (shots, frequencies):
        return values
    if values.shape[1:] == (frequencies, shots):
        return values.transpose(0, 2, 1)
    raise RuntimeError(
        f"Cannot map QICK data shape {values.shape} to "
        f"(2, {shots}, {frequencies})"
    )


def _frequency_snr(iq_shots):
    """Compute the QUA-compatible g/e readout SNR for every frequency."""
    iq_shots = np.asarray(iq_shots)
    if iq_shots.ndim != 3 or iq_shots.shape[0] != 2:
        raise ValueError("iq_shots must have shape (2, shot, frequency)")
    means = np.mean(iq_shots, axis=1)
    ddof = 1 if iq_shots.shape[1] > 1 else 0
    variance = (
        np.var(iq_shots[0].real, axis=0, ddof=ddof)
        + np.var(iq_shots[0].imag, axis=0, ddof=ddof)
        + np.var(iq_shots[1].real, axis=0, ddof=ddof)
        + np.var(iq_shots[1].imag, axis=0, ddof=ddof)
    ) / 4.0
    signal = np.abs(means[1] - means[0]) ** 2
    snr = np.divide(
        signal, 2.0 * variance,
        out=np.full(signal.shape, np.nan, dtype=float),
        where=variance > 0,
    )
    return means, variance, snr

class _ExcitedResonatorSpec(ResonatorSpec):
    """Internal excited-state spectrum used by :class:`DispersiveShift`."""

    EXPT_NAME = "resonator_spec_e"
    TITLE_PREFIX = "Resonator Spectroscopy |e>"

    def _create_program(self):
        """Create the QICK program for this experiment.

        Returns
        -------
        Any
            Result of the operation.
        """
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
        """Create the QICK program for this experiment.

        Raises
        ------
        NotImplementedError
            If the operation cannot be completed.
        """
        raise NotImplementedError("DispersiveShift runs separate |g> and |e> programs")

    def _extract_sweep_axis(self, prog):
        """Extract the primary sweep axis from the program.

        Parameters
        ----------
        prog : Any
            Value for ``prog``.

        Raises
        ------
        NotImplementedError
            If the operation cannot be completed.
        """
        raise NotImplementedError("Sweep axes are supplied by the two spectra")

    def run(self, py_avg: int, solve_type: str = "hm", **kwargs) -> ExperimentData:
        """Acquire g/e frequency shots, fit chi, and select maximum-SNR frequency.

        ``py_avg`` is the number of batches; ``cfg["snr_shots"]`` is the
        number of raw shots per state and frequency in each batch.
        The returned :class:`ExperimentData` contains
        ``best_readout_frequency_MHz`` in ``fit_result``.
        """
        required = ("qb_freq_ge", "pi_gain_ge", "sigma_ge")
        missing = [key for key in required if key not in self.cfg]
        if missing:
            raise KeyError(
                "DispersiveShift requires completed ge Rabi calibration; "
                f"missing config keys: {missing}"
            )
        batches = int(py_avg)
        if batches < 1:
            raise ValueError("py_avg must be positive")
        configured_shots = self.cfg.get("snr_shots")
        if configured_shots is None:
            shots_per_batch = batches
            batches = 1
        else:
            shots_per_batch = int(configured_shots)
        if shots_per_batch < 2:
            raise ValueError("snr_shots must be at least 2")
        total_shots = batches * shots_per_batch

        plot_alias = kwargs.pop("plot", None)
        plot_analysis = bool(kwargs.pop("plot_analysis", False))
        if plot_alias is not None:
            plot_analysis = bool(plot_alias)
        return_best_freq = bool(kwargs.pop("return_best_freq", False))
        progress = bool(kwargs.pop("progress", True))
        # Accepted for BaseExperiment.run call-site compatibility. Chi always
        # retains complex IQ shots and uses its own compact progress display.
        kwargs.pop("iq_process", None)
        kwargs.pop("liveplot", None)
        kwargs.pop("show_final_plot", None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported Chi.run options: {unknown}")

        run_cfg = dict(self.cfg)
        run_cfg["snr_shots"] = shots_per_batch
        program = ReadoutFrequencySNRProgram(
            self.soccfg, reps=1,
            final_delay=run_cfg["relax_delay"], cfg=run_cfg,
        )
        frequency = np.asarray(
            program.get_pulse_param("res_pulse", "freq", as_array=True),
            dtype=float,
        ).squeeze()
        if frequency.ndim != 1 or frequency.size != int(run_cfg["steps"]):
            unique = np.unique(frequency)
            if unique.size != int(run_cfg["steps"]):
                raise RuntimeError(
                    f"Could not resolve {run_cfg['steps']} frequency points from "
                    f"QICK axis shape {frequency.shape}"
                )
            frequency = unique

        chunks = []
        batch_iter = range(batches)
        if progress and batches > 1:
            batch_iter = tqdm(batch_iter, desc="Chi SNR batches")
        for _ in batch_iter:
            acquired = program.acquire(
                self.soc, rounds=1, progress=progress and batches == 1
            )[0]
            chunks.append(_decode_frequency_shots(
                acquired, shots=shots_per_batch, frequencies=frequency.size
            ))
        iq_shots = np.concatenate(chunks, axis=1)
        means, noise_variance, snr = _frequency_snr(iq_shots)
        if not np.any(np.isfinite(snr)):
            raise RuntimeError("All readout-frequency SNR values are invalid")
        best_index = int(np.nanargmax(snr))
        best_frequency = float(frequency[best_index])

        self._sweep_vals_x = frequency
        self._sweep_vals_y = np.array([0.0, 1.0])
        self.iqdata = means
        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=means,
            x_axis=frequency,
            y_axis=self._sweep_vals_y,
            axes={
                "frequency": {"values": frequency, "unit": "MHz"},
                "state": {"values": ["g", "e"]},
                "shot": {"values": np.arange(total_shots), "unit": "#"},
            },
            raw_data={
                "iq_shots": {
                    "values": iq_shots,
                    "dims": ["state", "shot", "frequency"],
                },
            },
            analysis_data={
                "readout_snr": {"values": snr, "dims": ["frequency"]},
                "noise_variance": {
                    "values": noise_variance, "dims": ["frequency"]
                },
            },
            dataset_dims={"iq": ["state", "frequency"]},
            metadata={
                "states": ["g", "e"],
                "solve_type": solve_type,
                "snr_shots_per_batch": shots_per_batch,
                "snr_batches": batches,
                "snr_total_shots_per_state": total_shots,
                "best_readout_frequency_MHz": best_frequency,
                "snr_formula": "abs(mean_e - mean_g)^2 / (2 * mean_IQ_variance)",
            },
            avg_count=total_shots,
            quality=QualityFlag.NO_INFORMATION,
            x_name=self.X_SAVE_NAME,
            x_unit=self.X_SAVE_UNIT,
            x_scale=self.X_SAVE_SCALE,
            y_name=self.Y_SAVE_NAME,
            y_unit=self.Y_SAVE_UNIT,
        )
        result = self.Analysis().run(result)
        result.fit_result.update({
            "best_readout_frequency_MHz": (best_frequency, None),
            "max_readout_snr": (float(snr[best_index]), None),
        })
        self.best_frequency = best_frequency
        self.result = result
        if plot_analysis:
            self.Analysis().plot(result)
        return (result, best_frequency) if return_best_freq else result
    def optimize_readout_frequency(
        self, shots: int, solve_type: str = "hm", **kwargs
    ) -> float:
        """Run Chi/SNR analysis and return only the maximum-SNR frequency."""
        kwargs.pop("return_best_freq", None)
        result = self.run(shots, solve_type=solve_type, **kwargs)
        return float(result["best_readout_frequency_MHz"])

# Concise notebook alias.
Chi = DispersiveShift

__all__ = ["Chi", "DispersiveShift", "ExcitedResonatorSpecProgram", "ReadoutFrequencySNRProgram"]
