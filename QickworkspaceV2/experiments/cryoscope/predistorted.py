"""Step 4: upload and verify a host-computed predistorted flux waveform."""

from __future__ import annotations

import numpy as np

from ._common import (
    CryoscopeExperimentBase,
    CryoscopeProgramBase,
    generator_sample_period_ns,
    pad_normalized_envelope,
    quantize_envelope,
)


class PredistortedCryoscopeProgram(CryoscopeProgramBase):
    """Play normalized samples produced by the FIR/IIR design functions."""

    def _add_flux_pulse(self, cfg):
        """Add flux pulse.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.

        Raises
        ------
        ValueError
            If the operation cannot be completed.
        """
        ch = cfg["flux_ch"]
        supplied = np.asarray(cfg["predistorted_waveform"], dtype=float)
        active_samples = int(cfg.get("predistorted_active_samples", supplied.size))
        if not 1 <= active_samples <= supplied.size:
            raise ValueError(
                "predistorted_active_samples must lie between 1 and the waveform length"
            )

        useful_waveform = supplied[:active_samples]
        padded = pad_normalized_envelope(self.soccfg, ch, useful_waveform)
        self.flux_waveform = padded
        self.flux_waveform_unpadded = useful_waveform
        self.flux_active_samples = active_samples
        self.flux_sample_period_ns = generator_sample_period_ns(self.soccfg, ch)

        self.add_envelope(
            ch=ch,
            name="predistorted_flux_env",
            idata=quantize_envelope(self.soccfg, ch, padded),
        )
        self.add_pulse(
            ch=ch,
            name="flux_pulse",
            style="arb",
            envelope="predistorted_flux_env",
            freq=0,
            phase=0,
            gain=cfg["flux_gain"],
        )


class PredistortedCryoscope(CryoscopeExperimentBase):
    """Apply one compensated waveform or verify many waveform prefixes."""

    EXPT_NAME = "cryoscope_predistorted"
    X_LABEL = "Applied waveform time (ns)"
    TITLE_PREFIX = "Cryoscope — predistorted flux pulse"
    X_SAVE_NAME = "Applied waveform time"
    X_SAVE_UNIT = "ns"
    X_SAVE_SCALE = 1.0

    def _create_program(self):
        """Create the QICK program for this experiment.

        Returns
        -------
        Any
            Result of the operation.
        """
        return PredistortedCryoscopeProgram(
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
        applied_ns = prog.flux_active_samples * prog.flux_sample_period_ns
        return np.array([applied_ns])

    def run_prefix_sweep(self, sample_counts, py_avg: int, *, acquire_xy=True, **run_kwargs):
        """Measure prefixes of one compensated waveform in a Python-side loop.

        Parameters
        ----------
        sample_counts : Any
            Value for ``sample_counts``.
        py_avg : int
            Number of Python-level acquisition averages.
        acquire_xy : Any, default: True
            Value for ``acquire_xy``.
        **run_kwargs : Any
            Value for ``run_kwargs``.

        Returns
        -------
        Any
            Result of the operation.

        Raises
        ------
        ValueError
            If the operation cannot be completed.
        """

        results = {}
        waveform = np.asarray(self.cfg["predistorted_waveform"], dtype=float)
        for sample_count in sample_counts:
            sample_count = int(sample_count)
            if not 1 <= sample_count <= waveform.size:
                raise ValueError(
                    f"sample count {sample_count} is outside [1, {waveform.size}]"
                )
            point_cfg = dict(self.cfg)
            point_cfg["predistorted_active_samples"] = sample_count
            point_experiment = type(self)(point_cfg)
            if acquire_xy:
                results[sample_count] = point_experiment.run_xy(py_avg, **run_kwargs)
            else:
                results[sample_count] = point_experiment.run(py_avg, **run_kwargs)
        return results


__all__ = ["PredistortedCryoscopeProgram", "PredistortedCryoscope"]
