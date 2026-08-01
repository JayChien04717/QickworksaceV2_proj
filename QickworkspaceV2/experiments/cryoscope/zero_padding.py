"""Step 2: sub-three-clock cryoscope points using an arb envelope."""

from __future__ import annotations

import numpy as np

from ._common import (
    CryoscopeExperimentBase,
    CryoscopeProgramBase,
    generator_sample_period_ns,
    make_zero_padded_rectangle,
    quantize_envelope,
)


class CryoscopeZeroPaddingProgram(CryoscopeProgramBase):
    """Play a short prefix and pad the descriptor with trailing zeros."""

    def _add_flux_pulse(self, cfg):
        ch = cfg["flux_ch"]
        active_samples = int(cfg["flux_active_samples"])
        normalized = make_zero_padded_rectangle(
            self.soccfg,
            ch,
            active_samples,
            amplitude=cfg.get("flux_envelope_amplitude", 1.0),
        )
        self.flux_waveform = normalized
        self.flux_active_samples = active_samples
        self.flux_sample_period_ns = generator_sample_period_ns(self.soccfg, ch)

        self.add_envelope(
            ch=ch,
            name="flux_zero_padded_env",
            idata=quantize_envelope(self.soccfg, ch, normalized),
        )
        self.add_pulse(
            ch=ch,
            name="flux_pulse",
            style="arb",
            envelope="flux_zero_padded_env",
            freq=0,
            phase=0,
            gain=cfg["flux_gain"],
        )


class CryoscopeZeroPadding(CryoscopeExperimentBase):
    """One short-pulse point, with optional Python-side sample-count sweep."""

    EXPT_NAME = "cryoscope_zero_padding"
    X_LABEL = "Active flux time (ns)"
    TITLE_PREFIX = "Cryoscope — zero-padded short flux pulse"
    X_SAVE_NAME = "Active flux time"
    X_SAVE_UNIT = "ns"
    X_SAVE_SCALE = 1.0

    def _create_program(self):
        return CryoscopeZeroPaddingProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        active_ns = prog.flux_active_samples * prog.flux_sample_period_ns
        return np.array([active_ns])

    def run_sample_sweep(self, active_samples, py_avg: int, *, acquire_xy=True, **run_kwargs):
        """Rebuild and run one envelope per requested sample count, including zero."""

        results = {}
        for sample_count in active_samples:
            sample_count = int(sample_count)
            point_cfg = dict(self.cfg)
            point_cfg["flux_active_samples"] = sample_count
            point_experiment = type(self)(point_cfg)
            if acquire_xy:
                results[sample_count] = point_experiment.run_xy(py_avg, **run_kwargs)
            else:
                results[sample_count] = point_experiment.run(py_avg, **run_kwargs)
        return results


__all__ = ["CryoscopeZeroPaddingProgram", "CryoscopeZeroPadding"]
