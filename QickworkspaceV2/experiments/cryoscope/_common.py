"""Shared building blocks for the cryoscope experiments.

This module intentionally contains the small amount of QICK plumbing that is
common to all three pulse styles.  The public, reader-facing examples live in
``const.py``, ``zero_padding.py``, and ``predistorted.py``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ...core.base_experiment import BaseExperiment
from ...core.base_program import BaseProgram


def generator_sample_period_ns(soccfg, ch: int) -> float:
    """Return the programmable envelope-sample interval for a generator."""

    gen_cfg = soccfg["gens"][ch]
    samples_per_clock = int(gen_cfg["samps_per_clk"])
    return 1000.0 / (float(gen_cfg["f_fabric"]) * samples_per_clock)


def minimum_envelope_samples(soccfg, ch: int) -> int:
    """QICK pulse descriptors must span at least three fabric clocks."""

    return 3 * int(soccfg["gens"][ch]["samps_per_clk"])


def pad_normalized_envelope(
    soccfg,
    ch: int,
    samples: Iterable[float],
    *,
    minimum_clocks: int = 3,
) -> np.ndarray:
    """Validate a normalized envelope and append the padding QICK requires.

    Padding is placed *after* the useful waveform so the physical rising edge
    stays at the requested start time.
    """

    waveform = np.asarray(samples, dtype=float)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("flux waveform must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(waveform)):
        raise ValueError("flux waveform contains NaN or infinite values")
    peak = float(np.max(np.abs(waveform)))
    if peak > 1.0 + 1e-12:
        raise ValueError(
            f"normalized flux waveform has peak {peak:.4g} > 1; "
            "reduce the target amplitude or scale the predistorted waveform"
        )

    samples_per_clock = int(soccfg["gens"][ch]["samps_per_clk"])
    required = max(minimum_clocks * samples_per_clock, waveform.size)
    remainder = required % samples_per_clock
    if remainder:
        required += samples_per_clock - remainder
    return np.pad(waveform, (0, required - waveform.size))


def quantize_envelope(soccfg, ch: int, normalized_samples) -> np.ndarray:
    """Convert a normalized real envelope to QICK's signed int16 format."""

    waveform = np.asarray(normalized_samples, dtype=float)
    maxv = int(soccfg.get_maxv(ch))
    return np.round(waveform * maxv).astype(np.int16)


def make_zero_padded_rectangle(
    soccfg,
    ch: int,
    active_samples: int,
    *,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Make a short rectangle followed by enough zeros for a legal pulse."""

    active_samples = int(active_samples)
    if active_samples < 0:
        raise ValueError("active_samples must be non-negative")
    if abs(amplitude) > 1:
        raise ValueError("amplitude must be in the normalized range [-1, 1]")
    # pad_normalized_envelope requires a non-empty array.  For the zero point,
    # a single zero is supplied and then padded to the same legal descriptor.
    active = np.full(max(1, active_samples), float(amplitude))
    if active_samples == 0:
        active[0] = 0.0
    return pad_normalized_envelope(soccfg, ch, active)


class CryoscopeProgramBase(BaseProgram):
    """Ramsey cryoscope sequence shared by all flux-pulse implementations."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.setup_standard_gates(cfg, prefix="ge")

        flux_ch = cfg["flux_ch"]
        if flux_ch in (cfg["qb_ch"], cfg["res_ch"]):
            raise ValueError("flux_ch must be separate from qb_ch and res_ch")
        gen_params = {"ch": flux_ch, "nqz": cfg.get("nqz_flux", 1)}
        gen_cfg = self.soccfg["gens"][flux_ch]
        has_mixer = gen_cfg.get(
            "has_mixer", gen_cfg.get("type") == "axis_sg_int4_v2"
        )
        if has_mixer and "flux_mixer" in cfg:
            gen_params["mixer_freq"] = cfg["flux_mixer"]
        self.declare_gen(**gen_params)
        self._add_flux_pulse(cfg)

    def _add_flux_pulse(self, cfg):
        raise NotImplementedError

    def _body(self, cfg):
        axis = str(cfg.get("cryoscope_axis", "X")).upper()
        analysis_gate = {"X": "y90m_ge", "Y": "x90_ge"}.get(axis)
        if analysis_gate is None:
            raise ValueError("cryoscope_axis must be 'X' or 'Y'")

        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)

        # First Ramsey pi/2 pulse.
        self.pulse(ch=cfg["qb_ch"], name="x90_ge", t=0)
        self.delay_auto(cfg.get("cryoscope_pre_delay", 0.01))

        # Flux pulse. delay_auto waits for the complete descriptor, including
        # trailing zero padding, before scheduling the analysis pulse.
        self.pulse(ch=cfg["flux_ch"], name="flux_pulse", t=0)
        self.delay_auto(cfg.get("cryoscope_post_delay", 0.01))

        # X and Y use the same physical sequence and differ only in this gate.
        self.pulse(ch=cfg["qb_ch"], name=analysis_gate, t=0)
        self.delay_auto(0.05)
        self.measure(cfg)


class CryoscopeExperimentBase(BaseExperiment):
    """Base experiment with a convenient two-quadrature acquisition helper."""

    TAG = "Cryoscope"
    Y_LABEL = "ADC Units"
    IQ_PROCESS = "all"
    LivePlot = False

    def run_xy(self, py_avg: int, **run_kwargs):
        """Acquire raw resonator IQ for the X and Y Ramsey quadratures.

        The returned values are still resonator IQ.  Convert them to Bloch X/Y
        with ground/excited IQ references before calling ``trace_from_xy``.
        """

        run_kwargs.setdefault("iq_process", "all")
        run_kwargs.setdefault("liveplot", False)
        run_kwargs.setdefault("show_final_plot", False)

        results = {}
        for axis in ("X", "Y"):
            axis_cfg = dict(self.cfg)
            axis_cfg["cryoscope_axis"] = axis
            axis_experiment = type(self)(axis_cfg)
            results[axis] = axis_experiment.run(py_avg=py_avg, **run_kwargs)
        self.xy_results = results
        return results


__all__ = [
    "CryoscopeProgramBase",
    "CryoscopeExperimentBase",
    "generator_sample_period_ns",
    "minimum_envelope_samples",
    "pad_normalized_envelope",
    "quantize_envelope",
    "make_zero_padded_rectangle",
]
