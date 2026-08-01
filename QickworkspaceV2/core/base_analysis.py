"""
BaseAnalysis — abstract base class for all analysis classes.

Each experiment family has an Analysis subclass that:
  1. Accepts an ExperimentData with raw_iq populated.
  2. Runs fitting.
  3. Fills fit_params, fit_errors, fit_result, quality, figures.
  4. Returns the mutated ExperimentData.

This decouples analysis from acquisition: stored HDF5 data can be
re-analysed without re-running the hardware experiment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Type

import numpy as np

if TYPE_CHECKING:
    from .experiment_data import ExperimentData, QualityFlag


class BaseAnalysis(ABC):
    """
    Abstract analysis class.

    Subclass contract
    -----------------
    1. Set ``thresholds`` — dict mapping param-name → ``{"min": ..., "max": ...}``.
       Used by :meth:`_assess_quality` to auto-assign ``QualityFlag``.
    2. Override :meth:`_run` — perform fitting on ``data.raw_iq``, populate
       ``data.fit_params``, ``data.fit_errors``, ``data.fit_result``,
       ``data.scalar_result`` and append figures to ``data.figures``.

    Usage
    -----
    ::

        result = T1Analysis().run(exp_data)
        print(result.quality, result.fit_result["T1"])
    """

    thresholds: dict = {}
    REQUIRED_CONFIG_KEYS: tuple[str, ...] = ()

    def run(self, data: "ExperimentData") -> "ExperimentData":
        """
        Run analysis on *data* and return the annotated ExperimentData.

        Calls :meth:`_run`, then :meth:`_assess_quality`.
        """
        if data.raw_iq is None:
            from .experiment_data import QualityFlag

            data.quality = QualityFlag.BAD
            data.quality_message = "No raw IQ data present"
            return data

        self._run(data)
        if data.quality.value == "bad":
            return data
        data.quality = self._assess_quality(data)
        return data

    def plot(self, data: "ExperimentData") -> None:
        """
        Show a fit overlay figure after analysis.  Override in subclasses.
        """
        pass

    # ── Helper ───────────────────────────────────────────────────────────────

    @staticmethod
    def _config_value(data: "ExperimentData", key: str, default=None):
        """Read persisted analysis context with legacy config fallback."""
        context = data.metadata.get("analysis_context") or {}
        if key in context:
            return context[key]
        return data.config.get(key, default)

    def _show_fit(
        self,
        data: "ExperimentData",
        simfunc,
        fit_params,
        *,
        xlabel: str = "x",
        title: str = "",
        result_text: str = "",
        extra_lines=None,
    ) -> None:
        """Render the shared fit dashboard via plot_utils."""
        from ..plotter.plot_utils import plot_fit_result

        if data.x_axis is None or data.raw_iq is None:
            return
        quality = data.quality.value if data.quality is not None else "no_information"
        fit_channel = data.metadata.get("fit_channel", "abs")
        fit_snr = data.metadata.get("fit_channel_snr")
        if fit_channel:
            channel_text = f"channel = {fit_channel}"
            if fit_snr is not None:
                channel_text += f"  SNR={fit_snr:.2f}"
            result_text = f"{result_text}\n{channel_text}" if result_text else channel_text
        plot_fit_result(
            data.x_axis, data.raw_iq, simfunc, fit_params,
            x_label=xlabel,
            title=title,
            result_text=result_text,
            quality=quality,
            extra_lines=extra_lines,
            fit_channel=fit_channel,
        )

    @staticmethod
    def _channel_data(iq_data, channel: str):
        """Return a real-valued fitting trace from complex IQ data."""
        channel = (channel or "abs").lower()
        aliases = {
            "amp": "abs",
            "amplitude": "abs",
            "i": "real",
            "avgi": "real",
            "q": "imag",
            "avgq": "imag",
        }
        channel = aliases.get(channel, channel)
        iq = np.asarray(iq_data)
        if channel == "abs":
            return np.abs(iq)
        if channel == "real":
            return np.real(iq)
        if channel == "imag":
            return np.imag(iq)
        if channel == "phase":
            return np.unwrap(np.angle(iq))
        raise ValueError(f"Unknown fit_channel '{channel}'")

    def _fit_channel(self, data, fitfunc, simfunc, *, fitparams=None, channels=None):
        """
        Fit one or more IQ channels and return the best result.

        ``data.config['fit_channel']`` defaults to ``'auto'``.  In auto mode the
        score is the fitted curve span divided by residual standard deviation.
        """
        if data.x_axis is None or data.raw_iq is None:
            raise ValueError("Missing x_axis or raw_iq")

        requested = str(data.config.get("fit_channel", "auto")).lower()
        if requested == "auto":
            channels = channels or ("abs", "real", "imag", "phase")
        else:
            channels = (requested,)

        x = np.asarray(data.x_axis)
        best = None
        errors = []
        for channel in channels:
            try:
                y = np.asarray(self._channel_data(data.raw_iq, channel), dtype=float)
                local_fitparams = list(fitparams) if fitparams is not None else None
                popt, pcov, _ = fitfunc(x, y, fitparams=local_fitparams)
                fit_y = simfunc(x, *popt)
                residual = y - fit_y
                noise = float(np.nanstd(residual))
                span = float(np.nanmax(fit_y) - np.nanmin(fit_y))
                score = span / max(noise, 1e-12)
                if not np.all(np.isfinite(popt)) or not np.isfinite(score):
                    raise RuntimeError("non-finite fit result")
                candidate = (score, channel, y, np.asarray(popt), pcov)
                if best is None or candidate[0] > best[0]:
                    best = candidate
            except Exception as exc:
                errors.append(f"{channel}: {exc}")

        if best is None:
            raise RuntimeError("; ".join(errors) if errors else "no channels fit")

        score, channel, y, popt, pcov = best
        data.metadata["fit_channel"] = channel
        data.metadata["fit_channel_snr"] = float(score)
        fit_y = np.asarray(simfunc(np.asarray(data.x_axis), *popt), dtype=float)
        data.analysis_data.update({
            "fit_input": {"values": np.asarray(y, dtype=float), "dims": ["x"]},
            "fit_curve": {"values": fit_y, "dims": ["x"]},
            "residual": {
                "values": np.asarray(y, dtype=float) - fit_y,
                "dims": ["x"],
            },
        })
        return y, popt, pcov, channel, float(score)

    @abstractmethod
    def _run(self, data: "ExperimentData") -> None:
        """
        Perform fitting and populate ``data.fit_params``, ``data.fit_errors``,
        ``data.fit_result``, ``data.scalar_result``, and ``data.figures``.

        Mutates *data* in place; return value is ignored.
        """
        ...

    def _assess_quality(self, data: "ExperimentData") -> "QualityFlag":
        """
        Compare ``data.fit_result`` against ``self.thresholds``.

        Returns ``GOOD`` if all thresholds pass, ``BAD`` if any fail,
        ``WARNING`` if fit is valid but marginal, ``NO_INFORMATION`` if
        no thresholds are defined.
        """
        from .experiment_data import QualityFlag

        if not self.thresholds:
            return QualityFlag.NO_INFORMATION

        checked = False
        for param, bounds in self.thresholds.items():
            val = data.fit_result.get(param)
            if val is None:
                continue
            checked = True
            v = val[0] if isinstance(val, (tuple, list)) else val
            lo = bounds.get("min")
            hi = bounds.get("max")
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                data.quality_message = f"{param}={v:.4g} out of [{lo}, {hi}]"
                return QualityFlag.BAD

        if not checked:
            data.quality_message = "No threshold parameters found in fit_result"
            return QualityFlag.NO_INFORMATION

        return QualityFlag.GOOD


class IdentityAnalysis(BaseAnalysis):
    """No-op analysis — passes data through unchanged."""

    def _run(self, data: "ExperimentData") -> None:
        pass
