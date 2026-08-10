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
from typing import TYPE_CHECKING, Optional

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
        """Run analysis on *data* and return the annotated ExperimentData.

                        Calls :meth:`_run`, then :meth:`_assess_quality`.

        Parameters
        ----------
        data : 'ExperimentData'
            Input data to process.

        Returns
        -------
        'ExperimentData'
            Result of the operation.
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
        """Show a fit overlay figure after analysis.  Override in subclasses.

        Parameters
        ----------
        data : 'ExperimentData'
            Input data to process.
        """
        pass

    def render(self, data: "ExperimentData") -> None:
        """Render analysis output, falling back to raw IQ when fitting failed.

        A fit that completed but failed a quality threshold still goes through
        :meth:`plot`, so users can inspect both the measured data and its poor
        fit. The raw-only fallback is reserved for a genuine fitting failure:
        BAD quality, no fit parameters, and no fit result.
        """
        from .experiment_data import QualityFlag

        fit_failed = (
            data.quality is QualityFlag.BAD
            and data.fit_params is None
            and not data.fit_result
        )
        if fit_failed:
            self._show_raw(data)
            return
        self.plot(data)

    def _show_raw(
        self,
        data: "ExperimentData",
        *,
        xlabel: Optional[str] = None,
    ) -> None:
        """Show a raw-IQ dashboard with the fitting failure reason."""
        from ..plotter.plot_utils import plot_fit_result

        trace = self._raw_plot_trace(data)
        if trace is None:
            return

        x, raw_iq, reduced = trace
        message = data.quality_message or "Fit failed; showing raw data."
        if reduced:
            message = f"{message}\n{reduced}"
        if xlabel is None:
            xlabel = data.x_name or "x"
            if data.x_unit:
                xlabel = f"{xlabel} ({data.x_unit})"
        figure = plot_fit_result(
            x,
            raw_iq,
            lambda values, *_: np.zeros_like(values, dtype=float),
            None,
            x_label=xlabel,
            title=f"{data.experiment_type or 'Analysis'} | Raw data (fit failed)",
            result_text=message,
            quality=data.quality.value,
            fit_channel=data.metadata.get("fit_channel", "abs"),
        )
        if all(id(saved) != id(figure) for saved in data.figures):
            data.figures.append(figure)
        return figure

    @staticmethod
    def _raw_plot_trace(data: "ExperimentData"):
        """Return ``(x, iq, note)`` suitable for a one-dimensional raw plot."""
        if data.x_axis is None or data.raw_iq is None:
            return None

        x = np.asarray(data.x_axis).reshape(-1)
        iq = np.asarray(data.raw_iq)
        if iq.ndim == 1 and iq.size == x.size:
            return x, iq, ""
        if iq.size == x.size:
            return x, iq.reshape(-1), ""
        if iq.ndim > 1 and iq.shape[-1] == x.size:
            rows = iq.reshape(-1, x.size)
            return x, np.nanmean(rows, axis=0), (
                f"Raw IQ averaged over {rows.shape[0]} traces for display."
            )
        return None


    @staticmethod
    def _config_value(data: "ExperimentData", key: str, default=None):
        """Read persisted analysis context with legacy config fallback.

        Parameters
        ----------
        data : 'ExperimentData'
            Input data to process.
        key : str
            Lookup key.
        default : Any, default: None
            Value for ``default``.

        Returns
        -------
        Any
            Result of the operation.
        """
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
        """Render the shared fit dashboard via plot_utils.

        Parameters
        ----------
        data : 'ExperimentData'
            Input data to process.
        simfunc : Any
            Value for ``simfunc``.
        fit_params : Any
            Value for ``fit_params``.
        xlabel : str, default: 'x'
            Value for ``xlabel``.
        title : str, default: ''
            Value for ``title``.
        result_text : str, default: ''
            Value for ``result_text``.
        extra_lines : Any, default: None
            Value for ``extra_lines``.
        """
        from ..plotter.plot_utils import plot_fit_result

        if data.x_axis is None or data.raw_iq is None:
            return
        quality = data.quality.value if data.quality is not None else "no_information"
        fit_channel = data.metadata.get("fit_channel", "abs")
        fit_snr = data.metadata.get("fit_channel_snr")
        channel_fit_curves = {}
        for channel, payload in (data.analysis_data.get("fit_curves") or {}).items():
            if isinstance(payload, dict) and "values" in payload:
                channel_fit_curves[channel] = payload["values"]
            else:
                channel_fit_curves[channel] = payload
        if fit_channel:
            channel_text = f"channel = {fit_channel}"
            if fit_snr is not None:
                channel_text += f"  SNR={fit_snr:.2f}"
            result_text = f"{result_text}\n{channel_text}" if result_text else channel_text
        figure = plot_fit_result(
            data.x_axis, data.raw_iq, simfunc, fit_params,
            x_label=xlabel,
            title=title,
            result_text=result_text,
            quality=quality,
            extra_lines=extra_lines,
            fit_channel=fit_channel,
            channel_fit_curves=channel_fit_curves,
        )
        if all(id(saved) != id(figure) for saved in data.figures):
            data.figures.append(figure)
        return figure

    @staticmethod
    def _channel_data(iq_data, channel: str):
        """Return a real-valued fitting trace from complex IQ data.

        Parameters
        ----------
        iq_data : Any
            Value for ``iq_data``.
        channel : str
            Value for ``channel``.

        Returns
        -------
        Any
            Result of the operation.

        Raises
        ------
        ValueError
            If the operation cannot be completed.
        """
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
        """Fit IQ channels, retain every successful curve, and return the primary fit.

        Auto mode selects the curve with the highest fitted-span/residual-noise
        score.  An explicitly requested channel remains primary, while the
        other channels are still fitted for analysis dashboards and the data
        library viewer.
        """
        if data.x_axis is None or data.raw_iq is None:
            raise ValueError("Missing x_axis or raw_iq")

        aliases = {"amplitude": "abs", "amp": "abs", "i": "real", "q": "imag"}
        if data.metadata.get("threshold_discrimination"):
            requested = "real"
            fit_channels = ("real",)
        else:
            requested_value = self._config_value(data, "fit_channel", "auto")
            requested = requested_value.lower() if isinstance(requested_value, str) else "auto"
            requested = aliases.get(requested or "auto", requested or "auto")
            fit_channels = tuple(channels or ("abs", "real", "imag", "phase"))
            fit_channels = tuple(dict.fromkeys(aliases.get(item, item) for item in fit_channels))
            if requested != "auto" and requested not in fit_channels:
                fit_channels = (requested, *fit_channels)

        x = np.asarray(data.x_axis)
        channel_results = {}
        errors = []
        for channel in fit_channels:
            try:
                y = np.asarray(self._channel_data(data.raw_iq, channel), dtype=float)
                local_fitparams = list(fitparams) if fitparams is not None else None
                popt, pcov, _ = fitfunc(x, y, fitparams=local_fitparams)
                popt = np.asarray(popt)
                fit_y = np.asarray(simfunc(x, *popt), dtype=float)
                residual = y - fit_y
                noise = float(np.nanstd(residual))
                span = float(np.nanmax(fit_y) - np.nanmin(fit_y))
                score = span / max(noise, 1e-12)
                if not np.all(np.isfinite(popt)) or not np.all(np.isfinite(fit_y)) or not np.isfinite(score):
                    raise RuntimeError("non-finite fit result")
                channel_results[channel] = (score, channel, y, popt, pcov, fit_y)
            except Exception as exc:
                errors.append(f"{channel}: {exc}")

        if not channel_results:
            raise RuntimeError("; ".join(errors) if errors else "no channels fit")

        if requested != "auto" and requested in channel_results:
            best = channel_results[requested]
        else:
            best = max(channel_results.values(), key=lambda candidate: candidate[0])
        score, channel, y, popt, pcov, fit_y = best
        data.metadata["fit_channel"] = channel
        data.metadata["fit_channel_snr"] = float(score)
        data.metadata["fit_channel_scores"] = {
            name: float(candidate[0]) for name, candidate in channel_results.items()
        }
        data.metadata["fit_channel_errors"] = errors
        data.analysis_data.update({
            "fit_input": {"values": np.asarray(y, dtype=float), "dims": ["x"]},
            "fit_curve": {"values": fit_y, "dims": ["x"]},
            "residual": {"values": np.asarray(y, dtype=float) - fit_y, "dims": ["x"]},
            "fit_inputs": {
                name: {"values": candidate[2], "dims": ["x"]}
                for name, candidate in channel_results.items()
            },
            "fit_curves": {
                name: {"values": candidate[5], "dims": ["x"]}
                for name, candidate in channel_results.items()
            },
        })
        return y, popt, pcov, channel, float(score)

    @abstractmethod
    def _run(self, data: "ExperimentData") -> None:
        """Perform fitting and populate ``data.fit_params``, ``data.fit_errors``,
                        ``data.fit_result``, ``data.scalar_result``, and ``data.figures``.

                        Mutates *data* in place; return value is ignored.

        Parameters
        ----------
        data : 'ExperimentData'
            Input data to process.
        """
        ...

    def _assess_quality(self, data: "ExperimentData") -> "QualityFlag":
        """Compare ``data.fit_result`` against ``self.thresholds``.

                        Returns ``GOOD`` if all thresholds pass, ``BAD`` if any fail,
                        ``WARNING`` if fit is valid but marginal, ``NO_INFORMATION`` if
                        no thresholds are defined.

        Parameters
        ----------
        data : 'ExperimentData'
            Input data to process.

        Returns
        -------
        'QualityFlag'
            Result of the operation.
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
        """Run the operation.

        Parameters
        ----------
        data : 'ExperimentData'
            Input data to process.
        """
        pass
