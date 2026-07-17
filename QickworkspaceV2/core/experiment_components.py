"""Focused runtime components used by :mod:`base_experiment`.

These classes keep the public ``BaseExperiment`` API stable while separating
session state, sweep resolution, acquisition, and result construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .acquisition import acquire_values
from .experiment_data import ExperimentData, QualityFlag


@dataclass
class RunContext:
    py_avg: int
    iq_process: str
    show_final_plot: bool
    liveplot: bool
    plot_analysis: bool
    kwargs: dict[str, Any]
    config_snapshot: dict


@dataclass
class SweepAxes:
    x: Optional[np.ndarray]
    y: Optional[np.ndarray]


@dataclass
class AcquisitionResult:
    raw_iq: Any = None
    interrupted: bool = False
    avg_count: int = 0
    quality: QualityFlag = QualityFlag.NO_INFORMATION
    quality_message: str = ""
    fit_params: Any = None
    fit_errors: Any = None
    fit_result: dict = field(default_factory=dict)
    scalar_result: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ExperimentRuntime:
    """QICK connection and output-path state shared by experiments."""

    soc: Any = None
    soccfg: Any = None
    data_path: Optional[str] = None
    session_name: Optional[str] = None

    def configure(self, soc, soccfg, data_path: str) -> None:
        self.soc = soc
        self.soccfg = soccfg
        self.data_path = data_path

    def require_hardware(self) -> tuple[Any, Any]:
        if self.soc is None:
            raise RuntimeError(
                "QICK session not initialised. Call BaseExperiment.connect_pyro4(...) "
                "or BaseExperiment.setup(soc, soccfg, data_path)."
            )
        return self.soc, self.soccfg

    def require_data_path(self) -> str:
        if self.data_path is None or str(self.data_path).strip() == "":
            raise RuntimeError("BaseExperiment data path is not configured.")
        return str(self.data_path)


class SweepDefinition:
    """Resolve an experiment program's x/y sweep axes."""

    def resolve(self, experiment, prog, ctx: RunContext) -> SweepAxes:
        steps = experiment.cfg.get("steps") if hasattr(experiment.cfg, "get") else None
        x_vals = experiment._resolve_axis(experiment._extract_sweep_axis(prog), steps)
        yoko_value = ctx.kwargs.get("yoko_value")
        if yoko_value is not None:
            y_vals = np.asarray(yoko_value, dtype=float)
        else:
            y_vals = experiment._resolve_axis(
                experiment._extract_sweep_axis_y(prog), steps
            )
        return SweepAxes(x=x_vals, y=y_vals)


class AcquisitionRunner:
    """Dispatch live or direct execution with a shared readout mode."""

    def acquire(self, experiment, prog, axes: SweepAxes, ctx: RunContext) -> AcquisitionResult:
        threshold = experiment._get_readout_threshold()
        if ctx.liveplot:
            return self._liveplot(experiment, prog, axes, ctx, threshold)
        return self._direct(experiment, prog, axes, ctx, threshold)

    def _liveplot(self, experiment, prog, axes, ctx, threshold) -> AcquisitionResult:
        from ..plotter.liveplot import liveplotfun

        instrument_manager, yoko_name = self._resolve_yoko(ctx)
        iqdata, interrupted, avg_count = liveplotfun(
            prog=prog, soc=experiment.soc, py_avg=ctx.py_avg,
            x_axis_vals=axes.x, y_axis_vals=axes.y,
            x_label=experiment.X_LABEL, y_label=experiment.Y_LABEL,
            title_prefix=experiment.TITLE_PREFIX,
            instrument_manager=instrument_manager, yoko_name=yoko_name,
            yoko_mode=ctx.kwargs.get("yoko_mode", "current"),
            yoko_voltage_ramp_step=experiment.YOKO_VOLTAGE_RAMP_STEP,
            yoko_current_ramp_step=experiment.YOKO_CURRENT_RAMP_STEP,
            yoko_ramp_interval=experiment.YOKO_RAMP_INTERVAL,
            show_final_plot=ctx.show_final_plot, iq_process=ctx.iq_process,
            threshold=threshold,
        )
        return self._result(
            experiment, iqdata, interrupted, avg_count, threshold
        )

    def _direct(self, experiment, prog, axes, ctx, threshold) -> AcquisitionResult:
        """Acquire without importing or entering the live-plot subsystem."""
        instrument_manager, yoko_name = self._resolve_yoko(ctx)
        if instrument_manager is not None and yoko_name is not None:
            if axes.y is None:
                raise ValueError("y_axis_vals must be provided for a Yoko sweep.")
            rows = []
            for value in axes.y:
                instrument_manager.set_value(
                    yoko_name, value, mode=ctx.kwargs.get("yoko_mode", "current")
                )
                rows.append(
                    acquire_values(
                        prog, experiment.soc, rounds=ctx.py_avg,
                        progress=False, threshold=threshold,
                    )
                )
            return self._result(
                experiment, np.asarray(rows), False, len(axes.y), threshold
            )

        values = acquire_values(
            prog, experiment.soc, rounds=ctx.py_avg,
            progress=True, threshold=threshold,
        )
        return self._result(
            experiment, values, False, ctx.py_avg, threshold
        )

    @staticmethod
    def _resolve_yoko(ctx: RunContext) -> tuple[Any, Any]:
        if ctx.kwargs.get("yoko_inst_addr") is not None:
            raise ValueError(
                "Direct yoko_inst_addr support has been removed. Register the Yoko "
                "with BaseInstrumentManager and pass instrument_manager=inst plus "
                "yoko_name='q1_flux' (or yoko_inst='q1_flux')."
            )
        instrument_manager = (
            ctx.kwargs.get("instrument_manager")
            or ctx.kwargs.get("baseinst")
            or ctx.kwargs.get("inst_manager")
        )
        yoko_name = (
            ctx.kwargs.get("yoko_name")
            or ctx.kwargs.get("yoko_inst_name")
            or ctx.kwargs.get("yoko_inst")
        )
        return instrument_manager, yoko_name

    @staticmethod
    def _result(
        experiment, iqdata, interrupted: bool, avg_count: int, threshold
    ) -> AcquisitionResult:
        if iqdata is None:
            return AcquisitionResult(
                interrupted=True, quality=QualityFlag.BAD,
                quality_message="No data acquired",
            )

        result = AcquisitionResult(
            raw_iq=iqdata, interrupted=interrupted, avg_count=avg_count
        )
        if threshold is not None:
            scalar = (
                float(np.asarray(iqdata).reshape(-1)[0])
                if np.size(iqdata) == 1 else None
            )
            result.fit_params = (
                np.array([scalar]) if scalar is not None else None
            )
            result.fit_result["population"] = (
                experiment._to_serializable(iqdata), None
            )
            result.scalar_result = scalar
            result.metadata = {
                "threshold": threshold,
                "threshold_discrimination": True,
            }
        return result


class ResultBuilder:
    """Construct ``ExperimentData`` from acquisition and fit state."""

    def build(self, experiment, acq: AcquisitionResult, axes: SweepAxes, ctx: RunContext) -> ExperimentData:
        metadata = {"iq_process": ctx.iq_process, **acq.metadata}
        common = dict(
            experiment_type=experiment.EXPT_NAME, quality=acq.quality,
            quality_message=acq.quality_message, config=ctx.config_snapshot,
            metadata=metadata, interrupted=acq.interrupted, avg_count=acq.avg_count,
        )
        if acq.raw_iq is None:
            return ExperimentData(**common)

        old_result = experiment._post_fit(axes.x)
        result = ExperimentData(
            **common, raw_iq=acq.raw_iq, x_axis=axes.x, y_axis=axes.y,
            fit_params=experiment.fit_params if experiment.fit_params is not None else acq.fit_params,
            fit_errors=experiment.fit_errors if experiment.fit_errors is not None else acq.fit_errors,
            fit_result=dict(acq.fit_result), scalar_result=acq.scalar_result,
            x_name=experiment.X_SAVE_NAME, x_unit=experiment.X_SAVE_UNIT,
            x_scale=experiment.X_SAVE_SCALE, y_name=experiment.Y_SAVE_NAME,
            y_unit=experiment.Y_SAVE_UNIT, y_scale=experiment.Y_SAVE_SCALE,
        )
        experiment._apply_old_result(result, old_result)
        if not result.fit_result and result.fit_params is not None:
            result.fit_result = experiment._build_fit_result()
        raw = np.asarray(acq.raw_iq)
        if axes.y is not None and raw.ndim >= 2:
            result.dataset_dims["iq"] = ["y", "x"]
        elif axes.x is not None and raw.ndim == 1:
            result.dataset_dims["iq"] = ["x"]
        return result
