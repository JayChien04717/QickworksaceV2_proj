"""
BaseExperiment — IBM/IQM-style base class for all experiment wrappers.

Key differences from qick_workspace BaseExperiment
----------------------------------------------------
* ``run()`` returns :class:`~core.experiment_data.ExperimentData` instead of
  raw fit tuples.
* Uses a shared QICK session initialised by ``connect_pyro4()`` or
  ``setup(soc, soccfg, data_path)``.
* Each subclass may declare an ``Analysis`` class attribute (a
  :class:`~core.base_analysis.BaseAnalysis` subclass) that is run
  automatically after ``_post_fit``.
* ``saveLabber`` and ``save`` both work; ``save`` uses the new HDF5 layout.

Backward Compatibility
----------------------
Old notebooks using ``BaseExperiment.setup(soc, soccfg, data_path)`` and
then calling ``expt.run()`` still work.  The returned ``ExperimentData``
supports tuple unpacking::

    fit_params, error = expt.run(py_avg)   # unchanged
    freq = float(expt.run(py_avg))         # unchanged

New API::

    result = expt.run(py_avg)
    result.is_good()
    result.fit_result["T1"]
    result.save("path/file.h5")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Type

import numpy as np

from .base_analysis import BaseAnalysis
from .experiment_data import ExperimentData, QualityFlag


@dataclass
class _RunContext:
    py_avg: int
    iq_process: str
    show_final_plot: bool
    liveplot: bool
    plot_analysis: bool
    kwargs: dict[str, Any]
    config_snapshot: dict


@dataclass
class _SweepAxes:
    x: Optional[np.ndarray]
    y: Optional[np.ndarray]


@dataclass
class _AcquisitionResult:
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


class BaseExperiment:
    """
    Base class for all experiment wrappers.

    Session initialisation (call once per notebook)::

        soc, soccfg = BaseExperiment.connect_pyro4(
            ns_host="192.168.10.82",
            ns_port=8888,
            proxy_name="myqick",
            data_path=r"D:\\Labber_Data\\Jay\\test",
        )

    Or, if you already created the QICK proxy yourself::

        BaseExperiment.setup(soc, soccfg, data_path)

    Subclass contract
    -----------------
    1. Set class-level metadata (``EXPT_NAME``, ``TAG``, ``X_LABEL``, etc.).
    2. Optionally set ``Analysis`` to a :class:`BaseAnalysis` subclass.
    3. Override :meth:`_create_program` — return the QICK program instance.
    4. Override :meth:`_extract_sweep_axis` — return the x-axis array.
    5. Optionally override :meth:`_post_fit` — perform fitting; populate
       ``self.fit_params``, ``self.fit_errors``, and return the old-style
       value (tuple or scalar) for backward compat.
    """

    # ── Legacy session state ─────────────────────────────────────────────────
    _soc = None
    _soccfg = None
    _data_path = None
    _session_name = None

    @classmethod
    def setup(cls, soc, soccfg, data_path: str):
        """Initialise shared QICK session (call once at notebook startup)."""
        data_path = cls._validate_data_path(data_path)
        cls._soc = soc
        cls._soccfg = soccfg
        cls._data_path = data_path

    @classmethod
    def connect_pyro4(
        cls,
        ns_host: str,
        ns_port: int = 8888,
        proxy_name: str = "myqick",
        data_path: Optional[str] = None,
    ):
        """Connect to QICK through Pyro4 and activate it for all experiments."""
        data_path = cls._validate_data_path(data_path)
        try:
            import Pyro4
            from qick.pyro import make_proxy
        except ImportError as exc:
            raise ImportError(
                "Pyro4 and qick must be installed to connect to QICK hardware."
            ) from exc

        Pyro4.config.SERIALIZER = "pickle"
        Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

        soc, soccfg = make_proxy(
            ns_host=ns_host,
            ns_port=ns_port,
            proxy_name=proxy_name,
        )
        cls.setup(soc, soccfg, data_path)
        cls._session_name = f"QICK@{ns_host}:{ns_port}/{proxy_name}"
        print(
            f"[BaseExperiment] Session activated: {cls._session_name}, "
            f"data_path={data_path!r}"
        )
        return soc, soccfg

    @classmethod
    def set_data_path(cls, data_path: str):
        cls._data_path = cls._validate_data_path(data_path)

    @staticmethod
    def _validate_data_path(data_path: Optional[str]) -> str:
        if data_path is None or str(data_path).strip() == "":
            raise ValueError(
                "data_path is required. Call BaseExperiment.setup(soc, soccfg, data_path=...) "
                "or BaseExperiment.connect_pyro4(..., data_path=...) before running experiments."
            )
        return str(data_path)

    @classmethod
    def _require_data_path(cls) -> str:
        if cls._data_path is None or str(cls._data_path).strip() == "":
            raise RuntimeError(
                "BaseExperiment data path is not configured. Call "
                "BaseExperiment.setup(soc, soccfg, data_path=...) or "
                "BaseExperiment.connect_pyro4(..., data_path=...) first."
            )
        return str(cls._data_path)

    # ── Subclass metadata ────────────────────────────────────────────────────
    EXPT_NAME: str = ""
    TAG: str = ""
    X_LABEL: str = ""
    Y_LABEL: str = "ADC Units"
    TITLE_PREFIX: str = ""
    SWEEP_KEYS_TO_REMOVE: list = []

    IQ_PROCESS: str = "all"
    LivePlot: bool = True

    YOKO_VOLTAGE_RAMP_STEP: float = 1e-5
    YOKO_CURRENT_RAMP_STEP: float = 1e-8
    YOKO_RAMP_INTERVAL: float = 0.01

    X_SAVE_NAME: str = ""
    X_SAVE_UNIT: str = ""
    X_SAVE_SCALE: float = 1.0

    Y_SAVE_NAME: str = ""
    Y_SAVE_UNIT: str = ""
    Y_SAVE_SCALE: float = 1.0

    # ── NEW: link to analysis class ──────────────────────────────────────────
    Analysis: Optional[Type[BaseAnalysis]] = None

    # ────────────────────────────────────────────────────────────────────────
    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict or ExperimentConfig
            Experiment configuration.
        """
        if BaseExperiment._soc is None:
            raise RuntimeError(
                "QICK session not initialised. "
                "Call BaseExperiment.connect_pyro4(...) or "
                "BaseExperiment.setup(soc, soccfg, data_path)."
            )
        self.soc = BaseExperiment._soc
        self.soccfg = BaseExperiment._soccfg

        self.cfg = config
        self.iqdata = None
        self.fit_params = None
        self.fit_errors = None
        self.result: Optional[ExperimentData] = None
        self._sweep_vals_x = None
        self._sweep_vals_y = None
        self._yoko_mode = None
        self._last_prog = None

    def prog_asm(self, use_last: bool = False):
        """
        Build and print the QICK program for this experiment.

        Useful in notebooks before acquisition:

            prog = expt.prog_asm()

        Parameters
        ----------
        use_last : bool, default False
            When True, print the most recently built program if available.

        Returns
        -------
        object
            The program object that was printed.
        """
        if use_last and self._last_prog is not None:
            prog = self._last_prog
        else:
            prog = self._create_program()

        self._last_prog = prog
        print(prog)
        return prog

    # ══════════════════════════════════════════════════════════════════════════
    # Unified entry point
    # ══════════════════════════════════════════════════════════════════════════

    def run(
        self,
        py_avg: int,
        iq_process: Optional[str] = None,
        show_final_plot: bool = False,
        liveplot: Optional[bool] = None,
        plot_analysis: bool = False,
        **kwargs,
    ) -> ExperimentData:
        """
        Execute the experiment, run analysis, and return an ExperimentData.

        Analysis plots are intentionally opt-in. Use ``expt.plot()`` after
        running, or pass ``plot_analysis=True`` for the old one-call behavior.

        Returns
        -------
        ExperimentData
            Populated result object.  Supports backward-compat unpacking::

                fit_params, error = expt.run(py_avg)
                freq = float(expt.run(py_avg))
        """
        ctx = self._prepare_run_options(
            py_avg=py_avg,
            iq_process=iq_process,
            show_final_plot=show_final_plot,
            liveplot=liveplot,
            plot_analysis=plot_analysis,
            kwargs=kwargs,
        )
        prog = self._build_program(ctx)
        axes = self._resolve_axes(prog, ctx)
        acq = self._acquire(prog, axes, ctx)
        result = self._finalize_result(acq, axes, ctx)
        return self._run_analysis(result, ctx)

    def plot(self, analyze: bool = True) -> ExperimentData:
        """
        Plot the latest result with this experiment's Analysis class.

        Parameters
        ----------
        analyze : bool, default True
            When True, rerun analysis on ``self.result`` before plotting.
            When False, only render the current fit/analysis state.
        """
        if self.result is None:
            raise RuntimeError("No result to plot. Call run() first.")
        if self.Analysis is None:
            raise RuntimeError(f"{self.__class__.__name__} has no Analysis class.")

        analysis_inst = self.Analysis()
        result = self.result
        if analyze:
            result = analysis_inst.run(result)
            self.result = result
        analysis_inst.plot(result)
        return result

    def _prepare_run_options(
        self,
        *,
        py_avg: int,
        iq_process: Optional[str],
        show_final_plot: bool,
        liveplot: Optional[bool],
        plot_analysis: bool,
        kwargs: dict,
    ) -> _RunContext:
        resolved_liveplot = self.LivePlot if liveplot is None else liveplot
        resolved_iq_process = iq_process if iq_process is not None else self.IQ_PROCESS
        self._yoko_mode = kwargs.get("yoko_mode", None)
        self.iqdata = None
        self.fit_params = None
        self.fit_errors = None
        config_snapshot = self._snapshot_config()
        return _RunContext(
            py_avg=py_avg,
            iq_process=resolved_iq_process,
            show_final_plot=show_final_plot,
            liveplot=resolved_liveplot,
            plot_analysis=plot_analysis,
            kwargs=dict(kwargs),
            config_snapshot=config_snapshot,
        )

    def _build_program(self, ctx: _RunContext):
        prog = self._create_program()
        self._last_prog = prog
        return prog

    def _resolve_axes(self, prog, ctx: _RunContext) -> _SweepAxes:
        steps = self.cfg.get("steps") if hasattr(self.cfg, "get") else None
        x_vals = BaseExperiment._resolve_axis(self._extract_sweep_axis(prog), steps)

        yoko_value = ctx.kwargs.get("yoko_value")
        if yoko_value is not None:
            y_vals = np.asarray(yoko_value, dtype=float)
        else:
            y_vals = BaseExperiment._resolve_axis(
                self._extract_sweep_axis_y(prog), steps
            )

        self._sweep_vals_x = x_vals
        self._sweep_vals_y = y_vals
        return _SweepAxes(x=x_vals, y=y_vals)

    def _acquire(self, prog, axes: _SweepAxes, ctx: _RunContext) -> _AcquisitionResult:
        threshold = self._get_readout_threshold()
        if threshold is not None:
            return self._acquire_threshold(prog, axes, ctx, threshold)
        return self._acquire_liveplot(prog, axes, ctx)

    def _acquire_liveplot(
        self, prog, axes: _SweepAxes, ctx: _RunContext
    ) -> _AcquisitionResult:
        from ..plotter.liveplot import liveplotfun

        yoko_addr = ctx.kwargs.get("yoko_inst_addr")
        yoko_alias = ctx.kwargs.get("yoko_inst")
        instrument_manager = (
            ctx.kwargs.get("instrument_manager")
            or ctx.kwargs.get("baseinst")
            or ctx.kwargs.get("inst_manager")
        )
        yoko_name = (
            ctx.kwargs.get("yoko_name")
            or ctx.kwargs.get("yoko_inst_name")
            or yoko_alias
        )
        if yoko_addr is not None:
            raise ValueError(
                "Direct yoko_inst_addr support has been removed. Register the Yoko "
                "with BaseInstrumentManager and pass instrument_manager=inst plus "
                "yoko_name='q1_flux' (or yoko_inst='q1_flux')."
            )

        iqdata, interrupted, avg_count = liveplotfun(
            prog=prog,
            soc=self.soc,
            py_avg=ctx.py_avg,
            x_axis_vals=axes.x,
            y_axis_vals=axes.y,
            x_label=self.X_LABEL,
            y_label=self.Y_LABEL,
            title_prefix=self.TITLE_PREFIX,
            instrument_manager=instrument_manager,
            yoko_name=yoko_name,
            yoko_mode=ctx.kwargs.get("yoko_mode", "current"),
            yoko_voltage_ramp_step=self.YOKO_VOLTAGE_RAMP_STEP,
            yoko_current_ramp_step=self.YOKO_CURRENT_RAMP_STEP,
            yoko_ramp_interval=self.YOKO_RAMP_INTERVAL,
            show_final_plot=ctx.show_final_plot,
            iq_process=ctx.iq_process,
            liveplot=ctx.liveplot,
        )
        self.iqdata = iqdata

        if iqdata is None:
            print("No data was acquired.")
            return _AcquisitionResult(
                raw_iq=None,
                interrupted=True,
                avg_count=0,
                quality=QualityFlag.BAD,
                quality_message="No data acquired",
            )

        if interrupted:
            print(
                f"Experiment interrupted at {avg_count} averages. "
                "Fit is based on partial data."
            )

        return _AcquisitionResult(
            raw_iq=iqdata,
            interrupted=interrupted,
            avg_count=avg_count,
        )

    def _acquire_threshold(
        self, prog, axes: _SweepAxes, ctx: _RunContext, threshold
    ) -> _AcquisitionResult:
        try:
            acquired = prog.acquire(
                self.soc,
                rounds=ctx.py_avg,
                threshold=threshold,
                progress=True,
            )
        except TypeError:
            acquired = prog.acquire(
                self.soc,
                threshold=threshold,
                progress=True,
            )

        i_values = self._threshold_to_real_values(acquired)
        self.iqdata = i_values

        scalar = None
        if np.size(i_values) == 1:
            scalar = float(np.asarray(i_values).reshape(-1)[0])

        return _AcquisitionResult(
            raw_iq=i_values,
            interrupted=False,
            avg_count=ctx.py_avg,
            fit_params=np.array([scalar]) if scalar is not None else None,
            fit_result={"population": (self._to_serializable(i_values), None)},
            scalar_result=scalar,
            metadata={
                "threshold": threshold,
                "threshold_discrimination": True,
            },
        )

    def _finalize_result(
        self, acq: _AcquisitionResult, axes: _SweepAxes, ctx: _RunContext
    ) -> ExperimentData:
        metadata = {"iq_process": ctx.iq_process}
        metadata.update(acq.metadata)

        if acq.raw_iq is None:
            result = ExperimentData(
                experiment_type=self.EXPT_NAME,
                quality=acq.quality,
                quality_message=acq.quality_message,
                config=ctx.config_snapshot,
                metadata=metadata,
                interrupted=acq.interrupted,
                avg_count=acq.avg_count,
            )
            self.result = result
            return result

        old_result = self._post_fit(axes.x)
        fit_params = self.fit_params if self.fit_params is not None else acq.fit_params
        fit_errors = self.fit_errors if self.fit_errors is not None else acq.fit_errors

        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=acq.raw_iq,
            x_axis=axes.x,
            y_axis=axes.y,
            fit_params=fit_params,
            fit_errors=fit_errors,
            fit_result=dict(acq.fit_result),
            scalar_result=acq.scalar_result,
            quality=acq.quality,
            quality_message=acq.quality_message,
            config=ctx.config_snapshot,
            metadata=metadata,
            interrupted=acq.interrupted,
            avg_count=acq.avg_count,
            x_name=self.X_SAVE_NAME,
            x_unit=self.X_SAVE_UNIT,
            x_scale=self.X_SAVE_SCALE,
            y_name=self.Y_SAVE_NAME,
            y_unit=self.Y_SAVE_UNIT,
            y_scale=self.Y_SAVE_SCALE,
        )

        self._apply_old_result(result, old_result)
        if result.fit_result == {} and result.fit_params is not None:
            result.fit_result = self._build_fit_result()

        self.result = result
        return result

    def _run_analysis(self, result: ExperimentData, ctx: _RunContext) -> ExperimentData:
        if self.Analysis is not None:
            analysis_inst = self.Analysis()
            result = analysis_inst.run(result)
            if ctx.plot_analysis:
                analysis_inst.plot(result)
        self.result = result
        return result

    def _apply_old_result(self, result: ExperimentData, old_result) -> None:
        if old_result is None:
            return
        if isinstance(old_result, (int, float, np.integer, np.floating)):
            result.scalar_result = float(old_result)
        elif isinstance(old_result, (tuple, list)) and len(old_result) == 2:
            pass
        elif isinstance(old_result, dict):
            result.fit_result = {k: (v, None) for k, v in old_result.items()}

    def _snapshot_config(self) -> dict:
        try:
            snapshot = dict(self.cfg)
        except Exception:
            return {}
        try:
            from ..tools.system_tool import clean_config

            snapshot = clean_config(snapshot)
        except Exception:
            pass
        return self._to_serializable(snapshot)

    @staticmethod
    def _to_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): BaseExperiment._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [BaseExperiment._to_serializable(v) for v in obj]
        if isinstance(obj, set):
            return [BaseExperiment._to_serializable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return BaseExperiment._to_serializable(obj.tolist())
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        sweep_info = {}
        for attr in ("loop", "name", "start", "stop", "step", "expts", "steps"):
            if hasattr(obj, attr):
                try:
                    sweep_info[attr] = BaseExperiment._to_serializable(
                        getattr(obj, attr)
                    )
                except Exception:
                    pass
        if sweep_info:
            sweep_info["type"] = type(obj).__name__
            return sweep_info

        return repr(obj)

    # =========================================================================
    # Save
    # =========================================================================
    def saveLabber(self, qb_idx, yoko_value=None, config_all=None, title=None):
        """Legacy Labber-format HDF5 save (unchanged from original)."""
        from ..tools.system_tool import (
            config_to_yaml,
            get_next_filename_labber,
            hdf5_generator,
        )

        if title is not None:
            expt_name = f"{self.EXPT_NAME}_{qb_idx}_{title}"
        else:
            expt_name = f"{self.EXPT_NAME}_{qb_idx}"

        save_dir = BaseExperiment._require_data_path()
        file_path = get_next_filename_labber(save_dir, expt_name, yoko_value)

        if config_all is not None:
            dict_val = config_all.to_yaml(q_id=qb_idx)
        else:
            dict_val = config_to_yaml(self.cfg)

        comment = self._save_comment(dict_val)

        x_info = {
            "name": self.X_SAVE_NAME,
            "unit": self.X_SAVE_UNIT,
            "values": self._sweep_vals_x * self.X_SAVE_SCALE,
        }
        y_info = None
        if self._sweep_vals_y is not None:
            y_info = {
                "name": self.Y_SAVE_NAME,
                "unit": self.Y_SAVE_UNIT,
                "values": self._sweep_vals_y * self.Y_SAVE_SCALE,
            }

        hdf5_generator(
            filepath=file_path,
            x_info=x_info,
            y_info=y_info,
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
            comment=comment,
            tag=self.TAG,
        )
        print(f"Data saved to {file_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _resolve_axis(vals, steps=None):
        """
        Convert whatever _extract_sweep_axis returns into a plain float array.

        get_pulse_param / get_time_param may return a QickParam sweep object
        instead of a numpy array, depending on the QICK version.  This method
        tries every known extraction path and catches RuntimeError from
        QickParam.__float__.

        Always call as  BaseExperiment._resolve_axis(vals, steps)  (not via
        self._resolve_axis) to avoid Python descriptor ambiguity.
        """
        if vals is None:
            return None

        # Already a plain numpy numeric array — fast path
        if isinstance(vals, np.ndarray) and np.issubdtype(vals.dtype, np.number):
            return vals.astype(float)

        # QickParam: try every known array-extraction method
        for method in ("to_array", "sweep_vals", "get_array"):
            fn = getattr(vals, method, None)
            if callable(fn):
                try:
                    return np.asarray(fn(), dtype=float)
                except Exception:
                    pass

        # QickParam with is_sweep(): extract from start/stop/step/expts/steps
        if hasattr(vals, "is_sweep") and callable(vals.is_sweep) and vals.is_sweep():
            start = getattr(vals, "start", None)
            stop = getattr(vals, "stop", None)
            step = getattr(vals, "step", None)
            # 'expts' or 'steps' for count
            n_raw = getattr(vals, "expts", None) or getattr(vals, "steps", None)
            n = int(n_raw) if n_raw is not None else (int(steps) if steps else 100)
            if start is not None and stop is not None:
                try:
                    return np.linspace(float(start), float(stop), n)
                except (TypeError, ValueError, RuntimeError):
                    pass
            if start is not None and step is not None:
                try:
                    return float(start) + np.arange(n) * float(step)
                except (TypeError, ValueError, RuntimeError):
                    pass

        # QickSweep1D-like (start + stop present, no is_sweep)
        start = getattr(vals, "start", None)
        stop = getattr(vals, "stop", None)
        if start is not None and stop is not None:
            n = int(steps) if steps is not None else 100
            try:
                return np.linspace(float(start), float(stop), n)
            except (TypeError, ValueError, RuntimeError):
                pass

        # Plain Python scalar
        if isinstance(vals, (int, float)):
            return np.array([float(vals)])

        # Try direct numpy cast — catches RuntimeError from QickParam.__float__
        try:
            return np.asarray(vals, dtype=float)
        except (TypeError, ValueError, RuntimeError):
            pass

        # Object array: resolve element-wise
        try:
            obj_arr = np.asarray(vals)
            resolved = []
            for v in obj_arr.flat:
                for method in ("to_array", "sweep_vals"):
                    fn = getattr(v, method, None)
                    if callable(fn):
                        try:
                            resolved.extend(np.asarray(fn(), dtype=float).tolist())
                            break
                        except Exception:
                            pass
                else:
                    s = getattr(v, "start", None)
                    e = getattr(v, "stop", None)
                    if s is not None and e is not None and steps:
                        resolved.extend(
                            np.linspace(float(s), float(e), int(steps)).tolist()
                        )
                    elif s is not None:
                        try:
                            resolved.append(float(s))
                        except (TypeError, ValueError, RuntimeError):
                            pass
                    else:
                        try:
                            resolved.append(float(v))
                        except (TypeError, ValueError, RuntimeError):
                            pass
            if resolved:
                return np.array(resolved)
        except Exception:
            pass

        raise ValueError(f"Cannot resolve sweep axis from {type(vals).__name__}")

    def _get_readout_threshold(self):
        """Return configured readout threshold, or None when disabled."""
        if not hasattr(self.cfg, "get"):
            return None
        return self.cfg.get("threshold")

    @staticmethod
    def _threshold_to_real_values(acquired):
        """
        Convert QICK threshold-acquire output to real-valued population data.

        Non-threshold acquisition returns I/Q pairs, which older code combines
        into complex values with ``dot([1, 1j])``. Threshold acquisition is
        already discriminated, so downstream code should see real values only.
        """
        try:
            data = acquired[0][0]
        except (IndexError, TypeError):
            data = acquired

        arr = np.asarray(data)
        if np.iscomplexobj(arr):
            return np.real(arr).squeeze()

        if arr.ndim > 0 and arr.shape[-1] == 2:
            try:
                return np.real(arr.dot([1, 1j])).squeeze()
            except (TypeError, ValueError):
                pass

        return arr.astype(float, copy=False).squeeze()

    def _run_threshold_acquire(
        self, prog, threshold=None, py_avg: int = 1
    ) -> ExperimentData:
        """
        Acquire with QICK's threshold discriminator and skip live plotting.

        QICK returns already-discriminated I/population values when threshold
        is supplied, so the result stores the returned I channel directly.
        """
        self._sweep_vals_x = BaseExperiment._resolve_axis(
            self._extract_sweep_axis(prog), self.cfg.get("steps")
        )
        self._sweep_vals_y = BaseExperiment._resolve_axis(
            self._extract_sweep_axis_y(prog), self.cfg.get("steps")
        )

        try:
            acquired = prog.acquire(
                self.soc,
                rounds=py_avg,
                threshold=threshold,
                progress=True,
            )
        except TypeError:
            acquired = prog.acquire(
                self.soc,
                threshold=threshold,
                progress=True,
            )

        i_values = self._threshold_to_real_values(acquired)
        self.iqdata = i_values

        scalar = None
        if np.size(i_values) == 1:
            scalar = float(np.asarray(i_values).reshape(-1)[0])

        fit_result = {"population": (self._to_serializable(i_values), None)}
        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=i_values,
            x_axis=self._sweep_vals_x,
            y_axis=self._sweep_vals_y,
            fit_params=np.array([scalar]) if scalar is not None else None,
            fit_errors=None,
            fit_result=fit_result,
            scalar_result=scalar,
            quality=QualityFlag.NO_INFORMATION,
            quality_message="Threshold discrimination used; live plot skipped.",
            config=self._snapshot_config(),
            metadata={
                "threshold": threshold,
                "threshold_discrimination": True,
            },
            interrupted=False,
            avg_count=py_avg,
            x_name=self.X_SAVE_NAME,
            x_unit=self.X_SAVE_UNIT,
            x_scale=self.X_SAVE_SCALE,
            y_name=self.Y_SAVE_NAME,
            y_unit=self.Y_SAVE_UNIT,
            y_scale=self.Y_SAVE_SCALE,
        )
        self.result = result
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # Subclass MUST override
    # ══════════════════════════════════════════════════════════════════════════

    def _create_program(self):
        raise NotImplementedError("Subclass must implement _create_program()")

    def _extract_sweep_axis(self, prog) -> np.ndarray:
        raise NotImplementedError("Subclass must implement _extract_sweep_axis()")

    def _extract_sweep_axis_y(self, prog) -> Optional[np.ndarray]:
        return None

    # ══════════════════════════════════════════════════════════════════════════
    # Subclass MAY override
    # ══════════════════════════════════════════════════════════════════════════

    def _post_fit(self, x_vals):
        """Optional: fit and return old-style value. Should set self.fit_params."""
        return None

    def _save_comment(self, dict_val: str) -> str:
        return str(dict_val)

    def _build_fit_result(self) -> dict:
        """Build named fit_result dict from self.fit_params. Override for clarity."""
        return {}
