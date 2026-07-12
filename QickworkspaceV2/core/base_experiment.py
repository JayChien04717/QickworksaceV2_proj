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

from typing import Optional, Type

import numpy as np

from .base_analysis import BaseAnalysis
from .experiment_data import ExperimentData
from .experiment_components import (
    AcquisitionResult as _AcquisitionResult,
    AcquisitionRunner,
    ExperimentRuntime,
    ResultBuilder,
    RunContext as _RunContext,
    SweepAxes as _SweepAxes,
    SweepDefinition,
)


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
    _runtime = ExperimentRuntime()

    @classmethod
    def setup(cls, soc, soccfg, data_path: str):
        """Initialise shared QICK session (call once at notebook startup)."""
        data_path = cls._validate_data_path(data_path)
        cls._soc = soc
        cls._soccfg = soccfg
        cls._data_path = data_path
        cls._runtime.configure(soc, soccfg, data_path)

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
        cls._runtime.session_name = cls._session_name
        print(
            f"[BaseExperiment] Session activated: {cls._session_name}, "
            f"data_path={data_path!r}"
        )
        return soc, soccfg

    @classmethod
    def set_data_path(cls, data_path: str):
        cls._data_path = cls._validate_data_path(data_path)
        cls._runtime.data_path = cls._data_path

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
        self.soc, self.soccfg = BaseExperiment._runtime.require_hardware()

        self.cfg = config
        self.iqdata = None
        self.fit_params = None
        self.fit_errors = None
        self.result: Optional[ExperimentData] = None
        self._sweep_vals_x = None
        self._sweep_vals_y = None
        self._yoko_mode = None
        self._last_prog = None
        self._sweep_definition = SweepDefinition()
        self._acquisition_runner = AcquisitionRunner()
        self._result_builder = ResultBuilder()

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
        prog = self._build_program()
        axes = self._resolve_axes(prog, ctx)
        acq = self._acquire(prog, axes, ctx)
        result = self._finalize_result(acq, axes, ctx)
        return self._run_analysis(result, ctx)

    def plot(
        self,
        analyze: Optional[bool] = None,
        *,
        plot_analysis: Optional[bool] = None,
    ) -> ExperimentData:
        """
        Plot the latest result with this experiment's Analysis class.

        This never reacquires hardware data. It only uses ``self.result`` from
        the most recent :meth:`run`, so it is safe to call from a later cell::

            expt.plot(plot_analysis=True)

        Parameters
        ----------
        analyze : bool or None, optional
            When True, rerun analysis on ``self.result`` before plotting.
            When False, only render the current fit/analysis state.
            Retained for backward compatibility.
        plot_analysis : bool or None, optional
            True reruns analysis and draws its fit plot. False draws only the
            stored measurement data. Defaults to True.
        """
        if analyze is not None and plot_analysis is not None and analyze != plot_analysis:
            raise ValueError("analyze and plot_analysis specify conflicting values")
        should_analyze = (
            plot_analysis
            if plot_analysis is not None
            else (analyze if analyze is not None else True)
        )
        if self.result is None:
            raise RuntimeError("No result to plot. Call run() first.")
        if self.Analysis is None:
            raise RuntimeError(f"{self.__class__.__name__} has no Analysis class.")

        result = self.result
        if not should_analyze:
            self._plot_raw_result(result)
            return result

        analysis_inst = self.Analysis()
        result = analysis_inst.run(result)
        self.result = result
        analysis_inst.plot(result)
        return result

    def _plot_raw_result(self, result: ExperimentData) -> None:
        """Plot stored data without fitting or rendering a fit curve."""
        import matplotlib.pyplot as plt

        if result.raw_iq is None:
            raise RuntimeError("No raw data to plot. Call run() first.")

        raw = np.asarray(result.raw_iq).squeeze()
        process = str(result.metadata.get("iq_process", "abs")).lower()
        if process in {"real", "i", "avgi"}:
            values, ylabel = np.real(raw), "I (ADC unit)"
        elif process in {"imag", "q", "avgq"}:
            values, ylabel = np.imag(raw), "Q (ADC unit)"
        elif process == "phase":
            values, ylabel = np.unwrap(np.angle(raw)), "Phase (rad)"
        else:
            values, ylabel = np.abs(raw), "|IQ| (ADC unit)"

        x = result.x_axis
        if values.ndim == 1:
            x = np.arange(values.size) if x is None else x
            plt.plot(x, values, "o-", markersize=3)
        elif values.ndim == 2 and x is not None and values.shape[-1] == len(x):
            for index, trace in enumerate(values):
                plt.plot(x, trace, label=f"Trace {index}")
            plt.legend()
        else:
            plt.imshow(values, aspect="auto", origin="lower")
            plt.colorbar(label=ylabel)

        plt.xlabel(result.x_name or self.X_LABEL or "Sweep")
        plt.ylabel(ylabel)
        plt.title(self.TITLE_PREFIX or result.experiment_type)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

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

    def _build_program(self):
        prog = self._create_program()
        self._last_prog = prog
        return prog

    def _resolve_axes(self, prog, ctx: _RunContext) -> _SweepAxes:
        axes = self._sweep_definition.resolve(self, prog, ctx)
        self._sweep_vals_x = axes.x
        self._sweep_vals_y = axes.y
        return axes

    def _acquire(self, prog, axes: _SweepAxes, ctx: _RunContext) -> _AcquisitionResult:
        result = self._acquisition_runner.acquire(self, prog, axes, ctx)
        self.iqdata = result.raw_iq
        if result.raw_iq is None:
            print("No data was acquired.")
        elif result.interrupted:
            print(
                f"Experiment interrupted at {result.avg_count} averages. "
                "Fit is based on partial data."
            )
        return result

    def _finalize_result(
        self, acq: _AcquisitionResult, axes: _SweepAxes, ctx: _RunContext
    ) -> ExperimentData:
        result = self._result_builder.build(self, acq, axes, ctx)
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

        # self.cfg is the effective configuration used by the program.  In
        # notebooks this is normally ``run_cfg``: a per-qubit copy containing
        # experiment-specific overrides.  ``config_all`` is retained in the
        # signature for notebook compatibility, but must not replace the
        # effective run configuration in the saved file.
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
