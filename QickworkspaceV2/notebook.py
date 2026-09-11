"""Reusable presentation and explicit working-calibration operations for notebooks.

Experiment choices and configuration remain in the notebook. The SDK owns
execution, progress, analysis, final figures and explicit working updates.
"""

from pathlib import Path
from numbers import Real
from copy import deepcopy
import math

from QickworkspaceV2.data.store import atomic_json
from QickworkspaceV2.calibration.updates import accepted_fit


def enable_notebook():
    """Configure inline rendering and SDK source reloading in an IPython kernel."""
    from IPython import get_ipython

    shell = get_ipython()
    if shell is not None:
        if "IPython.extensions.autoreload" not in shell.extension_manager.loaded:
            shell.run_line_magic("load_ext", "autoreload")
        shell.run_line_magic("autoreload", "2")
        shell.run_line_magic("matplotlib", "inline")


class NotebookLab:
    """Daily Notebook interface: settings and run calls, with SDK-owned execution.

    Construction loads settings only. connect() opens QICK; run/scan retain
    saved data, analysis, final plots and interrupted partial results.
    Working calibration writes are explicitly enabled and requested by the user.
    """

    def __init__(
        self,
        project="lab/project.yaml",
        *,
        qubit="Q1",
        data_path="data",
        working_config="lab/working_config.yaml",
        calibration_enabled=False,
    ):
        from QickworkspaceV2 import ExperimentConfig
        from QickworkspaceV2.device.models import ProjectConfig, read_yaml

        self.project = Path(project).resolve()
        self.working_config = Path(working_config).resolve()
        self.config = (
            ExperimentConfig.load(self.working_config)
            if self.working_config.exists()
            else ExperimentConfig.from_project(self.project)
        )
        self.qubit = qubit
        self.config[qubit]  # Reject a nonexistent target before connection.
        self.data_path = Path(data_path).resolve()
        self.connection = ProjectConfig.model_validate(read_yaml(self.project)).connection.model_dump()
        self.calibration_enabled = calibration_enabled
        self._measurement = None
        self.results = {}
        self._rb_reference = None

    def connect(self, **connection):
        from QickworkspaceV2.runtime.measurement import Measurement

        self.connection.update(connection)
        self._measurement = Measurement.from_pyro4(**self.connection, data_path=self.data_path)
        print(self._measurement.soccfg)

    @property
    def measurement(self):
        if self._measurement is None:
            raise RuntimeError("Call lab.connect() before measuring")
        return self._measurement

    @property
    def last_result(self):
        return self.measurement.last_result

    @property
    def soc(self):
        return self.measurement.soc

    @property
    def soccfg(self):
        return self.measurement.soccfg

    @property
    def store(self):
        return self.measurement.store

    def _analyzer(self, program):
        spec = program.__dict__.get("EXPERIMENT")
        return spec.analyze if spec else None

    def run(self, program, run_cfg, *, py_avg=None, analyze="auto", show=True, **options):
        from QickworkspaceV2.backends import AcquisitionCancelled
        from QickworkspaceV2.plotting import LivePlot

        progress = options.setdefault("on_progress", LivePlot())
        try:
            result = self.measurement.run(program, run_cfg, py_avg=py_avg, analyze=analyze, **options)
        except (KeyboardInterrupt, AcquisitionCancelled):
            result = self.measurement.last_result
            if result is None:
                raise
            print("Interrupted; retained current run:", result.run_id, result.path)
            analyzer = self._analyzer(program) if analyze == "auto" else analyze
            if analyzer is not None:
                result = self.measurement.analyze(result, analyzer)
        finally:
            if isinstance(progress, LivePlot):
                progress.close()
        if show:
            self.show(result)
        if result.experiment == "randomized_benchmarking":
            if run_cfg.get("interleaved", "none") == "none":
                self._rb_reference = result
            elif self._rb_reference is not None:
                try:
                    from QickworkspaceV2.experiments.randomized_benchmarking.analysis import interleaved_error

                    print(
                        "Interleaved gate error estimate (Markovian reference):",
                        interleaved_error(self._rb_reference, result, self.qubit),
                    )
                except ValueError as exc:
                    print("Interleaved estimate unavailable:", exc)
        return result

    def scan(self, program, run_cfg, parameter, values, *, py_avg=None, analyze="auto", show=True, **options):
        from QickworkspaceV2.backends import AcquisitionCancelled
        from QickworkspaceV2.analysis.diagnostics import summarize_traces
        from QickworkspaceV2.plotting import LivePlot

        progress = options.setdefault("on_progress", LivePlot())
        try:
            result = self.measurement.scan(
                program, run_cfg, parameter, values, py_avg=py_avg, analyze=analyze, **options
            )
        except (KeyboardInterrupt, AcquisitionCancelled):
            result = self.measurement.last_result
            if result is None:
                raise
            print("Interrupted; retained current scan:", result.run_id, result.path)
        finally:
            if isinstance(progress, LivePlot):
                progress.close()
        analyzer = summarize_traces if "child_runs" in result.metadata else self._analyzer(program)
        if analyzer is not None:
            result = self.measurement.analyze(result, analyzer)
        if show:
            self.show(result)
            if "child_runs" in result.metadata:
                child_table(self.measurement, result, parameter)
        return result

    def show(self, result, *, name=None):
        from QickworkspaceV2.experiments import default_registry

        try:
            result._plotter = default_registry().get(result.experiment).plot
        except KeyError:
            pass
        name = name or result.experiment
        self.results[result.run_id] = result
        report(name, result)

    def analyze(self, result, analyzer=None):
        from QickworkspaceV2.experiments import default_registry

        if analyzer is None:
            current = result if hasattr(result, "experiment") else self.load(result)
            analyzer = default_registry().get(current.experiment).analyze
        result = self.measurement.analyze(result, analyzer)
        self.show(result)
        return result

    def load(self, run_id, **options):
        return self.measurement.load(run_id, **options)

    def apply_fit(self, result):
        """Apply the measured experiment's local update rules, only when enabled."""
        from QickworkspaceV2.experiments import default_registry

        if not self.calibration_enabled:
            print("Working calibration writes disabled; acquired data and analysis remain saved")
            return False
        try:
            accepted_fit(result, self.qubit)
            spec = default_registry().get(result.experiment)
            if spec.updates is None:
                raise ValueError("This experiment does not declare calibration updates")
            return self._apply_updates(result, spec.updates(result, self.qubit))
        except ValueError as exc:
            print("Working calibration unchanged:", exc)
            return False

    def _apply_updates(self, result, updates):
        if not updates.working:
            raise ValueError(updates.reason)
        apply_working(result, self.config, self.working_config, target=self.qubit, **updates.working)
        return True

    def update_working(self, **values):
        if not self.calibration_enabled:
            print("Working calibration writes disabled")
            return
        self.config[self.qubit].update(values)
        self.config.save(self.working_config)

    def review(self):
        from IPython.display import display

        display(self.store.list(limit=100))
        print("Working configuration:", self.working_config)

    def show_settings(self):
        from IPython.display import display

        display(dict(self.config[self.qubit]))

    def reload_partial(self, result):
        loaded = self.load(result.run_id, partial=result.path.name == "partial.h5")
        self.show(loaded, name="reloaded_partial")
        return loaded

    def readout_grid(self, program, run_cfg, gains, lengths, *, margin_us=0.1, py_avg=None):
        from IPython.display import display
        from QickworkspaceV2.experiments.single_shot.analysis import plot_grid

        if not len(gains) or not len(lengths):
            raise ValueError("Readout grid needs nonempty gains and lengths")
        rows = []
        stop = False
        for length in lengths:
            for gain in gains:
                cfg = run_cfg.for_run(
                    ro_length=float(length), res_length=float(length) + margin_us, res_gain_ge=float(gain)
                )
                result = self.run(program, cfg, py_avg=py_avg, show=False)
                fit = result.fits.get(self.qubit)
                rows.append(
                    {
                        "length_us": float(length),
                        "gain": float(gain),
                        "fidelity": float(fit.parameters.get("fidelity", float("nan")))
                        if fit
                        else float("nan"),
                        "passed": bool(fit and fit.success),
                        "run_id": result.run_id,
                        "complete": result.metadata.get("acquisition_status") == "completed",
                    }
                )
                if not rows[-1]["complete"]:
                    stop = True
                    break
            if stop:
                break
        figure = plot_grid(rows, gains, lengths)
        figure.savefig(result.path.parent / "readout_optimization.png")
        atomic_json(result.path.parent / "readout_optimization.json", rows)
        display(figure)
        display(rows)
        return rows

    def apply_best_readout(self, rows):
        from QickworkspaceV2.experiments.single_shot.analysis import best_readout

        if not self.calibration_enabled:
            print("Working calibration writes disabled")
            return False
        try:
            return self.apply_fit(self.load(best_readout(rows)))
        except ValueError as exc:
            print("Working calibration unchanged:", exc)
            return False

    def ramsey_pair(self, program, run_cfg, *, center, offset=0.2, py_avg=None):
        if not math.isfinite(offset) or offset <= 0:
            raise ValueError("Ramsey pair offset must be finite and positive")
        results = {"center": center, "offset": offset}
        for name, sign in (("minus", -1), ("plus", 1)):
            cfg = run_cfg.for_run(qb_freq_ge=center + sign * offset, detuning_mhz=0)
            result = self.run(program, cfg, py_avg=py_avg)
            results[name] = result
            if result.metadata.get("acquisition_status") != "completed":
                break
        return results

    def apply_ramsey_pair(self, results):
        from QickworkspaceV2.experiments.ramsey_ge.analysis import updates_pair

        if not self.calibration_enabled:
            print("Working calibration writes disabled")
            return False
        try:
            updates = updates_pair(results, self.qubit)
            return self._apply_updates(results["plus"], updates)
        except ValueError as exc:
            print("Working calibration unchanged:", exc)
            return False

    def drag_scan(self, program, run_cfg, alphas, *, py_avg=None):
        from IPython.display import display

        if not len(alphas):
            raise ValueError("DRAG scan needs nonempty alphas")
        transition = program.TRANSITION
        analysis = self._drag_analysis(program.EXPERIMENT.id)
        runs = []
        for alpha in alphas:
            cfg = run_cfg.for_run(alpha=float(alpha), **{f"drag_alpha_{transition}": float(alpha)})
            result = self.run(program, cfg, py_avg=py_avg)
            runs.append({"alpha": float(alpha), "result": result})
            if result.metadata.get("acquisition_status") != "completed":
                break
        figure = analysis.plot_scan(runs, self.qubit)
        figure.savefig(runs[-1]["result"].path.parent / "drag_scan.png")
        display(figure)
        return runs

    @staticmethod
    def _drag_analysis(experiment):
        from QickworkspaceV2.experiments.drag_ge import analysis as ge
        from QickworkspaceV2.experiments.drag_ef import analysis as ef

        if experiment not in ("drag_ge", "drag_ef"):
            raise ValueError("Select a native GE or EF DRAG experiment")
        return {"drag_ge": ge, "drag_ef": ef}[experiment]

    def apply_best_drag(self, runs):
        if not self.calibration_enabled:
            print("Working calibration writes disabled")
            return False
        try:
            if not runs:
                raise ValueError("No DRAG runs to apply")
            analysis = self._drag_analysis(runs[0]["result"].experiment)
            return self.apply_fit(analysis.best_result(runs, self.qubit))
        except ValueError as exc:
            print("Working calibration unchanged:", exc)
            return False

    def reset_comparison(self, program, run_cfg, modes, *, py_avg=None):
        from IPython.display import display
        from QickworkspaceV2.experiments.active_reset_rabi.analysis import plot_comparison

        if not len(modes):
            raise ValueError("Reset comparison needs at least one mode")
        runs = {}
        for mode in modes:
            result = self.run(program, run_cfg.for_run(reset_mode=mode), py_avg=py_avg)
            runs[mode] = result
            if result.metadata.get("acquisition_status") != "completed":
                break
        figure = plot_comparison(runs, self.qubit)
        figure.savefig(result.path.parent / "reset_comparison.png")
        display(figure)
        return runs

    def flux_scan(
        self, program, run_cfg, values, *, enabled=False, address=None, limits=(-0.003, 0.003), py_avg=None
    ):
        if not enabled:
            print("External flux disabled; no external instrument connection")
            return None
        from QickworkspaceV2.instruments import BaseInstrumentManager

        instruments = BaseInstrumentManager()
        instruments.add_yoko(
            "flux", address, limits={"current": limits}, current_ramp_step=1e-8, ramp_interval=0.01
        )
        readback = instruments.value("flux")
        if readback["parameter"] != "current":
            raise ValueError("Configure the Yoko in current mode before scanning")
        results = []
        try:
            for value in values:
                instruments.set_value("flux", float(value), mode="current")
                result = self.run(program, run_cfg, py_avg=py_avg)
                atomic_json(result.path.parent / "flux_readback.json", instruments.value("flux"))
                results.append(result)
                if result.metadata.get("acquisition_status") != "completed":
                    break
        finally:
            instruments.set_value("flux", float(readback["value"]), mode="current")
        return results


def report(name, result, *, event=0, signal=None):
    """Display saved data, fit diagnostics and a final plot; save its PNG beside the run."""
    from IPython.display import display

    if not name or Path(name).name != name or any(c in name for c in "/\\:"):
        raise ValueError("Report name must be a filename without directory separators")
    print("Run:", result.run_id, "Data:", result.path)
    print("Acquisition:", result.metadata.get("acquisition_status", "unknown"))
    print("Analysis:", result.analysis_status, result.analysis_message)
    for target, fit in result.fits.items():
        print(target, fit.model, "PASS" if fit.success else "REVIEW", fit.parameters, fit.errors, fit.message)
    figure = result.plot(event=event, signal=signal)
    if result.path is not None:
        figure.savefig(result.path.parent / (name + ".png"), dpi=150)
    display(figure)


def apply_working(result, config, path, *, target, **updates):
    """Explicitly apply an accepted single-qubit result to a separate working file.

    The selected target must match the acquisition. File serialization happens
    before the live working view changes; a failed save leaves that view intact.
    """
    accepted_fit(result, target)
    if result.targets != (target,):
        raise ValueError("Select the measured qubit before applying working updates")
    if not all(math.isfinite(value) for value in updates.values() if isinstance(value, Real)):
        raise ValueError("Nonfinite working update")
    candidate = deepcopy(config)
    candidate[target].update(updates)
    candidate.save(path)
    config[target].update(updates)
    atomic_json(
        result.path.parent / "working_update.json",
        {"target": target, "source_run": result.run_id, "updates": updates},
    )
    print("Saved working calibration:", updates)


def child_table(lab, parent, key, *, target=None):
    """Display fit results for the actual saved points of a host scan."""
    from IPython.display import display

    if target is None:
        if len(parent.targets) != 1:
            raise ValueError("Select a target for a multi-qubit scan")
        target = parent.targets[0]
    rows = []
    for child_id in parent.metadata["child_runs"]:
        child = lab.load(child_id)
        fit = child.fits.get(target)
        rows.append(
            {
                "run_id": child_id,
                "value": child.metadata["parameters"].get(key),
                "passed": bool(fit and fit.success),
                "metrics": child.metrics,
            }
        )
    display(rows)
    return rows
