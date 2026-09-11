"""Direct notebook acquisition: native Program + editable run_cfg, without a recipe schema."""

from copy import deepcopy
from uuid import uuid4
import math
import numpy as np

from QickworkspaceV2.backends import QICKBackend, AcquisitionCancelled
from QickworkspaceV2.backends.qick import extract_coords
from QickworkspaceV2.device.editable import prepare_program_config
from QickworkspaceV2.experiments.base import ProgramPlan
from QickworkspaceV2.programs.sweeps import Sweep
from QickworkspaceV2.data.models import ExperimentData, TraceData
from QickworkspaceV2.data.store import RunStore, atomic_json
from QickworkspaceV2.data.serialization import jsonable
from .session import _hardware_lease, _code_record, _versions, _range


def infer_axes(program):
    """Bind each acquisition loop to a rounded pulse parameter or tagged delay."""
    cfg = program.cfg
    first = cfg["targets"][0]
    candidates = []
    for tag in sorted(program.time_dict, key=lambda name: (name != "evolution", name)):
        for parameter in program.list_time_params(tag):
            value = program.get_time_param(tag, parameter)
            if getattr(value, "spans", {}):
                candidates.append((value, "delay", "us", parameter, None, tag))
    pulses = sorted(
        program.pulses, key=lambda name: (not name.endswith(("qb_pulse", "res_pulse", "interaction")), name)
    )
    for pulse in pulses:
        # Select the first target's pulse; backend reads the corresponding pulse for every target.
        if "__" in pulse and not pulse.startswith(first + "__") and not pulse.endswith("__interaction"):
            continue
        available = program.list_pulse_params(pulse)
        for parameter in ("freq", "gain", "length", "phase"):
            if parameter not in available:
                continue
            value = program.get_pulse_param(pulse, parameter)
            if getattr(value, "spans", {}):
                name, unit = {
                    "freq": ("frequency", "MHz"),
                    "gain": ("gain", "normalized gain"),
                    "length": ("length", "us"),
                    "phase": ("phase", "deg"),
                }[parameter]
                binding = (
                    pulse.replace(first + "__", "{target}__", 1) if pulse.startswith(first + "__") else pulse
                )
                candidates.append((value, name, unit, parameter, binding, None))
    axes = []
    for loop, count, *_ in program.loops:
        if loop == "reps":
            continue
        match = next((c for c in candidates if set(c[0].spans) == {loop}), None)
        if match is None:
            raise ValueError(f"Cannot infer coordinate for loop {loop!r}; pass axes=[Sweep(...)] explicitly")
        value, name, unit, parameter, pulse, tag = match
        if any(axis.name == name for axis in axes):
            name = loop
        axes.append(
            Sweep(
                name,
                float(value.start),
                float(value.start + value.spans[loop]),
                count,
                unit,
                loop,
                parameter,
                pulse,
                tag,
            )
        )
    return tuple(axes)


def readout_frequency(program):
    """Read the compiled readout frequency for a direct or MUX host scan."""
    q = program.cfg["targets"][0]
    qc = program.cfg["qubits"][q]
    if qc["readout_mode"] == "mux":
        group = program.cfg["readout_groups"][qc["readout_group"]]
        slot = group["members"][q]["tone_slot"]
        return float(program.gen_chs[qc["res_ch"]]["mux_tones"][slot]["freq_rounded"])
    return float(program.get_pulse_param(q + "__res_pulse", "freq", as_array=True))


def validate_native_config(cfg):
    """Validate edited physical channels and gains without restricting experiment-specific keys."""
    adc = set()
    drives = {}
    for q, qc in cfg["qubits"].items():
        for key in ("ro_ch", "res_ch", "qb_ch", *(("qb_ch_ef",) if cfg["transition"] == "ef" else ())):
            if not isinstance(qc[key], int) or isinstance(qc[key], bool) or qc[key] < 0:
                raise ValueError(f"{q}/{key}: channel must be a nonnegative integer")
        if qc["ro_ch"] in adc:
            raise ValueError("Two targets cannot use the same digital ADC endpoint")
        adc.add(qc["ro_ch"])
        prefixes = ("ge", "ef") if cfg["transition"] == "ef" else ("ge",)
        for prefix in prefixes:
            ch = qc["qb_ch" if prefix == "ge" else "qb_ch_ef"]
            if ch in drives and drives[ch] != q:
                raise ValueError(f"{q} and {drives[ch]} share drive channel {ch}")
            drives[ch] = q
            for key in (f"qb_gain_{prefix}", f"pi_gain_{prefix}", f"pi2_gain_{prefix}"):
                if qc.get(key) is not None and (
                    not all(map(math.isfinite, _range(qc[key])))
                    or max(map(abs, _range(qc[key]))) > qc.get(f"max_gain_{prefix}", 1)
                ):
                    raise ValueError(f"{q}/{key}: gain must be finite and within configured limit")
        if not all(map(math.isfinite, _range(qc["res_gain_ge"]))) or max(
            map(abs, _range(qc["res_gain_ge"]))
        ) > qc.get("max_res_gain", 1):
            raise ValueError(f"{q}: readout gain exceeds configured limit")
    for group in cfg["readout_groups"].values():
        if group["port"]["channel"] in drives:
            raise ValueError("An active readout generator overlaps a qubit drive")
    if not isinstance(cfg["reps"], int) or isinstance(cfg["reps"], bool) or cfg["reps"] < 1:
        raise ValueError("reps must be a positive integer")


class Measurement:
    """Own one QICK connection. run(Program, run_cfg, py_avg=5) measures and saves.

    compile() uses the real QICK compiler without acquisition. No alternate backend
    or generated measurement data is supplied by this interface.
    """

    def __init__(self, soc, soccfg, *, data_path="data", resource_id="qick"):
        self.backend = QICKBackend(soc, soccfg, resource_id=resource_id)
        self.store = RunStore(data_path)
        self.last_program = None
        self.last_result = None
        self.last_run_id = None

    @classmethod
    def from_pyro4(cls, ns_host, ns_port=8888, proxy_name="myqick", *, data_path="data"):
        backend = QICKBackend.from_pyro4(ns_host, ns_port, proxy_name)
        return cls(backend.soc, backend.soccfg, data_path=data_path, resource_id=backend.resource_id)

    @property
    def soc(self):
        return self.backend.soc

    @property
    def soccfg(self):
        return self.backend.soccfg

    def compile(self, program, run_cfg):
        transition = (
            program.transition_for(run_cfg)
            if hasattr(program, "transition_for")
            else getattr(program, "TRANSITION", None)
        )
        cfg = prepare_program_config(run_cfg, transition)
        validate_native_config(cfg)
        processors = self.soccfg["tprocs"]
        if len(processors) != 1 or processors[0].get("type") != "qick_processor":
            raise ValueError("Measurement requires a QICK tProc v2 board")
        plan = ProgramPlan(program, cfg)
        self.last_program = self.backend.compile(plan)
        return self.last_program

    def run(
        self,
        program,
        run_cfg,
        py_avg=None,
        *,
        axes=None,
        analyze="auto",
        iq_process=None,
        capture_shots=None,
        readout_events=None,
        on_progress=None,
        cancel=None,
    ):
        if self.soc is None:
            raise RuntimeError("No QICK connection. Connect to the board before calling run().")
        averages = run_cfg.get("soft_avgs", 1) if py_avg is None else py_avg
        if not isinstance(averages, int) or isinstance(averages, bool) or averages < 1:
            raise ValueError("py_avg must be a positive integer")
        spec = program.__dict__.get("EXPERIMENT")
        identifier = spec.id if spec else program.__name__
        analyzer = (spec.analyze if spec else None) if analyze == "auto" else analyze
        run_id = uuid4().hex
        self.last_run_id, self.last_result = run_id, None
        metadata = {
            "targets": list(run_cfg.get("targets", [run_cfg.get("name", "Q1")])),
            "parameters": jsonable(run_cfg),
            "iq_process": iq_process or run_cfg.get("iq_process", "abs"),
            "backend": self.backend.capabilities(),
            "program_code": _code_record(program),
            "software_versions": _versions(),
            "py_avg": averages,
            "entry_point": "Measurement.run",
        }
        self.store.begin(run_id, identifier, metadata)
        directory = self.store.directory(run_id)
        try:
            with self.backend.lock, _hardware_lease(self.backend.resource_id):
                compiled = self.compile(program, run_cfg)
                cfg = compiled.cfg
                default_events = tuple(getattr(compiled, "READOUT_EVENTS", ()))
                events = tuple(default_events if readout_events is None else readout_events)
                counts = {ro["trigs"] for ro in compiled.ro_chs.values()}
                if len(counts) != 1 or not counts or next(iter(counts)) < 1:
                    raise ValueError("Each selected ADC must have the same positive readout count")
                count = next(iter(counts))
                if not events:
                    events = tuple(f"readout_{i}" for i in range(count))
                if len(events) != count:
                    raise ValueError("readout_events differ from compiled trigger count")
                bindings = infer_axes(compiled) if axes is None else tuple(axes)
                mode = getattr(program, "ACQUISITION_MODE", None)
                shot_mode = (
                    getattr(program, "CAPTURE_SHOTS", False) if capture_shots is None else capture_shots
                )
                if metadata["iq_process"] == "population":
                    if mode == "decimated":
                        raise ValueError("Decimated TOF traces do not contain single-shot populations")
                    if capture_shots is False:
                        raise ValueError("Population analysis requires capture_shots=True")
                    if any(qc.get("ro_threshold") is None for qc in cfg["qubits"].values()):
                        raise ValueError(
                            "Set each selected qubit's ro_threshold before population acquisition"
                        )
                    shot_mode = True
                plan = ProgramPlan(
                    program, cfg, bindings, events, shot_mode, metadata={"acquisition_mode": mode}
                )
                for target in cfg["targets"]:
                    extract_coords(compiled, plan, target)
                if set(compiled.ro_chs) != {qc["ro_ch"] for qc in cfg["qubits"].values()}:
                    raise ValueError("Declared ADCs must match the selected targets")
                metadata.update(
                    resolved_config=jsonable(cfg), targets=list(cfg["targets"]), transition=cfg["transition"]
                )
                atomic_json(
                    directory / "compiled.json",
                    {
                        "asm": str(compiled),
                        "readouts": compiled.ro_chs,
                        "generators": compiled.gen_chs,
                        "config": cfg,
                    },
                )
                scale = getattr(program, "AXIS_SCALE", 1)

                def progress(done, total, traces):
                    preview = deepcopy(traces)
                    if scale != 1:
                        for trace in preview.values():
                            trace.coords[bindings[0].name] *= scale
                    partial = ExperimentData(
                        identifier,
                        preview,
                        metadata={
                            **metadata,
                            "acquisition_status": "partial",
                            "completed_averages": done,
                            "requested_averages": total,
                        },
                        run_id=run_id,
                    )
                    partial.save(directory / "partial.h5")
                    self.last_result = partial
                    if on_progress:
                        on_progress(
                            {"state": "acquiring", "completed": done, "total": total, "result": partial}
                        )

                traces = self.backend.acquire(
                    compiled, plan, soft_avgs=averages, on_progress=progress, cancel=cancel
                )
                if scale != 1:
                    for trace in traces.values():
                        trace.coords[bindings[0].name] *= scale
                metadata.update(
                    acquisition_status="completed", completed_averages=averages, requested_averages=averages
                )
                result = ExperimentData(identifier, traces, metadata=metadata, run_id=run_id)
                self.store.save_acquisition(result)
                self.last_result = result
                if analyzer is not None:
                    try:
                        result.fits = analyzer(result)
                        result.analysis_status = "completed"
                        result.analysis_message = "; ".join(
                            f"{q}: {fit.message}" for q, fit in result.fits.items() if not fit.success
                        )
                    except Exception as exc:
                        result.analysis_status, result.analysis_message = "failed", str(exc)
                result.metadata["analysis_code"] = _code_record(analyzer)
                self.store.save_analysis(result)
                if on_progress:
                    on_progress({"state": "completed", "result": result})
                return result
        except BaseException as exc:
            if isinstance(exc, (AcquisitionCancelled, KeyboardInterrupt)) and self.last_result is not None:
                if self.last_result.metadata.get("acquisition_status") == "partial":
                    self.last_result.metadata["interrupted"] = True
                    self.last_result.save(directory / "partial.h5")
            self.store.status(
                run_id,
                "cancelled" if isinstance(exc, (AcquisitionCancelled, KeyboardInterrupt)) else "failed",
                str(exc),
            )
            raise
        finally:
            from QickworkspaceV2.plotting import LivePlot

            if isinstance(on_progress, LivePlot):
                on_progress.close()

    def load(self, run_id, **kwargs):
        return self.store.load(run_id, **kwargs)

    def analyze(self, result, analyzer):
        """Analyze saved acquisition without reacquiring; persist a separate revision."""
        if not callable(analyzer):
            raise TypeError("analyzer must be a callable taking ExperimentData")
        # Analyze a fresh raw copy so repeated analysis cannot accumulate mutations.
        run_id = result.run_id if isinstance(result, ExperimentData) else result
        partial = not (self.store.directory(run_id) / "acquisition.h5").exists()
        analyzed = self.load(run_id, revision=0, partial=partial)
        try:
            analyzed.fits = analyzer(analyzed)
            analyzed.analysis_status = "completed"
            analyzed.analysis_message = "; ".join(
                f"{q}: {fit.message}" for q, fit in analyzed.fits.items() if not fit.success
            )
        except Exception as exc:
            analyzed.analysis_status, analyzed.analysis_message = "failed", str(exc)
        analyzed.metadata["analysis_code"] = _code_record(analyzer)
        self.store.save_analysis(analyzed, update_status=not partial)
        if run_id == self.last_run_id:
            self.last_result = analyzed
        return analyzed

    def scan(
        self,
        program,
        run_cfg,
        parameter,
        values,
        *,
        axis_name=None,
        unit="",
        coordinate=None,
        py_avg=None,
        analyze="auto",
        on_progress=None,
        **run_options,
    ):
        """One host scan over a single target's editable key, saving every child run.

        Readout frequency coordinates come from the compiled direct/MUX tone.
        coordinate(program) is optional for custom host parameters. All other
        sweep axes must agree across points before a rectangular parent is saved.
        """
        values = np.asarray(values)
        if values.ndim != 1 or not len(values) or not np.isfinite(values.astype(float)).all():
            raise ValueError("Host scan values must be a nonempty finite vector")
        if "qubits" in run_cfg:
            raise ValueError("Edit one target per direct host scan")
        if coordinate is None and parameter == "res_freq_ge":
            coordinate = readout_frequency
        axis_name = axis_name or parameter
        parent_id = uuid4().hex
        self.last_result, self.last_run_id = None, None
        children, actual = [], []
        parent = None

        def assemble(status):
            traces = {}
            for q, first in children[0].traces.items():
                if axis_name in first.dims:
                    raise ValueError("Host axis conflicts with an inner dimension; child runs remain saved")
                for child in children[1:]:
                    trace = child[q]
                    if trace.dims != first.dims or any(
                        not np.array_equal(trace.coords[d], first.coords[d]) for d in first.dims
                    ):
                        raise ValueError("Inner coordinates differ; use the saved child runs")
                shots = None if first.shots is None else np.stack([child[q].shots for child in children])
                traces[q] = TraceData(
                    np.stack([child[q].iq for child in children]),
                    (axis_name, *first.dims),
                    {axis_name: np.asarray(actual), **first.coords},
                    {axis_name: unit, **first.units},
                    shots=shots,
                    shot_dims=(axis_name, *first.shot_dims) if shots is not None else (),
                    metadata=deepcopy(first.metadata),
                )
            metadata = {
                **children[0].metadata,
                "host_scan_parameter": parameter,
                "child_runs": [r.run_id for r in children],
                "requested_values": values.tolist(),
                "completed_points": len(children),
                "requested_points": len(values),
                "acquisition_status": status,
                "host_coordinate_source": "compiled" if coordinate else "requested",
            }
            return ExperimentData(children[0].experiment, traces, metadata=metadata, run_id=parent_id)

        try:
            for index, value in enumerate(values):
                cancel = run_options.get("cancel")
                if cancel is not None and cancel.is_set():
                    raise AcquisitionCancelled("Host scan cancelled between points")
                cfg = deepcopy(run_cfg)
                cfg[parameter] = value.item()
                child = self.run(program, cfg, py_avg=py_avg, analyze=analyze, **run_options)
                children.append(child)
                actual.append(float(coordinate(self.last_program) if coordinate else value))
                parent = assemble("partial")
                if index == 0:
                    self.store.begin(parent_id, parent.experiment, parent.metadata)
                parent.save(self.store.directory(parent_id) / "partial.h5")
                self.last_run_id, self.last_result = parent_id, parent
                if on_progress:
                    on_progress(
                        {"state": "acquiring", "completed": index + 1, "total": len(values), "result": parent}
                    )
            parent = assemble("completed")
            self.store.save_acquisition(parent)
            self.store.save_analysis(parent)
            self.last_run_id, self.last_result = parent_id, parent
            if on_progress:
                on_progress({"state": "completed", "result": parent})
            return parent
        except BaseException as exc:
            if parent is not None:
                if self.last_run_id != parent_id:
                    parent.metadata["interrupted_child_run"] = self.last_run_id
                parent.metadata["interrupted"] = isinstance(exc, (KeyboardInterrupt, AcquisitionCancelled))
                parent.save(self.store.directory(parent_id) / "partial.h5")
                self.store.status(
                    parent_id, "cancelled" if parent.metadata["interrupted"] else "failed", str(exc)
                )
                self.last_run_id, self.last_result = parent_id, parent
            raise
        finally:
            from QickworkspaceV2.plotting import LivePlot

            if isinstance(on_progress, LivePlot):
                on_progress.close()
