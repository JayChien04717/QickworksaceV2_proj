"""Direct notebook acquisition: native Program + editable run_cfg, without a recipe schema."""

from copy import deepcopy
from importlib import import_module
from uuid import uuid4
import math

from QickworkspaceV2.backends import QICKBackend, AcquisitionCancelled
from QickworkspaceV2.backends.qick import extract_coords
from QickworkspaceV2.device.editable import prepare_program_config
from QickworkspaceV2.experiments.base import ProgramPlan
from QickworkspaceV2.programs.sweeps import Sweep
from QickworkspaceV2.data.models import ExperimentData
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
        cfg = prepare_program_config(run_cfg, getattr(program, "TRANSITION", None))
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
        module = import_module(program.__module__)
        spec = (
            getattr(module, "experiment", None)
            if program.__module__.startswith("QickworkspaceV2.experiments.")
            else None
        )
        identifier = spec.id if spec else program.__name__
        analyzer = (spec.analyze if spec else None) if analyze == "auto" else analyze
        run_id = uuid4().hex
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
                default_events = ("ground", "excited") if identifier == "single_shot" else ()
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
                shot_mode = (identifier == "single_shot") if capture_shots is None else capture_shots
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
                    partial = ExperimentData(identifier, preview, metadata=metadata, run_id=run_id)
                    partial.save(directory / "partial.h5")
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
                result = ExperimentData(identifier, traces, metadata=metadata, run_id=run_id)
                self.store.save_acquisition(result)
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
            self.store.status(
                run_id,
                "cancelled" if isinstance(exc, (AcquisitionCancelled, KeyboardInterrupt)) else "failed",
                str(exc),
            )
            raise

    def load(self, run_id, **kwargs):
        return self.store.load(run_id, **kwargs)
