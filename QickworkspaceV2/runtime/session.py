"""One orchestration path for notebooks, custom experiments and the service worker."""

from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from uuid import uuid4
import inspect
import math
import tempfile
import warnings
import sys
from importlib.metadata import version, PackageNotFoundError
import numpy as np

from QickworkspaceV2.backends import QICKBackend, AcquisitionCancelled
from QickworkspaceV2.calibration import CalibrationStore, CalibrationProposal
from QickworkspaceV2.device.models import Device, ProjectConfig, RunDefaults, read_yaml
from QickworkspaceV2.experiments.base import BuildContext
from QickworkspaceV2.data.models import ExperimentData, TraceData
from QickworkspaceV2.data.serialization import digest, jsonable
from QickworkspaceV2.data.store import RunStore, atomic_json


@contextmanager
def _hardware_lease(resource_id):
    """OS-owned lock: released after crash; protects independent notebook/worker processes."""
    path = Path(tempfile.gettempdir()) / "qickworkspace-locks"
    path.mkdir(exist_ok=True)
    with (path / (digest(resource_id) + ".lock")).open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        import os

        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("QICK is in use by another process; submit through the same worker") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _range(value):
    if hasattr(value, "spans"):
        return value.start + sum(min(0, v) for v in value.spans.values()), value.start + sum(
            max(0, v) for v in value.spans.values()
        )
    return value, value


def _code_record(callable_object):
    if callable_object is None:
        return None
    callable_object = getattr(callable_object, "func", callable_object)
    try:
        source = inspect.getsource(callable_object)
    except (OSError, TypeError):
        source = None
    return {
        "module": getattr(callable_object, "__module__", None),
        "name": getattr(callable_object, "__qualname__", type(callable_object).__name__),
        "source": source,
        "sha256": digest(source) if source is not None else None,
    }


def _versions():
    result = {"python": sys.version.split()[0]}
    for name in ("qick", "numpy", "scipy", "h5py", "matplotlib", "pydantic"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


class Session:
    def __init__(
        self, device, backend, *, data_dir="data", calibration_db=None, defaults=None, registry=None
    ):
        from QickworkspaceV2.experiments import default_registry

        self.base_device, self.backend = device, backend
        self.defaults = RunDefaults.model_validate(defaults or {})
        self.registry = registry or default_registry()
        self.store = RunStore(data_dir)
        self.scope = digest(
            {
                "device_id": device.config.device_id,
                "wiring_revision": device.config.wiring_revision,
                "hardware": device.hardware.model_dump(),
                "backend": "qick",
            }
        )
        self.calibration = CalibrationStore(
            calibration_db or self.store.root / "calibration.sqlite3", scope=self.scope
        )
        self.project_path = None

    @classmethod
    def from_project(cls, path, *, backend=None):
        path = Path(path).resolve()
        project = ProjectConfig.model_validate(read_yaml(path))
        device = Device.from_files(path.parent / project.hardware, path.parent / project.device)
        if backend is None:
            backend = QICKBackend.from_pyro4(**project.connection.model_dump())
        session = cls(
            device,
            backend,
            data_dir=path.parent / project.data_dir,
            calibration_db=path.parent / project.calibration_db,
            defaults=project.defaults,
        )
        session.project_path = path
        from importlib import import_module

        for name in project.experiment_modules:
            module = import_module(name)
            for spec in module.EXPERIMENTS:
                session.register(spec)
        return session

    @property
    def device(self):
        _, values = self.calibration.snapshot()
        return self.base_device.with_updates(values)

    def register(self, spec, **kwargs):
        return self.registry.register(spec, **kwargs)

    def catalog(self):
        return self.registry.catalog()

    def _prepare(self, experiment, params, run_options=None):
        spec = self.registry.get(experiment)
        parsed = spec.parameters.model_validate(params)
        targets = tuple(q.strip() for q in parsed.target.split(","))
        revision, calibration_values = self.calibration.snapshot()
        device = self.base_device.with_updates(calibration_values)
        device.select(targets)
        defaults = RunDefaults.model_validate({**self.defaults.model_dump(), **(run_options or {})})
        plan = spec.build(BuildContext(device, targets, defaults), parsed)
        self._validate_plan(device, plan)
        self.backend.validate(device, plan)
        return spec, parsed, plan, device, defaults, revision

    def _validate_plan(self, device, plan):
        cfg = plan.cfg
        targets = device.select(cfg["targets"])
        occupied = {}
        transition = cfg.get("transition", "ge")
        for q in targets:
            spec = device.config.qubits[q]
            if transition not in spec.transitions:
                raise ValueError(f"{q}: missing {transition} transition")
            prefixes = ("ge", "ef") if transition == "ef" else (transition,)
            for prefix in prefixes:
                if prefix not in spec.transitions:
                    raise ValueError(f"{q}: EF preparation needs a GE transition")
                trans = spec.transitions[prefix]
                port_id = trans.drive or spec.drive
                port = device.hardware.generators[port_id]
                if port_id in occupied and occupied[port_id] != q:
                    raise ValueError(
                        f"{q} and {occupied[port_id]} share drive {port_id}; select nonconflicting targets"
                    )
                occupied[port_id] = q
                for name in (f"qb_gain_{prefix}", f"pi_gain_{prefix}", f"pi2_gain_{prefix}"):
                    value = (cfg if cfg.get("_native_single") else cfg["qubits"][q]).get(name)
                    if value is not None and max(map(abs, _range(value))) > port.max_gain:
                        raise ValueError(
                            f"{q}/{name}: requested gain exceeds {port_id} max_gain={port.max_gain}"
                        )
        for name, group in cfg["readout_groups"].items():
            port = group["generator"]
            if port in occupied:
                raise ValueError(
                    f"Readout {name} overlaps drive for {occupied[port]}; fix wiring before measurement"
                )
            occupied[port] = name
            for member in group["members"].values():
                if abs(member["gain"]) > device.hardware.generators[port].max_gain:
                    raise ValueError(f"Readout {name}: gain exceeds configured port limit")
        if cfg.get("edge"):
            edge = cfg["edge"]
            port_id = device.config.couplers[edge].drive
            if port_id in occupied:
                raise ValueError(f"Coupler {edge} overlaps active drive/readout {occupied[port_id]}")
            gain = cfg["couplers"][edge]["pulse"]["gain"]
            if max(map(abs, _range(gain))) > device.hardware.generators[port_id].max_gain:
                raise ValueError(f"Coupler {edge}: gain exceeds port limit")
            occupied[port_id] = edge
        for port_id, gain in cfg.get("auxiliary_ports", {}).items():
            if port_id in occupied:
                raise ValueError(f"Auxiliary port {port_id} overlaps {occupied[port_id]}")
            if max(map(abs, _range(gain))) > device.hardware.generators[port_id].max_gain:
                raise ValueError(f"Auxiliary port {port_id}: gain exceeds port limit")
            occupied[port_id] = "auxiliary"
        count = (
            math.prod(s.points for s in plan.sweeps) * cfg["reps"] * len(plan.readout_events) * len(targets)
        )
        if count > 50_000_000:
            raise ValueError(
                "Program exceeds 50 million accumulated IQ samples; split the scan into smaller runs"
            )
        if len(set(s.name for s in plan.sweeps)) != len(plan.sweeps) or len(
            set(s.loop for s in plan.sweeps)
        ) != len(plan.sweeps):
            raise ValueError("Sweep dimensions and loops must be unique")

    def check(self, experiment, *, targets=None, **params):
        if targets is not None:
            params["target"] = targets if isinstance(targets, str) else ",".join(targets)
        try:
            spec, parsed, plan, device, defaults, revision = self._prepare(experiment, params)
            program = self.backend.compile(plan)
            return {
                "ready": True,
                "compiled": True,
                "hardware_verified": False,
                "backend": "qick",
                "targets": list(plan.cfg["targets"]),
                "calibration_revision": revision,
                "readout_channels": list(program.ro_chs),
                "host_sweep": plan.metadata.get("host_sweep"),
                "message": "Compiled; no acquisition performed",
            }
        except Exception as exc:
            return {"ready": False, "message": str(exc), "error_type": type(exc).__name__}

    def compile(self, experiment, **params):
        _, _, plan, _, _, _ = self._prepare(experiment, params)
        return self.backend.compile(plan)

    def run(
        self,
        experiment,
        *,
        targets=None,
        reps=None,
        soft_avgs=None,
        relax_delay_us=None,
        iq_process=None,
        on_progress=None,
        cancel=None,
        **params,
    ):
        if targets is not None:
            if "target" in params:
                raise ValueError("Use either target or targets")
            params["target"] = targets if isinstance(targets, str) else ",".join(targets)
        options = {
            k: v
            for k, v in dict(
                reps=reps, soft_avgs=soft_avgs, relax_delay_us=relax_delay_us, iq_process=iq_process
            ).items()
            if v is not None
        }
        with self.backend.lock:
            with _hardware_lease(self.backend.resource_id):
                return self._run(experiment, params, options, on_progress, cancel)

    def _run(self, experiment, params, options, on_progress, cancel, *, context=None):
        spec, parsed, plan, device, defaults, revision = self._prepare(experiment, params, options)
        from QickworkspaceV2 import __version__

        try:
            source = inspect.getsource(plan.program)
        except (OSError, TypeError):
            source = None
        metadata = {
            "targets": list(plan.cfg["targets"]),
            "parameters": parsed.model_dump(),
            "transition": plan.cfg.get("transition", "ge"),
            "device_snapshot": device.snapshot(),
            "resolved_config": jsonable(plan.cfg),
            "calibration_revision": revision,
            "calibration_scope": self.scope,
            "experiment_version": spec.version,
            "sdk_version": __version__,
            "backend": self.backend.capabilities(),
            "run_options": defaults.model_dump(),
            "iq_process": defaults.iq_process,
            "program_source": source,
            "program_source_sha256": digest(source),
            **plan.metadata,
        }
        metadata["software_versions"] = _versions()
        metadata["build_code"] = _code_record(spec.build)
        if context:
            metadata["execution_context"] = context
        run_id = uuid4().hex
        self.store.begin(run_id, spec.id, metadata)

        def notify(event):
            if on_progress:
                try:
                    on_progress(event)
                except Exception as exc:
                    warnings.warn(f"Progress callback failed: {exc}", RuntimeWarning, stacklevel=2)

        notify({"state": "compiling", "run_id": run_id})
        try:
            if plan.metadata.get("host_sweep"):
                traces = self._host_acquire(plan, defaults, notify, cancel, spec.id, metadata, run_id)
            else:
                program = self.backend.compile(plan)
                self._check_readouts(program, plan)
                atomic_json(self.store.directory(run_id) / "compiled.json", self._compiled_record(program))

                def progress(done, total, traces):
                    ExperimentData(spec.id, traces, metadata=metadata, run_id=run_id).save(
                        self.store.directory(run_id) / "partial.h5"
                    )
                    notify(
                        {
                            "state": "acquiring",
                            "completed": done,
                            "total": total,
                            "result": ExperimentData(
                                spec.id, deepcopy(traces), metadata=metadata, run_id=run_id
                            ),
                        }
                    )

                traces = self.backend.acquire(
                    program, plan, soft_avgs=defaults.soft_avgs, on_progress=progress, cancel=cancel
                )
                scale = plan.metadata.get("axis_scale", 1)
                if scale != 1:
                    for trace in traces.values():
                        trace.coords[plan.sweeps[0].name] *= scale
            result = ExperimentData(spec.id, traces, metadata, run_id=run_id)
            self.store.save_acquisition(result)
            self._analyze(result, spec)
            notify({"state": "completed", "result": result})
            return result
        except BaseException as exc:
            state = "cancelled" if isinstance(exc, (AcquisitionCancelled, KeyboardInterrupt)) else "failed"
            self.store.status(run_id, state, str(exc))
            raise

    @staticmethod
    def _compiled_record(program):
        record = {"readouts": program.ro_chs, "generators": program.gen_chs, "asm": str(program)}
        return jsonable(record)

    @staticmethod
    def _check_readouts(program, plan):
        expected = len(plan.readout_events)
        channels = [plan.cfg["qubits"][q]["ro_ch"] for q in plan.cfg["targets"]]
        if set(channels) != set(program.ro_chs):
            raise ValueError("Declared readouts differ from the target mapping")
        for ch, cfg in program.ro_chs.items():
            if cfg["trigs"] != expected:
                raise ValueError(
                    f"ADC {ch}: program emits {cfg['trigs']} readouts, contract declares {expected}"
                )

    def _host_acquire(self, plan, defaults, notify, cancel, identifier, metadata, run_id):
        sweep = plan.metadata["host_sweep"]
        offsets = np.linspace(sweep["start"], sweep["stop"], sweep["points"])
        collected, coordinates = {}, {}
        for step, offset in enumerate(offsets):
            if cancel and cancel.is_set():
                raise AcquisitionCancelled("Host scan cancelled between points")
            point = deepcopy(plan)
            point.metadata.pop("host_sweep", None)
            for q, qc in point.cfg["qubits"].items():
                qc["res_freq_ge"] += float(offset)
            for group in point.cfg["readout_groups"].values():
                for q, member in group["members"].items():
                    if q in point.cfg["targets"]:
                        member["frequency_mhz"] += float(offset)
            point.cfg["host_offset"], point.cfg["host_span"] = (
                float(offset),
                float(sweep["stop"] - sweep["start"]),
            )
            point.cfg["host_center"] = float((sweep["stop"] + sweep["start"]) / 2)
            program = self.backend.compile(point)
            self._check_readouts(program, point)
            atomic_json(
                self.store.directory(run_id) / "compiled" / f"{step:05d}.json", self._compiled_record(program)
            )
            traces = self.backend.acquire(program, point, soft_avgs=defaults.soft_avgs, cancel=cancel)
            for q, trace in traces.items():
                collected.setdefault(q, []).append(trace.iq.copy())
                qc = point.cfg["qubits"][q]
                if qc["readout_mode"] == "direct":
                    actual = float(program.get_pulse_param(q + "__res_pulse", "freq", as_array=True))
                else:
                    group = point.cfg["readout_groups"][qc["readout_group"]]
                    tone = program.gen_chs[qc["res_ch"]]["mux_tones"][group["members"][q]["tone_slot"]]
                    actual = float(tone["freq_rounded"])
                coordinates.setdefault(q, []).append(actual)
            partial_traces = {
                q: TraceData(
                    np.stack(values),
                    (sweep["name"], "readout"),
                    {sweep["name"]: np.array(coordinates[q]), "readout": np.array(plan.readout_events)},
                    {sweep["name"]: sweep["unit"]},
                    metadata=traces[q].metadata,
                )
                for q, values in collected.items()
            }
            # Durable partial data survives interruption of a long host scan.
            ExperimentData(identifier, partial_traces, metadata, run_id=run_id).save(
                self.store.directory(run_id) / "partial.h5"
            )
            notify(
                {
                    "state": "acquiring",
                    "completed": step + 1,
                    "total": len(offsets),
                    "result": ExperimentData(identifier, partial_traces, metadata, run_id=run_id),
                }
            )
        return partial_traces

    def _analyze(self, result, spec):
        result._plotter = spec.plot
        result.metadata["analysis_code"] = _code_record(spec.analyze)
        result.metadata["analysis_experiment_version"] = spec.version
        if spec.analyze is None:
            result.analysis_status = "not_analyzed"
        else:
            try:
                result.fits = spec.analyze(result)
                result.analysis_status = "completed"
                failures = [f"{q}: {fit.message}" for q, fit in result.fits.items() if not fit.success]
                result.analysis_message = "; ".join(failures)
            except Exception as exc:
                result.analysis_status, result.analysis_message = "failed", str(exc)
        self.store.save_analysis(result)

    def reanalyze(self, run_id, *, experiment=None):
        with self.backend.lock:
            result = self.store.load(run_id, revision=0)
            self._analyze(result, self.registry.get(experiment or result.experiment))
            return result

    def scan(self, experiment, parameter, values, *, target="Q1", unit="", on_progress=None, **params):
        """Host outer parameter scan, e.g. readout punchout, using the same run path."""
        values = np.asarray(values)
        if values.ndim != 1 or not len(values) or not np.isfinite(values.astype(float)).all():
            raise ValueError("Host scan values must be a nonempty finite one-dimensional array")
        spec = self.registry.get(experiment)
        if parameter not in spec.parameters.model_fields or parameter == "target":
            raise ValueError("Select a numeric experiment parameter for the host scan")
        children = []
        for index, value in enumerate(values):
            child = self.run(spec, target=target, **{**params, parameter: value.item()})
            children.append(child)
            if on_progress:
                on_progress(
                    {"state": "acquiring", "completed": index + 1, "total": len(values), "result": child}
                )
        traces = {}
        for q, first in children[0].traces.items():
            if parameter in first.dims:
                raise ValueError("Host parameter name conflicts with an existing data dimension")
            for child in children[1:]:
                trace = child[q]
                if trace.dims != first.dims or any(
                    not np.array_equal(trace.coords[d], first.coords[d]) for d in first.dims
                ):
                    raise ValueError(
                        "Inner axes vary across points; use individual child runs instead of a rectangular array"
                    )
            shots = None if first.shots is None else np.stack([child[q].shots for child in children])
            traces[q] = TraceData(
                np.stack([child[q].iq for child in children]),
                (parameter, *first.dims),
                {parameter: values, **first.coords},
                {parameter: unit, **first.units},
                shots=shots,
                shot_dims=(parameter, *first.shot_dims) if shots is not None else (),
                metadata=first.metadata,
            )
        metadata = {
            **children[0].metadata,
            "host_scan_parameter": parameter,
            "child_runs": [r.run_id for r in children],
        }
        result = ExperimentData(spec.id, traces, metadata)
        self.store.begin(result.run_id, spec.id, metadata)
        self.store.save_acquisition(result)
        self.store.save_analysis(result)
        return result

    def load(self, run_id, **kwargs):
        result = self.store.load(run_id, **kwargs)
        result._plotter = self.registry.get(result.experiment).plot
        return result

    def scan_instrument(self, experiment, axis, values, **kwargs):
        from .instrument_scan import scan_instrument

        return scan_instrument(self, experiment, axis, values, **kwargs)

    def propose(self, result):
        if not result.is_good():
            raise ValueError("Calibration proposal requires successful, quality-accepted fits")
        if result.metadata["calibration_scope"] != self.scope:
            raise ValueError("Run belongs to a different device/backend calibration scope")
        transition = result.metadata.get("transition", "ge")
        updates = {}
        for q, fit in result.fits.items():
            root = f"qubits.{q}.transitions.{transition}"
            if result.experiment in ("qubit_spec_ge", "qubit_spec_ef"):
                updates[root + ".frequency_mhz"] = fit.parameters["center"]
            elif result.experiment in ("power_rabi_ge", "power_rabi_ef"):
                updates[root + ".pulse.pi_gain"] = fit.parameters["pi_gain"]
                updates[root + ".pulse.pi2_gain"] = fit.parameters["pi2_gain"]
            elif result.experiment in ("single_shot", "resonator_spec"):
                group = self.device.readout(q)[0]
                root = f"readout_groups.{group}.members.{q}"
                keys = (
                    {"rotation_deg": "rotation_deg", "threshold": "threshold"}
                    if result.experiment == "single_shot"
                    else {"frequency_mhz": "center"}
                )
                for key, metric in keys.items():
                    updates[root + "." + key] = fit.parameters[metric]
        if not updates:
            raise ValueError(
                "This experiment reports metrics without automatic parameter updates; create an explicit proposal after review"
            )
        return CalibrationProposal(
            updates,
            result.metadata["calibration_revision"],
            result.run_id,
            self.scope,
            source="measured",
        )

    def commit(self, proposal):
        return self.calibration.commit(proposal, validate=self.base_device.with_updates)
