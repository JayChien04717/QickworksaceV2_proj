"""Editable notebook configuration. Native QICK values stay native Python objects."""

from copy import deepcopy
from collections.abc import Mapping, MutableMapping
from pathlib import Path
import yaml

from .models import Device, ProjectConfig, RunDefaults, read_yaml


class RunConfig(dict):
    """A normal editable dict with a private copy of the readout-group context."""

    def __init__(self, values=(), *, groups=None):
        super().__init__(values)
        self.groups = deepcopy(groups or {})

    def copy(self):
        return deepcopy(self)

    def for_run(self, **overrides):
        """Fork this run's settings; native QICK sweep objects are preserved."""
        result = self.copy()
        result.update(overrides)
        return result


class QubitConfig(MutableMapping):
    """Live working values for one qubit; for_run() freezes an independent copy."""

    def __init__(self, owner, name):
        self._owner, self.name = owner, name

    def __getitem__(self, key):
        return self._owner._qubits[self.name][key]

    def __setitem__(self, key, value):
        self._owner._qubits[self.name][key] = deepcopy(value)

    def __delitem__(self, key):
        del self._owner._qubits[self.name][key]

    def __iter__(self):
        return iter(self._owner._qubits[self.name])

    def __len__(self):
        return len(self._owner._qubits[self.name])

    def __repr__(self):
        return repr(dict(self))

    def for_run(self, **overrides):
        return self._owner.for_run(self.name, **overrides)


class ExperimentConfig:
    """Editable settings selected by fixed qubit IDs, with independent run copies.

    Additional experiment-specific keys are allowed. No database commit or file
    write occurs until save() is explicitly called.
    """

    def __init__(self, config, *, defaults=None):
        self.groups = {}
        self.couplers = {}
        if isinstance(config, Device):
            self._qubits = {q: config.flat_qubit(q) for q in config.config.qubits}
            self.groups = {
                g: {**group.model_dump(), "port": config.hardware.generators[group.generator].model_dump()}
                for g, group in config.config.readout_groups.items()
            }
            self.couplers = {
                name: {**edge.model_dump(), "port": config.hardware.generators[edge.drive].model_dump()}
                for name, edge in config.config.couplers.items()
            }
            for q, flat in self._qubits.items():
                for prefix, transition in config.config.qubits[q].transitions.items():
                    flat[f"max_gain_{prefix}"] = config.hardware.generators[
                        transition.drive or config.config.qubits[q].drive
                    ].max_gain
                flat["max_res_gain"] = config.hardware.generators[config.readout(q)[1].generator].max_gain
        elif isinstance(config, Mapping):
            self._qubits = {q: {**deepcopy(values), "name": q} for q, values in config.items()}
        else:
            raise TypeError("ExperimentConfig requires a Device or a mapping keyed by qubit ID")
        run = RunDefaults.model_validate(defaults or {})
        common = {
            "reps": run.reps,
            "relax_delay": run.relax_delay_us,
            "soft_avgs": run.soft_avgs,
            "iq_process": run.iq_process,
            "steps": 81,
            "reset_wait_us": 200.0,
            "detuning_mhz": 0.2,
        }
        self._qubits = {
            q: RunConfig({**common, **values}, groups=self.groups) for q, values in self._qubits.items()
        }
        self.project_path = None

    @classmethod
    def from_project(cls, path="lab/project.yaml"):
        path = Path(path).resolve()
        project = ProjectConfig.model_validate(read_yaml(path))
        obj = cls(
            Device.from_files(path.parent / project.hardware, path.parent / project.device),
            defaults=project.defaults,
        )
        obj.project_path = path
        return obj

    @property
    def qubits(self):
        return tuple(self._qubits)

    def __getitem__(self, qubit):
        """Live view: config_all['Q1'].update(res_gain_ge=0.15, res_sigma=0.01)."""
        return QubitConfig(self, self._resolve(qubit))

    def update_all(self, **values):
        """Explicitly update shared working values on every configured qubit."""
        for q in self.qubits:
            self[q].update(values)
        return self

    def for_run(self, *qubits, **overrides):
        """Single/multiple targets, one independent set of run parameters."""
        names = tuple(self._resolve(q) for q in qubits)
        if not names or len(set(names)) != len(names):
            raise ValueError("Select nonempty, unique qubit IDs")
        if len(names) == 1:
            cfg = deepcopy(self._qubits[names[0]])
            cfg.groups = self._current_groups()
        else:
            first = self._qubits[names[0]]
            cfg = RunConfig(
                {
                    "targets": names,
                    "qubits": {q: self.for_run(q) for q in names},
                    "couplers": deepcopy(
                        {k: v for k, v in self.couplers.items() if set(v["qubits"]) <= set(names)}
                    ),
                    **{
                        k: first[k]
                        for k in (
                            "reps", "relax_delay", "soft_avgs", "iq_process", "steps",
                            "reset_wait_us", "detuning_mhz",
                        )
                    },
                },
                groups=self._current_groups(),
            )
        cfg.update(overrides)
        return cfg

    def _current_groups(self):
        groups = deepcopy(self.groups)
        for group in groups.values():
            for q, member in group["members"].items():
                flat = self._qubits[q]
                member.update(
                    frequency_mhz=flat["res_freq_ge"],
                    gain=flat["res_gain_ge"],
                    phase_deg=flat.get("res_phase", 0),
                    tone_slot=flat.get("tone_slot", member["tone_slot"]),
                )
        return groups

    def _resolve(self, qubit):
        if not isinstance(qubit, str) or qubit not in self._qubits:
            raise KeyError(f"Unknown qubit {qubit!r}; available: {self.qubits}")
        return qubit

    def save(self, path):
        """Save scalar working settings separately from the original wiring profile."""

        # YAML rejects live QickSweep objects instead of silently converting them to strings.
        def plain(value):
            if isinstance(value, dict):
                return {k: plain(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            return value

        payload = yaml.safe_dump(
            {
                "qubits": plain(self._qubits),
                "readout_groups": plain(self.groups),
                "couplers": plain(self.couplers),
            },
            sort_keys=False,
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        return path

    @classmethod
    def load(cls, path):
        data = read_yaml(path)
        obj = cls(data["qubits"])
        obj.groups = deepcopy(data.get("readout_groups", {}))
        obj.couplers = deepcopy(data.get("couplers", {}))
        for cfg in obj._qubits.values():
            cfg.groups = deepcopy(obj.groups)
        return obj


def prepare_program_config(values, transition=None):
    """Make the helper view from the edited dict, without rebuilding experiment parameters."""
    source = deepcopy(values)
    if "qubits" in source and "readout_groups" in source and not isinstance(source, RunConfig):
        # Existing recipe programs already carry a fully resolved group configuration.
        return source
    cfg = dict(source)
    if "qubits" in source:
        targets = tuple(source["targets"])
        flats = {q: dict(source["qubits"][q]) for q in targets}
    else:
        q = source.get("name", "Q1")
        targets = (q,)
        flats = {q: dict(source)}
        cfg["namespace"] = ""
    groups = deepcopy(getattr(source, "groups", {}))
    active_groups = {}
    for q, flat in flats.items():
        flat["namespace"] = q
        flat.setdefault("readout_mode", "direct")
        group_id = flat.get("readout_group", "R_" + q)
        if group_id not in active_groups:
            if flat["readout_mode"] == "mux" and group_id not in groups:
                raise ValueError(
                    "MUX requires the complete group context; use config_all.for_run(...)"
                )
            group = deepcopy(groups.get(group_id, {}))
            group.update(
                mode=flat["readout_mode"],
                generator=group.get("generator", "readout_" + group_id),
                length_us=flat["res_length"],
                integration_us=flat["ro_length"],
                trigger_us=flat["trig_time"],
                port={
                    "channel": flat["res_ch"],
                    "nqz": flat["nqz_res"],
                    "mixer_mhz": flat.get("res_mixer"),
                    "max_gain": flat.get("max_res_gain", 1.0),
                },
            )
            group.setdefault("members", {})
            active_groups[group_id] = group
        group = active_groups[group_id]
        if any(
            group[k] != flat[f]
            for k, f in (
                ("length_us", "res_length"),
                ("integration_us", "ro_length"),
                ("trigger_us", "trig_time"),
            )
        ):
            raise ValueError(f"{group_id}: selected members must agree on shared readout timing")
        if group["port"]["channel"] != flat["res_ch"]:
            raise ValueError(f"{group_id}: selected members disagree on the shared generator")
        member = group["members"].setdefault(q, {})
        member.update(
            frequency_mhz=flat["res_freq_ge"],
            gain=flat["res_gain_ge"],
            phase_deg=flat.get("res_phase", 0),
            tone_slot=flat.get("tone_slot", member.get("tone_slot", 0)),
            adc=str(flat["ro_ch"]),
        )
        flat["readout_group"] = group_id
    cfg.update(
        targets=targets,
        qubits=flats,
        readout_groups=active_groups,
        transition=transition or cfg.get("transition", "ge"),
        couplers=cfg.get("couplers", {}),
        pins=cfg.get("pins", []),
    )
    return cfg
