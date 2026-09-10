"""Versioned device models. Frequencies are QICK MHz; times are microseconds."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Annotated, Literal
import re

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Gain = Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]
Channel = Annotated[int, Field(ge=0, strict=True)]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class Generator(Model):
    channel: Channel
    nqz: Literal[1, 2] = 1
    mixer_mhz: Finite | None = None
    max_gain: Annotated[float, Field(gt=0, le=1)] = 1.0


class HardwareConfig(Model):
    schema_version: Literal[1] = 1
    name: str
    tproc: Literal["v2"] = "v2"
    generators: dict[str, Generator]
    readouts: dict[str, Channel]
    trigger_pins: list[Channel] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ports(self):
        channels = [p.channel for p in self.generators.values()]
        if len(channels) != len(set(channels)):
            raise ValueError("Generator channels must have one port ID; share that ID explicitly")
        if len(self.readouts.values()) != len(set(self.readouts.values())):
            raise ValueError("Digital readout channels must have one endpoint ID each")
        return self


class PulseConfig(Model):
    style: Literal["const", "arb", "flat_top"] = "arb"
    shape: Literal["gauss", "drag", "cosine"] = "gauss"
    length_us: Positive = 0.1
    sigma_us: Positive = 0.02
    sigma_count: Positive = 5.0
    phase_deg: Finite = 0.0
    gain: Gain = 0.1
    pi_gain: Gain | None = None
    pi2_gain: Gain | None = None
    drag_alpha: Finite | None = None
    drag_delta_mhz: Finite | None = None

    @model_validator(mode="after")
    def drag_parameters(self):
        if self.shape == "drag" and (self.drag_alpha is None or not self.drag_delta_mhz):
            raise ValueError("DRAG requires drag_alpha and nonzero drag_delta_mhz")
        return self


class Transition(Model):
    frequency_mhz: Finite
    drive: str | None = None
    pulse: PulseConfig = Field(default_factory=PulseConfig)


class QubitConfig(Model):
    drive: str
    kind: Literal["transmon", "fluxonium", "other"] = "transmon"
    enabled: bool = True
    transitions: dict[str, Transition]
    tags: list[str] = Field(default_factory=list)
    position: tuple[Finite, Finite] | None = None


class ReadoutMember(Model):
    adc: str
    tone_slot: Channel = 0
    frequency_mhz: Finite
    gain: Gain
    phase_deg: Finite = 0.0
    rotation_deg: Finite = 0.0
    threshold: Finite | None = None


class ReadoutGroup(Model):
    generator: str
    mode: Literal["direct", "mux"] = "direct"
    length_us: Positive
    integration_us: Positive
    trigger_us: Annotated[float, Field(ge=0)] = 0.5
    members: dict[str, ReadoutMember]

    @model_validator(mode="after")
    def valid_members(self):
        if not self.members:
            raise ValueError("Readout group must contain members")
        if self.mode == "direct" and len(self.members) != 1:
            raise ValueError("Direct readout needs one member; use mux for a shared generator")
        slots = [v.tone_slot for v in self.members.values()]
        if len(set(slots)) != len(slots):
            raise ValueError("tone_slot must be unique within a readout group")
        return self


class CouplerConfig(Model):
    qubits: tuple[str, str]
    drive: str
    pulse: PulseConfig
    frequency_mhz: Finite = 0.0
    calibrated: bool = False
    phase_corrections_deg: dict[str, Finite] = Field(default_factory=dict)


class DeviceConfig(Model):
    schema_version: Literal[1] = 1
    device_id: str
    wiring_revision: str = "1"
    qubits: dict[str, QubitConfig]
    readout_groups: dict[str, ReadoutGroup]
    couplers: dict[str, CouplerConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def references(self):
        for name in [*self.qubits, *self.readout_groups, *self.couplers]:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                raise ValueError(f"Invalid device entity ID {name!r}")
        seen = set()
        for group in self.readout_groups.values():
            for name in group.members:
                if name not in self.qubits:
                    raise ValueError(f"Unknown readout member {name}")
                if name in seen:
                    raise ValueError(f"{name} belongs to multiple readout groups")
                seen.add(name)
        if set(self.qubits) != seen:
            raise ValueError(f"Missing readout membership: {set(self.qubits) - seen}")
        for name, edge in self.couplers.items():
            if len(set(edge.qubits)) != 2 or not set(edge.qubits) <= self.qubits.keys():
                raise ValueError(f"{name}: coupler needs two distinct existing qubits")
            if not set(edge.phase_corrections_deg) <= set(edge.qubits):
                raise ValueError(f"{name}: phase corrections must refer to endpoints")
        return self


class RunDefaults(Model):
    reps: Annotated[int, Field(gt=0, strict=True)] = 100
    soft_avgs: Annotated[int, Field(gt=0, strict=True)] = 1
    relax_delay_us: Annotated[float, Field(ge=0)] = 100.0
    iq_process: Literal["abs", "real", "imag", "phase"] = "abs"


class ConnectionConfig(Model):
    ns_host: str = "192.168.10.82"
    ns_port: Annotated[int, Field(gt=0, le=65535)] = 8888
    proxy_name: str = "myqick"


class ProjectConfig(Model):
    schema_version: Literal[1] = 1
    hardware: str
    device: str
    data_dir: str = "data"
    calibration_db: str = "calibration.sqlite3"
    defaults: RunDefaults = Field(default_factory=RunDefaults)
    connection: ConnectionConfig = Field(default_factory=ConnectionConfig)
    experiment_modules: list[str] = Field(default_factory=list)


def read_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8-sig") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


class Device:
    """A validated wiring + device snapshot, with stable IDs independent of order."""

    def __init__(self, hardware: HardwareConfig | dict, config: DeviceConfig | dict):
        self.hardware = HardwareConfig.model_validate(deepcopy(hardware))
        self.config = DeviceConfig.model_validate(deepcopy(config))
        self.validate()

    @classmethod
    def from_files(cls, hardware, device):
        return cls(read_yaml(hardware), read_yaml(device))

    def validate(self):
        ports = self.hardware.generators
        readout_ports, adc_ids = set(), set()
        for q, spec in self.config.qubits.items():
            for port in [spec.drive, *(t.drive for t in spec.transitions.values() if t.drive)]:
                if port not in ports:
                    raise ValueError(f"{q}: unknown drive port {port}")
        for name, group in self.config.readout_groups.items():
            if group.generator not in ports or group.generator in readout_ports:
                raise ValueError(f"{name}: unknown or reused readout generator")
            readout_ports.add(group.generator)
            for member in group.members.values():
                if member.adc not in self.hardware.readouts or member.adc in adc_ids:
                    raise ValueError(f"{name}: unknown or reused digital ADC endpoint {member.adc}")
                adc_ids.add(member.adc)
                if abs(member.gain) > ports[group.generator].max_gain:
                    raise ValueError(f"{name}: readout gain exceeds port limit")
        for q, spec in self.config.qubits.items():
            for transition in spec.transitions.values():
                port = ports[transition.drive or spec.drive]
                for gain in [transition.pulse.gain, transition.pulse.pi_gain, transition.pulse.pi2_gain]:
                    if gain is not None and abs(gain) > port.max_gain:
                        raise ValueError(f"{q}: pulse gain exceeds port limit")
        for name, edge in self.config.couplers.items():
            if edge.drive not in ports:
                raise ValueError(f"{name}: unknown coupler drive")
            if abs(edge.pulse.gain) > ports[edge.drive].max_gain:
                raise ValueError(f"{name}: coupler gain exceeds port limit")
        return self

    def select(self, targets=None, *, tags=None) -> tuple[str, ...]:
        if targets is None:
            targets = [q for q, s in self.config.qubits.items() if s.enabled]
        if isinstance(targets, str):
            targets = [targets]
        names = tuple(targets)
        if not names or len(set(names)) != len(names):
            raise ValueError("targets must be nonempty and unique")
        for q in names:
            if q not in self.config.qubits or not self.config.qubits[q].enabled:
                raise ValueError(f"Unknown or disabled qubit {q}")
        if tags:
            names = tuple(q for q in names if set(tags) <= set(self.config.qubits[q].tags))
            if not names:
                raise ValueError("No targets match tags")
        return names

    def readout(self, qubit):
        for name, group in self.config.readout_groups.items():
            if qubit in group.members:
                return name, group, group.members[qubit]
        raise KeyError(qubit)

    def neighbors(self, qubit):
        self.select(qubit)
        return sorted(
            {q for e in self.config.couplers.values() if qubit in e.qubits for q in e.qubits if q != qubit}
        )

    def summary(self):
        return [
            {
                "qubit": q,
                "drive": s.drive,
                "readout_group": self.readout(q)[0],
                "adc": self.readout(q)[2].adc,
                "tone_slot": self.readout(q)[2].tone_slot,
                "transitions_mhz": {k: t.frequency_mhz for k, t in s.transitions.items()},
                "neighbors": self.neighbors(q) if s.enabled else [],
                "enabled": s.enabled,
            }
            for q, s in self.config.qubits.items()
        ]

    def snapshot(self):
        return {"hardware": self.hardware.model_dump(), "device": self.config.model_dump()}

    def with_updates(self, updates: dict):
        data = self.config.model_dump()
        for path, value in updates.items():
            cursor = data
            keys = path.split(".")
            for key in keys[:-1]:
                cursor = cursor[key]
            if keys[-1] not in cursor:
                raise KeyError(path)
            cursor[keys[-1]] = value
        return Device(self.hardware, data)

    def flat_qubit(self, qubit, transition="ge"):
        """Native helper view. New runs namespace pulses with the stable qubit ID."""
        spec = self.config.qubits[qubit]
        group_id, group, tone = self.readout(qubit)
        port = self.hardware.generators[group.generator]
        cfg = {
            "name": qubit,
            "namespace": qubit,
            "ro_ch": self.hardware.readouts[tone.adc],
            "res_ch": port.channel,
            "nqz_res": port.nqz,
            "res_mixer": port.mixer_mhz,
            "readout_group": group_id,
            "readout_mode": group.mode,
            "res_freq_ge": tone.frequency_mhz,
            "res_gain_ge": tone.gain,
            "res_phase": tone.phase_deg,
            "ro_phase": tone.rotation_deg,
            "ro_threshold": tone.threshold,
            "tone_slot": tone.tone_slot,
            "res_length": group.length_us,
            "res_pulse_type": "const",
            "res_sigma": 0.02,
            "res_length_mult": 5.0,
            "ro_length": group.integration_us,
            "trig_time": group.trigger_us,
            "pins": self.hardware.trigger_pins,
            "cooling": False,
            "transition": transition,
        }
        for pfx, trans in spec.transitions.items():
            p = trans.pulse
            drive = self.hardware.generators[trans.drive or spec.drive]
            suffix = "" if pfx == "ge" else f"_{pfx}"
            cfg.update(
                {
                    f"qb_ch{suffix}": drive.channel,
                    f"nqz_qb{suffix}": drive.nqz,
                    f"qb_mixer{suffix}": drive.mixer_mhz,
                    f"qb_freq_{pfx}": trans.frequency_mhz,
                    f"qb_gain_{pfx}": p.gain,
                    f"pi_gain_{pfx}": p.pi_gain,
                    f"pi2_gain_{pfx}": p.pi2_gain,
                    f"sigma_{pfx}": p.sigma_us,
                    f"length_mult_{pfx}": p.sigma_count,
                    f"qb_phase_{pfx}": p.phase_deg,
                    f"pulse_type_{pfx}": p.style,
                    f"shape_{pfx}": p.shape,
                    f"qb_length_{pfx}": p.length_us,
                    f"qb_flat_top_length_{pfx}": p.length_us,
                    f"drag_alpha_{pfx}": p.drag_alpha,
                    f"drag_delta_{pfx}": p.drag_delta_mhz,
                }
            )
        cfg["qb_phase"] = cfg.get("qb_phase_ge", 0)
        return cfg
