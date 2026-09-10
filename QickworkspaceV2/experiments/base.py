"""Small authoring contract: validated parameters + native program + optional analysis."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
from pydantic import ConfigDict, BaseModel, Field

from QickworkspaceV2.programs.sweeps import Sweep


class Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)
    target: str = Field(default="Q1", description="Qubit ID, or comma-separated IDs in measurement order")


@dataclass
class ProgramPlan:
    program: type
    cfg: dict
    sweeps: tuple[Sweep, ...] = ()
    readout_events: tuple[str, ...] = ("measurement",)
    capture_shots: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class BuildContext:
    device: object
    targets: tuple[str, ...]
    defaults: object

    def native_config(self, transition="ge"):
        """Single-target flat view for existing _initialize/_body code, with run metadata attached."""
        if len(self.targets) != 1:
            raise ValueError("native_config is a single-target view; use config() for multiple qubits")
        config = self.config(transition)
        flat = self.device.flat_qubit(self.targets[0], transition)
        flat["namespace"] = ""
        return {**config, **flat, "_native_single": True}

    def config(self, transition="ge"):
        device = self.device
        groups = {}
        for name, group in device.config.readout_groups.items():
            if set(group.members) & set(self.targets):
                groups[name] = {
                    **group.model_dump(),
                    "port": device.hardware.generators[group.generator].model_dump(),
                }
        couplers = {
            name: {**e.model_dump(), "port": device.hardware.generators[e.drive].model_dump()}
            for name, e in device.config.couplers.items()
            if set(e.qubits) <= set(self.targets)
        }
        return {
            "targets": self.targets,
            "transition": transition,
            "qubits": {q: device.flat_qubit(q, transition) for q in self.targets},
            "readout_groups": groups,
            "couplers": couplers,
            "reps": self.defaults.reps,
            "relax_delay": self.defaults.relax_delay_us,
            "pins": device.hardware.trigger_pins,
        }


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    parameters: type[Parameters]
    build: Callable[[BuildContext, Parameters], ProgramPlan]
    analyze: Callable | None = None
    plot: Callable | None = None
    description: str = ""
    version: str = "1.0.0"

    def schema(self):
        return {
            "id": self.id,
            "version": self.version,
            "description": self.description,
            "parameters": self.parameters.model_json_schema(),
        }


class ExperimentRegistry:
    def __init__(self, specs=()):
        self._specs = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec, *, replace=False):
        if not isinstance(spec, ExperimentSpec):
            raise TypeError("Register an ExperimentSpec")
        if spec.id in self._specs and not replace:
            raise ValueError(f"Experiment {spec.id} is already registered")
        self._specs[spec.id] = spec
        return spec

    def get(self, identifier):
        if isinstance(identifier, ExperimentSpec):
            return identifier
        try:
            return self._specs[identifier]
        except KeyError:
            raise KeyError(
                f"Unknown experiment {identifier!r}; available: {', '.join(self._specs)}"
            ) from None

    def catalog(self):
        return [spec.schema() for spec in self._specs.values()]
