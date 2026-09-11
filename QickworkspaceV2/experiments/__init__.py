"""One independently maintained module per experiment and transition."""

from importlib import import_module
from .base import ExperimentRegistry

EXPERIMENT_MODULES = (
    "time_of_flight",
    "resonator_spec",
    "resonator_punchout",
    "broadband_resonator_spectrum",
    "single_shot",
    "qubit_spec_ge",
    "qubit_spec_ef",
    "power_rabi_ge",
    "power_rabi_ef",
    "time_rabi_ge",
    "time_rabi_ef",
    "t1_ge",
    "t1_ef",
    "ramsey_ge",
    "ramsey_ef",
    "spin_echo_ge",
    "spin_echo_ef",
    "allxy",
    "randomized_benchmarking",
    "state_tomography",
    "conditional_ramsey",
    "coupler_chevron",
    "resonator_spec_e",
    "resonator_spec_f",
    "power_rabi_chevron_ge",
    "power_rabi_chevron_ef",
    "drag_ge",
    "drag_ef",
    "resonator_flux",
    "twpa_probe",
    "active_reset_rabi",
    "single_shot_gef",
    "qubit_temperature",
    "dispersive",
    "ckp",
)


def default_registry():
    return ExperimentRegistry(import_module(f"{__name__}.{name}").experiment for name in EXPERIMENT_MODULES)
