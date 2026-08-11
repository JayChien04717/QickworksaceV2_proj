from .experiment_data import ExperimentData, QualityFlag
from .base_analysis import BaseAnalysis
from .base_experiment import BaseExperiment
from .experiment_components import AcquisitionResult, SweepAxis
from .composite import run_batch, run_parallel, summarize_results


def __getattr__(name):
    """Return the getattr result.

    Parameters
    ----------
    name : Any
        Name of the target object.

    Returns
    -------
    Any
        Result of the operation.

    Raises
    ------
    AttributeError
        If the operation cannot be completed.
    """
    if name in {"BaseProgram", "GATE_ALIAS", "resolve_gate"}:
        from .base_program import BaseProgram, GATE_ALIAS, resolve_gate

        exports = {
            "BaseProgram": BaseProgram,
            "GATE_ALIAS": GATE_ALIAS,
            "resolve_gate": resolve_gate,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ExperimentData",
    "QualityFlag",
    "BaseAnalysis",
    "BaseProgram",
    "GATE_ALIAS",
    "resolve_gate",
    "BaseExperiment",
    "AcquisitionResult",
    "SweepAxis",
    "run_batch",
    "run_parallel",
    "summarize_results",
]
