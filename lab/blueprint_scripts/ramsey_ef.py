# Generated from ExperimentRegistry; edit the experiment module and re-export.
from typing import Annotated


def ramsey_ef(target: str = 'Q1', start: float = 0.0, stop: float = 30.0, detuning_mhz: float = 0.2, points: Annotated[int, (8, 10001)] = 81, request_id: str = "") -> dict:
    """RamseyEF: independently maintained native EF protocol. Reuse request_id when retrying the same measurement."""
    from QickworkspaceV2.integrations.nvidia import execute
    return execute('ramsey_ef', {'target': target, 'start': start, 'stop': stop, 'detuning_mhz': detuning_mhz, 'points': points}, request_id=request_id or None)
