# Generated from ExperimentRegistry; edit the experiment module and re-export.
from typing import Annotated


def conditional_ramsey(target: str = 'Q1,Q2', start: float = 0.0, stop: float = 30.0, points: Annotated[int, (8, 10001)] = 81, detuning_mhz: float = 0.2, reset_wait_us: float = 200.0, request_id: str = "") -> dict:
    """Two-qubit conditional Ramsey: control,target. Reuse request_id when retrying the same measurement."""
    from QickworkspaceV2.integrations.nvidia import execute
    return execute('conditional_ramsey', {'target': target, 'start': start, 'stop': stop, 'points': points, 'detuning_mhz': detuning_mhz, 'reset_wait_us': reset_wait_us}, request_id=request_id or None)
