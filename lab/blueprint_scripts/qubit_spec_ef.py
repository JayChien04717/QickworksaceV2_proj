# Generated from ExperimentRegistry; edit the experiment module and re-export.
from typing import Annotated


def qubit_spec_ef(target: str = 'Q1', start: float = -20.0, stop: float = 20.0, gain: Annotated[float, (-1, 1)] = 0.05, length_us: float = 2.0, points: Annotated[int, (8, 10001)] = 81, request_id: str = "") -> dict:
    """QubitSpecEF: independently maintained native EF protocol. Reuse request_id when retrying the same measurement."""
    from QickworkspaceV2.integrations.nvidia import execute
    return execute('qubit_spec_ef', {'target': target, 'start': start, 'stop': stop, 'gain': gain, 'length_us': length_us, 'points': points}, request_id=request_id or None)
