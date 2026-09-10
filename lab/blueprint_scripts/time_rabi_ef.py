# Generated from ExperimentRegistry; edit the experiment module and re-export.
from typing import Annotated


def time_rabi_ef(target: str = 'Q1', start: float = 0.02, stop: float = 2.0, points: Annotated[int, (8, 10001)] = 81, request_id: str = "") -> dict:
    """TimeRabiEF: independently maintained native EF protocol. Reuse request_id when retrying the same measurement."""
    from QickworkspaceV2.integrations.nvidia import execute
    return execute('time_rabi_ef', {'target': target, 'start': start, 'stop': stop, 'points': points}, request_id=request_id or None)
