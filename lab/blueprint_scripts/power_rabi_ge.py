# Generated from ExperimentRegistry; edit the experiment module and re-export.
from typing import Annotated


def power_rabi_ge(target: str = 'Q1', start: Annotated[float, (0, 1)] = 0.0, stop: Annotated[float, (0, 1)] = 0.6, points: Annotated[int, (8, 10001)] = 81, request_id: str = "") -> dict:
    """PowerRabiGE: independently maintained native GE protocol. Reuse request_id when retrying the same measurement."""
    from QickworkspaceV2.integrations.nvidia import execute
    return execute('power_rabi_ge', {'target': target, 'start': start, 'stop': stop, 'points': points}, request_id=request_id or None)
