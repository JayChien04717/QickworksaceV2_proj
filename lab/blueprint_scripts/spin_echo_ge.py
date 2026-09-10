# Generated from ExperimentRegistry; edit the experiment module and re-export.
from typing import Annotated


def spin_echo_ge(target: str = 'Q1', start: float = 0.0, stop: float = 100.0, points: Annotated[int, (8, 10001)] = 81, request_id: str = "") -> dict:
    """SpinEchoGE: independently maintained native GE protocol. Reuse request_id when retrying the same measurement."""
    from QickworkspaceV2.integrations.nvidia import execute
    return execute('spin_echo_ge', {'target': target, 'start': start, 'stop': stop, 'points': points}, request_id=request_id or None)
