# Generated from ExperimentRegistry; edit the experiment module and re-export.
from typing import Annotated


def coupler_chevron(target: str = 'Q1,Q2', coupler: str = 'C12', gain_start: Annotated[float, (-1, 1)] = 0.01, gain_stop: Annotated[float, (-1, 1)] = 0.2, gain_points: Annotated[int, (2, 501)] = 21, length_start_us: float = 0.02, length_stop_us: float = 2.0, length_points: Annotated[int, (2, 1001)] = 81, request_id: str = "") -> dict:
    """Two-dimensional native coupler interaction scan. Reuse request_id when retrying the same measurement."""
    from QickworkspaceV2.integrations.nvidia import execute
    return execute('coupler_chevron', {'target': target, 'coupler': coupler, 'gain_start': gain_start, 'gain_stop': gain_stop, 'gain_points': gain_points, 'length_start_us': length_start_us, 'length_stop_us': length_stop_us, 'length_points': length_points}, request_id=request_id or None)
