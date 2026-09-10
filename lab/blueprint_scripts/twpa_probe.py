# Generated from ExperimentRegistry; edit the experiment module and re-export.
from typing import Annotated


def twpa_probe(target: str = 'Q1', start_mhz: float = 4000.0, stop_mhz: float = 8000.0, points: Annotated[int, (2, 10001)] = 101, gain_scale: Annotated[float, (0, 5)] = 0.1, request_id: str = "") -> dict:
    """Broadband complex transmission; compare matched pump-on/off reference runs. Reuse request_id when retrying the same measurement."""
    from QickworkspaceV2.integrations.nvidia import execute
    return execute('twpa_probe', {'target': target, 'start_mhz': start_mhz, 'stop_mhz': stop_mhz, 'points': points, 'gain_scale': gain_scale}, request_id=request_id or None)
