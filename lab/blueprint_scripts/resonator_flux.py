# Generated from ExperimentRegistry; edit the experiment module and re-export.
from typing import Annotated


def resonator_flux(bias_port: str, target: str = 'Q1', bias_gain: Annotated[float, (-1, 1)] = 0.0, settle_us: float = 0.1, bias_length_us: float = 10.0, start: float = -3.0, stop: float = 3.0, points: Annotated[int, (8, 10001)] = 41, request_id: str = "") -> dict:
    """Readout spectroscopy under a timed QICK bias pulse. Reuse request_id when retrying the same measurement."""
    from QickworkspaceV2.integrations.nvidia import execute
    return execute('resonator_flux', {'bias_port': bias_port, 'target': target, 'bias_gain': bias_gain, 'settle_us': settle_us, 'bias_length_us': bias_length_us, 'start': start, 'stop': stop, 'points': points}, request_id=request_id or None)
