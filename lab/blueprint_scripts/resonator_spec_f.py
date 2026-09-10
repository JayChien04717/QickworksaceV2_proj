# Generated from ExperimentRegistry; edit the experiment module and re-export.
from typing import Annotated


def resonator_spec_f(target: str = 'Q1', start: float = -3.0, stop: float = 3.0, points: Annotated[int, (8, 10001)] = 81, gain_scale: Annotated[float, (0, 5)] = 1.0, request_id: str = "") -> dict:
    """Readout resonance after preparing |f>; direct and mux. Reuse request_id when retrying the same measurement."""
    from QickworkspaceV2.integrations.nvidia import execute
    return execute('resonator_spec_f', {'target': target, 'start': start, 'stop': stop, 'points': points, 'gain_scale': gain_scale}, request_id=request_id or None)
