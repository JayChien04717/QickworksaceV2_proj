# Generated from ExperimentRegistry; edit the experiment module and re-export.
from typing import Annotated


def drag_ef(delta_mhz: float, target: str = 'Q1', alpha: float = 0.0, iterations: Annotated[int, (1, 101)] = 5, reset_wait_us: float = 200.0, request_id: str = "") -> dict:
    """EF DRAG alpha point with repeated X/-X and reference states. Reuse request_id when retrying the same measurement."""
    from QickworkspaceV2.integrations.nvidia import execute
    return execute('drag_ef', {'delta_mhz': delta_mhz, 'target': target, 'alpha': alpha, 'iterations': iterations, 'reset_wait_us': reset_wait_us}, request_id=request_id or None)
