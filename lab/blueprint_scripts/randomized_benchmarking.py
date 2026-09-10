# Generated from ExperimentRegistry; edit the experiment module and re-export.
from typing import Annotated


def randomized_benchmarking(target: str = 'Q1', reset_wait_us: float = 200.0, depths: str = '0,1,2,4,8,12,16,24', sequences_per_depth: Annotated[int, (2, 20)] = 2, seed: int = 42, request_id: str = "") -> dict:
    """randomized benchmarking: independent native protocol. Reuse request_id when retrying the same measurement."""
    from QickworkspaceV2.integrations.nvidia import execute
    return execute('randomized_benchmarking', {'target': target, 'reset_wait_us': reset_wait_us, 'depths': depths, 'sequences_per_depth': sequences_per_depth, 'seed': seed}, request_id=request_id or None)
