# Generated from ExperimentRegistry; edit the experiment module and re-export.
def single_shot(target: str = 'Q1', reset_wait_us: float = 200.0, request_id: str = "") -> dict:
    """Paired ground/excited shots with held-out discrimination. Reuse request_id when retrying the same measurement."""
    from QickworkspaceV2.integrations.nvidia import execute
    return execute('single_shot', {'target': target, 'reset_wait_us': reset_wait_us}, request_id=request_id or None)
