# Generated from ExperimentRegistry; edit the experiment module and re-export.
def allxy(target: str = 'Q1', reset_wait_us: float = 200.0, request_id: str = "") -> dict:
    """allxy: independent native protocol. Reuse request_id when retrying the same measurement."""
    from QickworkspaceV2.integrations.nvidia import execute
    return execute('allxy', {'target': target, 'reset_wait_us': reset_wait_us}, request_id=request_id or None)
