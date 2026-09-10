# Generated from ExperimentRegistry; edit the experiment module and re-export.
def state_tomography(target: str = 'Q1', reset_wait_us: float = 200.0, preparation: str = 'plus', request_id: str = "") -> dict:
    """state tomography: independent native protocol. Reuse request_id when retrying the same measurement."""
    from QickworkspaceV2.integrations.nvidia import execute
    return execute('state_tomography', {'target': target, 'reset_wait_us': reset_wait_us, 'preparation': preparation}, request_id=request_id or None)
