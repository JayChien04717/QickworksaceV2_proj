# Generated from ExperimentRegistry; edit the experiment module and re-export.
def time_of_flight(target: str = 'Q1', request_id: str = "") -> dict:
    """Raw decimated I/Q; one hardware rep and software averaging. Reuse request_id when retrying the same measurement."""
    from QickworkspaceV2.integrations.nvidia import execute
    return execute('time_of_flight', {'target': target}, request_id=request_id or None)
