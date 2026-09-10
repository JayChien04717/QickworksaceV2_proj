"""Reference-normalized transmission with strict frequency-axis alignment."""

import numpy as np
from QickworkspaceV2.data.models import ExperimentData, TraceData


def normalize_transmission(measured, reference):
    traces = {}
    if measured.targets != reference.targets:
        raise ValueError("Measurement and reference must use the same target order")
    for q, trace in measured.traces.items():
        ref = reference[q]
        if trace.dims[-len(ref.dims) :] != ref.dims:
            raise ValueError("Reference dimensions must match the inner measurement axes")
        if any(not np.array_equal(trace.coords[d], ref.coords[d]) for d in ref.dims):
            raise ValueError("Reference coordinates differ; acquire the same actual frequency grid")
        if not np.isfinite(ref.iq).all() or np.any(abs(ref.iq) <= np.finfo(float).eps):
            raise ValueError("Reference contains zero or nonfinite transmission")
        traces[q] = TraceData(
            trace.iq / ref.iq,
            trace.dims,
            trace.coords,
            trace.units,
            metadata={**trace.metadata, "quantity": "transmission ratio"},
        )
    return ExperimentData(
        measured.experiment,
        traces,
        {
            **measured.metadata,
            "iq_process": "db",
            "source_run_id": measured.run_id,
            "reference_run_id": reference.run_id,
            "quantity": "relative transmission",
        },
    )
