"""Signal summaries for maps without a supported physical fit model."""

import numpy as np
from QickworkspaceV2.data.models import FitResult


def summarize_traces(result):
    fits = {}
    for q, trace in result.traces.items():
        values = np.abs(trace.iq)
        if not values.size or not np.isfinite(values).all():
            fits[q] = FitResult("signal_summary", False, message="No finite IQ data")
            continue
        fits[q] = FitResult("signal_summary", False,
                            {"min_amplitude": float(values.min()), "max_amplitude": float(values.max()),
                             "amplitude_span": float(np.ptp(values))},
                            units={"min_amplitude": "ADC", "max_amplitude": "ADC", "amplitude_span": "ADC"},
                            message="Data summary only; no physical calibration inferred from this map")
    return fits
