"""Apply numerical fitters to SDK trace selections."""
import numpy as np
from labtools.fitting import FIT_FUNCTIONS, FitResult, fit_resonator

def analyze_traces(result, fitter, *, signal="abs", event=0, **fit_options):
    """Apply a specific fit function to each target's selected trace."""
    if not callable(fitter):
        raise TypeError("Pass a fit function, such as fit_exponential")
    settings = dict(result.metadata.get("parameters", {}).get("fit", {}))
    settings.update(fit_options)
    selection = settings.pop("model", None)
    if selection is not None:
        fitter = FIT_FUNCTIONS[selection]
    label = next((name for name, fn in FIT_FUNCTIONS.items() if fn is fitter), fitter.__name__)
    fits = {}
    for q, trace in result.traces.items():
        dims = [d for d in trace.dims if d != "readout"]
        if len(dims) != 1 or "readout" not in trace.dims:
            fits[q] = FitResult(
                label, False, message="Select one sweep axis and one readout event before fitting"
            )
            continue
        dim = dims[0]
        values = (
            trace.iq
            if fitter is fit_resonator
            else trace.signal(signal, rotation_deg=trace.metadata.get("rotation_deg", 0))
        )
        y = np.take(values, event, axis=trace.dims.index("readout"))
        fits[q] = fitter(trace.coords[dim], y, unit=trace.units.get(dim, ""), **settings)
    return fits
