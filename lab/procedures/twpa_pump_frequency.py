"""Pump-frequency sweep with explicit instrument units and limits."""

from QickworkspaceV2 import InstrumentAxis
from QickworkspaceV2.analysis.transmission import normalize_transmission


def run(
    lab,
    frequencies_hz,
    *,
    read_frequency,
    set_frequency,
    minimum_hz,
    maximum_hz,
    resource_id,
    reference,
    target="Q1",
    **probe_parameters,
):
    axis = InstrumentAxis(
        "pump_frequency", "Hz", read_frequency, set_frequency, minimum_hz, maximum_hz, resource_id
    )
    measured = lab.scan_instrument("twpa_probe", axis, frequencies_hz, target=target, **probe_parameters)
    gain = normalize_transmission(measured, reference)
    gain.save(lab.store.directory(measured.run_id) / "relative_gain.h5")
    return measured, gain
