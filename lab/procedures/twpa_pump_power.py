"""Pump-power sweep. The caller configures pump frequency and output state."""

from QickworkspaceV2 import InstrumentAxis
from QickworkspaceV2.analysis.transmission import normalize_transmission


def run(
    lab,
    powers_dbm,
    *,
    read_power,
    set_power,
    minimum_dbm,
    maximum_dbm,
    resource_id,
    reference,
    target="Q1",
    **probe_parameters,
):
    axis = InstrumentAxis("pump_power", "dBm", read_power, set_power, minimum_dbm, maximum_dbm, resource_id)
    measured = lab.scan_instrument("twpa_probe", axis, powers_dbm, target=target, **probe_parameters)
    gain = normalize_transmission(measured, reference)
    gain.save(lab.store.directory(measured.run_id) / "relative_gain.h5")
    return measured, gain
