"""Resonator spectrum against a measured DC-source setpoint."""

from QickworkspaceV2 import InstrumentAxis


def run(
    lab,
    values,
    *,
    read_bias,
    set_bias,
    unit,
    minimum,
    maximum,
    resource_id,
    target="Q1",
    **spectrum_parameters,
):
    axis = InstrumentAxis("dc_bias", unit, read_bias, set_bias, minimum, maximum, resource_id)
    return lab.scan_instrument("resonator_spec", axis, values, target=target, **spectrum_parameters)
