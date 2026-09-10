"""TWPA bias sweep; set_bias must perform your instrument's ramp and settling."""

from QickworkspaceV2 import InstrumentAxis


def run(
    lab,
    bias_values,
    *,
    read_bias,
    set_bias,
    unit,
    minimum,
    maximum,
    resource_id,
    target="Q1",
    **probe_parameters,
):
    axis = InstrumentAxis("pump_bias", unit, read_bias, set_bias, minimum, maximum, resource_id)
    return lab.scan_instrument("twpa_probe", axis, bias_values, target=target, **probe_parameters)
