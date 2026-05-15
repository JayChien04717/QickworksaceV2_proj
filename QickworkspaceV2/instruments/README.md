# Instruments

This folder contains PyVISA instrument drivers and the notebook-friendly
`BaseInstrumentManager` registry.

The main idea is:

- Drivers know how to talk to one hardware model.
- `BaseInstrumentManager` knows which physical instruments are connected,
  what names they have in the notebook, and what safety limits to enforce.

## Common Instrument Language

All source instruments should be controlled with the same outer vocabulary:

```python
inst.on(name)
inst.off(name)
inst.value(name)
inst.set_value(name, value, mode=None)
inst.set_values(targets)
inst.status
```

The meaning of `value` depends on the instrument type:

- Yoko/DC source: `current` or `voltage`
- RF source: `power`

RF frequency is not treated as the main output value. Set it explicitly:

```python
inst.set("pump", "frequency", 6.0e9)
```

Yoko/DC sources also expose ramp settings:

```python
inst.configure_ramp(
    "q1_flux",
    current_step=1e-8,
    voltage_step=1e-5,
    interval=0.01,
)

inst.ramp("q1_flux")
```

## Quick Start

```python
from QickworkspaceV2.instruments import BaseInstrumentManager

inst = BaseInstrumentManager()

inst.add_yoko(
    "q1_flux",
    "GPIB0::1::INSTR",
    limits={
        "current": (-3e-3, 3e-3),
        "voltage": (-1.0, 1.0),
    },
    current_ramp_step=1e-8,
    voltage_ramp_step=1e-5,
    ramp_interval=0.01,
)

inst.add_sgs100(
    "pump",
    "192.168.0.10",
    limits={
        "frequency": (1e6, 20e9),
        "power": (-80, 10),
    },
)

inst.status
```

Example status output:

```text
yoko: q1_flux address: GPIB0::1::INSTR | output: on | value: 0.0 A
sgs100a: pump address: TCPIP::192.168.0.10::INSTR | output: off | value: frequency=6000000000.0 Hz, power=-40.0 dBm
```

## Multiple Instruments of the Same Type

Use a unique name for each physical instrument. For example, two Yokogawa GS200
sources can be registered as `q1_flux` and `q2_flux`:

```python
inst.add_yoko(
    "q1_flux",
    "GPIB0::1::INSTR",
    limits={"current": (-3e-3, 3e-3)},
)

inst.add_yoko(
    "q2_flux",
    "GPIB0::2::INSTR",
    limits={"current": (-2e-3, 2e-3)},
)
```

Control a specific Yoko by name:

```python
inst.set_value("q1_flux", 0.5e-3, mode="current")
inst.set_value("q2_flux", -0.2e-3, mode="current")
```

Set multiple independent instruments from one notebook cell. This runs in
parallel by default, so two Yoko/DC flux ramps can happen at the same time:

```python
inst.set_values({
    "q1_flux": {"value": 0.5e-3, "mode": "current"},
    "q2_flux": {"value": -0.2e-3, "mode": "current"},
})
```

`set_values(...)` is intentionally for Yoko/DC sources only. RF sources usually
do not need slow ramp synchronization; use `inst.set_value("pump", -40)` or
`inst.set("pump", "frequency", 6.0e9)` for RF control.

Get one raw driver when you need model-specific methods:

```python
yoko_q1 = inst.get("q1_flux")
yoko_q2 = inst.get("q2_flux")
```

Prefer `inst.set_value(...)` for normal use because it checks the registered
limits before writing to hardware.

## Common Manager API

```python
inst.names
inst.status
inst.help()
inst.help("q1_flux")
inst.limits()
inst.limits("pump")
inst.get("pump")
```

Set common parameters with limit checks:

```python
inst.set_value("q1_flux", 0.0, mode="current")
inst.set_value("pump", -40)
inst.set("pump", "frequency", 6.0e9)
```

Output control:

```python
inst.on("pump")
inst.off("pump")
inst.off("q1_flux")
```

Close connections:

```python
inst.close("pump")
inst.close()
```

## Use With BaseExperiment Liveplot

The old liveplot Yoko path still accepts a VISA address:

```python
expt.run(py_avg=5, yoko_inst_addr="GPIB0::1::INSTR", yoko_value=flux_values)
```

The new optional path uses `BaseInstrumentManager` names instead:

```python
inst.add_yoko("q1_flux", "GPIB0::1::INSTR", limits={"current": (-3e-3, 3e-3)})

expt.run(
    py_avg=5,
    instrument_manager=inst,
    yoko_name="q1_flux",
    yoko_value=flux_values,
    yoko_mode="current",
)
```

Shortcut form:

```python
expt.run(
    py_avg=5,
    instrument_manager=inst,
    yoko_inst="q1_flux",
    yoko_value=flux_values,
    yoko_mode="current",
)
```

When the manager path is used, the registered safety limits and ramp settings
from `BaseInstrumentManager` are used during the Yoko sweep.

## Supported Drivers

Current drivers:

- `yoko.py`: Yokogawa GS200 property-style driver
- `YOKOGS200.py`: Legacy Yokogawa GS200 `SetVoltage` / `SetCurrent` driver
- `sgs100a.py`: Rohde & Schwarz SGS100A RF source
- `mg3692.py`: Anritsu MG3692 RF source

For SGS100A and MG3692, you can pass either a full VISA resource string or a
plain IP address. Plain IPs are converted to `TCPIP::<ip>::INSTR`.

Recommended new code should use:

```python
from QickworkspaceV2.instruments import BaseInstrumentManager
```

and register instruments through `add_yoko`, `add_sgs100`, `add_sgs100a`, or
`add_mg3692`.

## Driver Style

New drivers should expose a small common API when possible:

```python
class MyInstrument:
    KIND = "my_kind"
    MODEL = "Vendor Model"
    DEFAULT_LIMITS = {
        "frequency": (1e6, 20e9),
        "power": (-120, 10),
    }

    def __init__(self, address: str):
        self.address = address

    def idn(self) -> str:
        ...

    def close(self) -> None:
        ...

    def get_limits(self) -> dict:
        return dict(self.DEFAULT_LIMITS)

    def snapshot(self) -> dict:
        return {
            "output": self.status,
            "frequency": self.frequency,
            "power": self.power,
        }
```

For source instruments, also provide:

```python
def on(self) -> None:
    ...

def off(self) -> None:
    ...
```

RF source drivers should use property names:

```python
source.frequency = 6.0e9
source.power = -40
source.status
```

DC source drivers should use:

```python
yoko.mode
yoko.current
yoko.voltage
yoko.output
yoko.value
yoko.ramp_rate
```

## Safety Limits

Driver `DEFAULT_LIMITS` are hardware-oriented defaults. Manager-level `limits`
are lab safety limits and override the defaults:

```python
inst.add_sgs100(
    "pump",
    "192.168.0.10",
    limits={"power": (-80, 0)},
)
```

Then this is allowed:

```python
inst.set("pump", "power", -20)
```

but this raises `ValueError`:

```python
inst.set("pump", "power", 10)
```

Use narrow lab limits for flux lines and pump sources. Do not rely only on the
instrument's full hardware range.
