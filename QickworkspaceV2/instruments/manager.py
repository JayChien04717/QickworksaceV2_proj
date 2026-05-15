"""Notebook-friendly instrument registry and safety helper."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .base import InstrumentSpec, LimitMap, merge_limits


class TextReport:
    """Plain-text report that displays cleanly in notebooks and REPLs."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text

    def _repr_pretty_(self, printer, cycle: bool) -> None:
        printer.text("..." if cycle else self.text)


class BaseInstrumentManager:
    """
    Register lab instruments under friendly names and expose one control API.

    Examples
    --------
    >>> inst = BaseInstrumentManager()
    >>> inst.add_yoko("q1_flux", "USB0::...", limits={"current": (-3e-3, 3e-3)})
    >>> inst.add_sgs100a("pump", "TCPIP::192.168.0.10::INSTR")
    >>> inst.status
    """

    def __init__(self) -> None:
        self._instruments: dict[str, InstrumentSpec] = {}

    def add_instrument(
        self,
        name: str,
        driver: type | Callable[..., Any],
        address: str,
        *,
        kind: str | None = None,
        limits: Mapping[str, tuple[float, float]] | None = None,
        notes: str = "",
        **driver_kwargs: Any,
    ) -> Any:
        """Create a driver and register it by name."""

        if name in self._instruments:
            raise ValueError(f"Instrument name already exists: {name!r}")
        obj = driver(address, **driver_kwargs)
        driver_limits = self._driver_limits(obj)
        spec = InstrumentSpec(
            name=name,
            kind=kind or getattr(obj, "KIND", obj.__class__.__name__),
            address=self._driver_address(obj, address),
            driver=obj,
            limits=merge_limits(driver_limits, limits),
            notes=notes,
        )
        self._instruments[name] = spec
        return obj

    def add_yoko(
        self,
        name: str,
        address: str,
        *,
        limits: Mapping[str, tuple[float, float]] | None = None,
        voltage_ramp_step: float | None = None,
        current_ramp_step: float | None = None,
        ramp_interval: float | None = None,
        notes: str = "",
    ) -> Any:
        """Add a Yokogawa GS200 DC source."""

        import pyvisa as visa

        from .yoko import YOKOGS200

        rm = visa.ResourceManager()
        yoko = self.add_instrument(
            name,
            lambda addr: YOKOGS200(addr, rm),
            address,
            kind="yoko",
            limits=limits,
            notes=notes,
        )
        if voltage_ramp_step is not None:
            yoko.voltage_ramp_step = voltage_ramp_step
        if current_ramp_step is not None:
            yoko.current_ramp_step = current_ramp_step
        if ramp_interval is not None:
            yoko.ramp_interval = ramp_interval
        return yoko

    def add_sgs100a(
        self,
        name: str,
        address: str,
        *,
        limits: Mapping[str, tuple[float, float]] | None = None,
        notes: str = "",
    ) -> Any:
        """Add a Rohde & Schwarz SGS100A RF source."""

        from .sgs100a import RohdeSchwarzSGS100A

        return self.add_instrument(
            name,
            RohdeSchwarzSGS100A,
            address,
            kind="sgs100a",
            limits=limits,
            notes=notes,
        )

    def add_mg3692(
        self,
        name: str,
        address: str,
        *,
        limits: Mapping[str, tuple[float, float]] | None = None,
        notes: str = "",
    ) -> Any:
        """Add an Anritsu MG3692 RF source."""

        from .mg3692 import AnritsuMG3692

        return self.add_instrument(
            name,
            AnritsuMG3692,
            address,
            kind="mg3692",
            limits=limits,
            notes=notes,
        )

    # Notebook aliases matching the compact style often used at the bench.
    addyoko = add_yoko
    addsgs100a = add_sgs100a
    addsgs100 = add_sgs100a
    addmg3692 = add_mg3692

    def get(self, name: str) -> Any:
        return self._instruments[name].driver

    def spec(self, name: str) -> InstrumentSpec:
        return self._instruments[name]

    @property
    def names(self) -> list[str]:
        return list(self._instruments)

    @property
    def status(self) -> TextReport:
        """Return a printable status table for all registered instruments."""

        if not self._instruments:
            return TextReport("No instruments registered.")
        return TextReport(
            "\n".join(self._format_status_line(spec) for spec in self._instruments.values())
        )

    def help(self, name: str | None = None) -> TextReport:
        """Return output ranges and common commands."""

        if name is None:
            if not self._instruments:
                return TextReport("No instruments registered.")
            return TextReport("\n\n".join(str(self.help(item)) for item in self._instruments))

        spec = self.spec(name)
        lines = [
            f"{spec.kind}: {spec.name}",
            f"address: {spec.address}",
            "ranges:",
        ]
        if spec.limits:
            for param, (low, high) in spec.limits.items():
                lines.append(f"  {param}: {low:g} to {high:g}")
        else:
            lines.append("  no limits registered")
        snapshot = self._safe_snapshot(spec.driver)
        if snapshot:
            lines.append("snapshot:")
            for key, value in snapshot.items():
                lines.append(f"  {key}: {value}")
        if spec.notes:
            lines.append(f"notes: {spec.notes}")
        lines.append(f"common: inst.get({name!r}), inst.on({name!r}), inst.off({name!r})")
        return TextReport("\n".join(lines))

    def limits(self, name: str | None = None) -> dict[str, LimitMap] | LimitMap:
        if name is not None:
            return dict(self.spec(name).limits)
        return {item: dict(spec.limits) for item, spec in self._instruments.items()}

    def set(self, name: str, parameter: str, value: Any) -> None:
        """Set a driver property after checking registered limits."""

        spec = self.spec(name)
        self._validate_range(spec, parameter, value)
        if not hasattr(spec.driver, parameter):
            raise AttributeError(f"{name!r} has no settable parameter {parameter!r}")
        setattr(spec.driver, parameter, value)

    def set_yoko(self, name: str, value: float, mode: str = "current") -> None:
        spec = self.spec(name)
        if mode not in {"current", "voltage"}:
            raise ValueError("mode must be 'current' or 'voltage'")
        self._validate_range(spec, mode, value)
        driver = spec.driver
        if hasattr(driver, "mode"):
            driver.mode = mode
        setattr(driver, mode, value)

    def on(self, name: str) -> None:
        self.get(name).on()

    def off(self, name: str) -> None:
        self.get(name).off()

    def close(self, name: str | None = None) -> None:
        """Close one instrument or every registered instrument."""

        names = [name] if name is not None else list(self._instruments)
        for item in names:
            driver = self.get(item)
            close = getattr(driver, "close", None)
            if callable(close):
                close()

    def _validate_range(self, spec: InstrumentSpec, parameter: str, value: Any) -> None:
        if parameter not in spec.limits:
            return
        low, high = spec.limits[parameter]
        if not (low <= float(value) <= high):
            raise ValueError(
                f"{spec.name}.{parameter}={value:g} is outside allowed range "
                f"{low:g} to {high:g}"
            )

    def _format_status_line(self, spec: InstrumentSpec) -> str:
        driver = spec.driver
        output = self._safe_get(driver, "output")
        if output is None:
            output = self._safe_get(driver, "status")
        value = self._status_value(driver)
        return (
            f"{spec.kind}: {spec.name} address: {spec.address} | "
            f"output: {output if output is not None else 'unknown'} | "
            f"value: {value}"
        )

    def _status_value(self, driver: Any) -> str:
        get_value = getattr(driver, "GetValue", None)
        if callable(get_value):
            try:
                info = get_value()
                return f"{info.get('value')} {info.get('unit')}"
            except Exception as exc:
                return f"error({exc})"

        pieces = []
        for attr, unit in (("frequency", "Hz"), ("power", "dBm")):
            value = self._safe_get(driver, attr)
            if value is not None:
                pieces.append(f"{attr}={value} {unit}")
        return ", ".join(pieces) if pieces else "unknown"

    def _safe_snapshot(self, driver: Any) -> dict[str, Any]:
        snapshot = getattr(driver, "snapshot", None)
        if callable(snapshot):
            try:
                return dict(snapshot())
            except Exception:
                return {}
        return {}

    def _safe_get(self, driver: Any, attr: str) -> Any:
        try:
            value = getattr(driver, attr)
        except Exception:
            return None
        if callable(value):
            try:
                return value()
            except TypeError:
                return None
            except Exception:
                return None
        return value

    def _driver_limits(self, driver: Any) -> LimitMap:
        get_limits = getattr(driver, "get_limits", None)
        if callable(get_limits):
            return dict(get_limits())
        return dict(getattr(driver, "DEFAULT_LIMITS", {}))

    def _driver_address(self, driver: Any, fallback: str) -> str:
        for attr in ("resource_name", "VISAaddress", "address"):
            value = getattr(driver, attr, None)
            if value:
                return str(value)
        instrument = getattr(driver, "instrument", None)
        value = getattr(instrument, "resource_name", None)
        return str(value or fallback)


InstrumentManager = BaseInstrumentManager
