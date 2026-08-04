from __future__ import annotations

import pyvisa as visa
import numpy as np
import time
from tqdm.auto import tqdm
from typing import Literal, Union

from .base import DCSourceInstrument
from ..tools.units import auto_unit


class YOKOGS200(DCSourceInstrument):
    """
    PyVISA driver for the Yokogawa GS200 DC Source (property-based API).

    Ramping is built into the 'voltage' and 'current' property setters
    for safe operation.
    """

    KIND = "yoko"
    MODEL = "Yokogawa GS200"
    DEFAULT_LIMITS = {
        "voltage": (-10.0, 10.0),
        "current": (-0.01, 0.01),
    }

    def __init__(self, VISAaddress: str, rm: visa.ResourceManager):
        """Initialize the YOKOGS200 instance.

        Parameters
        ----------
        VISAaddress : str
            Value for ``VISAaddress``.
        rm : visa.ResourceManager
            Value for ``rm``.

        Raises
        ------
        ConnectionError
            If the operation cannot be completed.
        """
        self.VISAaddress = VISAaddress
        self.rm = rm
        try:
            self.session = rm.open_resource(VISAaddress)
            self.session.read_termination = "\n"
            self.session.write_termination = "\n"
        except visa.Error as ex:
            raise ConnectionError(f"Couldn't connect to '{VISAaddress}'. Error: {ex}")

        self.voltage_ramp_step = 1e-4
        self.current_ramp_step = 1e-8
        self.ramp_interval = 0.01
        self.show_ramp_progress = True
        self.ramp_progress_leave = False

        self._output_map = {
            "on": "1", "1": "1", 1: "1", True: "1",
            "off": "0", "0": "0", 0: "0", False: "0",
        }
        self._output_map_inv = {"1": "on", "0": "off"}
        self._mode_map = {"voltage": "VOLT", "current": "CURR"}
        self._mode_map_inv = {"VOLT": "voltage", "CURR": "current"}

        self.connect_message()

    def connect_message(self) -> None:
        """Connect message."""
        try:
            idn = self.session.query("*IDN?")
            print(f"Connected to: {idn.strip()}")
        except visa.Error as e:
            print(f"Could not query IDN. Error: {e}")

    def idn(self) -> str:
        """Return the instrument identification string.

        Returns
        -------
        str
            Result of the operation.
        """
        return self.session.query("*IDN?").strip()

    def get_limits(self) -> dict:
        """Return limits.

        Returns
        -------
        dict
            Result of the operation.
        """
        return dict(self.DEFAULT_LIMITS)

    def discover_limits(self) -> dict:
        # Keep Yoko limits conservative by default. Hardware full scale is not
        # necessarily safe for a flux line, so lab limits should override these.
        """Discover limits.

        Returns
        -------
        dict
            Result of the operation.
        """
        return self.get_limits()

    @property
    def ramp_rate(self) -> dict:
        """Return the ramp rate result.

        Returns
        -------
        dict
            Result of the operation.
        """
        return {
            "voltage_step": self.voltage_ramp_step,
            "current_step": self.current_ramp_step,
            "interval": self.ramp_interval,
        }

    def configure_ramp(
        self,
        *,
        voltage_step: float | None = None,
        current_step: float | None = None,
        interval: float | None = None,
    ) -> None:
        """Return the configure ramp result.

        Parameters
        ----------
        voltage_step : float | None, default: None
            Value for ``voltage_step``.
        current_step : float | None, default: None
            Value for ``current_step``.
        interval : float | None, default: None
            Value for ``interval``.
        """
        if voltage_step is not None:
            self.voltage_ramp_step = voltage_step
        if current_step is not None:
            self.current_ramp_step = current_step
        if interval is not None:
            self.ramp_interval = interval

    def close(self) -> None:
        """Close the operation."""
        print(f"Disconnecting from {self.VISAaddress}")
        self.session.close()

    @property
    def output(self) -> str:
        """Return the current output setting.

        Returns
        -------
        str
            Result of the operation.
        """
        val = self.session.query("OUTPut?").strip()
        return self._output_map_inv.get(val, f"unknown_state_{val}")

    @output.setter
    def output(self, value: Union[str, int, bool]):
        """Return the current output setting.

        Parameters
        ----------
        value : Union[str, int, bool]
            Value to apply.

        Raises
        ------
        ValueError
            If the operation cannot be completed.
        """
        val_str = str(value).lower()
        cmd_val = self._output_map.get(val_str)
        if cmd_val is None:
            raise ValueError(f"Invalid output value: {value}. Use 'on', 'off', 1, or 0.")
        self.session.write(f"OUTPut {cmd_val}")

    def on(self) -> None:
        """Enable the instrument output."""
        self.output = "on"

    def off(self) -> None:
        """Disable the instrument output."""
        self.output = "off"

    @property
    def mode(self) -> str:
        """Return the mode result.

        Returns
        -------
        str
            Result of the operation.
        """
        val = self.session.query("SOURce:FUNCtion?").strip()
        return self._mode_map_inv.get(val, f"unknown_mode_{val}")

    @mode.setter
    def mode(self, value: Literal["voltage", "current"]):
        """Return the mode result.

        Parameters
        ----------
        value : Literal['voltage', 'current']
            Value to apply.

        Raises
        ------
        ValueError
            If the operation cannot be completed.
        """
        val_str = str(value).lower()
        cmd_val = self._mode_map.get(val_str)
        if cmd_val is None:
            raise ValueError(f"Invalid mode: {value}. Use 'voltage' or 'current'.")
        self.session.write(f"SOURce:FUNCtion {cmd_val}")

    @property
    def level(self) -> float:
        """Return the level result.

        Returns
        -------
        float
            Result of the operation.
        """
        result = self.session.query("SOURce:LEVel?")
        return float(result.strip())

    @level.setter
    def level(self, value: float):
        """Return the level result.

        Parameters
        ----------
        value : float
            Value to apply.
        """
        self.session.write(f":SOURce:LEVel:AUTO {value:.8f}")

    @property
    def voltage(self) -> float:
        """Return the voltage result.

        Returns
        -------
        float
            Result of the operation.
        """
        self.mode = "voltage"
        return self.level

    @voltage.setter
    def voltage(self, new_voltage: float):
        """Return the voltage result.

        Parameters
        ----------
        new_voltage : float
            Value for ``new_voltage``.
        """
        self.mode = "voltage"
        self._ramp_to(new_voltage, unit="V", step=self.voltage_ramp_step)

    @property
    def current(self) -> float:
        """Return the current result.

        Returns
        -------
        float
            Result of the operation.
        """
        self.mode = "current"
        return self.level

    @current.setter
    def current(self, new_current: float):
        """Return the current result.

        Parameters
        ----------
        new_current : float
            Value for ``new_current``.
        """
        self.mode = "current"
        self._ramp_to(new_current, unit="A", step=self.current_ramp_step)

    def _ramp_to(self, target: float, *, unit: str, step: float) -> None:
        """Return the ramp to result.

        Parameters
        ----------
        target : float
            Value for ``target``.
        unit : str
            Value for ``unit``.
        step : float
            Value for ``step``.
        """
        start = self.level
        stop = target
        steps = max(1, round(abs(stop - start) / step))
        values = np.linspace(start, stop, num=steps + 1, endpoint=True)
        self.on()
        desc = (
            f"Yoko {self.mode} "
            f"{self._format_ramp_value(start, unit)} -> "
            f"{self._format_ramp_value(stop, unit)}"
        )
        iterator = values
        if self.show_ramp_progress:
            iterator = tqdm(
                values,
                desc=desc,
                unit="step",
                leave=self.ramp_progress_leave,
                dynamic_ncols=True,
            )
        for value in iterator:
            self.level = value
            if self.show_ramp_progress:
                iterator.set_postfix_str(f"now={self._format_ramp_value(value, unit)}")
            time.sleep(self.ramp_interval)

    @staticmethod
    def _format_ramp_value(value: float, unit: str) -> str:
        """Format ramp value.

        Parameters
        ----------
        value : float
            Value to apply.
        unit : str
            Value for ``unit``.

        Returns
        -------
        str
            Result of the operation.
        """
        scaled = auto_unit(value, unit)
        return f"{float(scaled['value']):.4g} {scaled['unit']}"

    def GetValue(self) -> dict:
        """Return the GetValue result.

        Returns
        -------
        dict
            Result of the operation.
        """
        current_mode = self.mode
        current_level = self.level
        if current_mode == "voltage":
            return dict(unit="V", value=current_level)
        else:
            return dict(unit="A", value=current_level)

    @property
    def value(self) -> dict:
        """Return the value result.

        Returns
        -------
        dict
            Result of the operation.
        """
        return self.GetValue()

    def snapshot(self) -> dict:
        """Return the current instrument settings.

        Returns
        -------
        dict
            Result of the operation.
        """
        value = self.GetValue()
        return {
            "output": self.output,
            "mode": self.mode,
            "value": value["value"],
            "unit": value["unit"],
            "voltage_ramp_step": self.voltage_ramp_step,
            "current_ramp_step": self.current_ramp_step,
            "ramp_interval": self.ramp_interval,
            "show_ramp_progress": self.show_ramp_progress,
        }
