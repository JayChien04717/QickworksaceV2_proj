import pyvisa
from typing import Union, Tuple

from .base import RFSourceInstrument

ON_OFF_MAP = {
    "on": "1", "1": "1", "true": "1", 1: "1",
    "off": "0", "0": "0", "false": "0", 0: "0",
}
ON_OFF_MAP_INV = {"1": "on", "0": "off"}


class AnritsuMG3692(RFSourceInstrument):
    """
    Driver for the Anritsu MG3692 signal generator (property-based PyVISA API).

    Parameters
    ----------
    address : str
        VISA resource address or plain IP (auto-prefixed with TCPIP::).
    """

    KIND = "mg3692"
    MODEL = "Anritsu MG3692"
    DEFAULT_LIMITS = {
        "frequency": (2e9, 20e9),
        "power": (-120, 27),
    }

    def __init__(self, address: str) -> None:
        """Initialize the AnritsuMG3692 instance.

        Parameters
        ----------
        address : str
            Instrument resource address.

        Raises
        ------
        ValueError
            If the operation cannot be completed.
        """
        self.address = address
        self.rm = pyvisa.ResourceManager()
        if "::" not in address:
            self.resource_name = f"TCPIP::{address}::INSTR"
        else:
            self.resource_name = address
        try:
            self.instrument = self.rm.open_resource(self.resource_name)
            self.instrument.timeout = 5000
        except pyvisa.Error as e:
            self.instrument = None
            raise ValueError(f"Could not connect to {self.resource_name}. Error: {e}")

        self.instrument.read_termination = "\n"
        self.instrument.write_termination = "\n"
        self.connect_message()

    def connect_message(self) -> None:
        """Connect message."""
        if self.instrument:
            try:
                idn = self.instrument.query("*IDN?")
                print(f"Connected to: {idn.strip()}")
            except pyvisa.Error as e:
                print(f"Could not query IDN. Error: {e}")

    def idn(self) -> str:
        """Return the instrument identification string.

        Returns
        -------
        str
            Result of the operation.
        """
        return self.query("*IDN?")

    def close(self) -> None:
        """Close the operation."""
        if self.instrument:
            print(f"Disconnecting from {self.resource_name}")
            self.instrument.close()
            self.rm.close()

    def reconnect(self) -> None:
        """Return the reconnect result.

        Raises
        ------
        ConnectionError
            If the operation cannot be completed.
        """
        print(f"[mg3692] Reconnecting to {self.resource_name} ...")
        try:
            try:
                self.instrument.close()
            except Exception:
                pass
            self.instrument = self.rm.open_resource(self.resource_name)
            self.instrument.timeout = 5000
            self.instrument.read_termination  = "\n"
            self.instrument.write_termination = "\n"
            print("[mg3692] Reconnected.")
        except pyvisa.Error as e:
            raise ConnectionError(f"[mg3692] Reconnect failed: {e}") from e

    def write(self, cmd: str) -> None:
        """Write the operation.

        Parameters
        ----------
        cmd : str
            Instrument command string.
        """
        if self.instrument:
            try:
                self.instrument.write(cmd)
            except pyvisa.errors.VisaIOError as e:
                if e.error_code == pyvisa.constants.StatusCode.error_connection_lost:
                    self.reconnect()
                    self.instrument.write(cmd)
                else:
                    raise

    def query(self, cmd: str) -> str:
        """Query the instrument and return its response.

        Parameters
        ----------
        cmd : str
            Instrument command string.

        Returns
        -------
        str
            Result of the operation.
        """
        if self.instrument:
            try:
                return self.instrument.query(cmd).strip()
            except pyvisa.errors.VisaIOError as e:
                if e.error_code == pyvisa.constants.StatusCode.error_connection_lost:
                    self.reconnect()
                    return self.instrument.query(cmd).strip()
                else:
                    raise
        return ""

    def check_error(self) -> str:
        """Return the latest instrument error.

        Returns
        -------
        str
            Result of the operation.
        """
        err_msg = ""
        if self.instrument:
            err_msg = self.query("SYST:ERR?")
            print(f"Instrument Status: {err_msg}")
        return err_msg

    def get_limit(self, parameter: str) -> Tuple[float, float]:
        """Return limit.

        Parameters
        ----------
        parameter : str
            Value for ``parameter``.

        Returns
        -------
        Tuple[float, float]
            Result of the operation.

        Raises
        ------
        ValueError
            If the operation cannot be completed.
        """
        param_lower = parameter.lower()
        if param_lower in self.DEFAULT_LIMITS:
            return self.DEFAULT_LIMITS[param_lower]
        raise ValueError(f"Limits not defined for parameter '{parameter}'.")

    def get_limits(self) -> dict:
        """Return limits.

        Returns
        -------
        dict
            Result of the operation.
        """
        return dict(self.DEFAULT_LIMITS)

    def discover_limits(self) -> dict:
        """Discover limits.

        Returns
        -------
        dict
            Result of the operation.
        """
        limits = self.get_limits()
        queries = {
            "frequency": ("FREQ? MIN", "FREQ? MAX"),
            "power": ("POW? MIN", "POW? MAX"),
        }
        for key, (min_query, max_query) in queries.items():
            try:
                limits[key] = (float(self.query(min_query)), float(self.query(max_query)))
            except Exception:
                pass
        return limits

    def _map_and_write(self, cmd_template: str, value: Union[str, int, bool], name: str) -> None:
        """Return the map and write result.

        Parameters
        ----------
        cmd_template : str
            Value for ``cmd_template``.
        value : Union[str, int, bool]
            Value to apply.
        name : str
            Name of the target object.

        Raises
        ------
        ValueError
            If the operation cannot be completed.
        """
        try:
            mapped_val = ON_OFF_MAP[str(value).lower()]
            self.write(cmd_template.format(mapped_val))
        except KeyError:
            raise ValueError(f"Invalid {name} value: {value}. Use 'on' or 'off'.")

    def _query_and_map(self, cmd: str) -> str:
        """Return and map.

        Parameters
        ----------
        cmd : str
            Instrument command string.

        Returns
        -------
        str
            Result of the operation.
        """
        val = self.query(cmd)
        return ON_OFF_MAP_INV.get(val, f"unknown_val_{val}")

    @property
    def frequency(self) -> float:
        """Return the current frequency setting.

        Returns
        -------
        float
            Result of the operation.
        """
        try:
            return float(self.query("FREQ?"))
        except ValueError:
            return 0.0

    @frequency.setter
    def frequency(self, value: float) -> None:
        """Return the current frequency setting.

        Parameters
        ----------
        value : float
            Value to apply.
        """
        min_f, max_f = self.get_limit("frequency")
        if not (min_f <= value <= max_f):
            print(f"Warning: Frequency {value} Hz is outside normal range ({min_f}, {max_f})")
        self.write(f"FREQ {value} HZ")

    @property
    def power(self) -> float:
        """Return the current power setting.

        Returns
        -------
        float
            Result of the operation.
        """
        try:
            return float(self.query("POW?"))
        except ValueError:
            return -999.0

    @power.setter
    def power(self, value: float) -> None:
        """Return the current power setting.

        Parameters
        ----------
        value : float
            Value to apply.
        """
        min_p, max_p = self.get_limit("power")
        if not (min_p <= value <= max_p):
            print(f"Warning: Power {value} dBm is outside normal range ({min_p}, {max_p})")
        self.write(f"POW {value} DBM")

    @property
    def status(self) -> str:
        """Return the current status.

        Returns
        -------
        str
            Result of the operation.
        """
        return self._query_and_map("OUTP?")

    @status.setter
    def status(self, value: Union[str, int, bool]) -> None:
        """Return the current status.

        Parameters
        ----------
        value : Union[str, int, bool]
            Value to apply.
        """
        self._map_and_write("OUTP {}", value, "status")

    def output(self, state: Union[bool, None] = None) -> Union[bool, None]:
        """Return the current output setting.

        Parameters
        ----------
        state : Union[bool, None], default: None
            Value for ``state``.

        Returns
        -------
        Union[bool, None]
            Result of the operation.
        """
        if state is None:
            return self.status == "on"
        self.status = state

    def on(self) -> None:
        """Enable the instrument output."""
        self.output(True)

    def off(self) -> None:
        """Disable the instrument output."""
        self.output(False)

    def snapshot(self) -> dict:
        """Return the current instrument settings.

        Returns
        -------
        dict
            Result of the operation.
        """
        return {
            "output": self.status,
            "frequency": self.frequency,
            "power": self.power,
        }


MG3692Driver = AnritsuMG3692
