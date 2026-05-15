import pyvisa as visa
import numpy as np
import time
from tqdm.auto import tqdm

try:
    from ..tools.system_tool import auto_unit
except Exception:
    def auto_unit(value, base_unit=""):
        prefixes = {
            -12: "p", -9: "n", -6: "u", -3: "m",
            0: "", 3: "k", 6: "M", 9: "G",
        }
        arr = np.array(value, dtype=float)
        maxval = np.max(np.abs(arr))
        if maxval == 0:
            exp = 0
        else:
            exp = int(np.floor(np.log10(maxval) / 3) * 3)
            exp = max(min(exp, 9), -12)
        return {"unit": f"{prefixes[exp]}{base_unit}", "value": arr / (10**exp)}


class YOKOGS200:
    """Legacy Yokogawa GS200 driver with SetVoltage/GetVoltage API."""

    def __init__(self, VISAaddress, rm):
        self.VISAaddress = VISAaddress
        try:
            self.session = rm.open_resource(VISAaddress)
        except visa.Error as ex:
            import sys
            sys.stderr.write("Couldn't connect to '%s', exiting now..." % VISAaddress)
            sys.exit()
        self.voltage_ramp_step = 1e-5
        self.current_ramp_step = 1e-8
        self.ramp_interval = 0.01
        self.show_ramp_progress = True
        self.ramp_progress_leave = False

    def OutputOn(self):
        self.session.write("OUTPut 1")

    def OutputOff(self):
        self.session.write("OUTPut 0")

    def SetVoltage(self, voltage):
        start = self.GetVoltage()
        stop = voltage
        steps = max(1, round(abs(stop - start) / self.voltage_ramp_step))
        tempvolts = np.linspace(start, stop, num=steps + 1, endpoint=True)
        self.OutputOn()
        iterator = self._ramp_iterator(tempvolts, start, stop, "V", "voltage")
        for tempvolt in iterator:
            self.session.write(":SOURce:LEVel:AUTO %.8f" % tempvolt)
            self._update_ramp_progress(iterator, tempvolt, "V")
            time.sleep(self.ramp_interval)

    def SetCurrent(self, current):
        start = self.GetCurrent()
        stop = current
        steps = max(1, round(abs(stop - start) / self.current_ramp_step))
        tempcurrents = np.linspace(start, stop, num=steps)
        self.OutputOn()
        iterator = self._ramp_iterator(tempcurrents, start, stop, "A", "current")
        for tempcurrent in iterator:
            self.session.write(":SOURce:LEVel:AUTO %.8f" % tempcurrent)
            self._update_ramp_progress(iterator, tempcurrent, "A")
            time.sleep(self.ramp_interval)

    def _ramp_iterator(self, values, start, stop, unit, label):
        if not self.show_ramp_progress:
            return values
        desc = (
            f"Yoko {label} "
            f"{self._format_ramp_value(start, unit)} -> "
            f"{self._format_ramp_value(stop, unit)}"
        )
        return tqdm(
            values,
            desc=desc,
            unit="step",
            leave=self.ramp_progress_leave,
            dynamic_ncols=True,
        )

    def _update_ramp_progress(self, iterator, value, unit):
        if self.show_ramp_progress and hasattr(iterator, "set_postfix_str"):
            iterator.set_postfix_str(f"now={self._format_ramp_value(value, unit)}")

    @staticmethod
    def _format_ramp_value(value, unit):
        scaled = auto_unit(value, unit)
        return f"{float(scaled['value']):.4g} {scaled['unit']}"

    def SetMode(self, mode):
        import sys
        if not (mode == "voltage" or mode == "current"):
            sys.stderr.write("Unknown output mode %s." % mode)
            return
        self.session.write("SOURce:FUNCtion %s" % mode)

    def GetVoltage(self):
        self.session.write("SOURce:FUNCtion VOLTage")
        self.session.write("SOURce:LEVel?")
        result = self.session.read()
        return float(result.rstrip("\n"))

    def GetCurrent(self):
        self.session.write("SOURce:FUNCtion CURRent")
        self.session.write("SOURce:LEVel?")
        result = self.session.read()
        return float(result.rstrip("\n"))

    def GetValue(self):
        mode = self.GetMode()
        if mode == "voltage":
            value = self.GetVoltage()
            return dict(unit="V", value=value)
        else:
            value = self.GetCurrent()
            return dict(unit="A", value=value)

    def GetMode(self):
        self.session.write("SOURce:FUNCtion?")
        result = self.session.read()
        result = result.rstrip("\n")
        if result == "VOLT":
            return "voltage"
        else:
            return "current"
