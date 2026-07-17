"""Small, dependency-light helpers for display units."""

import numpy as np


def auto_unit(value, base_unit=""):
    """Scale numeric values to the most appropriate SI metric prefix."""
    prefixes = {
        -12: "p",
        -9: "n",
        -6: "u",
        -3: "m",
        0: "",
        3: "k",
        6: "M",
        9: "G",
    }
    values = np.asarray(value, dtype=float)
    maximum = np.max(np.abs(values))
    if maximum == 0:
        exponent = 0
    else:
        exponent = int(np.floor(np.log10(maximum) / 3) * 3)
        exponent = max(min(exponent, 9), -12)
    return {
        "unit": f"{prefixes[exponent]}{base_unit}",
        "value": values / (10**exponent),
    }


__all__ = ["auto_unit"]
