"""Portable metadata serialization; never pickle experiment data."""
import dataclasses
import hashlib
import json
from pathlib import Path
import numpy as np


def jsonable(value):
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return {"real": jsonable(value.real), "imag": jsonable(value.imag)}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "start") and hasattr(value, "spans"):
        return {"qick_param": {"start": value.start, "spans": value.spans}}
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def dumps(value):
    return json.dumps(jsonable(value), sort_keys=True, ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()
