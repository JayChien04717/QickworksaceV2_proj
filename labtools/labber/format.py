"""Labber HDF5 schema initialization adapted from the supplied Labber_saver.py.

Uses h5py directly; no Labber installation or V1 result model is required.
"""

import os
import time
import h5py
import numpy as np

_STEP_NAME_API = "Step index API"
vlen_bytes = h5py.special_dtype(vlen=bytes)
interface_enum = h5py.special_dtype(
    enum=(
        np.int16,
        {"GPIB": 0, "None": 7, "Other": 6, "PXI": 3, "Serial": 4, "TCPIP": 1, "USB": 2, "VISA": 5},
    )
)
startup_enum = h5py.special_dtype(enum=(np.int16, {"Do nothing": 2, "Get config": 1, "Set config": 0}))
range_type_enum = h5py.special_dtype(enum=(np.int16, {"Center - Span": 2, "Single": 0, "Start - Stop": 1}))
step_type_enum = h5py.special_dtype(enum=(np.int16, {"Fixed # of pts": 1, "Fixed step": 0}))
interp_enum = h5py.special_dtype(
    enum=(np.int16, {"Linear": 0, "Log": 1, "Log, #/decade": 2, "Lorentzian": 3})
)
step_unit_enum = h5py.special_dtype(enum=(np.int16, {"Instrument": 0, "Physical": 1}))
after_last_enum = h5py.special_dtype(
    enum=(np.int16, {"Goto first point": 0, "Goto value...": 2, "Stay at final": 1})
)
sweep_mode_enum = h5py.special_dtype(enum=(np.int16, {"Between points": 1, "Continuous": 2, "Off": 0}))


def _to_bytes(s):
    if isinstance(s, str):
        return s.encode("utf-8")
    return s


def create_file(name, log_channels, step_channels):
    base, ext = os.path.splitext(name)
    if ext.lower() != ".hdf5":
        name = base + ".hdf5"
    resolved_path = os.path.abspath(name)
    parent_dir = os.path.dirname(resolved_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    for ch in log_channels:
        if "vector" not in ch:
            ch["vector"] = True
    bAllVector = all([ch.get("vector", True) for ch in log_channels])
    actual_step_channels = list(step_channels)
    if bAllVector:
        step_index_api_ch = {"name": _STEP_NAME_API, "values": np.array([1.0])}
        actual_step_channels = [step_index_api_ch] + actual_step_channels
    with h5py.File(resolved_path, "w") as f:
        step_dims = [len(ch["values"]) for ch in actual_step_channels]
        f.attrs["Step dimensions"] = step_dims
        f.attrs["arm_trig_mode"] = False
        f.attrs["comment"] = b""
        f.attrs["creation_time"] = time.time()
        f.attrs["hardware_loop"] = False
        f.attrs["log_name"] = _to_bytes(os.path.splitext(os.path.basename(resolved_path))[0])
        f.attrs["log_parallel"] = True
        f.attrs["logger_mode"] = False
        f.attrs["time_per_point"] = 0.0
        f.attrs["trig_channel"] = b""
        f.attrs["version"] = b"1.8.6"
        f.attrs["wait_between"] = 0.01
        tags_grp = f.create_group("Tags")
        tags_grp.attrs["Project"] = np.array([b""], dtype=object)
        tags_grp.attrs["Tags"] = np.array([])
        tags_grp.attrs["User"] = np.array([b""], dtype=object)
        f.create_group("Settings")
        inst_cfg_grp = f.create_group("Instrument config")
        dtype_instruments = np.dtype(
            [
                ("hardware", vlen_bytes),
                ("version", vlen_bytes),
                ("id", vlen_bytes),
                ("model", vlen_bytes),
                ("name", vlen_bytes),
                ("interface", interface_enum),
                ("address", vlen_bytes),
                ("server", vlen_bytes),
                ("startup", startup_enum),
                ("lock", "?"),
                ("show_advanced", "?"),
                ("Timeout", "<f8"),
                ("Term. character", vlen_bytes),
                ("Send end on write", "?"),
                ("Lock VISA resource", "?"),
                ("Suppress end bit termination on read", "?"),
                ("Use specific TCP port", "?"),
                ("TCP port", "<f8"),
                ("Use VICP protocol", "?"),
                ("Baud rate", "<f8"),
                ("Data bits", "<f8"),
                ("Stop bits", "<f8"),
                ("Parity", vlen_bytes),
                ("GPIB board number", "<f8"),
                ("Send GPIB go to local at close", "?"),
                ("PXI chassis", "<f8"),
                ("Run in 32-bit mode", "?"),
            ]
        )
        step_inst_id = b"Generic - GPIB: , Step channels at localhost"
        log_inst_id = b"Generic - GPIB: , Log channels at localhost"
        inst_data = [
            (
                b"Generic",
                b"1.0",
                step_inst_id,
                b"",
                b"Step channels",
                0,
                b"",
                b"",
                0,
                False,
                False,
                10.0,
                b"Auto",
                True,
                False,
                False,
                False,
                0.0,
                False,
                9600.0,
                8.0,
                1.0,
                b"No parity",
                0.0,
                False,
                1.0,
                False,
            ),
            (
                b"Generic",
                b"1.0",
                log_inst_id,
                b"",
                b"Log channels",
                0,
                b"",
                b"",
                0,
                False,
                False,
                10.0,
                b"Auto",
                True,
                False,
                False,
                False,
                0.0,
                False,
                9600.0,
                8.0,
                1.0,
                b"No parity",
                0.0,
                False,
                1.0,
                False,
            ),
        ]
        f.create_dataset("Instruments", data=np.array(inst_data, dtype=dtype_instruments))
        dtype_channels = np.dtype(
            [
                ("name", vlen_bytes),
                ("instrument", vlen_bytes),
                ("quantity", vlen_bytes),
                ("unitPhys", vlen_bytes),
                ("unitInstr", vlen_bytes),
                ("gain", "<f8"),
                ("offset", "<f8"),
                ("amp", "<f8"),
                ("highLim", "<f8"),
                ("lowLim", "<f8"),
                ("outputChannel", vlen_bytes),
                ("limit_action", vlen_bytes),
                ("limit_run_script", "?"),
                ("limit_script", vlen_bytes),
                ("use_log_interval", "?"),
                ("log_interval", "<f8"),
                ("limit_run_always", "?"),
            ]
        )
        chn_data = []
        for ch in actual_step_channels:
            name_bytes = _to_bytes(ch["name"])
            unit_bytes = _to_bytes(ch.get("unit", ""))
            chn_data.append(
                (
                    name_bytes,
                    step_inst_id,
                    name_bytes,
                    unit_bytes,
                    unit_bytes,
                    1.0,
                    0.0,
                    1.0,
                    float("inf"),
                    float("-inf"),
                    b"",
                    b"Nothing",
                    False,
                    b"",
                    False,
                    1.0,
                    False,
                )
            )
        for ch in log_channels:
            name_bytes = _to_bytes(ch["name"])
            unit_bytes = _to_bytes(ch.get("unit", ""))
            is_vector = ch.get("vector", True)
            high_lim = 0.0 if is_vector else float("inf")
            low_lim = 0.0 if is_vector else float("-inf")
            chn_data.append(
                (
                    name_bytes,
                    log_inst_id,
                    name_bytes,
                    unit_bytes,
                    unit_bytes,
                    1.0,
                    0.0,
                    1.0,
                    high_lim,
                    low_lim,
                    b"",
                    b"Nothing",
                    False,
                    b"",
                    False,
                    1.0,
                    False,
                )
            )
        f.create_dataset("Channels", data=np.array(chn_data, dtype=dtype_channels))
        dtype_log_list = np.dtype([("channel_name", vlen_bytes)])
        log_list_data = [(_to_bytes(ch["name"]),) for ch in log_channels]
        f.create_dataset("Log list", data=np.array(log_list_data, dtype=dtype_log_list))
        dtype_step_list = np.dtype(
            [
                ("channel_name", vlen_bytes),
                ("step_unit", step_unit_enum),
                ("wait_after", "<f8"),
                ("after_last", after_last_enum),
                ("final_value", "<f8"),
                ("use_relations", "?"),
                ("equation", vlen_bytes),
                ("show_advanced", "?"),
                ("sweep_mode", sweep_mode_enum),
                ("use_outside_sweep_rate", "?"),
                ("sweep_rate_outside", "<f8"),
                ("alternate_direction", "?"),
            ]
        )
        step_list_data = []
        for ch in actual_step_channels:
            step_list_data.append(
                (
                    _to_bytes(ch["name"]),
                    0,
                    float(ch.get("wait_after", 0.0)),
                    0,
                    0.0,
                    False,
                    b"x",
                    False,
                    0,
                    False,
                    0.0,
                    False,
                )
            )
        f.create_dataset("Step list", data=np.array(step_list_data, dtype=dtype_step_list))
        step_cfg_grp = f.create_group("Step config")
        step_inst_cfg = inst_cfg_grp.create_group("Generic - GPIB: , Step channels at localhost")
        step_inst_cfg.attrs["Installed options"] = np.array([])
        for ch in actual_step_channels:
            name = ch["name"]
            step_inst_cfg.attrs[name] = 0.0
            single_step_grp = step_cfg_grp.create_group(name)
            opt_grp = single_step_grp.create_group("Optimizer")
            opt_grp.attrs["Enabled"] = False
            vals = ch["values"]
            if name == _STEP_NAME_API:
                vals = np.linspace(1.0, 2.0, 51)
            start = vals[0]
            stop = vals[-1]
            span = abs(stop - start)
            opt_grp.attrs["Initial step size"] = float(0.2 * span) if span > 0 else 1.0
            opt_grp.attrs["Max value"] = float(max(start, stop))
            opt_grp.attrs["Min value"] = float(min(start, stop))
            opt_grp.attrs["Precision"] = float(0.0001 * span) if span > 0 else 0.0001
            opt_grp.attrs["Start value"] = float(start)
            dtype_rel = np.dtype(
                [("variable", vlen_bytes), ("channel_name", vlen_bytes), ("use_lookup", "?")]
            )
            single_step_grp.create_dataset(
                "Relation parameters", data=np.array([(b"x", b"Step values", False)], dtype=dtype_rel)
            )
            dtype_items = np.dtype(
                [
                    ("range_type", range_type_enum),
                    ("step_type", step_type_enum),
                    ("single", "<f8"),
                    ("start", "<f8"),
                    ("stop", "<f8"),
                    ("center", "<f8"),
                    ("span", "<f8"),
                    ("step", "<f8"),
                    ("n_pts", "<i4"),
                    ("interp", interp_enum),
                    ("sweep_rate", "<f8"),
                ]
            )
            if len(vals) > 1 and name != _STEP_NAME_API:
                item_data = (1, 1, start, start, stop, 0.0, 0.0, 0.0, len(vals), 0, 0.0)
            else:
                item_data = (0, 1, start, start, start + 1.0, 0.0, 0.0, 0.0, len(vals), 0, 0.0)
            single_step_grp.create_dataset("Step items", data=np.array([item_data], dtype=dtype_items))
        log_inst_cfg = inst_cfg_grp.create_group("Generic - GPIB: , Log channels at localhost")
        log_inst_cfg.attrs["Installed options"] = np.array([])
        for ch in log_channels:
            name = ch["name"]
            is_vector = ch.get("vector", True)
            is_complex = ch.get("complex", False)
            if is_vector:
                x_name = ch.get("x_name", "Time" if "x_unit" in ch or "x_name" in ch else "Index")
                x_unit = ch.get("x_unit", "s" if "x_unit" in ch or "x_name" in ch else "")
                log_inst_cfg.attrs[f"___{name}___x_name"] = _to_bytes(x_name)
                log_inst_cfg.attrs[f"___{name}___x_unit"] = _to_bytes(x_unit)
                log_inst_cfg.attrs[name] = np.array([], dtype=complex if is_complex else float)
            else:
                log_inst_cfg.attrs[name] = 0j if is_complex else 0.0
    return resolved_path
