from __future__ import annotations

import datetime
import os
import pprint
import re
import warnings
from collections import defaultdict
from typing import Any

import numpy as np
import yaml
from addict import Dict as AddictDict

from .units import auto_unit


def get_next_filename_labber(
    dest_path: str, exp_name: str, yoko_value: dict[str, Any] | None = None
) -> str:
    """Generate the next HDF5 filename for a Labber-compatible log file.

            Files are saved inside a date-structured subdirectory under *dest_path*
            (``<dest_path>/<YYYY>/<MM>/Data_<MMDD>/``).

    Parameters
    ----------
    dest_path : str
        Filesystem location for dest path.
    exp_name : str
        Name of the exp.
    yoko_value : dict[str, Any] | None, default: None
        Value for ``yoko_value``.

    Returns
    -------
    str
        Result of the operation.

    Raises
    ------
    ValueError
        If the operation cannot be completed.
    """
    dest_path = os.path.abspath(dest_path)
    yy, mm, dd = datetime.datetime.today().strftime("%Y-%m-%d").split("-")
    save_path = os.path.join(dest_path, yy, mm, f"Data_{mm}{dd}")
    os.makedirs(save_path, exist_ok=True)

    if yoko_value is not None:
        if not isinstance(yoko_value, dict):
            raise ValueError(
                "yoko_value must be a dict with 'value' and 'unit' keys. Got: %s"
                % type(yoko_value).__name__
            )
        try:
            value = yoko_value["value"]
            value = auto_unit(value)
            unit = yoko_value["unit"]
            filename = f"{exp_name}_{value['value']:.2f}{value['unit']}{unit}"
            return os.path.join(save_path, filename)
        except KeyError:
            raise ValueError(
                "yoko_value dictionary must contain 'value' and 'unit' keys"
            )
    else:
        max_index = 0
        pattern = re.compile(rf"^{re.escape(exp_name)}_(\d+)\.hdf5$")
        for root, dirs, files in os.walk(dest_path):
            for f in files:
                match = pattern.match(f)
                if match:
                    current_index = int(match.group(1))
                    max_index = max(max_index, current_index)
        next_index = max_index + 1
        final_filename = f"{exp_name}_{next_index:03d}"
        return os.path.join(save_path, final_filename)


def hdf5_generator(
    filepath,
    x_info,
    z_info,
    y_info=None,
    comment=None,
    tag=None,
    *,
    result=None,
    embed_native=True,
    save_plots=True,
    data_root=None,
    filename_mode="random",
    figures=None,
):
    """Save one combined Labber + native HDF5 file.

            ``filename_mode`` is ``"random"`` by default and appends the native
            experiment ID. Set it to ``"sequential"`` to retain ``_001``, ``_002``, ...
            filenames returned by :func:`get_next_filename_labber`.

    Parameters
    ----------
    filepath : Any
        Value for ``filepath``.
    x_info : Any
        Value for ``x_info``.
    z_info : Any
        Value for ``z_info``.
    y_info : Any, default: None
        Value for ``y_info``.
    comment : Any, default: None
        Value for ``comment``.
    tag : Any, default: None
        Value for ``tag``.
    result : Any, default: None
        Experiment result to process.
    embed_native : Any, default: True
        Value for ``embed_native``.
    save_plots : Any, default: True
        Value for ``save_plots``.
    data_root : Any, default: None
        Value for ``data_root``.
    filename_mode : {'random', 'sequential'}, default: 'random'
        File-naming strategy. ``'random'`` replaces the temporary
        three-digit suffix with the sortable experiment ID. ``'sequential'``
        keeps the traditional ``_001``, ``_002``, ... suffix.
    figures : Any, default: None
        Value for ``figures``.

    Returns
    -------
    Any
        Result of the operation.

    Raises
    ------
    ImportError
        If the operation cannot be completed.
    """
    try:
        from ..tools.Labber_saver import LabberHDF5Saver
    except ImportError as exc:
        raise ImportError("Labber_saver is required to save HDF5 files.") from exc

    if data_root is None:
        try:
            from ..core.base_experiment import BaseExperiment
            data_root = BaseExperiment._data_path
        except Exception:
            data_root = None

    if figures is None and result is not None:
        figures = getattr(result, "figures", None)

    saver = LabberHDF5Saver(
        filename_mode=filename_mode,
        embed_native=embed_native,
        save_plots=save_plots,
    )
    return saver.save(
        filepath,
        x_info,
        z_info,
        y_info=y_info,
        comment=comment,
        tag=tag,
        result=result,
        data_root=data_root,
        figures=figures,
        catalog=True,
    )


def clean_config(config):
    """Recursively clean a config dict for serialization (removes QickParam, converts numpy).

    Parameters
    ----------
    config : Any
        Experiment configuration.

    Returns
    -------
    Any
        Result of the operation.
    """
    try:
        from qick.asm_v2 import QickParam

        _qick_param_types = (QickParam,)
    except ImportError:
        _qick_param_types = ()

    def _clean(obj):
        """Return the clean result.

        Parameters
        ----------
        obj : Any
            Value for ``obj``.

        Returns
        -------
        Any
            Result of the operation.
        """
        if isinstance(obj, (dict, AddictDict)):
            return {
                k: _clean(v)
                for k, v in obj.items()
                if not (_qick_param_types and isinstance(v, _qick_param_types))
            }
        elif isinstance(obj, (list, tuple)):
            return [_clean(i) for i in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        return obj

    return _clean(config)


def config_to_yaml(config: dict) -> str:
    """Clean a config dict and serialize it to a YAML string.

    Parameters
    ----------
    config : dict
        Experiment configuration.

    Returns
    -------
    str
        Result of the operation.
    """
    return yaml.dump(
        clean_config(config),
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


class ExperimentConfig:
    """
    Manager for nested multi-qubit experiment configurations.

    Parameters
    ----------
    data : list of dict or dict
        One configuration dict per qubit, or a single dict.
    keys_to_unify : list of str, optional
        Keys whose values are collapsed to a scalar when all qubits share the same value.
    """

    def __init__(self, data: list | dict, keys_to_unify: list[str] | None = None):
        """Initialize the ExperimentConfig instance.

        Parameters
        ----------
        data : list | dict
            Input data to process.
        keys_to_unify : list[str] | None, default: None
            Value for ``keys_to_unify``.
        """
        self._raw_list = data
        self.keys_to_unify = keys_to_unify or [
            "reps",
            "res_length",
            "ro_length",
            "trig_time",
            "relax_delay",
            "nqz_res",
        ]
        self._name_map = {}
        if isinstance(self._raw_list, list):
            for idx, cfg in enumerate(self._raw_list):
                name = cfg.get("name")
                if name:
                    self._name_map[name] = idx
        self._dirty = True
        self._unified_cache = None

    @property
    def unified_config(self) -> AddictDict:
        """Return the unified config result.

        Returns
        -------
        AddictDict
            Result of the operation.
        """
        if self._dirty or self._unified_cache is None:
            raw_collected = self._collect_all_key_values(self._raw_list)
            self._unified_cache = self._refine_cfg(raw_collected)
            self._dirty = False
        return self._unified_cache

    def _mark_dirty(self) -> None:
        """Return the mark dirty result."""
        self._dirty = True

    def __repr__(self) -> str:
        """Return a human-readable representation.

        Returns
        -------
        str
            Result of the operation.
        """
        names = [cfg.get("name", f"idx{i}") for i, cfg in enumerate(self._raw_list)]
        return f"ExperimentConfig(qubits={names})"

    def qubit_names(self) -> list:
        """Return the list of qubit names in config order.

        Returns
        -------
        list
            Result of the operation.
        """
        return [cfg.get("name", f"idx{i}") for i, cfg in enumerate(self._raw_list)]

    def get_qubit(self, q_id: int | str) -> AddictDict:
        """Return a flat configuration dict for a single qubit.

        Parameters
        ----------
        q_id : int | str
            Value for ``q_id``.

        Returns
        -------
        AddictDict
            Result of the operation.
        """
        indices = self._resolve_indices(q_id)
        idx = indices[0]
        selected = AddictDict()
        for key, value in self.unified_config.items():
            if isinstance(value, list):
                if idx < len(value):
                    selected[key] = value[idx]
            else:
                selected[key] = value
        return selected

    def muxconfig(
        self, qb_list, mux_ro_ch_start=2, mux_gen=12, LO_ext=None
    ) -> AddictDict:
        """Build a mux configuration for selected qubits.

                        The mux generator tones and readout channels remain slot-stable. For
                        example, Q1/Q3 with mux_ro_ch_start=2 uses readout channels [2, 4].

        Parameters
        ----------
        qb_list : Any
            Value for ``qb_list``.
        mux_ro_ch_start : Any, default: 2
            Value for ``mux_ro_ch_start``.
        mux_gen : Any, default: 12
            Value for ``mux_gen``.
        LO_ext : Any, default: None
            Value for ``LO_ext``.

        Returns
        -------
        AddictDict
            Result of the operation.
        """
        active_slots = self._resolve_indices(qb_list)
        active_set = set(active_slots)
        all_slots = list(range(len(self._raw_list)))
        selected = AddictDict()

        for key, value in self.unified_config.items():
            if isinstance(value, list):
                selected[key] = [
                    value[i] if i < len(value) else None for i in all_slots
                ]
            else:
                selected[key] = value

        all_names = self.qubit_names()
        active_names = [all_names[i] for i in active_slots]
        active_ro_chs = [i + mux_ro_ch_start for i in active_slots]

        res_freqs = list(selected.get("res_freq_ge", []))
        res_gains = list(selected.get("res_gain_ge", []))
        res_phases = list(selected.get("res_phase", [0] * len(all_slots)))
        ro_phases = list(selected.get("ro_phase", [0] * len(all_slots)))

        def _pad(values, default):
            """Return the pad result.

            Parameters
            ----------
            values : Any
                Values to process.
            default : Any
                Value for ``default``.

            Returns
            -------
            Any
                Result of the operation.
            """
            values = list(values)
            if len(values) < len(all_slots):
                values.extend([default] * (len(all_slots) - len(values)))
            return values[: len(all_slots)]

        res_freqs = _pad(res_freqs, None)
        if LO_ext is not None:
            res_freqs = [
                freq - LO_ext if freq is not None else None for freq in res_freqs
            ]
        res_gains = _pad(res_gains, 0)
        res_phases = _pad(res_phases, 0)
        ro_phases = _pad(ro_phases, 0)
        res_gains = [
            gain if slot in active_set else 0 for slot, gain in enumerate(res_gains)
        ]

        selected["qubit_names"] = active_names
        selected["active_slots"] = active_slots
        selected["mask"] = all_slots
        selected["res_ch"] = mux_gen
        selected["ro_chs"] = active_ro_chs
        selected["active_ro_chs"] = active_ro_chs
        selected["res_freqs"] = res_freqs
        selected["res_gains"] = res_gains
        selected["res_phases"] = res_phases
        selected["ro_phases"] = ro_phases
        selected["LO_ext"] = LO_ext
        selected["trig_time"] = 0.75

        # Backward-compatible aliases for older notebooks.
        selected["mux_ro_chs"] = active_ro_chs
        selected["gen_mask"] = list(all_slots)
        selected["mux_gen"] = mux_gen
        selected["mux_ro_phases"] = [ro_phases[i] for i in active_slots]

        active_freqs = [res_freqs[i] for i in active_slots if res_freqs[i] is not None]
        if active_freqs:
            selected["mixer_freq"] = int(round(np.mean(active_freqs)))
        return selected

    def to_yaml_mux(self, qb_list, **kwargs) -> str:
        """Convert the value to yaml mux.

        Parameters
        ----------
        qb_list : Any
            Value for ``qb_list``.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        str
            Result of the operation.
        """
        indices = self._resolve_indices(qb_list)

        def _get_mux_val(key, path_parts):
            """Return mux val.

            Parameters
            ----------
            key : Any
                Lookup key.
            path_parts : Any
                Value for ``path_parts``.

            Returns
            -------
            Any
                Result of the operation.
            """
            vals = []
            for idx in indices:
                cfg = self._raw_list[idx]
                curr = cfg
                for p in path_parts:
                    if isinstance(curr, dict):
                        curr = curr.get(p)
                    else:
                        curr = None
                        break
                vals.append(curr)
            vals = [v for v in vals if v is not None]
            if not vals:
                return None
            if key in self.keys_to_unify:
                unique = set(v for v in vals if not isinstance(v, (dict, list)))
                if len(unique) == 1:
                    return list(unique)[0]
            return vals

        def _recursive_fill(template_node, path=[]):
            """Return the recursive fill result.

            Parameters
            ----------
            template_node : Any
                Value for ``template_node``.
            path : Any, default: []
                Filesystem path.

            Returns
            -------
            Any
                Result of the operation.
            """
            if isinstance(template_node, dict):
                res = {}
                for k, v in template_node.items():
                    if isinstance(v, dict):
                        res[k] = _recursive_fill(v, path + [k])
                    else:
                        res[k] = _get_mux_val(k, path + [k])
                return res
            return template_node

        template = self._clean_data(self._raw_list[indices[0]])
        mux_nested = _recursive_fill(template)
        mux_gen = kwargs.get("mux_gen", 12)
        mux_ro_ch_start = kwargs.get("mux_ro_ch_start", 2)

        if "name" in mux_nested:
            all_names = [self._raw_list[i].get("name", f"Q{i}") for i in indices]
            mux_nested["name"] = ", ".join(all_names)
        if "ch" in mux_nested:
            mux_nested["ch"].pop("res_ch", None)
            mux_nested["ch"].pop("ro_ch", None)
            mux_nested["ch"]["mux_res_ch"] = mux_gen
            mux_nested["ch"]["mux_ro_ch"] = [i + mux_ro_ch_start for i in indices]

        flat = self.muxconfig(qb_list, **kwargs)
        if "gen_mask" in flat:
            mux_nested["gen_mask"] = flat["gen_mask"]
        if "mixer_freq" in flat:
            mux_nested["mixer_freq"] = flat["mixer_freq"]

        return self._dump_dict_with_spacing(mux_nested)

    def update_mux(self, param, value=None, q_index=None) -> None:
        """Update mux.

        Parameters
        ----------
        param : Any
            Value for ``param``.
        value : Any, default: None
            Value to apply.
        q_index : Any, default: None
            Value for ``q_index``.
        """
        self.update(param, value, q_index)

    def update(self, param, value=None, q_index=None) -> None:
        """Unified update: dict merge, list-of-tuples, auto-search, or dot-path.

        Parameters
        ----------
        param : Any
            Value for ``param``.
        value : Any, default: None
            Value to apply.
        q_index : Any, default: None
            Value for ``q_index``.

        Raises
        ------
        TypeError
            If the operation cannot be completed.
        """
        target_indices = self._resolve_indices(q_index)

        if isinstance(param, (list, tuple)) and all(
            isinstance(item, (list, tuple)) and len(item) == 2 for item in param
        ):
            for k, v in param:
                self.update(k, v, q_index=q_index)
            return

        if isinstance(param, dict):
            updated_count = 0
            for idx in target_indices:
                raw_nested_cfg = self._raw_list[idx]
                for k, v in param.items():
                    if self._recursive_update(raw_nested_cfg, k, v):
                        updated_count += 1
            if q_index is not None:
                print(
                    f"Merged dictionary into {q_index}. Updated {updated_count} parameters."
                )

        elif isinstance(param, str) and "." not in param:
            leaf_key = param
            is_list_val = isinstance(value, (list, np.ndarray))
            should_distribute = is_list_val and (len(value) == len(target_indices))
            has_key = [
                self._find_key_path(self._raw_list[j], leaf_key) is not None
                for j in range(len(self._raw_list))
            ]
            for j in range(len(self._raw_list)):
                if not has_key[j] and j not in target_indices:
                    self._raw_list[j][leaf_key] = None
            for i, cfg_idx in enumerate(target_indices):
                cfg = self._raw_list[cfg_idx]
                val_to_set = value[i] if should_distribute else value
                if isinstance(val_to_set, np.generic):
                    val_to_set = val_to_set.item()
                if not has_key[cfg_idx]:
                    cfg[leaf_key] = val_to_set
                else:
                    self._recursive_update(cfg, leaf_key, val_to_set)

        elif isinstance(param, str):
            key_path = param
            keys = key_path.split(".")
            leaf_key = keys[-1]
            is_list_val = isinstance(value, (list, np.ndarray))
            should_distribute = is_list_val and (len(value) == len(target_indices))

            def _key_exists_in(nested, path_keys):
                """Return the key exists in result.

                Parameters
                ----------
                nested : Any
                    Value for ``nested``.
                path_keys : Any
                    Value for ``path_keys``.

                Returns
                -------
                Any
                    Result of the operation.
                """
                curr = nested
                for k in path_keys[:-1]:
                    if not isinstance(curr, dict) or k not in curr:
                        return False
                    curr = curr[k]
                return isinstance(curr, dict) and path_keys[-1] in curr

            has_key = [
                _key_exists_in(self._raw_list[j], keys)
                for j in range(len(self._raw_list))
            ]
            for j in range(len(self._raw_list)):
                if not has_key[j] and j not in target_indices:
                    cfg_j = self._raw_list[j]
                    node = cfg_j
                    for k in keys[:-1]:
                        if isinstance(node, dict):
                            node = node.setdefault(k, {})
                        else:
                            node = getattr(node, k)
                    if isinstance(node, dict):
                        node[leaf_key] = None
                    else:
                        setattr(node, leaf_key, None)

            for i, cfg_idx in enumerate(target_indices):
                cfg = self._raw_list[cfg_idx]
                target = cfg
                for k in keys[:-1]:
                    if isinstance(target, dict):
                        if k not in target:
                            warnings.warn(
                                f"Creating new nested key '{k}' in path '{key_path}'",
                                stacklevel=2,
                            )
                        target = target.setdefault(k, {})
                    else:
                        target = getattr(target, k)
                val_to_set = value[i] if should_distribute else value
                if isinstance(val_to_set, np.generic):
                    val_to_set = val_to_set.item()
                if isinstance(target, dict):
                    target[leaf_key] = val_to_set
                else:
                    setattr(target, leaf_key, val_to_set)
        else:
            raise TypeError(
                "First argument must be a string (key path) or a dict (config)."
            )

        self._mark_dirty()

    def _find_key_path(self, nested, target_key, _path=()):
        """Return key path.

        Parameters
        ----------
        nested : Any
            Value for ``nested``.
        target_key : Any
            Value for ``target_key``.
        _path : Any, default: ()
            Filesystem location for  path.

        Returns
        -------
        Any
            Result of the operation.
        """
        if isinstance(nested, dict):
            if target_key in nested:
                return _path + (target_key,)
            for k, v in nested.items():
                if isinstance(v, (dict, list)):
                    result = self._find_key_path(v, target_key, _path + (k,))
                    if result is not None:
                        return result
        elif isinstance(nested, list):
            for item in nested:
                if isinstance(item, (dict, list)):
                    result = self._find_key_path(item, target_key, _path)
                    if result is not None:
                        return result
        return None

    def _recursive_update(self, nested_data, target_key, new_value) -> bool:
        """Return the recursive update result.

        Parameters
        ----------
        nested_data : Any
            Value for ``nested_data``.
        target_key : Any
            Value for ``target_key``.
        new_value : Any
            Value for ``new_value``.

        Returns
        -------
        bool
            Result of the operation.
        """
        found = False
        if isinstance(nested_data, dict):
            if target_key in nested_data:
                val_to_set = new_value
                if isinstance(val_to_set, np.generic):
                    val_to_set = val_to_set.item()
                if nested_data[target_key] != val_to_set:
                    nested_data[target_key] = val_to_set
                    return True
                return False
            for v in nested_data.values():
                if isinstance(v, (dict, list)):
                    if self._recursive_update(v, target_key, new_value):
                        found = True
        elif isinstance(nested_data, list):
            for item in nested_data:
                if isinstance(item, (dict, list)):
                    if self._recursive_update(item, target_key, new_value):
                        found = True
        return found

    def read_config(self, q_id) -> dict:
        """Return config.

        Parameters
        ----------
        q_id : Any
            Value for ``q_id``.

        Returns
        -------
        dict
            Result of the operation.
        """
        indices = self._resolve_indices(q_id)
        return self._clean_data(self._raw_list[indices[0]])

    read_qubit_config = read_config

    def save_to_py(self, filename: str = "latest_cfg.py") -> None:
        """Save to py.

        Parameters
        ----------
        filename : str, default: 'latest_cfg.py'
            Value for ``filename``.
        """
        clean_data = self._clean_data(self._raw_list)
        with open(filename, "w", encoding="utf-8") as f:
            f.write("# Auto-generated configuration file\n")
            f.write("config_list = ")
            pprint.pprint(clean_data, stream=f, width=120, sort_dicts=False)
            f.write("\n")
        print(f"Configuration saved to {filename}")

    def save_qubit_config(self, q_id, filename=None, var_name="config") -> None:
        """Save qubit config.

        Parameters
        ----------
        q_id : Any
            Value for ``q_id``.
        filename : Any, default: None
            Value for ``filename``.
        var_name : Any, default: 'config'
            Name of the var.
        """
        indices = self._resolve_indices(q_id)
        raw_nested_cfg = self._raw_list[indices[0]]
        clean_data = self._clean_data(raw_nested_cfg)
        if filename is None:
            name = clean_data.get("name", f"Q{indices[0]}")
            filename = f"{name}_config.py"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(
                f"# Auto-generated configuration file for {clean_data.get('name', q_id)}\n"
            )
            f.write("from addict import Dict\n\n")
            f.write(f"{var_name} = ")
            pprint.pprint(clean_data, stream=f, width=120, sort_dicts=False)
            f.write("\n")
        print(f"Saved {q_id} configuration to {filename}")

    def to_yaml(self, q_id=None) -> str:
        """Convert the value to yaml.

        Parameters
        ----------
        q_id : Any, default: None
            Value for ``q_id``.

        Returns
        -------
        str
            Result of the operation.

        Raises
        ------
        ValueError
            If the operation cannot be completed.
        """
        if q_id is not None:
            indices = self._resolve_indices(q_id)
            if not indices:
                raise ValueError(f"No qubit found for identifier {q_id}")
            clean_data = self._clean_data(self._raw_list[indices[0]])
            return self._dump_dict_with_spacing(clean_data)
        else:
            clean_data = self._clean_data(self._raw_list)
            if isinstance(clean_data, list):
                yaml_parts = []
                for item in clean_data:
                    part = self._dump_dict_with_spacing(item, is_list_item=True)
                    yaml_parts.append(part)
                return "\n\n\n".join(yaml_parts) + "\n"
            else:
                return self._dump_dict_with_spacing(clean_data)

    def _dump_dict_with_spacing(self, data, is_list_item=False, indent=0) -> str:
        """Return the dump dict with spacing result.

        Parameters
        ----------
        data : Any
            Input data to process.
        is_list_item : Any, default: False
            Whether to is list item.
        indent : Any, default: 0
            Value for ``indent``.

        Returns
        -------
        str
            Result of the operation.
        """
        if not isinstance(data, dict):
            if isinstance(data, list) and all(
                not isinstance(x, (dict, list)) for x in data
            ):
                return yaml.dump(data, default_flow_style=True, sort_keys=False).strip()
            return yaml.dump(data, default_flow_style=False, sort_keys=False).strip()

        parts = []
        keys = list(data.keys())
        for i, key in enumerate(keys):
            val = data[key]
            if is_list_item and i == 0:
                line_pfx = ("  " * indent) + "- "
                eff_indent = indent + 1
            elif is_list_item:
                line_pfx = ("  " * indent) + "  "
                eff_indent = indent + 1
            else:
                line_pfx = "  " * indent
                eff_indent = indent

            if isinstance(val, dict):
                parts.append(f"{line_pfx}{key}:")
                parts.append(self._dump_dict_with_spacing(val, indent=eff_indent + 1))
            elif isinstance(val, list) and all(
                not isinstance(x, (dict, list)) for x in val
            ):
                list_str = yaml.dump(
                    val, default_flow_style=True, sort_keys=False
                ).strip()
                parts.append(f"{line_pfx}{key}: {list_str}")
            else:
                dumped = yaml.dump(
                    {key: val}, default_flow_style=False, sort_keys=False
                ).strip()
                lines = dumped.split("\n")
                parts.append(f"{line_pfx}{lines[0]}")
                for line in lines[1:]:
                    parts.append(f"{'  ' * eff_indent}{line}")

            if i < len(keys) - 1:
                next_val = data[keys[i + 1]]
                if isinstance(val, (dict, list)) or isinstance(next_val, (dict, list)):
                    parts.append("")

        return "\n".join(parts)

    def to_yaml_file(self, filename, q_id=None):
        """Convert the value to yaml file.

        Parameters
        ----------
        filename : Any
            Value for ``filename``.
        q_id : Any, default: None
            Value for ``q_id``.
        """
        yaml_str = self.to_yaml(q_id=q_id)
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(yaml_str)
            print(f"Configuration saved to {filename}")

    def _resolve_indices(self, q_identifier) -> list[int]:
        """Resolve indices.

        Parameters
        ----------
        q_identifier : Any
            Value for ``q_identifier``.

        Returns
        -------
        list[int]
            Result of the operation.

        Raises
        ------
        TypeError
            If the operation cannot be completed.
        ValueError
            If the operation cannot be completed.
        """
        if q_identifier is None:
            return list(range(len(self._raw_list)))
        if isinstance(q_identifier, (int, np.integer)):
            return [int(q_identifier)]
        if isinstance(q_identifier, str):
            if q_identifier in self._name_map:
                return [self._name_map[q_identifier]]
            else:
                raise ValueError(f"Qubit name '{q_identifier}' not found.")
        if isinstance(q_identifier, list):
            resolved = []
            for x in q_identifier:
                if isinstance(x, str) and x in self._name_map:
                    resolved.append(self._name_map[x])
                elif isinstance(x, (int, np.integer)):
                    resolved.append(int(x))
                else:
                    raise ValueError(f"Invalid identifier: {x}")
            return resolved
        raise TypeError("Invalid q_index type.")

    def _clean_data(self, data: Any) -> Any:
        """Return the clean data result.

        Parameters
        ----------
        data : Any
            Input data to process.

        Returns
        -------
        Any
            Result of the operation.
        """
        try:
            from qick.asm_v2 import QickParam

            _qp = (QickParam,)
        except ImportError:
            _qp = ()

        def _clean(obj):
            """Return the clean result.

            Parameters
            ----------
            obj : Any
                Value for ``obj``.

            Returns
            -------
            Any
                Result of the operation.
            """
            if isinstance(obj, (dict, AddictDict)):
                return {
                    k: _clean(v)
                    for k, v in obj.items()
                    if not (_qp and isinstance(v, _qp))
                }
            elif isinstance(obj, list):
                return [_clean(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, complex):
                return {"real": obj.real, "imag": obj.imag}
            return obj

        return _clean(data)

    def _collect_all_key_values(self, data) -> defaultdict:
        """Return the collect all key values result.

        Parameters
        ----------
        data : Any
            Input data to process.

        Returns
        -------
        defaultdict
            Result of the operation.
        """
        result = defaultdict(list)
        if isinstance(data, (dict, AddictDict)):
            for key, value in data.items():
                if isinstance(value, (dict, list, AddictDict)):
                    nested_results = self._collect_all_key_values(value)
                    for k, v_list in nested_results.items():
                        result[k].extend(v_list)
                else:
                    result[key].append(value)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list, AddictDict)):
                    nested_results = self._collect_all_key_values(item)
                    for k, v_list in nested_results.items():
                        result[k].extend(v_list)
        return result

    def _refine_cfg(self, collected_data) -> AddictDict:
        """Return the refine cfg result.

        Parameters
        ----------
        collected_data : Any
            Value for ``collected_data``.

        Returns
        -------
        AddictDict
            Result of the operation.
        """
        refined_dict = AddictDict(collected_data)
        for key, value_list in refined_dict.items():
            if (
                key in self.keys_to_unify
                and isinstance(value_list, list)
                and value_list
            ):
                unique_values = set(value_list)
                if len(unique_values) == 1:
                    single_value = unique_values.pop()
                    if isinstance(single_value, np.generic):
                        single_value = single_value.item()
                    refined_dict[key] = single_value
        return refined_dict

    def __getitem__(self, item):
        """Return the getitem result.

        Parameters
        ----------
        item : Any
            Value for ``item``.

        Returns
        -------
        Any
            Result of the operation.

        Raises
        ------
        TypeError
            If the operation cannot be completed.
        """
        if isinstance(item, str):
            if item in self._name_map:
                return self.get_qubit(item)
            return self.unified_config.get(item)
        if isinstance(item, int):
            return self.get_qubit(item)
        raise TypeError("Index must be int or str")
