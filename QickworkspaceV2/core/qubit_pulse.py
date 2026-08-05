"""Qubit generator, pulse-envelope, and standard-gate helpers."""


GATE_ALIAS = {
    "X": "x180_{pfx}",
    "Y": "y180_{pfx}",
    "X/2": "x90_{pfx}",
    "-X/2": "x90m_{pfx}",
    "Y/2": "y90_{pfx}",
    "-Y/2": "y90m_{pfx}",
    "x180": "x180_{pfx}",
    "y180": "y180_{pfx}",
    "x90": "x90_{pfx}",
    "x90m": "x90m_{pfx}",
    "y90": "y90_{pfx}",
    "y90m": "y90m_{pfx}",
}

STANDARD_GATES = (
    ("x180", 0, "pi_gain"),
    ("y180", 90, "pi_gain"),
    ("x90", 0, "pi2_gain"),
    ("x90m", 180, "pi2_gain"),
    ("y90", 90, "pi2_gain"),
    ("y90m", -90, "pi2_gain"),
)


def resolve_gate(name, prefix="ge"):
    """Resolve a shorthand gate name to its transition-qualified pulse name.

    Parameters
    ----------
    name : Any
        Name of the target object.
    prefix : Any, default: 'ge'
        Value for ``prefix``.

    Returns
    -------
    Any
        Result of the operation.
    """
    if name in ("I", "-I", None, "None"):
        return name
    if name in GATE_ALIAS:
        return GATE_ALIAS[name].format(pfx=prefix)
    return name


class QubitPulseMixin:
    """Add transition-aware qubit pulse helpers to a QICK program."""

    def setup_qubit_gen(self, cfg, prefix="ge"):
        """Declare the qubit generator for a given transition prefix.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        prefix : Any, default: 'ge'
            Value for ``prefix``.
        """
        if prefix == "ge":
            ch, nqz_key, mixer_key = cfg["qb_ch"], "nqz_qb", "qb_mixer"
        else:
            ch = cfg[f"qb_ch_{prefix}"]
            nqz_key = f"nqz_qb_{prefix}"
            mixer_key = f"qb_mixer_{prefix}"

        gen_params = {"ch": ch, "nqz": cfg[nqz_key]}
        gen_cfg = self.soccfg["gens"][ch]
        has_mixer = gen_cfg.get(
            "has_mixer", gen_cfg.get("type") == "axis_sg_int4_v2"
        )
        if has_mixer:
            gen_params["mixer_freq"] = cfg[mixer_key]
        self.declare_gen(**gen_params)

    def setup_qb_pulse(
        self,
        cfg,
        prefix="ge",
        pulse_type=None,
        shape="gauss",
        name="qb_pulse",
        phase=None,
        gain_key=None,
        gain_override=None,
        ch=None,
        length_mult=5,
    ):
        """Add a transition-aware qubit pulse to the QICK pulse library.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        prefix : Any, default: 'ge'
            Value for ``prefix``.
        pulse_type : Any, default: None
            Value for ``pulse_type``.
        shape : Any, default: 'gauss'
            Value for ``shape``.
        name : Any, default: 'qb_pulse'
            Name of the target object.
        phase : Any, default: None
            Value for ``phase``.
        gain_key : Any, default: None
            Value for ``gain_key``.
        gain_override : Any, default: None
            Value for ``gain_override``.
        ch : Any, default: None
            Value for ``ch``.
        length_mult : Any, default: 5
            Value for ``length_mult``.

        Raises
        ------
        ValueError
            If the operation cannot be completed.
        """
        pulse_type = (
            cfg.get("pulse_type", "arb") if pulse_type is None else pulse_type
        )
        if pulse_type not in {"const", "arb", "flat_top", "drag"}:
            raise ValueError(f"Unknown qubit pulse type: {pulse_type}")

        if pulse_type == "drag":
            shape = "drag"
        if ch is None:
            ch = cfg["qb_ch"] if prefix == "ge" else cfg[f"qb_ch_{prefix}"]
        phase = cfg["qb_phase"] if phase is None else phase
        if gain_override is not None:
            gain = gain_override
        elif gain_key is not None:
            gain = cfg[gain_key]
        else:
            gain = cfg[f"qb_gain_{prefix}"]

        pulse_params = {
            "ch": ch,
            "name": name,
            "style": "arb" if pulse_type == "drag" else pulse_type,
            "freq": cfg[f"qb_freq_{prefix}"],
            "phase": phase,
            "gain": gain,
        }
        if pulse_type == "const":
            pulse_params["length"] = cfg[f"qb_length_{prefix}"]
        else:
            pulse_params["envelope"] = self._ensure_qb_envelope(
                cfg, prefix, ch, shape, length_mult
            )
            if pulse_type == "flat_top":
                pulse_params["length"] = cfg[f"qb_flat_top_length_{prefix}"]
        self.add_pulse(**pulse_params)

    def _ensure_qb_envelope(self, cfg, prefix, ch, shape, length_mult):
        """Create an envelope once per channel and return its QICK name.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        prefix : Any
            Value for ``prefix``.
        ch : Any
            Value for ``ch``.
        shape : Any
            Value for ``shape``.
        length_mult : Any
            Value for ``length_mult``.

        Returns
        -------
        Any
            Result of the operation.

        Raises
        ------
        KeyError
            If the operation cannot be completed.
        ValueError
            If the operation cannot be completed.
        """
        envelope = f"env_{prefix}_{shape}"
        envelope_key = (ch, envelope)
        added_envelopes = self.__dict__.setdefault("_added_envs", set())
        if envelope_key in added_envelopes:
            return envelope

        sigma = cfg[f"sigma_{prefix}"]
        if shape in ("gauss", "gaussian"):
            self.add_gauss(
                ch=ch,
                name=envelope,
                sigma=sigma,
                length=sigma * length_mult,
                even_length=True,
            )
        elif shape in ("cos", "cosine"):
            self.add_cosine(
                ch=ch, name=envelope, length=sigma, even_length=True
            )
        elif shape == "drag":
            if "drag_alpha" not in cfg:
                raise KeyError(
                    "no parameter 'drag_alpha' found in cfg — calibrate DRAG first"
                )
            self.add_DRAG(
                ch=ch,
                name=envelope,
                sigma=sigma,
                length=sigma * length_mult,
                delta=cfg["qb_freq_ge"] - cfg["qb_freq_ef"],
                alpha=cfg["drag_alpha"],
                even_length=True,
            )
        else:
            raise ValueError(f"Unknown pulse shape: {shape}")

        added_envelopes.add(envelope_key)
        return envelope

    def setup_standard_gates(
        self, cfg, prefix="ge", pulse_type=None, shape="gauss"
    ):
        """Register the six standard calibration gates.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        prefix : Any, default: 'ge'
            Value for ``prefix``.
        pulse_type : Any, default: None
            Value for ``pulse_type``.
        shape : Any, default: 'gauss'
            Value for ``shape``.
        """
        for gate, phase, gain in STANDARD_GATES:
            self.setup_qb_pulse(
                cfg,
                prefix=prefix,
                pulse_type=pulse_type,
                shape=shape,
                name=f"{gate}_{prefix}",
                phase=phase,
                gain_key=f"{gain}_{prefix}",
            )


__all__ = [
    "GATE_ALIAS",
    "STANDARD_GATES",
    "QubitPulseMixin",
    "resolve_gate",
]
