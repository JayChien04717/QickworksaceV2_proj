"""Shared QICK program setup for single-qubit experiments."""

from qick.asm_v2 import AveragerProgramV2


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
    """Resolve a shorthand gate to its transition-qualified pulse name."""
    if name in ("I", "-I", None, "None"):
        return name
    if name in GATE_ALIAS:
        return GATE_ALIAS[name].format(pfx=prefix)
    return name


class BaseProgram(AveragerProgramV2):
    """Base class for QICK programs in this framework.

    Subclasses declare experiment-specific resources in ``_initialize`` and
    emit their pulse sequence in ``_body``. This class owns the common
    resonator, cooling, measurement, and feedback-reset details.
    """

    def setup_qubit_gen(self, cfg, prefix="ge"):
        """Declare the qubit generator for a transition."""
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
        """Add a transition-aware qubit pulse to the QICK pulse library."""
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
        """Create an envelope once per channel and return its QICK name."""
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
                    "no parameter 'drag_alpha' found in cfg -- calibrate DRAG first"
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
        """Register the six standard calibrated gates."""
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

    def setup_resonator(self, cfg, prefix="ge"):
        """Declare one resonator generator, readout, and flat-top pulse."""
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        frequency = cfg[f"res_freq_{prefix}"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        self.add_readoutconfig(
            ch=ro_ch,
            name="myro",
            freq=frequency,
            gen_ch=res_ch,
        )
        self.add_gauss(
            ch=res_ch,
            name="readout",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=res_ch,
            name="res_pulse",
            ro_ch=ro_ch,
            style="flat_top",
            envelope="readout",
            length=cfg["res_length"],
            freq=frequency,
            phase=cfg["res_phase"],
            gain=cfg[f"res_gain_{prefix}"],
        )

    def apply_cool(self, cfg, style="flat_top"):
        """Declare every configured cooling generator and pulse."""
        for index in (1, 2):
            channel_key = f"cool_ch{index}"
            if channel_key not in cfg:
                continue

            channel = cfg[channel_key]
            generator = {
                "ch": channel,
                "nqz": cfg.get(f"nqz_cool_ch{index}", 2),
            }
            generator_config = self.soccfg["gens"][channel]
            has_mixer = generator_config.get(
                "has_mixer",
                generator_config.get("type") == "axis_sg_int4_v2",
            )
            if has_mixer:
                generator["mixer_freq"] = cfg[f"cool_mixer{index}"]
            self.declare_gen(**generator)

            pulse = {
                "ch": channel,
                "name": f"cool_pulse{index}",
                "style": style,
                "length": cfg["cool_length"],
                "freq": cfg[f"cool_freq_{index}"],
                "phase": 0,
                "gain": cfg[f"cool_gain_{index}"],
            }
            if style == "flat_top":
                envelope = f"cooling{index}"
                self.add_gauss(
                    ch=channel,
                    name=envelope,
                    sigma=cfg["res_sigma"],
                    length=5 * cfg["res_sigma"],
                    even_length=True,
                )
                pulse["envelope"] = envelope
            self.add_pulse(**pulse)

    def cooling_body(self, cfg, ring_down=0.5):
        """Play configured cooling pulses and wait for resonator ring-down."""
        if not cfg.get("cooling", False):
            return False

        applied = False
        for index in (1, 2):
            channel_key = f"cool_ch{index}"
            if channel_key not in cfg:
                continue
            self.pulse(
                ch=cfg[channel_key],
                name=f"cool_pulse{index}",
                t=0,
            )
            applied = True
        if applied:
            self.delay_auto(ring_down, tag="Ring down")
        return applied

    def measure(self, cfg, *, pins=None):
        """Play the standard resonator pulse and trigger its ADC."""
        if pins is None:
            pins = [0]
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(
            ros=[cfg["ro_ch"]],
            pins=pins,
            t=cfg["trig_time"],
        )

    def setup_active_reset(self, cfg, *, include_idle=True):
        """Configure feedback classification and the reset pulse slot.

        ``threshold`` uses the length-normalized units returned by QICK. The
        tProc compares its raw accumulator, so the register stores threshold
        times the readout length.
        """
        ro_ch = cfg["ro_ch"]
        tproc_ch = self.soccfg["readouts"][ro_ch].get("tproc_ch")
        if tproc_ch is None or int(tproc_ch) < 0:
            raise RuntimeError(
                f"readout channel {ro_ch} has no tProc feedback input "
                "('tproc_ch'); measurement-based active reset is unavailable"
            )

        self.setup_qb_pulse(
            cfg,
            "ge",
            name="reset_pi",
            gain_key="pi_gain_ge",
        )
        if include_idle:
            self.setup_qb_pulse(
                cfg,
                "ge",
                name="reset_idle",
                gain_override=0,
            )

        reset_mode = str(cfg.get("reset_mode", "conditional")).lower()
        if reset_mode not in {"conditional", "always", "never"}:
            raise ValueError(
                "reset_mode must be 'conditional', 'always', or 'never'"
            )

        component = str(cfg.get("reset_component", "I")).upper()
        if component not in {"I", "Q"}:
            raise ValueError("reset_component must be 'I' or 'Q'")

        excited_if = str(cfg.get("reset_excited_if", ">=")).strip()
        if excited_if == ">=":
            ground_test = "<"
        elif excited_if == "<":
            ground_test = ">="
        else:
            raise ValueError(
                "reset_excited_if must be '>=' or '<'; these are the "
                "comparators supported by QICK read_and_jump()"
            )

        threshold = float(cfg["threshold"])
        readout_length = float(cfg["ro_length"])
        self.reset_mode = reset_mode
        self.reset_component = component
        self.ground_test = ground_test
        self.reset_threshold_normalized = threshold
        self.reset_threshold_raw = int(round(threshold * readout_length))
        self.reset_readout_length = readout_length
        self.add_reg("reset_threshold")
        self.write_reg("reset_threshold", self.reset_threshold_raw)

    def activate_reset(self, cfg):
        """Measure and apply ``reset_pi`` only to non-ground shots."""
        self.measure(cfg, pins=cfg.get("reset_trigger_pins", [0]))
        self.wait_auto(
            float(cfg.get("read_wait", 0.15)),
            gens=True,
            ros=True,
        )
        self.resync(float(cfg.get("feedback_slack", 0.05)))
        self.read_and_jump(
            ro_ch=cfg["ro_ch"],
            component=self.reset_component,
            threshold="reset_threshold",
            test=self.ground_test,
            label="AFTER_ACTIVE_RESET",
        )
        self.pulse(ch=cfg["qb_ch"], name="reset_pi", t=0)
        self.label("AFTER_ACTIVE_RESET")


__all__ = ["BaseProgram", "GATE_ALIAS", "resolve_gate"]
