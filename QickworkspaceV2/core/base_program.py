"""Shared QICK program setup for single-qubit experiments."""

from qick.asm_v2 import AveragerProgramV2

from .qubit_pulse import GATE_ALIAS, QubitPulseMixin, resolve_gate


class BaseProgram(QubitPulseMixin, AveragerProgramV2):
    """Base class for QICK programs in this framework.

    Subclasses declare experiment-specific resources in ``_initialize`` and
    emit their pulse sequence in ``_body``. This class owns the common
    resonator, cooling, measurement, and feedback-reset details.
    """

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
