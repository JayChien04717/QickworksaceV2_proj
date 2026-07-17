"""
BaseProgram: Base class for all QICK programs.

Centralizes generator declaration, readout, qubit pulse setup (prefix-aware),
standard gates, cooling, and measurement.
Subclasses only need to implement _initialize() and _body().
"""

from qick.asm_v2 import AveragerProgramV2

from .qubit_pulse import GATE_ALIAS, QubitPulseMixin, resolve_gate


class BaseProgram(QubitPulseMixin, AveragerProgramV2):
    """
    Base class for all QICK programs in this framework.

    Centralises generator declaration, readout configuration, qubit pulse
    setup (prefix-aware for ge/ef transitions), standard calibration gates,
    active cooling, and the measurement trigger sequence.  Subclasses only
    need to implement :meth:`_initialize` and :meth:`_body`.
    """

    def setup_resonator(self, cfg, prefix="ge"):
        """Configure the resonator readout channel and flat_top readout pulse."""
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        self.add_readoutconfig(ch=ro_ch, name="myro", freq=cfg[f"res_freq_{prefix}"], gen_ch=res_ch)
        self.add_gauss(ch=res_ch, name="readout", sigma=cfg["res_sigma"],
                       length=5 * cfg["res_sigma"], even_length=True)
        self.add_pulse(
            ch=res_ch, name="res_pulse", ro_ch=ro_ch, style="flat_top",
            envelope="readout", length=cfg["res_length"],
            freq=cfg[f"res_freq_{prefix}"], phase=cfg["res_phase"], gain=cfg[f"res_gain_{prefix}"],
        )

    def apply_cool(self, cfg, style="flat_top"):
        """Configure active-reset cooling channels and pulses."""
        for i in [1, 2]:
            ch_key = f"cool_ch{i}"
            if ch_key not in cfg:
                continue
            ch = cfg[ch_key]
            nqz = cfg.get(f"nqz_cool_ch{i}", 2)
            gen_params = {"ch": ch, "nqz": nqz}
            gen_cfg = self.soccfg["gens"][ch]
            has_mixer = gen_cfg.get(
                "has_mixer", gen_cfg.get("type") == "axis_sg_int4_v2"
            )
            if has_mixer:
                gen_params["mixer_freq"] = cfg[f"cool_mixer{i}"]
            self.declare_gen(**gen_params)

            if style == "flat_top":
                env_name = f"cooling{i}"
                self.add_gauss(ch=ch, name=env_name, sigma=cfg["res_sigma"],
                               length=cfg["res_sigma"] * 5, even_length=True)
                self.add_pulse(ch=ch, name=f"cool_pulse{i}", envelope=env_name, style="flat_top",
                               length=cfg["cool_length"], freq=cfg[f"cool_freq_{i}"], phase=0, gain=cfg[f"cool_gain_{i}"])
            else:
                self.add_pulse(ch=ch, name=f"cool_pulse{i}", style="const",
                               length=cfg["cool_length"], freq=cfg[f"cool_freq_{i}"], phase=0, gain=cfg[f"cool_gain_{i}"])

    def cooling_body(self, cfg, ring_down=0.5):
        """Execute the active-reset cooling pulse sequence inside _body."""
        if not cfg.get("cooling", False):
            return False
        self.pulse(ch=cfg["cool_ch1"], name="cool_pulse1", t=0)
        self.pulse(ch=cfg["cool_ch2"], name="cool_pulse2", t=0)
        self.delay_auto(ring_down, tag="Ring down")
        return True

    def measure(self, cfg):
        """Execute the standard readout pulse and ADC trigger."""
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


__all__ = ["BaseProgram", "GATE_ALIAS", "resolve_gate"]
