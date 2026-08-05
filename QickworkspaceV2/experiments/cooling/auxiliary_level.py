"""
Fluxonium auxiliary-level cooling experiments.

Sequence family inspired by arXiv:2402.06267:
prepare |e0>, move to a cooling flux bias, drive the auxiliary ef tone, and
let the readout cavity dissipate |g1> back to |g0>.
"""

from __future__ import annotations

import numpy as np

from ...analysis.resonator import LorentzianAnalysis
from ...core.base_experiment import BaseExperiment
from ...core.base_program import BaseProgram


def _flux_enabled(cfg):
    return "flux_ch" in cfg and "flux_gain" in cfg


def _add_flux_pulse(prog, cfg):
    if not _flux_enabled(cfg):
        return
    prog.declare_gen(ch=cfg["flux_ch"], nqz=cfg.get("nqz_flux", 1))
    prog.add_pulse(
        ch=cfg["flux_ch"], name="cool_flux", style="const",
        length=cfg["flux_length"], freq=0, phase=0, gain=cfg["flux_gain"],
    )


def _play_flux_pulse(prog, cfg):
    if not _flux_enabled(cfg):
        return
    prog.pulse(ch=cfg["flux_ch"], name="cool_flux", t=0)
    prog.delay(cfg.get("flux_settle", 0.01))


def _add_aux_pulse(prog, cfg, name="aux_cool"):
    ch = cfg.get("aux_cool_ch", cfg["qb_ch_ef"])
    prog.setup_qubit_gen(cfg, "ef")
    style = cfg.get("aux_cool_style", "flat_top")
    if style == "flat_top":
        env = f"{name}_env"
        sigma = cfg.get("aux_cool_sigma", cfg["sigma_ef"])
        prog.add_gauss(ch=ch, name=env, sigma=sigma, length=5 * sigma, even_length=True)
        prog.add_pulse(
            ch=ch, name=name, style="flat_top", envelope=env,
            length=cfg["aux_cool_length"], freq=cfg["aux_cool_freq"],
            phase=cfg.get("aux_cool_phase", 0), gain=cfg["aux_cool_gain"],
        )
    else:
        prog.add_pulse(
            ch=ch, name=name, style="const",
            length=cfg["aux_cool_length"], freq=cfg["aux_cool_freq"],
            phase=cfg.get("aux_cool_phase", 0), gain=cfg["aux_cool_gain"],
        )


class AuxiliaryEfSpecProgram(BaseProgram):
    """EF spectroscopy while the flux pulse parks the qubit at cooling bias."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        _add_flux_pulse(self, cfg)
        self.add_loop("freqloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_ge_pi", gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, "ef", name="qb_ef_spec", pulse_type="flat_top")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=cfg["qb_ch"], name="qb_ge_pi", t=0)
        self.delay_auto(0.02)
        _play_flux_pulse(self, cfg)
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_ef_spec", t=0)
        self.delay_auto(cfg.get("aux_cool_ringdown", 0.05))
        if cfg.get("ge_ref", False):
            self.pulse(ch=cfg["qb_ch"], name="qb_ge_pi", t=0)
            self.delay_auto(0.02)
        self.measure(cfg)


class AuxiliaryEfSpec(BaseExperiment):
    """Find the ef frequency at the cooling flux bias."""

    EXPT_NAME = "s020_aux_ef_spec"
    TAG = "Cooling"
    X_LABEL = "EF Frequency (MHz)"
    TITLE_PREFIX = "Auxiliary EF Spectrum at Cooling Bias"
    SWEEP_KEYS_TO_REMOVE = ["qb_freq_ef"]
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    Analysis = LorentzianAnalysis

    def _create_program(self):
        return AuxiliaryEfSpecProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("qb_ef_spec", "freq", as_array=True)


class AuxiliarySidebandSpecProgram(BaseProgram):
    """Prepare |e0>, apply candidate cooling tone, then measure reset strength."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        _add_flux_pulse(self, cfg)
        self.add_loop("freqloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_ge_pi", gain_key="pi_gain_ge")
        _add_aux_pulse(self, cfg)

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("prepare_e", True):
            self.pulse(ch=cfg["qb_ch"], name="qb_ge_pi", t=0)
            self.delay_auto(0.02)
        _play_flux_pulse(self, cfg)
        self.delay(cfg.get("aux_cool_pre", 0.01))
        self.pulse(ch=cfg.get("aux_cool_ch", cfg["qb_ch_ef"]), name="aux_cool", t=0)
        self.delay_auto(cfg.get("aux_cool_ringdown", 0.05))
        if cfg.get("ge_ref", False):
            self.pulse(ch=cfg["qb_ch"], name="qb_ge_pi", t=0)
            self.delay_auto(0.02)
        self.measure(cfg)


class AuxiliarySidebandSpec(BaseExperiment):
    """Sweep the auxiliary cooling tone and look for max |e0> -> |g0> reset."""

    EXPT_NAME = "s021_aux_sideband_spec"
    TAG = "Cooling"
    X_LABEL = "Cooling Tone Frequency (MHz)"
    TITLE_PREFIX = "Auxiliary-Level Sideband Cooling Spectrum"
    SWEEP_KEYS_TO_REMOVE = ["aux_cool_freq"]
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    def _create_program(self):
        return AuxiliarySidebandSpecProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("aux_cool", "freq", as_array=True)


class AuxiliaryLevelCoolingProgram(AuxiliarySidebandSpecProgram):
    """Fixed-frequency auxiliary-level cooling/reset pulse."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        _add_flux_pulse(self, cfg)
        self.setup_qb_pulse(cfg, "ge", name="qb_ge_pi", gain_key="pi_gain_ge")
        _add_aux_pulse(self, cfg)


class AuxiliaryLevelCooling(BaseExperiment):
    """Run a calibrated auxiliary-level cooling pulse and read out the result."""

    EXPT_NAME = "s022_aux_level_cooling"
    TAG = "Cooling"
    X_LABEL = "Shot"
    TITLE_PREFIX = "Auxiliary-Level Cooling"
    X_SAVE_NAME = "Shot"
    X_SAVE_UNIT = ""
    X_SAVE_SCALE = 1.0
    LivePlot = False

    def _create_program(self):
        return AuxiliaryLevelCoolingProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return np.array([0.0])


__all__ = [
    "AuxiliaryEfSpecProgram", "AuxiliaryEfSpec",
    "AuxiliarySidebandSpecProgram", "AuxiliarySidebandSpec",
    "AuxiliaryLevelCoolingProgram", "AuxiliaryLevelCooling",
]
