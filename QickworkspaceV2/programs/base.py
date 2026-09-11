"""Thin helpers over AveragerProgramV2; native QICK methods stay available."""

from __future__ import annotations

from contextlib import contextmanager

try:
    from qick.asm_v2 import AveragerProgramV2
except ImportError:

    class AveragerProgramV2:
        def __init__(self, *args, **kwargs):
            raise ImportError("Install qickworkspace[hardware] to compile QICK programs")


class BaseProgram(AveragerProgramV2):
    """Subclass _initialize(cfg) and _body(cfg), exactly as in QICK."""

    def __init__(self, soccfg, reps=None, final_delay=None, *, cfg=None, **kwargs):
        from QickworkspaceV2.device.editable import prepare_program_config

        resolved = prepare_program_config(cfg or {}, getattr(self, "TRANSITION", None))
        super().__init__(
            soccfg,
            reps=resolved.get("reps", 100) if reps is None else reps,
            final_delay=resolved.get("relax_delay", 200.0) if final_delay is None else final_delay,
            cfg=resolved,
            **kwargs,
        )

    def make_program(self):
        # Reset compile-time gate frames on every compilation, without declaring
        # hardware. Each experiment initializes its channels and pulses explicitly.
        self.qubits = {q: Qubit(self, self.cfg["qubits"][q], self.cfg["transition"])
                       for q in self.cfg["targets"]}
        return super().make_program()

    def add_sweep_loop(self, cfg, value, default="sweep"):
        """Use the loop name written in the user's QickSweep1D, without translating it."""
        loops = tuple(getattr(value, "spans", {}))
        if len(loops) > 1:
            raise ValueError(
                "This experiment expects one sweep loop; define multiple loops in a custom program"
            )
        self.add_loop(loops[0] if loops else default, cfg["steps"])

    @staticmethod
    def pulse_name(cfg, name):
        return f"{cfg['namespace']}__{name}" if cfg.get("namespace") else name

    def _declare_once(self, ch, **kwargs):
        if not hasattr(self, "_declared_generators"):
            self._declared_generators = {}
        if ch in self._declared_generators:
            if self._declared_generators[ch] != kwargs:
                raise ValueError(f"Generator {ch} declared with incompatible settings")
            return
        self.declare_gen(ch=ch, **kwargs)
        self._declared_generators[ch] = kwargs

    def declare_gen_auto(self, ch, nqz, mixer_key=None, cfg=None):
        kwargs = {"nqz": nqz}
        gencfg = self.soccfg["gens"][ch]
        if gencfg.get("has_mixer", "int4" in gencfg.get("type", "")):
            mixer = (cfg or {}).get(mixer_key)
            if mixer is None:
                raise ValueError(f"Generator {ch} requires an explicit mixer frequency")
            kwargs["mixer_freq"] = mixer
        self._declare_once(ch, **kwargs)

    def setup_qubit_gen(self, cfg, prefix="ge"):
        suffix = "" if prefix == "ge" else f"_{prefix}"
        self.declare_gen_auto(cfg[f"qb_ch{suffix}"], cfg[f"nqz_qb{suffix}"], f"qb_mixer{suffix}", cfg)

    def setup_resonator(self, cfg, prefix="ge"):
        if cfg.get("readout_mode") == "mux":
            raise ValueError("Use setup_readout for a multiplexed readout group")
        self.declare_gen_auto(cfg["res_ch"], cfg["nqz_res"], "res_mixer", cfg)
        ch = cfg["ro_ch"]
        freq = cfg.get(f"res_freq_{prefix}", cfg["res_freq_ge"])
        name = self.pulse_name(cfg, "myro")
        if "tproc_ctrl" in self.soccfg["readouts"][ch]:
            self.declare_readout(ch=ch, length=cfg["ro_length"])
            self.add_readoutconfig(ch=ch, name=name, freq=freq, gen_ch=cfg["res_ch"])
        else:
            if hasattr(freq, "spans") and freq.spans:
                raise ValueError("This readout has no tProc frequency control; use a host sweep")
            self.declare_readout(ch=ch, length=cfg["ro_length"], freq=freq, gen_ch=cfg["res_ch"])
        style = cfg.get("res_pulse_type", "const")
        pulse = dict(
            ch=cfg["res_ch"],
            name=self.pulse_name(cfg, "res_pulse"),
            ro_ch=ch,
            style=style,
            freq=freq,
            phase=cfg.get("res_phase", 0),
            gain=cfg.get(f"res_gain_{prefix}", cfg["res_gain_ge"]),
        )
        if style in ("arb", "flat_top"):
            envelope = self.pulse_name(cfg, "res_envelope")
            self.add_gauss(
                ch=cfg["res_ch"],
                name=envelope,
                sigma=cfg["res_sigma"],
                length=cfg["res_sigma"] * cfg.get("res_length_mult", 5),
                even_length=True,
            )
            pulse["envelope"] = envelope
        if style in ("const", "flat_top"):
            pulse["length"] = cfg["res_length"]
        if style not in ("const", "arb", "flat_top"):
            raise ValueError(f"Unknown readout pulse style {style}")
        self.add_pulse(**pulse)

    def setup_qb_pulse(
        self,
        cfg,
        prefix="ge",
        pulse_type=None,
        shape=None,
        name="qb_pulse",
        phase=None,
        gain_key=None,
        gain_override=None,
        ch=None,
        length_mult=None,
    ):
        ch = cfg["qb_ch" if prefix == "ge" else f"qb_ch_{prefix}"] if ch is None else ch
        pulse_type = pulse_type or cfg.get("pulse_type", cfg.get(f"pulse_type_{prefix}", "arb"))
        shape = shape or cfg.get(f"shape_{prefix}", "gauss")
        if pulse_type == "drag":
            pulse_type, shape = "arb", "drag"
        gain = gain_override if gain_override is not None else cfg[gain_key or f"qb_gain_{prefix}"]
        if gain is None:
            raise ValueError(f"{cfg.get('name', 'qubit')}: missing {gain_key or 'pulse gain'}")
        phase = cfg.get(f"qb_phase_{prefix}", cfg.get("qb_phase", 0)) if phase is None else phase
        full_name = self.pulse_name(cfg, name)
        params = dict(
            ch=ch, name=full_name, style=pulse_type, freq=cfg[f"qb_freq_{prefix}"], phase=phase, gain=gain
        )
        if pulse_type not in ("const", "arb", "flat_top"):
            raise ValueError(f"Unsupported pulse style {pulse_type}")
        if pulse_type in ("arb", "flat_top"):
            sigma = cfg[f"sigma_{prefix}"]
            length = sigma * (length_mult or cfg.get(f"length_mult_{prefix}", 5))
            if shape in ("gauss", "gaussian"):
                shape, create = "gauss", self.add_gauss
                envelope_args = {"sigma": sigma, "length": length}
            elif shape == "drag":
                alpha, delta = (
                    cfg.get(f"drag_alpha_{prefix}", cfg.get("drag_alpha")),
                    cfg.get(f"drag_delta_{prefix}"),
                )
                if alpha is None or not delta:
                    raise ValueError("DRAG needs explicit alpha and nonzero delta_mhz")
                create = self.add_DRAG
                envelope_args = {"sigma": sigma, "length": length, "delta": delta, "alpha": alpha}
            elif shape in ("cos", "cosine"):
                shape, create = "cosine", self.add_cosine
                envelope_args = {"length": length}
            else:
                raise ValueError(f"Unsupported envelope {shape}")
            # Gain and phase belong to pulses; identical envelopes share generator RAM.
            key = (ch, shape, tuple(envelope_args.items()))
            cache = self.__dict__.setdefault("_qubit_envelopes", {})
            env = cache.get(key)
            if env is None:
                env = full_name + "__env"
                create(ch=ch, name=env, **envelope_args, even_length=True)
                cache[key] = env
            params["envelope"] = env
        if pulse_type == "const":
            params["length"] = cfg[f"qb_length_{prefix}"]
        elif pulse_type == "flat_top":
            params["length"] = cfg[f"qb_flat_top_length_{prefix}"]
        self.add_pulse(**params)
        return full_name

    def setup_standard_gates(self, cfg, prefix="ge", pulse_type=None, shape=None):
        for name, phase, gain in [
            ("x", 0, "pi"),
            ("y", 90, "pi"),
            ("halfx", 0, "pi2"),
            ("halfy", 90, "pi2"),
            ("mhalfx", 180, "pi2"),
            ("mhalfy", -90, "pi2"),
        ]:
            self.setup_qb_pulse(
                cfg,
                prefix,
                pulse_type=pulse_type,
                shape=shape,
                name=f"{name}_{prefix}",
                phase=phase + cfg.get(f"qb_phase_{prefix}", 0),
                gain_key=f"{gain}_gain_{prefix}",
            )

    def setup_readout(self, cfg):
        """Declare resonator generators, ADCs and readout pulses together.

        Qubit drive channels and gate pulses are declared by the experiment.
        """
        self._using_device_helpers = True
        for group_id, group in cfg["readout_groups"].items():
            active = [q for q in cfg["targets"] if q in group["members"]]
            if not active:
                continue
            if group["mode"] == "direct":
                self.setup_resonator(cfg["qubits"][active[0]])
                continue
            port = group["port"]
            if any(cfg["qubits"][q].get("res_pulse_type", "const") != "const" for q in active):
                raise ValueError(
                    "MUX readout supports constant masked pulses; res_sigma is for direct readout"
                )
            size = max(m["tone_slot"] for m in group["members"].values()) + 1
            freqs, gains, phases = [0.0] * size, [0.0] * size, [0.0] * size
            for member in group["members"].values():
                slot = member["tone_slot"]
                freqs[slot], gains[slot], phases[slot] = (
                    member["frequency_mhz"],
                    member["gain"],
                    member["phase_deg"],
                )
            kwargs = dict(
                nqz=port["nqz"], mux_freqs=freqs, mux_gains=gains, ro_ch=cfg["qubits"][active[0]]["ro_ch"]
            )
            capability = self.soccfg["gens"][port["channel"]]
            if capability.get("has_phase", True):
                kwargs["mux_phases"] = phases
            elif any(phases):
                raise ValueError("This mux generator does not support nonzero tone phases")
            if capability.get("has_mixer", False):
                if port["mixer_mhz"] is None:
                    raise ValueError("Mux generator requires mixer_mhz")
                kwargs["mixer_freq"] = port["mixer_mhz"]
            self._declare_once(port["channel"], **kwargs)
            for q in active:
                qc = cfg["qubits"][q]
                ch = qc["ro_ch"]
                if "tproc_ctrl" in self.soccfg["readouts"][ch]:
                    self.declare_readout(ch=ch, length=qc["ro_length"])
                    self.add_readoutconfig(
                        ch=ch, name=self.pulse_name(qc, "myro"), freq=qc["res_freq_ge"], gen_ch=qc["res_ch"]
                    )
                else:
                    self.declare_readout(
                        ch=ch, length=qc["ro_length"], freq=qc["res_freq_ge"], gen_ch=qc["res_ch"]
                    )
            self.add_pulse(
                ch=port["channel"],
                name=f"{group_id}__readout",
                style="const",
                length=group["length_us"],
                mask=[group["members"][q]["tone_slot"] for q in active],
            )

    def qubit(self, name):
        return self.qubits[name]

    def wait_pulses(self):
        """Advance past active pulses/readouts with one tProc timing tick of margin.

        Generator envelope length and tProc time use different clocks. A zero
        auto-delay can round down, leaving a fraction of a tick on the drive.
        Raw delay_auto remains available unchanged for user-written timing.
        """
        self.delay_auto(self.soccfg.cycles2us(1))

    @contextmanager
    def parallel(self):
        """Explicit simultaneous gate layer; each drive may occur only once."""
        if getattr(self, "_parallel_channels", None) is not None:
            raise ValueError("Nested parallel layers are not supported")
        self._parallel_channels = set()
        try:
            yield
        finally:
            self._parallel_channels = None
            self.wait_pulses()

    def measure(self, cfg=None):
        cfg = self.cfg if cfg is None else cfg
        if not getattr(self, "_using_device_helpers", False):
            ch = cfg["ro_ch"]
            if "tproc_ctrl" in self.soccfg["readouts"][ch]:
                self.send_readoutconfig(ch=ch, name=self.pulse_name(cfg, "myro"), t=0)
            self.pulse(ch=cfg["res_ch"], name=self.pulse_name(cfg, "res_pulse"), t=0)
            self.trigger(ros=[ch], pins=cfg.get("pins", []), t=cfg["trig_time"])
            return
        for q in cfg["targets"]:
            qc = cfg["qubits"][q]
            if "tproc_ctrl" in self.soccfg["readouts"][qc["ro_ch"]]:
                self.send_readoutconfig(ch=qc["ro_ch"], name=self.pulse_name(qc, "myro"), t=0)
        triggers = {}
        for group_id, group in cfg["readout_groups"].items():
            active = [q for q in cfg["targets"] if q in group["members"]]
            if not active:
                continue
            pulse = (
                f"{group_id}__readout"
                if group["mode"] == "mux"
                else self.pulse_name(cfg["qubits"][active[0]], "res_pulse")
            )
            self.pulse(ch=group["port"]["channel"], name=pulse, t=0)
            triggers.setdefault(group["trigger_us"], []).extend(cfg["qubits"][q]["ro_ch"] for q in active)
        for index, (t, ros) in enumerate(sorted(triggers.items())):
            self.trigger(ros=ros, pins=cfg.get("pins", []) if index == 0 else [], t=t)

    def setup_coupler(self, cfg, edge):
        spec = cfg["couplers"][edge]
        p, port = spec["pulse"], spec["port"]
        flat = dict(
            namespace=edge,
            qb_ch=port["channel"],
            nqz_qb=port["nqz"],
            qb_mixer=port["mixer_mhz"],
            qb_freq_ge=spec["frequency_mhz"],
            qb_gain_ge=p["gain"],
            qb_phase=p["phase_deg"],
            sigma_ge=p["sigma_us"],
            qb_length_ge=p["length_us"],
            qb_flat_top_length_ge=p["length_us"],
            pulse_type_ge=p["style"],
            shape_ge=p["shape"],
            length_mult_ge=p["sigma_count"],
            drag_alpha_ge=p["drag_alpha"],
            drag_delta_ge=p["drag_delta_mhz"],
        )
        self.setup_qubit_gen(flat)
        self.setup_qb_pulse(flat, name="interaction")

    def cz(self, edge):
        spec = self.cfg["couplers"][edge]
        if not spec["calibrated"]:
            raise ValueError(f"{edge}: CZ pulse is not calibrated; use an explicit interaction scan")
        self.interaction(edge)
        for q, phase in spec["phase_corrections_deg"].items():
            self.qubit(q).z(phase)

    def interaction(self, edge):
        if getattr(self, "_parallel_channels", None) is not None:
            raise ValueError("Interactions must be outside single-qubit parallel layers")
        self.pulse(ch=self.cfg["couplers"][edge]["port"]["channel"], name=f"{edge}__interaction", t=0)
        self.wait_pulses()

    def apply_cool(self, cfg, style="flat_top"):
        """Native extension: caller provides the measured cooling configuration."""
        for i in (1, 2):
            ch = cfg[f"cool_ch{i}"]
            self.declare_gen_auto(ch, cfg.get(f"nqz_cool_ch{i}", 2), f"cool_mixer{i}", cfg)
            name = self.pulse_name(cfg, f"cool_pulse{i}")
            kwargs = dict(
                ch=ch,
                name=name,
                style=style,
                length=cfg["cool_length"],
                freq=cfg[f"cool_freq_{i}"],
                phase=0,
                gain=cfg[f"cool_gain_{i}"],
            )
            if style == "flat_top":
                env = name + "__env"
                self.add_gauss(
                    ch=ch, name=env, sigma=cfg["res_sigma"], length=5 * cfg["res_sigma"], even_length=True
                )
                kwargs["envelope"] = env
            self.add_pulse(**kwargs)

    def cooling_body(self, cfg, ring_down=0.5):
        if cfg.get("cooling", False):
            for i in (1, 2):
                self.pulse(ch=cfg[f"cool_ch{i}"], name=self.pulse_name(cfg, f"cool_pulse{i}"), t=0)
            self.delay_auto(ring_down)


class Qubit:
    """Gate view. Default calls are sequential; parallel() is explicit."""

    def __init__(self, program, cfg, transition):
        self.program, self.cfg, self.transition = program, cfg, transition
        self.frame = 0.0
        self._counter = 0

    def _gate(self, name, phase, gain):
        p, cfg, prefix = self.program, self.cfg, self.transition
        channel = cfg["qb_ch" if prefix == "ge" else f"qb_ch_{prefix}"]
        parallel = getattr(p, "_parallel_channels", None)
        if parallel is not None:
            if channel in parallel:
                raise ValueError(f"Drive {channel} used twice in one simultaneous layer")
            parallel.add(channel)
        if self.frame != 0:
            self._counter += 1
            name = f"frame_{self._counter}_{prefix}"
            p.setup_qb_pulse(
                cfg,
                prefix,
                name=name,
                phase=phase + self.frame + cfg.get(f"qb_phase_{prefix}", 0),
                gain_key=f"{gain}_gain_{prefix}",
            )
        else:
            name = f"{name}_{prefix}"
        p.pulse(ch=channel, name=p.pulse_name(cfg, name), t=0)
        if parallel is None:
            p.wait_pulses()

    def x(self):
        self._gate("x", 0, "pi")

    def y(self):
        self._gate("y", 90, "pi")

    def halfx(self):
        self._gate("halfx", 0, "pi2")

    def halfy(self):
        self._gate("halfy", 90, "pi2")

    def mhalfx(self):
        self._gate("mhalfx", 180, "pi2")

    def mhalfy(self):
        self._gate("mhalfy", -90, "pi2")

    def h(self):
        """Hadamard up to global phase, from calibrated Y90 followed by X180."""
        if getattr(self.program, "_parallel_channels", None) is not None:
            raise ValueError("h uses two pulse layers; place halfy and x in separate parallel contexts")
        self.halfy()
        self.x()

    def z(self, degrees):
        # A compile-time virtual frame. Reset naturally when _body is compiled.
        self.frame = (self.frame + float(degrees)) % 360

    def wait(self, time_us):
        if getattr(self.program, "_parallel_channels", None) is not None:
            raise ValueError("wait must be outside a parallel gate layer")
        self.program.delay_auto(time_us)
