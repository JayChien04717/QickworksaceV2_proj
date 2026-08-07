"""Chi-kappa-power characterization of a dispersive readout resonator."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


from ...core.acquisition import acquire_values
from ...core.base_analysis import BaseAnalysis
from ...core.base_experiment import BaseExperiment
from ...core.base_program import BaseProgram
from ...core.experiment_data import ExperimentData, QualityFlag


def _ridge_lorentzian(frequency, offset, amplitude, center, half_width):
    return offset + amplitude / (1.0 + ((frequency - center) / half_width) ** 2)


def _process_ckp_iq(iq, channel):
    """Return a real IQ projection and optional per-state PCA axes."""
    aliases = {
        "amp": "abs",
        "amplitude": "abs",
        "i": "real",
        "avgi": "real",
        "q": "imag",
        "avgq": "imag",
        "optimal": "pca",
    }
    channel = aliases.get(str(channel).lower(), str(channel).lower())
    iq = np.asarray(iq)
    if channel == "pca":
        projected = np.empty(iq.shape, dtype=float)
        axes = []
        for state in range(iq.shape[0]):
            samples = np.column_stack(
                (iq[state].real.ravel(), iq[state].imag.ravel())
            )
            centered = samples - np.mean(samples, axis=0, keepdims=True)
            _, _, vectors = np.linalg.svd(centered, full_matrices=False)
            axis = vectors[0]
            projected[state] = (
                axis[0] * iq[state].real + axis[1] * iq[state].imag
            )
            axes.append(complex(axis[0], axis[1]))
        return projected, channel, axes
    if channel == "abs":
        return np.abs(iq), channel, None
    if channel == "real":
        return np.real(iq), channel, None
    if channel == "imag":
        return np.imag(iq), channel, None
    if channel == "phase":
        return np.unwrap(np.angle(iq), axis=1), channel, None
    raise ValueError(
        "ckp_fit_channel must be one of: pca, abs, real, imag, phase"
    )

def _joint_ckp_model(
    coordinates, qubit_idle, resonator_mid, chi, kappa, peak_stark_shift
):
    """Five-parameter CKP model using ordinary frequencies in MHz."""
    state, resonator_frequency = coordinates
    state_resonance = resonator_mid + (2.0 * state - 1.0) * chi
    return qubit_idle + peak_stark_shift / (
        1.0 + (2.0 * (resonator_frequency - state_resonance) / kappa) ** 2
    )


def _decode_ckp_map(values, resonator_points, qubit_points):
    """Convert QICK loop order (resonator, qubit) to plot order (qubit, resonator)."""
    array = np.asarray(values, dtype=complex).squeeze()
    qick_shape = (resonator_points, qubit_points)
    plot_shape = (qubit_points, resonator_points)
    if array.shape == qick_shape:
        return array.T
    # Compatibility with data that was already transposed by an acquisition
    # wrapper. This branch is intentionally disabled for square maps, where
    # QICK loop order must win because the shapes are indistinguishable.
    if qick_shape != plot_shape and array.shape == plot_shape:
        return array
    if array.size == resonator_points * qubit_points:
        return array.reshape(qick_shape).T
    raise RuntimeError(
        f"Cannot map QICK data shape {array.shape} to CKP map {plot_shape}"
    )


class CKPProgram(BaseProgram):
    """Prepare g/e, Stark-drive the resonator, probe the qubit, then read out."""

    RESONATOR_LOOP = "ckp_resfreqloop"
    QUBIT_LOOP = "ckp_qbfreqloop"

    def _initialize(self, cfg):
        self.declare_gen(ch=cfg["res_ch"], nqz=cfg["nqz_res"])
        self.declare_readout(ch=cfg["ro_ch"], length=cfg["ro_length"])
        self.add_readoutconfig(
            ch=cfg["ro_ch"],
            name="myro",
            freq=cfg["res_freq_ge"],
            gen_ch=cfg["res_ch"],
        )
        self.add_loop(self.RESONATOR_LOOP, cfg["ckp_res_steps"])
        self.add_loop(self.QUBIT_LOOP, cfg["ckp_qb_steps"])

        self.add_gauss(
            ch=cfg["res_ch"],
            name="ckp_readout_env",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        common_res = {
            "ch": cfg["res_ch"],
            "style": "flat_top",
            "envelope": "ckp_readout_env",
            "phase": cfg["res_phase"],
        }
        self.add_pulse(
            **common_res,
            name="ckp_stark_pulse",
            length=cfg.get("ckp_res_length", 1.0),
            freq=cfg["ckp_res_freq"],
            gain=cfg.get("ckp_res_gain", cfg["res_gain_ge"]),
        )
        self.add_pulse(
            **common_res,
            name="res_pulse",
            length=cfg["res_length"],
            ro_ch=cfg["ro_ch"],
            freq=cfg["res_freq_ge"],
            gain=cfg["res_gain_ge"],
        )

        self.setup_qubit_gen(cfg, prefix="ge")
        envelope = self._ensure_qb_envelope(
            cfg, "ge", cfg["qb_ch"], "gauss", length_mult=5
        )
        common_qb = {
            "ch": cfg["qb_ch"],
            "style": "arb",
            "envelope": envelope,
            "phase": cfg["qb_phase"],
        }
        self.add_pulse(
            **common_qb,
            name="ckp_prepare_pi",
            freq=cfg["qb_freq_ge"],
            gain=cfg["pi_gain_ge"],
        )
        self.add_pulse(
            **common_qb,
            name="ckp_probe",
            freq=cfg["ckp_qb_freq"],
            gain=cfg.get("ckp_qb_gain", 0.8 * cfg["pi_gain_ge"]),
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        if cfg["ckp_prepare_e"]:
            self.pulse(ch=cfg["qb_ch"], name="ckp_prepare_pi", t=0)
            self.delay_auto(cfg.get("ckp_prepare_delay", 0.02))

        # The probe is scheduled near the end of the long resonator pulse, so
        # it sees the steady-state ac Stark shift while the resonator is driven.
        self.pulse(ch=cfg["res_ch"], name="ckp_stark_pulse", t=0)
        self.pulse(
            ch=cfg["qb_ch"],
            name="ckp_probe",
            t=cfg.get(
                "ckp_probe_delay",
                max(0.0, cfg.get("ckp_res_length", 1.0) - 5 * cfg["sigma_ge"]),
            ),
        )
        self.delay_auto(cfg.get("ckp_ringdown", 0.2), tag="ckp_ringdown")
        self.measure(cfg)


class CKPAnalysis(BaseAnalysis):
    """Fit both CKP ridges and derive chi, kappa, drive power, and nbar."""

    thresholds = {
        "joint_fit_r2": {"min": 0.85},
        "nbar_resonant": {"min": 0.0},
    }

    def _run(self, data):
        raw_iq = np.asarray(data.raw_iq)
        context = data.metadata.get("analysis_context", {})
        try:
            response, fit_channel, projection_axes = _process_ckp_iq(
                raw_iq, context.get("ckp_fit_channel", "pca")
            )
        except ValueError as exc:
            self._fail(data, str(exc))
            return
        res_freq = np.asarray(data.x_axis, dtype=float)
        qb_freq = np.asarray(data.y_axis, dtype=float)
        expected = (2, qb_freq.size, res_freq.size)
        if response.shape != expected:
            self._fail(data, f"Expected raw_iq shape {expected}, got {response.shape}")
            return
        if min(qb_freq.size, res_freq.size) < 7:
            self._fail(data, "CKP needs at least seven points on each frequency axis")
            return

        contrast = np.ptp(response, axis=1)
        configured_contrast = context.get("ckp_min_slice_contrast")
        min_contrast = (
            0.05 * float(np.nanmedian(contrast))
            if configured_contrast is None
            else float(configured_contrast)
        )
        ridge, ridge_error, contrast = self._fit_slice_ridges(
            response, qb_freq, min_contrast
        )
        valid = np.isfinite(ridge)
        data.analysis_data.update(
            {
                "ckp_response": {
                    "values": response,
                    "dims": ["state", "qubit_frequency", "resonator_frequency"],
                },
                "qubit_resonance_ridge_MHz": {
                    "values": ridge,
                    "dims": ["state", "resonator_frequency"],
                },
                "qubit_resonance_ridge_error_MHz": {
                    "values": ridge_error,
                    "dims": ["state", "resonator_frequency"],
                },
                "slice_contrast": {
                    "values": contrast,
                    "dims": ["state", "resonator_frequency"],
                },
            }
        )
        if np.any(np.sum(valid, axis=1) < max(5, res_freq.size // 4)):
            self._fail(
                data,
                "Too few valid qubit-spectrum slices; increase averaging or "
                "widen the qubit-frequency sweep",
            )
            return

        p0 = self._initial_guess(ridge, res_freq, qb_freq)
        states = np.broadcast_to(np.arange(2)[:, None], ridge.shape)[valid]
        res_points = np.broadcast_to(res_freq[None, :], ridge.shape)[valid]
        ridge_points = ridge[valid]
        res_step = float(np.median(np.abs(np.diff(res_freq))))
        res_span = float(np.ptp(res_freq))
        qb_span = float(np.ptp(qb_freq))
        try:
            popt, pcov = curve_fit(
                _joint_ckp_model,
                (states, res_points),
                ridge_points,
                p0=p0,
                bounds=(
                    [
                        qb_freq.min() - qb_span,
                        res_freq.min(),
                        -res_span,
                        res_step / 10,
                        -2 * qb_span,
                    ],
                    [
                        qb_freq.max() + qb_span,
                        res_freq.max(),
                        res_span,
                        4 * res_span,
                        2 * qb_span,
                    ],
                ),
                maxfev=50_000,
            )
        except (RuntimeError, ValueError) as exc:
            self._fail(data, f"CKP joint fit failed: {exc}")
            return

        error = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        qubit_idle, resonator_mid, chi, kappa, peak_shift = popt
        if abs(chi) <= np.finfo(float).eps:
            self._fail(data, "CKP fit returned chi indistinguishable from zero")
            return
        predicted = _joint_ckp_model((states, res_points), *popt)
        residual = ridge_points - predicted
        denominator = np.sum((ridge_points - np.mean(ridge_points)) ** 2)
        r2 = 1 - np.sum(residual**2) / denominator if denominator > 0 else np.nan

        nbar_resonant = float(peak_shift / (2 * chi))
        nbar_error = (
            abs(nbar_resonant)
            * np.hypot(error[4] / peak_shift, error[2] / chi)
            if peak_shift != 0
            else np.nan
        )
        resonances = np.array([resonator_mid - chi, resonator_mid + chi])
        context = data.metadata["analysis_context"]
        readout_frequency = float(context["res_freq_ge"])
        nbar_readout = nbar_resonant / (
            1 + (2 * (readout_frequency - resonances) / kappa) ** 2
        )
        gain = float(context["ckp_res_gain"])
        photons_per_gain2 = nbar_resonant / gain**2 if gain else np.nan

        # In input-output normalization, |A|^2 is incident photon flux and
        # n_res = 4|A|^2/kappa. kappa_MHz here means kappa/(2*pi).
        photon_flux = nbar_resonant * (2 * np.pi * abs(kappa) * 1e6) / 4
        incident_power_w = (
            6.62607015e-34 * resonator_mid * 1e6 * photon_flux
        )
        incident_power_dbm = (
            10 * np.log10(incident_power_w / 1e-3)
            if incident_power_w > 0
            else np.nan
        )
        lifetime_ns = 1e3 / (2 * np.pi * abs(kappa))
        fit_grid = np.vstack(
            [
                _joint_ckp_model(
                    (np.full(res_freq.shape, state), res_freq), *popt
                )
                for state in range(2)
            ]
        )

        data.fit_params = np.asarray(popt)
        data.fit_errors = error
        data.fit_result = {
            "qubit_idle_MHz": (qubit_idle, error[0]),
            "resonator_mid_MHz": (resonator_mid, error[1]),
            "f_res_g_MHz": (resonances[0], np.hypot(error[1], error[2])),
            "f_res_e_MHz": (resonances[1], np.hypot(error[1], error[2])),
            "chi_MHz": (chi, error[2]),
            "two_chi_MHz": (2 * chi, 2 * error[2]),
            "kappa_MHz": (abs(kappa), error[3]),
            "resonator_lifetime_ns": (lifetime_ns, None),
            "peak_ac_stark_shift_MHz": (peak_shift, error[4]),
            "nbar_resonant": (nbar_resonant, nbar_error),
            "nbar_readout_g": (float(nbar_readout[0]), None),
            "nbar_readout_e": (float(nbar_readout[1]), None),
            "photons_per_gain_squared": (photons_per_gain2, None),
            "incident_photon_flux_per_s": (photon_flux, None),
            "incident_power_W": (incident_power_w, None),
            "incident_power_dBm": (incident_power_dbm, None),
            "joint_fit_r2": (float(r2), None),
        }
        data.analysis_data["ckp_fit_ridge_MHz"] = {
            "values": fit_grid,
            "dims": ["state", "resonator_frequency"],
        }
        data.metadata.update(
            {
                "fit_model": "CKP five-parameter joint Lorentzian",
                "frequency_convention": (
                    "ordinary frequency MHz; kappa_MHz = kappa/(2*pi)"
                ),
                "photon_number_kind": "coherent-state steady-state mean",
                "valid_ridge_points": int(np.count_nonzero(valid)),
                "fit_channel": fit_channel,
                "ckp_iq_projection_axes": projection_axes,
                "ckp_min_slice_contrast_used": min_contrast,
            }
        )
        data.scalar_result = chi

    @staticmethod
    def _fit_slice_ridges(response, qb_freq, min_contrast):
        ridge = np.full((2, response.shape[2]), np.nan)
        error = np.full_like(ridge, np.nan)
        contrast = np.ptp(response, axis=1)
        step = float(np.median(np.abs(np.diff(qb_freq))))
        span = float(np.ptp(qb_freq))
        for state in range(2):
            for index in range(response.shape[2]):
                trace = response[state, :, index]
                if not np.all(np.isfinite(trace)) or np.ptp(trace) < min_contrast:
                    continue
                edge = max(1, qb_freq.size // 8)
                offset = float(np.median(np.r_[trace[:edge], trace[-edge:]]))
                peak = int(np.argmax(np.abs(trace - offset)))
                try:
                    popt, pcov = curve_fit(
                        _ridge_lorentzian,
                        qb_freq,
                        trace,
                        p0=[
                            offset,
                            trace[peak] - offset,
                            qb_freq[peak],
                            max(step, span / 20),
                        ],
                        bounds=(
                            [-np.inf, -np.inf, qb_freq.min(), step / 10],
                            [np.inf, np.inf, qb_freq.max(), span],
                        ),
                        maxfev=20_000,
                    )
                except (RuntimeError, ValueError):
                    continue
                ridge[state, index] = popt[2]
                if np.all(np.isfinite(pcov)):
                    error[state, index] = np.sqrt(max(pcov[2, 2], 0))
        return ridge, error, contrast

    @staticmethod
    def _initial_guess(ridge, res_freq, qb_freq):
        edge = max(2, res_freq.size // 6)
        qubit_idle = float(
            np.nanmedian(
                np.concatenate((ridge[:, :edge].ravel(), ridge[:, -edge:].ravel()))
            )
        )
        resonances = [
            res_freq[np.nanargmax(np.abs(ridge[state] - qubit_idle))]
            for state in range(2)
        ]
        chi = float((resonances[1] - resonances[0]) / 2)
        if abs(chi) < np.finfo(float).eps:
            chi = float(np.ptp(res_freq) / 20)
        peak_shift = float(
            np.nanmedian(
                [
                    ridge[state, np.nanargmax(np.abs(ridge[state] - qubit_idle))]
                    - qubit_idle
                    for state in range(2)
                ]
            )
        )
        if abs(peak_shift) < np.finfo(float).eps:
            peak_shift = -float(np.ptp(qb_freq)) / 5
        return [
            qubit_idle,
            float(np.mean(resonances)),
            chi,
            max(4 * np.median(np.abs(np.diff(res_freq))), np.ptp(res_freq) / 5),
            peak_shift,
        ]

    @staticmethod
    def _fail(data, message):
        data.quality = QualityFlag.BAD
        data.quality_message = message

    def plot(self, data):
        """Plot prepared-g/e maps with fitted qubit-resonance ridges."""
        import matplotlib.pyplot as plt

        response_entry = data.analysis_data.get("ckp_response", {})
        response = np.asarray(response_entry.get("values", []))
        if response.ndim != 3:
            return
        fit_channel = data.metadata.get("fit_channel", "abs")
        ridge = np.asarray(
            data.analysis_data.get("qubit_resonance_ridge_MHz", {}).get(
                "values", []
            )
        )
        fit = np.asarray(
            data.analysis_data.get("ckp_fit_ridge_MHz", {}).get("values", [])
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        for state, ax in enumerate(axes):
            image = ax.pcolormesh(
                data.x_axis,
                data.y_axis,
                response[state],
                shading="auto",
                cmap="viridis",
            )
            if ridge.shape == (2, len(data.x_axis)):
                ax.plot(data.x_axis, ridge[state], "w.", ms=4, label="slice fits")
            if fit.shape == (2, len(data.x_axis)):
                ax.plot(data.x_axis, fit[state], color="tab:red", lw=2, label="CKP fit")
            ax.set(
                title=f"Prepared |{'g' if state == 0 else 'e'}>",
                xlabel="Resonator drive frequency (MHz)",
                ylabel="Qubit probe frequency (MHz)",
            )
            ax.legend(frameon=False)
        fig.colorbar(image, ax=axes, label=f"{fit_channel}(IQ)", pad=0.02)
        chi = data.get_param("chi_MHz")
        kappa = data.get_param("kappa_MHz")
        nbar = data.get_param("nbar_resonant")
        if None not in (chi, kappa, nbar):
            fig.suptitle(
                f"CKP: chi={chi:.4g} MHz, kappa/2pi={kappa:.4g} MHz, "
                f"nbar(res)={nbar:.3g}"
            )
        if all(id(saved) != id(fig) for saved in data.figures):
            data.figures.append(fig)
        plt.show()
        return fig


class CKP(BaseExperiment):
    """Measure dispersive shift, resonator linewidth, and readout photon number."""

    EXPT_NAME = "resonator_ckp"
    TAG = "CKP"
    X_LABEL = "Resonator drive frequency (MHz)"
    Y_LABEL = "Qubit probe frequency (MHz)"
    TITLE_PREFIX = "Chi-Kappa-Power"
    SWEEP_KEYS_TO_REMOVE = ["ckp_res_freq", "ckp_qb_freq"]
    X_SAVE_NAME = "Resonator Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6
    Y_SAVE_NAME = "Qubit Frequency"
    Y_SAVE_UNIT = "Hz"
    Y_SAVE_SCALE = 1e6
    Analysis = CKPAnalysis

    def _create_program(self):
        return self._program(prepare_e=False)

    def _program(self, prepare_e):
        cfg = dict(self.cfg)
        required = (
            "qb_freq_ge",
            "pi_gain_ge",
            "sigma_ge",
            "res_freq_ge",
            "res_gain_ge",
            "ckp_res_freq",
            "ckp_qb_freq",
            "ckp_res_steps",
            "ckp_qb_steps",
        )
        missing = [key for key in required if cfg.get(key) is None]
        if missing:
            raise KeyError(f"CKP missing required config keys: {missing}")
        cfg["ckp_prepare_e"] = bool(prepare_e)
        return CKPProgram(
            self.soccfg,
            reps=cfg["reps"],
            final_delay=cfg["relax_delay"],
            cfg=cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("ckp_stark_pulse", "freq", as_array=True)

    def _extract_sweep_axis_y(self, prog):
        return prog.get_pulse_param("ckp_probe", "freq", as_array=True)

    def run(self, py_avg, **kwargs):
        """Acquire prepared-g/e CKP maps and return analyzed ExperimentData."""
        plot_analysis = bool(kwargs.pop("plot_analysis", False))
        progress = bool(kwargs.pop("progress", True))
        fit_channel = kwargs.pop(
            "ckp_fit_channel",
            kwargs.pop("iq_process", self.cfg.get("ckp_fit_channel", "pca")),
        )
        kwargs.pop("liveplot", None)
        kwargs.pop("show_final_plot", None)
        if kwargs:
            raise TypeError(f"Unsupported CKP.run options: {', '.join(sorted(kwargs))}")
        rounds = int(py_avg)
        if rounds < 1:
            raise ValueError("py_avg must be positive")

        programs = [self._program(False), self._program(True)]
        self._last_prog = programs[-1]
        res_freq = self._axis(
            programs[0].get_pulse_param(
                "ckp_stark_pulse", "freq", as_array=True
            ),
            self.cfg["ckp_res_steps"],
        )
        qb_freq = self._axis(
            programs[0].get_pulse_param("ckp_probe", "freq", as_array=True),
            self.cfg["ckp_qb_steps"],
        )
        iq_maps = []
        for program in programs:
            values = acquire_values(
                program,
                self.soc,
                rounds=rounds,
                progress=progress,
            )
            iq_maps.append(
                _decode_ckp_map(values, res_freq.size, qb_freq.size)
            )
        iq_maps = np.asarray(iq_maps)

        self._sweep_vals_x = res_freq
        self._sweep_vals_y = qb_freq
        self.iqdata = iq_maps
        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=iq_maps,
            x_axis=res_freq,
            y_axis=qb_freq,
            axes={
                "state": {"values": ["g", "e"]},
                "qubit_frequency": {"values": qb_freq, "unit": "MHz"},
                "resonator_frequency": {"values": res_freq, "unit": "MHz"},
            },
            dataset_dims={
                "iq": ["state", "qubit_frequency", "resonator_frequency"]
            },
            metadata={
                "states": ["g", "e"],
                "iq_process": fit_channel,
                "analysis_context": {
                    "ckp_fit_channel": fit_channel,
                    "ckp_res_gain": self.cfg.get(
                        "ckp_res_gain", self.cfg["res_gain_ge"]
                    ),
                    "res_freq_ge": self.cfg["res_freq_ge"],
                    "ckp_min_slice_contrast": self.cfg.get(
                        "ckp_min_slice_contrast"
                    ),
                },
                "pulse_timing_us": {
                    "resonator_drive": self.cfg.get("ckp_res_length", 1.0),
                    "probe_start": self.cfg.get(
                        "ckp_probe_delay",
                        max(
                            0.0,
                            self.cfg.get("ckp_res_length", 1.0)
                            - 5 * self.cfg["sigma_ge"],
                        ),
                    ),
                    "ringdown": self.cfg.get("ckp_ringdown", 0.2),
                },
            },
            avg_count=rounds,
            quality=QualityFlag.NO_INFORMATION,
            x_name=self.X_SAVE_NAME,
            x_unit=self.X_SAVE_UNIT,
            x_scale=self.X_SAVE_SCALE,
            y_name=self.Y_SAVE_NAME,
            y_unit=self.Y_SAVE_UNIT,
            y_scale=self.Y_SAVE_SCALE,
        )
        result = self.Analysis().run(result)
        self.fit_params = result.fit_params
        self.fit_errors = result.fit_errors
        self.result = result
        if plot_analysis:
            self._render_and_capture_analysis(self.Analysis().plot, result)
        return result

    @staticmethod
    def _axis(values, steps):
        axis = np.asarray(
            BaseExperiment._resolve_axis(values, steps), dtype=float
        ).squeeze()
        if axis.ndim != 1:
            axis = np.unique(axis)
        if axis.size != int(steps):
            unique = np.unique(axis)
            if unique.size != int(steps):
                raise RuntimeError(
                    f"Expected {steps} CKP sweep points, got shape {axis.shape}"
                )
            axis = unique
        return axis


ChiKappaPower = CKP

__all__ = [
    "CKP",
    "CKPAnalysis",
    "CKPProgram",
    "ChiKappaPower",
    "_decode_ckp_map",
    "_joint_ckp_model",
]




