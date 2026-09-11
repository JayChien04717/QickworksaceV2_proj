"""
Setup/single_shot — s000: Single-shot readout (g/e/f) and optimization.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ...core.base_program import BaseProgram
from ...core.experiment_data import ExperimentData, QualityFlag
from ...tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml
from .singleshot_utils import (
    general_hist,
    hist,
    fast_histogram_metrics,
    plot_hist,
)



class SingleShotProgram_gef(BaseProgram):
    """QICK program for g/e/f single-shot readout with multi-trigger body."""

    def _initialize(self, cfg):
        """Initialize pulse and acquisition resources.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, 'ge')
        # A physical generator has one mixer configuration. If ge and ef share
        # the channel, declare it once and let the per-pulse DDS set each
        # absolute transition frequency.
        if cfg.get("shot_f", False) and cfg["qb_ch_ef"] != cfg["qb_ch"]:
            self.setup_qubit_gen(cfg, 'ef')
        self.add_loop("shotloop", cfg["shots"])
        self.setup_qb_pulse(cfg, 'ge', name="qb_ge_pulse", gain_key="pi_gain_ge")
        if cfg.get("shot_f", False):
            self.setup_qb_pulse(cfg, 'ef', name="qb_ef_pulse", gain_key="pi_gain_ef")

    def _body(self, cfg):
        """Execute one iteration of the pulse sequence.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])
        self.delay_auto(cfg["relax_delay"], tag="relax_wait")
        self.pulse(ch=cfg["qb_ch"], name="qb_ge_pulse", t=0)
        self.delay_auto(0.01, tag="wait")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])
        if cfg.get("shot_f", False):
            self.delay_auto(cfg["relax_delay"], tag="relax_wait2")
            self.pulse(ch=cfg["qb_ch"], name="qb_ge_pulse", t=0)
            self.delay_auto(0.01, tag="wait1")
            self.pulse(ch=cfg["qb_ch_ef"], name="qb_ef_pulse", t=0)
            self.delay_auto(0.01)
            self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
            self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class SingleShotOptProgram(SingleShotProgram_gef):
    """Single-shot optimizer program; intentionally identical to acquisition."""


class SingleShot_gef:
    """Single-shot readout for g/e/f state discrimination."""

    def __init__(self, config):
        """Initialize the SingleShot_gef instance.

        Parameters
        ----------
        config : Any
            Experiment configuration.

        Raises
        ------
        RuntimeError
            If the operation cannot be completed.
        """
        from ...core.base_experiment import BaseExperiment
        if BaseExperiment._soc is None:
            raise RuntimeError("Call BaseExperiment.setup(soc, soccfg, data_path) first.")
        self.soc = BaseExperiment._soc
        self.soccfg = BaseExperiment._soccfg
        self.cfg = config

    def run(self, SHOTS, shot_f=False):
        """Run the operation.

        Parameters
        ----------
        SHOTS : Any
            Value for ``SHOTS``.
        shot_f : Any, default: False
            Value for ``shot_f``.

        Returns
        -------
        Any
            Result of the operation.
        """
        self.cfg["shots"] = SHOTS
        self.cfg["shot_f"] = shot_f
        prog = SingleShotProgram_gef(
            self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg
        )
        iq_list = prog.acquire(self.soc, rounds=1, progress=True)
        Ig = iq_list[0][0, :, 0]
        Qg = iq_list[0][0, :, 1]
        Ie = iq_list[0][1, :, 0]
        Qe = iq_list[0][1, :, 1]
        if shot_f:
            If = iq_list[0][2, :, 0]
            Qf = iq_list[0][2, :, 1]
            self.data = {"Ig": Ig, "Qg": Qg, "Ie": Ie, "Qe": Qe, "If": If, "Qf": Qf}
        else:
            self.data = {"Ig": Ig, "Qg": Qg, "Ie": Ie, "Qe": Qe}
        states = ["g", "e", "f"] if shot_f else ["g", "e"]
        iq_by_state = [Ig + 1j * Qg, Ie + 1j * Qe]
        if shot_f:
            iq_by_state.append(If + 1j * Qf)
        self.result = ExperimentData(
            experiment_type="s000_singleshot_gef" if shot_f else "s000_singleshot_ge",
            raw_iq=np.stack(iq_by_state, axis=0),
            metadata={"qubit": self.cfg.get("name"), "states": states, "shots": int(SHOTS)},
            axes={
                "state": {"values": states, "label": "Prepared state"},
                "shot": {"values": np.arange(SHOTS), "label": "Shot", "unit": "#"},
            },
            dataset_dims={"iq": ["state", "shot"]},
            data_kind="single_shot",
            analysis_id="single_shot",
            plot_id="single_shot_iq",
            quality=QualityFlag.GOOD,
            avg_count=1,
        )
        return self.data

    def plot(self, fid_avg=False, verbose=True, *, plot_analysis=True):
        """Plot the operation.

        Parameters
        ----------
        fid_avg : Any, default: False
            Value for ``fid_avg``.
        verbose : Any, default: True
            Value for ``verbose``.
        plot_analysis : Any, default: True
            Value for ``plot_analysis``.

        Returns
        -------
        Any
            Result of the operation.
        """
        before_figures = set(plt.get_fignums())
        analyzed = hist(self.data, plot=True, verbose=verbose, fid_avg=fid_avg)
        if getattr(self, "result", None) is not None:
            known = {id(figure) for figure in self.result.figures}
            for number in plt.get_fignums():
                if number not in before_figures:
                    figure = plt.figure(number)
                    if id(figure) not in known:
                        self.result.figures.append(figure)
                        known.add(id(figure))
        if getattr(self, "result", None) is not None:
            fidelity = float(analyzed[0][0])
            thresholds = np.asarray(analyzed[1], dtype=float)
            rotation_deg = float(analyzed[2])
            confusion = np.asarray(analyzed[3], dtype=float)
            self.result.fit_result.update({
                "fidelity": (fidelity, None),
                "rotation_deg": (rotation_deg, None),
                "threshold": (float(thresholds[0]), None) if thresholds.size else (None, None),
            })
            self.result.analysis_data.update({
                "thresholds": {"values": thresholds, "dims": ["threshold"]},
                "confusion_matrix_pct": {
                    "values": confusion,
                    "dims": ["prepared_state", "declared_state"],
                },
            })
            states = self.result.metadata.get("states", ["g", "e"])
            self.result.axes.update({
                "threshold": {"values": np.arange(thresholds.size)},
                "prepared_state": {"values": states},
                "declared_state": {"values": states},
            })
            self.result.quality = QualityFlag.GOOD if fidelity >= 0.85 else QualityFlag.WARNING
        return analyzed

    def saveLabber(self, qb_idx, yoko_value=None, config_all=None, filename_mode="random"):
        """Save Labber.

        Parameters
        ----------
        qb_idx : Any
            Value for ``qb_idx``.
        yoko_value : Any, default: None
            Value for ``yoko_value``.
        config_all : Any, default: None
            Value for ``config_all``.
        filename_mode : Any, default: 'random'
            Value for ``filename_mode``.

        Returns
        -------
        Any
            Result of the operation.
        """
        from ...core.base_experiment import BaseExperiment
        has_f = "If" in self.data
        expt_name = ("s000_singleshot_gef" if has_f else "s000_singleshot_ge") + f"_{qb_idx}"
        save_dir = BaseExperiment._require_data_path()
        file_path = get_next_filename_labber(save_dir, expt_name, yoko_value)
        print("Current data file: " + file_path)
        dict_val = (
            config_all.to_yaml(q_id=qb_idx)
            if config_all is not None
            else config_to_yaml(self.cfg)
        )
        shotdata = np.array([
            self.data["Ig"] + 1j * self.data["Qg"],
            self.data["Ie"] + 1j * self.data["Qe"],
        ] + ([self.data["If"] + 1j * self.data["Qf"]] if has_f else []))
        states = [0, 1, 2] if has_f else [0, 1]
        saved_path = hdf5_generator(
            filepath=file_path,
            x_info={"name": "# shot", "unit": "#", "values": np.arange(self.cfg["shots"])},
            y_info={"name": "State", "unit": "", "values": states},
            z_info={"name": "Signal", "unit": "ADC unit", "values": shotdata},
            comment=f"{dict_val}", tag="SingleShot",
            result=self.result,
            filename_mode=filename_mode,
        )
        return str(saved_path)



class SingleShot_ge_opt:
    """Minimal empirical optimization of readout length and gain.

    Frequency stays fixed at ``cfg['res_freq_ge']``.  Every grid point uses
    the same robust median rotation and exact all-shot threshold; no GMM,
    Gaussian process, Pareto filter, or histogram-bin tuning is involved.
    """

    def __init__(self, config):
        from ...core.base_experiment import BaseExperiment
        if BaseExperiment._soc is None:
            raise RuntimeError("Call BaseExperiment.setup(soc, soccfg, data_path) first.")
        self.soc = BaseExperiment._soc
        self.soccfg = BaseExperiment._soccfg
        self.cfg = config

    @staticmethod
    def _axis(values, fallback, name):
        if values is None:
            values = [fallback]
        elif np.isscalar(values):
            values = [values]
        result = np.asarray(values, dtype=float)
        if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a non-empty finite 1-D sweep")
        return result

    def run(self, SHOTS, sweep_para: dict, shot_f=False):
        """Acquire a length x gain grid for prepared g and e states."""
        if shot_f:
            raise ValueError("The minimal optimizer is ge-only; use SingleShot_gef for f-state analysis.")
        if int(SHOTS) < 1:
            raise ValueError("SHOTS must be positive")
        extra = set(sweep_para) - {"length", "gain"}
        if extra:
            raise ValueError("Only 'length' and 'gain' are swept; frequency stays fixed.")

        self.length_sweep = self._axis(
            sweep_para.get("length"), self.cfg["ro_length"], "length"
        )
        self.gain_sweep = self._axis(
            sweep_para.get("gain"), self.cfg["res_gain_ge"], "gain"
        )
        self.fixed_frequency = float(self.cfg["res_freq_ge"])
        self.shots = int(SHOTS)
        self.cfg.update({"shots": self.shots, "shot_f": False})

        shape = (len(self.length_sweep), len(self.gain_sweep), self.shots)
        self.I_g_array = np.empty(shape)
        self.Q_g_array = np.empty(shape)
        self.I_e_array = np.empty(shape)
        self.Q_e_array = np.empty(shape)

        total = len(self.length_sweep) * len(self.gain_sweep)
        points = (
            (li, gi, length, gain)
            for li, length in enumerate(self.length_sweep)
            for gi, gain in enumerate(self.gain_sweep)
        )
        for li, gi, length, gain in tqdm(points, total=total, desc="Length x gain"):
            self.cfg.update({
                "steps": self.shots,
                "ro_length": float(length),
                "res_gain_ge": float(gain),
                "res_freq_ge": self.fixed_frequency,
            })
            program = SingleShotOptProgram(
                self.soccfg, reps=1,
                final_delay=self.cfg["relax_delay"], cfg=self.cfg,
            )
            iq = program.acquire(self.soc, rounds=1, progress=False)[0]
            self.I_g_array[li, gi] = iq[0, :, 0]
            self.Q_g_array[li, gi] = iq[0, :, 1]
            self.I_e_array[li, gi] = iq[1, :, 0]
            self.Q_e_array[li, gi] = iq[1, :, 1]

        self.data = {
            "Ig": self.I_g_array, "Qg": self.Q_g_array,
            "Ie": self.I_e_array, "Qe": self.Q_e_array,
        }
        return self.data

    def analyze(self):
        """Choose the largest ``fidelity * e_core * e_survival`` score."""
        if not hasattr(self, "data"):
            raise RuntimeError("Call run() before analyze().")

        shape = (len(self.length_sweep), len(self.gain_sweep))
        names = (
            "fidelity", "e_to_g", "g_to_e", "e_core", "g_core",
            "e_tail", "g_tail", "score", "threshold", "rotation_deg",
        )
        arrays = {name: np.empty(shape) for name in names}

        for li in range(shape[0]):
            for gi in range(shape[1]):
                metrics = fast_histogram_metrics({
                    "Ig": self.I_g_array[li, gi],
                    "Qg": self.Q_g_array[li, gi],
                    "Ie": self.I_e_array[li, gi],
                    "Qe": self.Q_e_array[li, gi],
                })
                arrays["fidelity"][li, gi] = metrics["fid"]
                arrays["e_to_g"][li, gi] = metrics["e_to_g_error"]
                arrays["g_to_e"][li, gi] = metrics["g_to_e_error"]
                arrays["e_core"][li, gi] = metrics["e_core_fraction"]
                arrays["g_core"][li, gi] = metrics["g_core_fraction"]
                arrays["e_tail"][li, gi] = metrics["e_tail_fraction"]
                arrays["g_tail"][li, gi] = metrics["g_tail_fraction"]
                arrays["score"][li, gi] = metrics["readout_score"]
                arrays["threshold"][li, gi] = metrics["threshold"]
                arrays["rotation_deg"][li, gi] = metrics["rotation_deg"]

        best_idx = np.unravel_index(np.nanargmax(arrays["score"]), shape)
        li, gi = best_idx
        self.metrics = arrays
        self.best_index = best_idx
        self.fid_Array = arrays["fidelity"]
        self.leakage_array = arrays["e_to_g"]
        self.thermal_array = arrays["g_to_e"]
        self.selection_score_array = arrays["score"]
        self.best = {
            "length": float(self.length_sweep[li]),
            "gain": float(self.gain_sweep[gi]),
            "frequency": self.fixed_frequency,
            "fidelity": float(arrays["fidelity"][best_idx]),
            "score": float(arrays["score"][best_idx]),
            "e_to_g_error": float(arrays["e_to_g"][best_idx]),
            "g_to_e_error": float(arrays["g_to_e"][best_idx]),
            "e_core_fraction": float(arrays["e_core"][best_idx]),
            "e_tail_fraction": float(arrays["e_tail"][best_idx]),
            "threshold": float(arrays["threshold"][best_idx]),
            "rotation_deg": float(arrays["rotation_deg"][best_idx]),
        }
        self.cfg.update({
            "ro_length": self.best["length"],
            "res_gain_ge": self.best["gain"],
        })

        raw_iq = np.stack((
            self.I_g_array + 1j * self.Q_g_array,
            self.I_e_array + 1j * self.Q_e_array,
        ), axis=2)
        analysis_data = {
            name: {"values": values, "dims": ["length", "gain"]}
            for name, values in arrays.items()
        }
        self.result = ExperimentData(
            experiment_type="s000_singleshot_ge_opt",
            raw_iq=raw_iq,
            metadata={
                "qubit": self.cfg.get("name"),
                "states": ["g", "e"],
                "shots": self.shots,
                "best": self.best,
                "optimizer_method": "empirical_length_gain",
                "score_formula": "fidelity * e_core_fraction * (1 - e_to_g_error)",
            },
            axes={
                "length": {"values": self.length_sweep, "unit": "us"},
                "gain": {"values": self.gain_sweep, "unit": "DAC unit"},
                "state": {"values": ["g", "e"]},
                "shot": {"values": np.arange(self.shots), "unit": "#"},
            },
            dataset_dims={"iq": ["length", "gain", "state", "shot"]},
            analysis_data=analysis_data,
            fit_result={
                "best_length": (self.best["length"], None),
                "best_gain": (self.best["gain"], None),
                "best_fidelity": (self.best["fidelity"], None),
                "best_score": (self.best["score"], None),
                "best_e_to_g_error": (self.best["e_to_g_error"], None),
                "best_e_tail_fraction": (self.best["e_tail_fraction"], None),
            },
            data_kind="single_shot_optimization",
            analysis_id="single_shot_optimization",
            plot_id="single_shot_optimization",
            quality=QualityFlag.GOOD,
            avg_count=1,
        )

        print("\n--- Best empirical readout ---")
        print(
            f"length={self.best['length']:.3f} us  gain={self.best['gain']:.6f}  "
            f"freq(fixed)={self.fixed_frequency:.6f} MHz"
        )
        print(
            f"score={self.best['score']:.4f}  fidelity={self.best['fidelity']:.4f}  "
            f"e->g={self.best['e_to_g_error']:.4f}  "
            f"e-tail={self.best['e_tail_fraction']:.4f}"
        )
        return self.best["length"], self.best["gain"]

    def plot_grid_analysis(self):
        """Plot only the four metrics that drive the practical choice."""
        if not hasattr(self, "metrics"):
            self.analyze()
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        panels = (
            ("score", "Useful score", "viridis"),
            ("fidelity", "All-shot fidelity", "RdYlGn"),
            ("e_to_g", "e -> g error (T1-sensitive)", "Reds"),
            ("e_tail", "e main-peak tail", "Oranges"),
        )
        extent = [
            self.gain_sweep[0], self.gain_sweep[-1],
            self.length_sweep[-1], self.length_sweep[0],
        ]
        for ax, (key, title, cmap) in zip(axes.flat, panels):
            image = ax.imshow(
                self.metrics[key], origin="upper", aspect="auto",
                extent=extent, cmap=cmap,
            )
            ax.scatter(self.best["gain"], self.best["length"], marker="*", s=130,
                       color="cyan", edgecolor="black")
            ax.set(title=title, xlabel="Readout gain", ylabel="Readout length (us)")
            fig.colorbar(image, ax=ax)
        fig.suptitle(
            f"Q{str(self.cfg.get('name', '')).lstrip('Q')} readout optimization  "
            f"(frequency fixed at {self.fixed_frequency:.4f} MHz)"
        )
        fig.tight_layout()
        return fig

    plot = plot_grid_analysis

    def plot_best_histogram(self):
        """Show the ordinary single-shot diagnostic for the chosen point."""
        if not hasattr(self, "best_index"):
            self.analyze()
        li, gi = self.best_index
        return hist({
            "Ig": self.I_g_array[li, gi], "Qg": self.Q_g_array[li, gi],
            "Ie": self.I_e_array[li, gi], "Qe": self.Q_e_array[li, gi],
        }, plot=True, verbose=True, title=(
            f"Best L={self.best['length']:.3f} us, G={self.best['gain']:.6f}"
        ))


__all__ = [
    "SingleShotProgram_gef",
    "SingleShot_gef",
    "SingleShotOptProgram",
    "SingleShot_ge_opt",
    "plot_hist",
    "general_hist",
    "hist",
]