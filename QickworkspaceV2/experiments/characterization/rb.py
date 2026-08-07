"""
Characterization/rb — s015: Single Qubit RB, Interleaved RB, AutoRB.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment
from ...core.acquisition import acquire_values
from ...core.experiment_data import ExperimentData, QualityFlag
from ...analysis.rb import RBAnalysis
from ...tools.fitting import fitrb, rb_func, rb_error, error_fit_err


_GEN_TO_QICK = {
    "I":    None,
    "X":    "x180_{pfx}",
    "Y":    "y180_{pfx}",
    "X/2":  "x90_{pfx}",
    "-X/2": "x90m_{pfx}",
    "Y/2":  "y90_{pfx}",
    "-Y/2": "y90m_{pfx}",
}

_INTERLEAVED_FILE_SUFFIX = {
    "X":    "X",
    "Y":    "Y",
    "X/2":  "halfX",
    "-X/2": "halfXm",
    "Y/2":  "halfY",
    "-Y/2": "halfYm",
}


def _safe_rb_file_suffix(label):
    """Return the safe rb file suffix result.

    Parameters
    ----------
    label : Any
        Value for ``label``.

    Returns
    -------
    Any
        Result of the operation.
    """
    suffix = _INTERLEAVED_FILE_SUFFIX.get(label, str(label))
    for char in '<>:"/\\|?*':
        suffix = suffix.replace(char, "_")
    return suffix


def _rb_sample_matrix(raw, n_depths, n_samples, iq_process,
                      threshold_discrimination=False):
    """Return one processed scalar for each (depth, randomized sample)."""
    values = np.asarray(raw)
    processed = np.real(values) if iq_process == "real" else np.abs(values)
    expected = int(n_depths) * int(n_samples)
    if expected <= 0 or processed.size % expected:
        raise ValueError(
            "RB data cannot be reshaped to "
            f"(depth={n_depths}, sample={n_samples}); got {values.shape}"
        )
    grouped = processed.reshape(n_depths, n_samples, -1)
    if threshold_discrimination:
        # Older QICK scalar returns can retain [population, Q-placeholder].
        # Select the population rather than averaging it with the placeholder.
        above_threshold = (
            grouped[..., 0] if grouped.shape[2] == 2 else grouped.mean(axis=2)
        )
        # QICK reports the above-threshold/e declaration; RB plots g survival.
        return 1.0 - above_threshold
    return grouped.mean(axis=2)


class RBProgram(BaseProgram):
    """QICK program that unrolls a Clifford gate sequence at compile time."""

    def _initialize(self, cfg):
        """Initialize pulse and acquisition resources.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        prefix = cfg.get("prefix", "ge")
        self.setup_resonator(cfg, prefix=prefix)
        self.setup_qubit_gen(cfg, prefix=prefix)
        self.setup_standard_gates(cfg, prefix=prefix)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)

    def _body(self, cfg):
        """Execute one iteration of the pulse sequence.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.

        Raises
        ------
        ValueError
            If the operation cannot be completed.
        """
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.cooling_body(cfg)
        pfx = cfg.get("prefix", "ge")
        for gate in cfg["gate_seq"]:
            if gate == "I":
                self.delay_auto(cfg[f"sigma_{pfx}"] * 5)
            else:
                template = _GEN_TO_QICK.get(gate)
                if template is None:
                    raise ValueError(f"Unknown gate '{gate}' in gate_seq")
                self.pulse(ch=cfg["qb_ch"], name=template.format(pfx=pfx), t=0)
                self.delay_auto(0.01)
        self.delay_auto(0.05)
        self.measure(cfg)


class RandomizedBenchmarking(BaseExperiment):
    """Single-qubit RB (s015): standard and interleaved."""

    EXPT_NAME = "s015_RB"
    Analysis = RBAnalysis

    def __init__(self, config):
        """Initialize the RandomizedBenchmarking instance.

        Parameters
        ----------
        config : Any
            Experiment configuration.
        """
        super().__init__(config)
        self.x = None
        self.rb_result = None
        self._number_sample = None
        self._interleaved = None
        self._iq_process = "abs"
        self._threshold_discrimination = False

    def run(
        self,
        py_avg: int,
        max_circuit_depth: int,
        delta_clifford: int,
        number_sample: int,
        interleaved_gate: str | None = None,
        seed: int | None = None,
        prefix: str = "ge",
        iq_process: str = "abs",
        randomize_depth_order: bool = False,
    ) -> ExperimentData:
        """Run the operation.

        Parameters
        ----------
        py_avg : int
            Number of Python-level acquisition averages.
        max_circuit_depth : int
            Value for ``max_circuit_depth``.
        delta_clifford : int
            Value for ``delta_clifford``.
        number_sample : int
            Value for ``number_sample``.
        interleaved_gate : str | None, default: None
            Value for ``interleaved_gate``.
        seed : int | None, default: None
            Value for ``seed``.
        prefix : str, default: 'ge'
            Value for ``prefix``.
        iq_process : str, default: 'abs'
            IQ processing mode.
        randomize_depth_order : bool, default: False
            Value for ``randomize_depth_order``.

        Returns
        -------
        ExperimentData
            Result of the operation.

        Raises
        ------
        ValueError
            If the operation cannot be completed.
        """
        from ...tools.rb_generator import single_qb_rb, INTERLEAVE_GATES

        threshold = self._get_readout_threshold()
        threshold_discrimination = threshold is not None
        self._threshold_discrimination = threshold_discrimination
        self._iq_process = "real" if threshold_discrimination else iq_process
        if threshold_discrimination:
            print(
                f"[RB] QICK threshold={threshold!r}; "
                "reporting ground survival P(g)=1-P(I>threshold)."
            )
        self.x = np.arange(1, max_circuit_depth, delta_clifford)
        self._number_sample = number_sample
        self._interleaved = interleaved_gate
        self._prefix = prefix

        if interleaved_gate is not None and interleaved_gate not in INTERLEAVE_GATES:
            raise ValueError(
                f"interleaved_gate '{interleaved_gate}' not in {list(INTERLEAVE_GATES.keys())}"
            )

        is_irb = interleaved_gate is not None
        desc = f"IRB ({interleaved_gate})" if is_irb else "Standard RB"
        rng = np.random.default_rng(seed)
        n_depths = len(self.x)
        seeds_matrix = [
            [int(rng.integers(0, 2**31)) for _ in range(number_sample)]
            for _ in range(n_depths)
        ]
        sequences_matrix = [[None] * number_sample for _ in range(n_depths)]
        programs_matrix = [[None] * number_sample for _ in range(n_depths)]
        depth_indices = np.arange(n_depths)
        if randomize_depth_order:
            rng.shuffle(depth_indices)

        for idx in tqdm(depth_indices, desc=f"Compile {desc}", leave=False):
            depth = self.x[idx]
            for sample_idx in range(number_sample):
                sequence = single_qb_rb(
                    n_clifford=depth,
                    n_sample=1,
                    interleave=interleaved_gate,
                    seed=seeds_matrix[idx][sample_idx],
                )[0]
                sequences_matrix[idx][sample_idx] = sequence
                program_cfg = dict(self.cfg)
                program_cfg["gate_seq"] = sequence
                program_cfg["prefix"] = prefix
                programs_matrix[idx][sample_idx] = RBProgram(
                    self.soccfg,
                    reps=program_cfg["reps"],
                    final_delay=program_cfg["relax_delay"],
                    cfg=program_cfg,
                )

        rb_accum = [[None] * number_sample for _ in range(n_depths)]
        for _ in tqdm(range(py_avg), desc="Software Average"):
            for idx in tqdm(depth_indices, desc=desc, leave=False):
                for sample_idx in tqdm(
                    range(number_sample), desc="Samples", leave=False
                ):
                    iq_data = acquire_values(
                        programs_matrix[idx][sample_idx],
                        self.soc,
                        rounds=1,
                        progress=False,
                        threshold=threshold,
                        scalar_readout=True,
                    )
                    previous = rb_accum[idx][sample_idx]
                    rb_accum[idx][sample_idx] = (
                        iq_data if previous is None else previous + iq_data
                    )

        self.rb_result = [
            [rb_accum[idx][s_i] / py_avg for s_i in range(number_sample)]
            for idx in range(n_depths)
        ]

        raw_iq = np.asarray(self.rb_result)
        avg = _rb_sample_matrix(
            raw_iq, n_depths, number_sample, self._iq_process,
            threshold_discrimination,
        ).mean(axis=1)
        metadata = {
            "qubit": self.cfg.get("name"),
            "iq_process": self._iq_process,
            "number_sample": number_sample,
            "interleaved_gate": interleaved_gate,
            "prefix": prefix,
            "seeds": seeds_matrix,
            "gate_sequences": sequences_matrix,
            "randomized_depth_order": self.x[depth_indices].tolist(),
        }
        if threshold_discrimination:
            metadata.update({
                "threshold": threshold,
                "threshold_discrimination": True,
                "raw_threshold_population": "above_threshold",
                "reported_population": "ground_survival",
            })
        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=raw_iq,
            x_axis=self.x.astype(float),
            y_axis=avg,
            metadata=metadata,
            axes={
                "depth": {"values": self.x.astype(float), "label": "Circuit depth", "unit": "# Cliffords"},
                "sample": {"values": np.arange(number_sample), "unit": "#"},
            },
            dataset_dims={"iq": ["depth", "sample"]},
            analysis_data={
                "mean_signal": {"values": avg, "dims": ["depth"]},
            },
            data_kind="rb",
            analysis_id="rb",
            plot_id="rb_decay",
            avg_count=py_avg,
            quality=QualityFlag.NO_INFORMATION,
        )
        if self.Analysis is not None:
            result = self.Analysis().run(result)
        self.result = result
        return result

    def plot(
        self, label: str = "RB", color=None, ax=None, marker="o",
        show_individual=False, *, plot_analysis=True,
    ):
        """Plot the operation.

        Parameters
        ----------
        label : str, default: 'RB'
            Value for ``label``.
        color : Any, default: None
            Value for ``color``.
        ax : Any, default: None
            Matplotlib axes on which to draw.
        marker : Any, default: 'o'
            Value for ``marker``.
        show_individual : Any, default: False
            Whether to show individual.
        plot_analysis : Any, default: True
            Value for ``plot_analysis``.

        Returns
        -------
        Any
            Result of the operation.

        Raises
        ------
        RuntimeError
            If the operation cannot be completed.
        """
        if self.x is None or self.rb_result is None:
            raise RuntimeError("Call run() first.")
        _proc = np.real if self._iq_process == "real" else np.abs
        raw = np.array(self.rb_result)
        # Acquisition can retain singleton/readout dimensions after the
        # (depth, randomized-sample) axes. Treat every value belonging to a
        # depth consistently for the mean, SEM, and individual traces.
        threshold_discrimination = getattr(
            self, "_threshold_discrimination",
            self._get_readout_threshold() is not None,
        )
        samples = _rb_sample_matrix(
            raw, len(self.x), self._number_sample, self._iq_process,
            threshold_discrimination,
        )
        avg = samples.mean(axis=1)
        pOpt, pCov = fitrb(self.x, avg)
        p_fit = pOpt[0]
        p_fit_err = float(np.sqrt(np.diag(pCov))[0]) if pCov is not None else 0.0
        epc = rb_error(p_fit, d=2)
        epc_err = float(np.sqrt(error_fit_err(pCov[0, 0], d=2))) if pCov is not None else 0.0
        print(f"\n--- {label} ---")
        print(f"  p   = {p_fit*100:.6f} ± {p_fit_err*100:.6f} %")
        print(f"  EPC = {epc*100:.6f} ± {epc_err*100:.6f} %")
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))
        c = color or "steelblue"
        if show_individual:
            for s in range(samples.shape[1]):
                ax.scatter(self.x, samples[:, s], s=6, color="gray", alpha=0.25, linewidths=0, zorder=1)
        xfit = np.linspace(self.x.min(), self.x.max(), 400)
        ax.plot(xfit, rb_func(xfit, *pOpt), color=c, linewidth=2.0, zorder=3)
        sem = samples.std(axis=1) / np.sqrt(samples.shape[1])
        ax.errorbar(self.x, avg, yerr=sem,
                    fmt="none", ecolor=c, capsize=3, zorder=4)
        ax.scatter(self.x, avg, s=60, color=c, marker=marker,
                   edgecolors="black", label=label, zorder=5)
        result = getattr(self, "result", None)
        if result is not None and all(id(figure) != id(ax.figure) for figure in result.figures):
            result.figures.append(ax.figure)
        return epc, epc_err, p_fit, p_fit_err, pCov

    def saveLabber(self, qb_idx, config_all=None, yoko_value=None, title=None):
        """Save Labber.

        Parameters
        ----------
        qb_idx : Any
            Value for ``qb_idx``.
        config_all : Any, default: None
            Value for ``config_all``.
        yoko_value : Any, default: None
            Value for ``yoko_value``.
        title : Any, default: None
            Value for ``title``.

        Raises
        ------
        RuntimeError
            If the operation cannot be completed.
        """
        if self.x is None or self.rb_result is None:
            raise RuntimeError("Call run() first.")
        from ...tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml
        if title is not None:
            expt_name = f"s015_RB_{_safe_rb_file_suffix(title)}_{qb_idx}"
        elif self._interleaved is not None:
            suffix = _safe_rb_file_suffix(self._interleaved)
            expt_name = f"s015_RB_{suffix}_{qb_idx}"
        else:
            expt_name = f"s015_RB_{qb_idx}_ref"
        save_dir = BaseExperiment._data_path
        file_path = get_next_filename_labber(save_dir, expt_name, yoko_value)
        dict_val = (
            config_all.to_yaml(q_id=qb_idx)
            if config_all is not None
            else config_to_yaml(self.cfg)
        )
        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Circuit Depth", "unit": "", "values": self.x.astype(float)},
            y_info={"name": "Sample Number", "unit": "", "values": np.arange(self._number_sample, dtype=float)},
            z_info={
                "name": "Signal", "unit": "ADC unit",
                "values": _rb_sample_matrix(
                    self.rb_result, len(self.x), self._number_sample,
                    self._iq_process,
                    getattr(
                        self, "_threshold_discrimination",
                        self._get_readout_threshold() is not None,
                    ),
                ).T,
            },
            comment=str(dict_val), tag="RB",
            result=self.result,
            figures=self._analysis_figures_for_save(),
        )
        print(f"RB data saved to {file_path}")


def _gate_fidelity(p_ref, p_irb, d=2):
    """Return the gate fidelity result.

    Parameters
    ----------
    p_ref : Any
        Value for ``p_ref``.
    p_irb : Any
        Value for ``p_irb``.
    d : Any, default: 2
        Value for ``d``.

    Returns
    -------
    Any
        Result of the operation.
    """
    epc = (d - 1) / d * (1 - p_irb / p_ref)
    return 1 - epc, epc


def _gate_fidelity_err(p_ref, p_irb, var_p_ref, var_p_irb, d=2):
    """Return the gate fidelity err result.

    Parameters
    ----------
    p_ref : Any
        Value for ``p_ref``.
    p_irb : Any
        Value for ``p_irb``.
    var_p_ref : Any
        Value for ``var_p_ref``.
    var_p_irb : Any
        Value for ``var_p_irb``.
    d : Any, default: 2
        Value for ``d``.

    Returns
    -------
    Any
        Result of the operation.
    """
    c = (d - 1) / d
    depc_dpref = c * p_irb / p_ref**2
    depc_dpirb = -c / p_ref
    return float(np.sqrt(depc_dpref**2 * var_p_ref + depc_dpirb**2 * var_p_irb))


class AutoRB:
    """Automated Standard + Interleaved RB in one call (s015)."""

    def __init__(self, config):
        """Initialize the AutoRB instance.

        Parameters
        ----------
        config : Any
            Experiment configuration.
        """
        self.cfg = config
        self._rb_kwargs: dict = {}
        self.results: dict = {}
        self._rb_objects: dict = {}

    def run(
        self,
        py_avg: int,
        max_circuit_depth: int,
        delta_clifford: int,
        number_sample: int,
        interleaved_gates: list[str] | None = None,
        seed: int | None = None,
        prefix: str = "ge",
        iq_process: str = "abs",
    ):
        """Run the operation.

        Parameters
        ----------
        py_avg : int
            Number of Python-level acquisition averages.
        max_circuit_depth : int
            Value for ``max_circuit_depth``.
        delta_clifford : int
            Value for ``delta_clifford``.
        number_sample : int
            Value for ``number_sample``.
        interleaved_gates : list[str] | None, default: None
            Value for ``interleaved_gates``.
        seed : int | None, default: None
            Value for ``seed``.
        prefix : str, default: 'ge'
            Value for ``prefix``.
        iq_process : str, default: 'abs'
            IQ processing mode.
        """
        from ...tools.hdf5_store import generate_experiment_id

        session_id = generate_experiment_id()
        self._rb_kwargs = dict(
            max_circuit_depth=max_circuit_depth,
            delta_clifford=delta_clifford,
            number_sample=number_sample,
            seed=seed, prefix=prefix, iq_process=iq_process,
        )
        gates_to_run = [None] + (interleaved_gates or [])
        for gate in tqdm(gates_to_run, desc="AutoRB"):
            label = "ref" if gate is None else gate
            rb = RandomizedBenchmarking(self.cfg)
            rb.run(py_avg, interleaved_gate=gate, **self._rb_kwargs)
            rb.result.parent_id = session_id
            rb.result.session_id = session_id
            self._rb_objects[label] = rb

    def plot(self, show_individual=False, *, plot_analysis=True):
        """Plot the operation.

        Parameters
        ----------
        show_individual : Any, default: False
            Whether to show individual.
        plot_analysis : Any, default: True
            Value for ``plot_analysis``.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        ref_rb = self._rb_objects.get("ref")
        if ref_rb is None:
            print("No reference RB — call run() first.")
            return

        ref_epc, ref_epc_err, p_ref, p_ref_err, ref_cov = ref_rb.plot(
            "Reference RB", color=colors[0], ax=ax, show_individual=show_individual,
            plot_analysis=plot_analysis,
        )
        self.results["ref"] = dict(epc=ref_epc, epc_err=ref_epc_err, p=p_ref, p_err=p_ref_err)

        for i, (label, rb) in enumerate(self._rb_objects.items()):
            if label == "ref":
                continue
            epc, epc_err, p_irb, p_irb_err, irb_cov = rb.plot(
                f"IRB ({label})", color=colors[(i + 1) % len(colors)],
                ax=ax, show_individual=show_individual,
            )
            var_ref = ref_cov[0, 0] if ref_cov is not None else 0
            var_irb = irb_cov[0, 0] if irb_cov is not None else 0
            f_gate, epc_gate = _gate_fidelity(p_ref, p_irb)
            epc_gate_err = _gate_fidelity_err(p_ref, p_irb, var_ref, var_irb)
            self.results[label] = dict(
                fidelity=f_gate, epc=epc_gate, epc_err=epc_gate_err,
                p=p_irb, p_err=p_irb_err,
            )
            print(f"  Gate '{label}': F = {f_gate*100:.4f}%, EPC = {epc_gate*100:.4f} ± {epc_gate_err*100:.4f} %")

        ax.set_xlabel("Circuit Depth (# Cliffords)")
        if getattr(
            ref_rb, "_threshold_discrimination",
            ref_rb._get_readout_threshold() is not None,
        ):
            ax.set_ylabel("Ground-state survival probability")
            ax.set_ylim(-0.02, 1.02)
        else:
            ax.set_ylabel("Signal (a.u.)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()

    def summary(self) -> str:
        """Return a summary of the current state.

        Returns
        -------
        str
            Result of the operation.
        """
        lines = ["AutoRB Summary", "=" * 50]
        for key, val in self.results.items():
            if "fidelity" in val:
                lines.append(
                    f"  {key:<10s}  F={val['fidelity']*100:.4f}%  "
                    f"EPC={val['epc']*100:.5f}%"
                )
            else:
                lines.append(f"  {key:<10s}  EPC={val['epc']*100:.5f}%")
        return "\n".join(lines)

    def saveLabber(self, qb_idx, config_all=None, yoko_value=None):
        """Save Labber.

        Parameters
        ----------
        qb_idx : Any
            Value for ``qb_idx``.
        config_all : Any, default: None
            Value for ``config_all``.
        yoko_value : Any, default: None
            Value for ``yoko_value``.
        """
        for label, rb in self._rb_objects.items():
            rb.saveLabber(qb_idx, config_all=config_all, yoko_value=yoko_value, title=label)
