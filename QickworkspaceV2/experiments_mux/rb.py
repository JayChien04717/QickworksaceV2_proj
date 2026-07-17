"""
Mux randomized benchmarking.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from qick.asm_v2 import AveragerProgramV2
from tqdm.auto import tqdm

from ..core.base_experiment import BaseExperiment
from ..core.base_program import resolve_gate
from ..core.experiment_data import ExperimentData, QualityFlag
from ..tools.fitting import error_fit_err, fitrb, rb_error, rb_func


class MuxRBProgram(AveragerProgramV2):
    """One RB sequence applied to all armed qubits with mux readout."""

    def _initialize(self, cfg):
        res_ch = cfg["res_ch"]
        ro_chs = list(cfg["active_ro_chs"])

        self.declare_gen(
            ch=res_ch,
            nqz=1,
            ro_ch=ro_chs[0],
            mixer_freq=cfg.get("mixer_freq", 0),
            mux_freqs=cfg["res_freqs"],
            mux_gains=cfg["res_gains"],
            mux_phases=cfg["res_phases"],
        )
        for ch, slot in zip(ro_chs, cfg["active_slots"]):
            self.declare_readout(
                ch=ch,
                length=cfg["ro_length"],
                freq=cfg["res_freqs"][slot],
                phase=cfg["ro_phases"][slot],
                gen_ch=res_ch,
            )

        for slot, name in zip(cfg["active_slots"], cfg["qubit_names"]):
            self.declare_gen(ch=cfg["qb_ch"][slot], nqz=cfg["nqz_qb"][slot])

        self.add_pulse(
            ch=res_ch,
            name="mux_readout",
            style="const",
            length=cfg["res_length"],
            mask=cfg["mask"],
        )

        for slot, name in zip(cfg["active_slots"], cfg["qubit_names"]):
            self._add_standard_gates(cfg, slot, name)

    def _add_standard_gates(self, cfg, slot, name):
        gates = [
            ("x180_ge", 0, "pi_gain_ge"),
            ("y180_ge", 90, "pi_gain_ge"),
            ("x90_ge", 0, "pi2_gain_ge"),
            ("x90m_ge", 180, "pi2_gain_ge"),
            ("y90_ge", 90, "pi2_gain_ge"),
            ("y90m_ge", -90, "pi2_gain_ge"),
        ]
        for gate_name, phase, gain_key in gates:
            self._add_gate_pulse(cfg, slot, f"{name}_{gate_name}", phase, cfg[gain_key][slot])

    def _add_gate_pulse(self, cfg, slot, pulse_name, phase, gain):
        qb_ch = cfg["qb_ch"][slot]
        pulse_type = cfg["pulse_type"][slot]
        if pulse_type == "const":
            self.add_pulse(
                ch=qb_ch,
                name=pulse_name,
                style="const",
                length=cfg["qb_flat_top_length_ge"][slot],
                freq=cfg["qb_freq_ge"][slot],
                phase=phase,
                gain=gain,
            )
        elif pulse_type == "arb":
            env_name = f"{pulse_name}_env"
            self.add_gauss(
                ch=qb_ch,
                name=env_name,
                sigma=cfg["sigma_ge"][slot],
                length=5 * cfg["sigma_ge"][slot],
                even_length=True,
            )
            self.add_pulse(
                ch=qb_ch,
                name=pulse_name,
                style="arb",
                envelope=env_name,
                freq=cfg["qb_freq_ge"][slot],
                phase=phase,
                gain=gain,
            )
        else:
            env_name = f"{pulse_name}_env"
            self.add_gauss(
                ch=qb_ch,
                name=env_name,
                sigma=cfg["sigma_ge"][slot],
                length=5 * cfg["sigma_ge"][slot],
                even_length=True,
            )
            self.add_pulse(
                ch=qb_ch,
                name=pulse_name,
                style="flat_top",
                envelope=env_name,
                length=cfg["qb_flat_top_length_ge"][slot],
                freq=cfg["qb_freq_ge"][slot],
                phase=phase,
                gain=gain,
            )

    def _pulse_gate(self, cfg, gate):
        resolved = resolve_gate(gate)
        if resolved in ("I", "-I", None, "None"):
            self.delay_auto(cfg["sigma_ge"][cfg["active_slots"][0]] * 5)
            return
        for slot, name in zip(cfg["active_slots"], cfg["qubit_names"]):
            self.pulse(ch=cfg["qb_ch"][slot], name=f"{name}_{resolved}", t=0)
        self.delay_auto(0.01)

    def _body(self, cfg):
        for gate in cfg["gate_seq"]:
            self._pulse_gate(cfg, gate)
        self.delay_auto(0.05)
        self.trigger(ros=cfg["active_ro_chs"], pins=[0], t=cfg["trig_time"])
        self.pulse(ch=cfg["res_ch"], name="mux_readout", t=0)


class MuxRandomizedBenchmarking(BaseExperiment):
    """Mux standard and interleaved single-qubit RB."""

    EXPT_NAME = "s015_mux_rb"
    TAG = "MuxRB"
    TITLE_PREFIX = "Mux RB"

    def __init__(self, config):
        super().__init__(config)
        self.x = None
        self.rb_result = None
        self._iq_process = "abs"
        self._number_sample = None
        self._interleaved = None

    @staticmethod
    def _extract_iq(iq_list, n_trace):
        vals = []
        for idx in range(n_trace):
            arr = np.asarray(iq_list[idx][0]).squeeze()
            if arr.ndim > 0 and arr.shape[-1] == 2:
                vals.append(np.asarray(arr).dot([1, 1j]).reshape(-1)[0])
            else:
                vals.append(np.asarray(arr, dtype=complex).reshape(-1)[0])
        return np.asarray(vals, dtype=complex)

    @staticmethod
    def _process_plot_data(iqdata, iq_process):
        iq_process = (iq_process or "abs").lower()
        if iq_process in {"real", "i", "avgi"}:
            return np.real(iqdata)
        if iq_process in {"imag", "q", "avgq"}:
            return np.imag(iqdata)
        if iq_process == "phase":
            return np.unwrap(np.angle(iqdata), axis=-1)
        return np.abs(iqdata)

    def run(
        self,
        py_avg,
        max_circuit_depth,
        delta_clifford,
        number_sample,
        interleaved_gate=None,
        seed=None,
        iq_process="abs",
        randomize_depth_order=False,
        plot=True,
    ):
        from ..tools.rb_generator import INTERLEAVE_GATES, single_qb_rb

        if interleaved_gate is not None and interleaved_gate not in INTERLEAVE_GATES:
            raise ValueError(f"interleaved_gate must be one of {list(INTERLEAVE_GATES.keys())}")

        cfg = dict(self.cfg)
        qubit_names = list(cfg["qubit_names"])
        trace_count = len(cfg["active_ro_chs"])
        self._iq_process = iq_process
        self._number_sample = number_sample
        self._interleaved = interleaved_gate
        self.x = np.arange(1, max_circuit_depth, delta_clifford)

        rng = np.random.default_rng(seed)
        seeds_matrix = [
            [int(rng.integers(0, 2**31)) for _ in range(number_sample)]
            for _ in range(len(self.x))
        ]
        sequences_matrix = [[None] * number_sample for _ in range(len(self.x))]
        depth_indices = np.arange(len(self.x))
        if randomize_depth_order:
            rng.shuffle(depth_indices)

        accum = np.zeros((trace_count, len(self.x), number_sample), dtype=complex)
        interrupted = False

        try:
            for avg_i in tqdm(range(py_avg), desc="Mux RB average"):
                for d_idx in tqdm(depth_indices, desc="Depth", leave=False):
                    depth = int(self.x[d_idx])
                    for sample_idx in tqdm(range(number_sample), desc="Samples", leave=False):
                        seq = single_qb_rb(
                            n_clifford=depth,
                            n_sample=1,
                            interleave=interleaved_gate,
                            seed=seeds_matrix[d_idx][sample_idx],
                        )[0]
                        sequences_matrix[d_idx][sample_idx] = seq
                        run_cfg = dict(cfg)
                        run_cfg["gate_seq"] = seq
                        prog = MuxRBProgram(
                            self.soccfg,
                            reps=run_cfg["reps"],
                            final_delay=run_cfg["relax_delay"],
                            cfg=run_cfg,
                        )
                        iq_list = prog.acquire(self.soc, rounds=1, progress=False)
                        accum[:, d_idx, sample_idx] += self._extract_iq(iq_list, trace_count)
        except KeyboardInterrupt:
            interrupted = True

        self.rb_result = accum / max(py_avg, 1)
        plot_data = self._process_plot_data(self.rb_result, iq_process)

        fit_result = {}
        fit_params = {}
        for idx, name in enumerate(qubit_names):
            avg = plot_data[idx].mean(axis=1)
            try:
                popt, pcov, _ = fitrb(self.x, avg)
                p_fit = float(popt[0])
                p_err = float(np.sqrt(np.diag(pcov))[0]) if pcov is not None else None
                epc = rb_error(p_fit, d=2)
                epc_err = (
                    float(np.sqrt(error_fit_err(pcov[0, 0], d=2)))
                    if pcov is not None
                    else None
                )
                fit_result[f"{name}_p"] = (round(p_fit, 8), p_err)
                fit_result[f"{name}_epc"] = (round(float(epc), 8), epc_err)
                fit_result[f"{name}_fidelity"] = (round(float(1 - epc), 8), epc_err)
                fit_params[name] = np.asarray(popt, dtype=float).tolist()
            except Exception:
                continue

        figures = []
        if plot:
            fig, axes = plt.subplots(
                trace_count, 1, figsize=(8, max(3, 2.8 * trace_count)), squeeze=False
            )
            for idx, (ax, name) in enumerate(zip(axes[:, 0], qubit_names)):
                vals = plot_data[idx]
                avg = vals.mean(axis=1)
                err = vals.std(axis=1) / np.sqrt(max(vals.shape[1], 1))
                ax.errorbar(self.x, avg, yerr=err, fmt="o", capsize=3, label=name)
                if name in fit_params:
                    xfit = np.linspace(self.x.min(), self.x.max(), 400)
                    ax.plot(xfit, rb_func(xfit, *fit_params[name]), "-", label="fit")
                epc = fit_result.get(f"{name}_epc", (None,))[0]
                if epc is not None:
                    ax.set_title(f"{name} | EPC = {epc:.6g}")
                else:
                    ax.set_title(name)
                ax.set_xlabel("Circuit depth (# Cliffords)")
                ax.set_ylabel("Signal")
                ax.grid(True, alpha=0.3)
                ax.legend()
            title = self.TITLE_PREFIX if interleaved_gate is None else f"{self.TITLE_PREFIX} IRB {interleaved_gate}"
            axes[0, 0].figure.suptitle(title + (" (Interrupted)" if interrupted else ""))
            fig.tight_layout()
            figures.append(fig)
            plt.show()

        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=self.rb_result,
            x_axis=self.x.astype(float),
            y_axis=plot_data.mean(axis=2),
            fit_result=fit_result,
            metadata={
                "qubit_names": qubit_names,
                "number_sample": number_sample,
                "interleaved_gate": interleaved_gate,
                "fit_params": fit_params,
                "seeds": seeds_matrix,
                "gate_sequences": sequences_matrix,
                "randomized_depth_order": self.x[depth_indices].tolist(),
            },
            axes={
                "qubit": {"values": qubit_names},
                "depth": {"values": self.x.astype(float), "unit": "# Cliffords"},
                "sample": {"values": np.arange(number_sample), "unit": "#"},
            },
            dataset_dims={"iq": ["qubit", "depth", "sample"]},
            analysis_data={
                "mean_signal": {"values": plot_data.mean(axis=2), "dims": ["qubit", "depth"]},
                "standard_error": {
                    "values": plot_data.std(axis=2) / np.sqrt(max(number_sample, 1)),
                    "dims": ["qubit", "depth"],
                },
            },
            data_kind="rb",
            analysis_id="rb",
            plot_id="rb_decay",
            figures=figures,
            quality=QualityFlag.GOOD if self.rb_result is not None else QualityFlag.BAD,
            interrupted=interrupted,
            avg_count=py_avg,
        )
        self.result = result
        return result

    def plot(self, show_individual=False):
        if self.result is None:
            raise RuntimeError("Call run() first.")
        return self.result.figures


MuxRB = MuxRandomizedBenchmarking


class MuxAutoRB:
    """Run mux reference RB plus optional interleaved RB gates."""

    def __init__(self, config):
        self.cfg = config
        self.results = {}
        self.rb_objects = {}

    def run(
        self,
        py_avg,
        max_circuit_depth,
        delta_clifford,
        number_sample,
        interleaved_gates=None,
        seed=None,
        iq_process="abs",
        plot=True,
    ):
        from ..tools.hdf5_store import generate_experiment_id

        session_id = generate_experiment_id()
        gates = [None] + list(interleaved_gates or [])
        for gate in tqdm(gates, desc="Mux AutoRB"):
            label = "ref" if gate is None else gate
            rb = MuxRandomizedBenchmarking(self.cfg)
            result = rb.run(
                py_avg=py_avg,
                max_circuit_depth=max_circuit_depth,
                delta_clifford=delta_clifford,
                number_sample=number_sample,
                interleaved_gate=gate,
                seed=seed,
                iq_process=iq_process,
                plot=plot,
            )
            result.parent_id = session_id
            result.session_id = session_id
            self.rb_objects[label] = rb
            self.results[label] = result.fit_result
        return self.results

    def summary(self):
        lines = ["Mux AutoRB Summary"]
        for label, fit in self.results.items():
            lines.append(f"[{label}]")
            for key, value in fit.items():
                if key.endswith("_epc") or key.endswith("_fidelity"):
                    lines.append(f"  {key}: {value[0]}")
        return "\n".join(lines)


__all__ = ["MuxAutoRB", "MuxRB", "MuxRBProgram", "MuxRandomizedBenchmarking"]
