"""
Mux AllXY gate diagnostic.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from qick.asm_v2 import AveragerProgramV2
from tqdm.auto import tqdm

from ..core.base_experiment import BaseExperiment
from ..core.base_program import resolve_gate
from ..core.experiment_data import ExperimentData, QualityFlag


ALLXY_SEQUENCE = [
    ("I", "I"),
    ("X", "X"),
    ("Y", "Y"),
    ("X", "Y"),
    ("Y", "X"),
    ("X/2", "I"),
    ("Y/2", "I"),
    ("X/2", "Y/2"),
    ("Y/2", "X/2"),
    ("X/2", "Y"),
    ("Y/2", "X"),
    ("X", "Y/2"),
    ("Y", "X/2"),
    ("X/2", "X"),
    ("X", "X/2"),
    ("Y/2", "Y"),
    ("Y", "Y/2"),
    ("X", "I"),
    ("Y", "I"),
    ("X/2", "X/2"),
    ("Y/2", "Y/2"),
]


class MuxAllXYProgram(AveragerProgramV2):
    """One AllXY gate pair applied to all armed qubits with mux readout."""

    def _initialize(self, cfg):
        """Initialize pulse and acquisition resources.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
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
        """Add standard gates.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        slot : Any
            Value for ``slot``.
        name : Any
            Name of the target object.
        """
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
        """Add gate pulse.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        slot : Any
            Value for ``slot``.
        pulse_name : Any
            Name of the pulse.
        phase : Any
            Value for ``phase``.
        gain : Any
            Value for ``gain``.
        """
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
        """Return the pulse gate result.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        gate : Any
            Value for ``gate``.
        """
        resolved = resolve_gate(gate)
        if resolved in ("I", "-I", None, "None"):
            return
        for slot, name in zip(cfg["active_slots"], cfg["qubit_names"]):
            self.pulse(ch=cfg["qb_ch"][slot], name=f"{name}_{resolved}", t=0)

    def _body(self, cfg):
        """Execute one iteration of the pulse sequence.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        gate1, gate2 = cfg["allxy_gates"]
        self._pulse_gate(cfg, gate1)
        self.delay_auto(0.01)
        self._pulse_gate(cfg, gate2)
        self.delay_auto(0.05)
        self.trigger(ros=cfg["active_ro_chs"], pins=[0], t=cfg["trig_time"])
        self.pulse(ch=cfg["res_ch"], name="mux_readout", t=0)


class MuxAllXY(BaseExperiment):
    """Mux AllXY for all armed qubits."""

    EXPT_NAME = "s014_mux_allxy_ge"
    TAG = "MuxAllXY"
    TITLE_PREFIX = "Mux AllXY"

    def __init__(self, config):
        """Initialize the MuxAllXY instance.

        Parameters
        ----------
        config : Any
            Experiment configuration.
        """
        super().__init__(config)
        self.iqdata = None

    @staticmethod
    def _extract_iq(iq_list, n_trace):
        """Extract iq.

        Parameters
        ----------
        iq_list : Any
            Value for ``iq_list``.
        n_trace : Any
            Value for ``n_trace``.

        Returns
        -------
        Any
            Result of the operation.
        """
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
        """Prepare acquired data for plotting.

        Parameters
        ----------
        iqdata : Any
            Value for ``iqdata``.
        iq_process : Any
            IQ processing mode.

        Returns
        -------
        Any
            Result of the operation.
        """
        iq_process = (iq_process or "abs").lower()
        if iq_process in {"real", "i", "avgi"}:
            return np.real(iqdata)
        if iq_process in {"imag", "q", "avgq"}:
            return np.imag(iqdata)
        if iq_process == "phase":
            return np.unwrap(np.angle(iqdata), axis=-1)
        return np.abs(iqdata)

    def run(self, py_avg=1, iq_process="abs", plot=True):
        """Run the operation.

        Parameters
        ----------
        py_avg : Any, default: 1
            Number of Python-level acquisition averages.
        iq_process : Any, default: 'abs'
            IQ processing mode.
        plot : Any, default: True
            Value for ``plot``.

        Returns
        -------
        Any
            Result of the operation.
        """
        cfg = dict(self.cfg)
        qubit_names = list(cfg["qubit_names"])
        trace_count = len(cfg["active_ro_chs"])
        data = []
        interrupted = False

        try:
            for gates in tqdm(ALLXY_SEQUENCE, desc="Mux AllXY"):
                run_cfg = dict(cfg)
                run_cfg["allxy_gates"] = gates
                prog = MuxAllXYProgram(
                    self.soccfg,
                    reps=run_cfg["reps"],
                    final_delay=run_cfg["relax_delay"],
                    cfg=run_cfg,
                )
                iq_list = prog.acquire(self.soc, rounds=py_avg, progress=False)
                data.append(self._extract_iq(iq_list, trace_count))
        except KeyboardInterrupt:
            interrupted = True

        self.iqdata = np.asarray(data, dtype=complex).T if data else None

        fit_result = {}
        if self.iqdata is not None:
            plot_data = self._process_plot_data(self.iqdata, iq_process)
            ideal = np.array(
                [0, 0, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0],
                dtype=float,
            )
            for idx, name in enumerate(qubit_names):
                trace = plot_data[idx]
                span = np.nanmax(trace) - np.nanmin(trace)
                norm = (trace - np.nanmin(trace)) / (span + 1e-12)
                fit_result[f"{name}_allxy_error"] = (
                    round(float(np.nanmean(np.abs(norm - ideal))), 6),
                    None,
                )

        figures = []
        if plot and self.iqdata is not None:
            plot_data = self._process_plot_data(self.iqdata, iq_process)
            fig, axes = plt.subplots(
                trace_count, 1, figsize=(9, max(3, 2.5 * trace_count)), squeeze=False
            )
            x = np.arange(len(ALLXY_SEQUENCE))
            for ax, name, trace in zip(axes[:, 0], qubit_names, plot_data):
                ax.plot(x, trace, "o-", markersize=4, label="data")
                ax.set_ylabel(name)
                ax.grid(True, alpha=0.3)
            axes[-1, 0].set_xticks(x)
            axes[-1, 0].set_xticklabels(
                [f"{a},{b}" for a, b in ALLXY_SEQUENCE], rotation=45, ha="right"
            )
            axes[0, 0].set_title(self.TITLE_PREFIX + (" (Interrupted)" if interrupted else ""))
            fig.tight_layout()
            figures.append(fig)
            plt.show()

        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=self.iqdata,
            x_axis=np.arange(len(ALLXY_SEQUENCE), dtype=float),
            y_axis=self._process_plot_data(self.iqdata, iq_process) if self.iqdata is not None else None,
            fit_result=fit_result,
            metadata={"qubit_names": qubit_names, "sequence": ALLXY_SEQUENCE},
            figures=figures,
            quality=QualityFlag.GOOD if self.iqdata is not None else QualityFlag.BAD,
            interrupted=interrupted,
            avg_count=py_avg,
        )
        self.result = result
        return result


__all__ = ["ALLXY_SEQUENCE", "MuxAllXY", "MuxAllXYProgram"]
