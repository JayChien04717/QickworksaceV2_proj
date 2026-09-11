"""
Characterization/allxy — s014: AllXY gate error diagnostic.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ...core.base_program import BaseProgram, resolve_gate
from ...core.base_experiment import BaseExperiment
from ...core.acquisition import acquire_values
from ...core.experiment_data import ExperimentData, QualityFlag
from ...analysis.rb import AllXYAnalysis


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


class AllXYProgram(BaseProgram):
    """QICK program for a single AllXY gate-pair measurement."""

    def _initialize(self, cfg):
        """Initialize pulse and acquisition resources.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.setup_standard_gates(cfg, prefix="ge")

    def _body(self, cfg):
        """Execute one iteration of the pulse sequence.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        gate1, gate2 = cfg["allxy_gates"]
        g1 = resolve_gate(gate1)
        g2 = resolve_gate(gate2)
        if g1 != "I":
            self.pulse(ch=cfg["qb_ch"], name=g1, t=0)
        self.delay_auto(0.01)
        if g2 != "I":
            self.pulse(ch=cfg["qb_ch"], name=g2, t=0)
        self.delay_auto(0.05)
        self.measure(cfg)


class AllXY(BaseExperiment):
    """AllXY gate error diagnostic (s014): 21 gate-pair sequences."""

    EXPT_NAME = "s014_AllXY_ge"
    TAG = "AllXY"
    X_LABEL = "Gate pair"
    TITLE_PREFIX = "AllXY"
    X_SAVE_NAME = "Gate pair"
    X_SAVE_UNIT = "index"
    Analysis = AllXYAnalysis

    def __init__(self, config):
        """Initialize the AllXY instance.

        Parameters
        ----------
        config : Any
            Experiment configuration.
        """
        super().__init__(config)
        self.allxy_lst: np.ndarray | None = None
        self._iq_process = "abs"

    def run(self, py_avg: int, iq_process: str = "abs") -> ExperimentData:
        """Run the operation.

        Parameters
        ----------
        py_avg : int
            Number of Python-level acquisition averages.
        iq_process : str, default: 'abs'
            IQ processing mode.

        Returns
        -------
        ExperimentData
            Result of the operation.
        """
        self._iq_process = iq_process
        allxy_lst = []
        threshold = self._get_readout_threshold()
        for gate in tqdm(ALLXY_SEQUENCE, desc="AllXY"):
            self.cfg["allxy_gates"] = gate
            prog = AllXYProgram(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )
            value = acquire_values(
                prog,
                self.soc,
                rounds=py_avg,
                progress=False,
                threshold=threshold,
                scalar_readout=True,
            )
            allxy_lst.append(np.asarray(value).reshape(-1)[0])
        self.allxy_lst = np.array(allxy_lst)
        self.iqdata = self.allxy_lst

        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=self.allxy_lst,
            x_axis=np.arange(len(ALLXY_SEQUENCE), dtype=float),
            x_name=self.X_SAVE_NAME,
            x_unit=self.X_SAVE_UNIT,
            metadata={
                "iq_process": "real" if threshold is not None else iq_process,
                "threshold": threshold,
                "threshold_discrimination": threshold is not None,
            },
            avg_count=py_avg,
            dataset_dims={"iq": ["x"]},
            quality=QualityFlag.NO_INFORMATION,
        )
        if self.Analysis is not None:
            result = self.Analysis().run(result)
        self.result = result
        return result

    def plot(self, *, plot_analysis=True):
        """Plot the operation.

        Parameters
        ----------
        plot_analysis : Any, default: True
            Value for ``plot_analysis``.

        Raises
        ------
        RuntimeError
            If the operation cannot be completed.
        """
        if self.allxy_lst is None:
            raise RuntimeError("Call run() first.")
        _proc = np.real if self._iq_process == "real" else np.abs
        amp = _proc(self.allxy_lst)
        if amp[0] < amp[-1]:
            ref = (
                [np.min(amp)] * 5
                + [(np.max(amp) + np.min(amp)) / 2] * 12
                + [np.max(amp)] * 4
            )
        else:
            ref = (
                [np.max(amp)] * 5
                + [(np.max(amp) + np.min(amp)) / 2] * 12
                + [np.min(amp)] * 4
            )
        if len(ref) != len(amp):
            ref = (
                ref[: len(amp)]
                if len(ref) > len(amp)
                else ref + [ref[-1]] * (len(amp) - len(ref))
            )
        plt.figure(figsize=(10, 5))
        plt.plot(amp, "bo", label="Data")
        plt.plot(ref, "r-", label="Reference Line")
        plt.xticks(np.arange(len(ALLXY_SEQUENCE)), ALLXY_SEQUENCE, rotation=45)
        plt.ylabel(r"$F_{\left|1\right\rangle}$")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def saveLabber(self, qb_idx, yoko_value=None, config_all=None):
        """Save Labber.

        Parameters
        ----------
        qb_idx : Any
            Value for ``qb_idx``.
        yoko_value : Any, default: None
            Value for ``yoko_value``.
        config_all : Any, default: None
            Value for ``config_all``.
        """
        from ...tools.system_tool import (
            hdf5_generator,
            get_next_filename_labber,
            config_to_yaml,
        )

        save_dir = BaseExperiment._data_path
        file_path = get_next_filename_labber(
            save_dir, f"{self.EXPT_NAME}_{qb_idx}", yoko_value
        )
        config_yaml = (
            config_all.to_yaml(q_id=qb_idx)
            if config_all is not None
            else config_to_yaml(self.cfg)
        )
        hdf5_generator(
            filepath=file_path,
            x_info={
                "name": "Sequence",
                "unit": "None",
                "values": np.arange(len(ALLXY_SEQUENCE)),
            },
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.allxy_lst},
            comment=str(config_yaml),
            tag=self.TAG,
            result=self.result,
        )
        print(f"Data saved to {file_path}")
