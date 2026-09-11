"""
RB analysis — randomized benchmarking exponential decay fit.
"""

from __future__ import annotations

import numpy as np

from ..core.base_analysis import BaseAnalysis
from ..core.experiment_data import ExperimentData, QualityFlag


class RBAnalysis(BaseAnalysis):
    """
    Exponential decay fit for randomized benchmarking.

    Extracts average Clifford gate fidelity (r) and error per Clifford (EPC).
    """

    thresholds = {
        "epc": {"max": 0.05},
    }

    def _run(self, data: ExperimentData) -> None:
        """Run the operation.

        Parameters
        ----------
        data : ExperimentData
            Input data to process.
        """
        if data.x_axis is None or data.raw_iq is None:
            return
        from ..tools.fitting import fitrb, rb_func

        x = data.x_axis  # Clifford lengths
        threshold_discrimination = bool(
            data.metadata.get("threshold_discrimination")
        )
        fit_channel = "real" if threshold_discrimination else "abs"
        raw_iq = np.asarray(data.raw_iq)
        raw = (
            1.0 - np.real(raw_iq)
            if threshold_discrimination
            else np.abs(raw_iq)
        )
        y = raw if raw.ndim == 1 else raw.reshape(len(data.x_axis), -1).mean(axis=1)

        try:
            popt, pcov = fitrb(x, y)
            err = np.sqrt(np.diag(pcov))
            # RB model: A * p^m + B  where p = 1 - EPC * d/(d-1), d=2 for single qubit
            p, A, B = popt
            # EPC = (1 - p) * (d-1)/d  for d=2 → EPC = (1-p)/2
            d = 2
            epc = (1 - p) * (d - 1) / d
            fidelity = 1 - epc

            data.fit_params = np.array(popt)
            data.fit_errors = err
            data.fit_result = {
                "epc": (round(epc, 6), None),
                "fidelity": (round(fidelity, 6), None),
                "p": (p, err[0]),
                "A": (A, err[1]),
                "B": (B, err[2]),
            }
            data.scalar_result = fidelity
            fit_curve = rb_func(x, *popt)
            data.metadata.update({
                "fit_model": "rb_decay",
                "fit_channel": fit_channel,
            })
            data.analysis_data.update({
                "fit_input": {"values": y, "dims": ["x"]},
                "fit_curve": {"values": fit_curve, "dims": ["x"]},
                "residual": {"values": y - fit_curve, "dims": ["x"]},
            })
        except Exception as exc:
            data.quality = QualityFlag.BAD
            data.quality_message = f"RB fit failed: {exc}"


class AllXYAnalysis(BaseAnalysis):
    """AllXY pulse-sequence gate fidelity assessment."""

    thresholds = {
        "allxy_error": {"max": 0.1},
    }

    def _run(self, data: ExperimentData) -> None:
        """Run the operation.

        Parameters
        ----------
        data : ExperimentData
            Input data to process.
        """
        if data.raw_iq is None:
            return
        # AllXY ideal values for the 21 sequences: pattern of 0, 0.5, 1
        ideal = np.array([
            0, 0, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0
        ])
        measured = np.abs(data.raw_iq)
        # Normalize measured to [0, 1]
        measured_norm = (measured - measured.min()) / (measured.max() - measured.min() + 1e-12)

        if len(measured_norm) == len(ideal):
            error = float(np.mean(np.abs(measured_norm - ideal)))
            data.fit_result = {"allxy_error": (error, None)}
            data.scalar_result = error
        else:
            data.quality = QualityFlag.NO_INFORMATION
