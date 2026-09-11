"""single_shot_gef: analysis."""

from QickworkspaceV2.calibration.updates import CalibrationUpdates

import numpy as np
from QickworkspaceV2.data.models import FitResult


def analyze(result):
    fits = {}
    for q, trace in result.traces.items():
        try:
            if trace.shot_dims != ("shot", "readout") or trace.shots.shape[1] != 3 or len(trace.shots) < 40:
                raise ValueError("Expected at least 40 aligned g/e/f shots")
            if list(trace.coords["readout"]) != ["ground", "excited", "second_excited"]:
                raise ValueError("Expected ordered ground/excited/second_excited labels")
            if not np.isfinite(trace.shots).all():
                raise ValueError("Nonfinite g/e/f shots")
            centers = trace.shots[::2].mean(axis=0)
            if min(abs(centers[i] - centers[j]) for i in range(3) for j in range(i)) <= 1e-12:
                raise ValueError("Prepared states have no centroid separation")
            validation = trace.shots[1::2]
            predictions = np.argmin(abs(validation[:, :, None] - centers[None, None, :]), axis=-1)
            confusion = [[float(np.mean(predictions[:, i] == j)) for j in range(3)] for i in range(3)]
            fidelity = float(np.trace(confusion) / 3)
            trace.metadata.update(
                confusion_matrix=confusion,
                classifier_centers=centers,
                classifier="held-out nearest centroid in complex IQ",
            )
            fits[q] = FitResult(
                "readout_gef",
                fidelity >= 0.8,
                {"fidelity": fidelity},
                message="Held-out three-state assignment; includes preparation errors",
            )
        except (ValueError, AttributeError) as exc:
            fits[q] = FitResult("readout_gef", False, message=str(exc))
    return fits


from ..single_shot.analysis import plot as plot


def updates(result, target):
    """Declare this experiment's explicit calibration updates; never write files."""
    return CalibrationUpdates(
        reason="Three-state classification does not define a single binary feedback threshold"
    )


__all__ = ["analyze", "plot", "updates"]
