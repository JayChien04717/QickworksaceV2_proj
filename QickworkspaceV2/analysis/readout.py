"""Single-shot discrimination and joint probabilities; no averaged-IQ inference."""

from dataclasses import dataclass
from itertools import product
import numpy as np
from QickworkspaceV2.data.models import FitResult


@dataclass(frozen=True)
class ReadoutClassifier:
    rotation_deg: float
    threshold: float
    fidelity: float
    confusion: list[list[float]]

    def predict(self, iq):
        return (
            np.real(np.asarray(iq) * np.exp(-1j * np.deg2rad(self.rotation_deg))) > self.threshold
        ).astype(int)


def train_classifier(ground, excited, *, rotation_deg=None):
    ground, excited = np.asarray(ground).ravel(), np.asarray(excited).ravel()
    if min(len(ground), len(excited)) < 40 or not np.isfinite(ground).all() or not np.isfinite(excited).all():
        raise ValueError("Classifier requires at least 40 finite shots per prepared state")
    g, e = ground[::2], excited[::2]
    delta = e.mean() - g.mean()
    if abs(delta) <= np.finfo(float).eps:
        raise ValueError("Prepared states have no IQ separation")
    # Pooled within-state covariance gives a linear discriminant for anisotropic IQ noise.
    gxy, exy = np.column_stack([g.real, g.imag]), np.column_stack([e.real, e.imag])
    covariance = (np.cov(gxy.T) + np.cov(exy.T)) / 2
    direction = np.linalg.pinv(covariance + np.eye(2) * max(np.trace(covariance), 1e-15) * 1e-8) @ np.array(
        [delta.real, delta.imag]
    )
    rotation = (float(np.rad2deg(np.arctan2(direction[1], direction[0])))
                if rotation_deg is None else float(rotation_deg))
    if not np.isfinite(rotation):
        raise ValueError("Classifier rotation must be finite")
    threshold = float(np.real((g.mean() + e.mean()) / 2 * np.exp(-1j * np.deg2rad(rotation))))
    model = ReadoutClassifier(rotation, threshold, 0, [])
    pg, pe = model.predict(ground[1::2]), model.predict(excited[1::2])
    confusion = [
        [float(np.mean(pg == 0)), float(np.mean(pg == 1))],
        [float(np.mean(pe == 0)), float(np.mean(pe == 1))],
    ]
    return ReadoutClassifier(rotation, threshold, (confusion[0][0] + confusion[1][1]) / 2, confusion)


def analyze_single_shot(result, *, threshold=None, rotation_deg=0.0, fixed_axis=False):
    """Train a discriminator, or evaluate an explicitly configured rotated-I threshold.

    The fixed threshold does not alter stored IQ/shots or recalibrate configuration.
    """
    if threshold is not None and (not np.isfinite(threshold) or not np.isfinite(rotation_deg)):
        raise ValueError("threshold and rotation_deg must be finite")
    fits = {}
    for q, trace in result.traces.items():
        try:
            if trace.shots is None or trace.shot_dims != ("shot", "readout") or trace.shots.shape[1] != 2:
                raise ValueError("Expected paired ground/excited single shots with two readout events")
            labels = list(trace.coords["readout"])
            if labels != ["ground", "excited"]:
                raise ValueError("Expected ground/excited labels in preparation order")
            if threshold is None:
                classifier = train_classifier(trace.shots[:, 0], trace.shots[:, 1],
                                              rotation_deg=rotation_deg if fixed_axis else None)
                mode = "Held-out assignment"
            else:
                if len(trace.shots) < 40 or not np.isfinite(trace.shots).all():
                    raise ValueError("Fixed discrimination requires at least 40 finite shots per state")
                classifier = ReadoutClassifier(float(rotation_deg), float(threshold), 0, [])
                pg, pe = classifier.predict(trace.shots[:, 0]), classifier.predict(trace.shots[:, 1])
                confusion = [[float(np.mean(pg == 0)), float(np.mean(pg == 1))],
                             [float(np.mean(pe == 0)), float(np.mean(pe == 1))]]
                classifier = ReadoutClassifier(classifier.rotation_deg, classifier.threshold,
                                               (confusion[0][0] + confusion[1][1]) / 2, confusion)
                mode = "Configured-threshold assignment"
            fits[q] = FitResult(
                "readout",
                classifier.fidelity >= 0.8,
                {
                    "rotation_deg": classifier.rotation_deg,
                    "threshold": classifier.threshold,
                    "fidelity": classifier.fidelity,
                },
                units={"rotation_deg": "deg", "threshold": "ADC", "fidelity": ""},
                message=f"{mode} fidelity {classifier.fidelity:.4f}",
            )
            trace.metadata["confusion_matrix"] = classifier.confusion
        except ValueError as exc:
            fits[q] = FitResult("readout", False, message=str(exc))
    return fits


def joint_probabilities(shots: dict, classifiers: dict[str, ReadoutClassifier]):
    targets = tuple(shots)
    if not targets or len(targets) > 20:
        raise ValueError("Joint probabilities require 1–20 targets")
    values = [np.asarray(shots[q]) for q in targets]
    if any(v.ndim != 1 for v in values) or len({len(v) for v in values}) != 1 or not len(values[0]):
        raise ValueError("Joint readout needs aligned, nonempty one-dimensional shots")
    if not all(np.isfinite(v).all() for v in values):
        raise ValueError("Joint readout shots must be finite")
    states = np.column_stack([classifiers[q].predict(shots[q]) for q in targets])
    return {
        "".join(map(str, bits)): float(np.mean(np.all(states == bits, axis=1)))
        for bits in product((0, 1), repeat=len(targets))
    }
