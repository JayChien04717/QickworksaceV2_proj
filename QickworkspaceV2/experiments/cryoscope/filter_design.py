"""Step 3: turn measured cryoscope X/Y into a predistortion filter.

The FIR path is the recommended starting point: it is causal, regularized, and
does not hide an unstable exact inverse.  The IIR path is provided for compact
models, but deliberately refuses to return an unstable inverse by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter, savgol_filter


@dataclass(frozen=True)
class CryoscopeTrace:
    """Processed two-quadrature cryoscope data."""

    time_ns: np.ndarray
    x: np.ndarray
    y: np.ndarray
    wrapped_phase_rad: np.ndarray
    unwrapped_phase_rad: np.ndarray
    smooth_phase_rad: np.ndarray
    detuning_mhz: np.ndarray
    normalized_step: np.ndarray


@dataclass(frozen=True)
class InverseFIRDesign:
    """Regularized causal FIR inverse and the response used to design it."""

    taps: np.ndarray
    line_impulse: np.ndarray
    delay_samples: int
    regularization: float
    smoothness: float


@dataclass(frozen=True)
class InverseIIRDesign:
    """Identified line filter and its exact coefficient-swapped inverse."""

    line_b: np.ndarray
    line_a: np.ndarray
    inverse_b: np.ndarray
    inverse_a: np.ndarray
    line_poles: np.ndarray
    inverse_poles: np.ndarray


def as_complex_iq(iq) -> np.ndarray:
    """Accept complex IQ or QICK's final ``[..., I, Q]`` representation."""

    values = np.asarray(iq)
    if np.iscomplexobj(values):
        return np.squeeze(values.astype(complex))
    if values.ndim > 0 and values.shape[-1] == 2:
        return np.squeeze(values[..., 0] + 1j * values[..., 1])
    return np.squeeze(values.astype(complex))


def project_iq_to_expectation(iq, iq_ground: complex, iq_excited: complex) -> np.ndarray:
    """Project resonator IQ onto the calibrated g-e axis and return <sigma_z>."""

    iq = as_complex_iq(iq)
    calibration_vector = complex(iq_excited) - complex(iq_ground)
    if abs(calibration_vector) == 0:
        raise ValueError("iq_ground and iq_excited must be different")
    excited_population = np.real(
        (iq - complex(iq_ground)) * np.conj(calibration_vector)
    ) / abs(calibration_vector) ** 2
    return 1.0 - 2.0 * excited_population


def merge_xy_segments(*segments):
    """Merge ``(time_ns, x, y)`` segments and average duplicate time points.

    A normal workflow passes the zero reference, zero-padded short points, and
    the longer const sweep as three segments.
    """

    if not segments:
        raise ValueError("at least one (time_ns, x, y) segment is required")
    times = []
    x_values = []
    y_values = []
    for time_ns, x, y in segments:
        time_ns = np.atleast_1d(np.asarray(time_ns, dtype=float))
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if time_ns.shape != x.shape or time_ns.shape != y.shape:
            raise ValueError("each segment's time_ns, x, and y shapes must match")
        times.append(time_ns)
        x_values.append(x)
        y_values.append(y)

    times = np.concatenate(times)
    x_values = np.concatenate(x_values)
    y_values = np.concatenate(y_values)
    order = np.argsort(times)
    times = times[order]
    x_values = x_values[order]
    y_values = y_values[order]

    unique_times, inverse = np.unique(times, return_inverse=True)
    x_merged = np.zeros(unique_times.size)
    y_merged = np.zeros(unique_times.size)
    counts = np.zeros(unique_times.size)
    np.add.at(x_merged, inverse, x_values)
    np.add.at(y_merged, inverse, y_values)
    np.add.at(counts, inverse, 1)
    return unique_times, x_merged / counts, y_merged / counts


def _odd_window(requested: int, length: int, polyorder: int) -> int:
    window = min(int(requested), length if length % 2 else length - 1)
    minimum = polyorder + 2
    if minimum % 2 == 0:
        minimum += 1
    window = max(window, minimum)
    if window > length:
        raise ValueError("trace is too short for the requested smoothing polynomial")
    return window


def trace_from_xy(
    time_ns,
    x,
    y,
    *,
    smooth_window: int = 15,
    polyorder: int = 3,
    tail_points: int = 10,
) -> CryoscopeTrace:
    """Convert Bloch X/Y traces into phase, detuning, and a unit step response.

    ``x`` and ``y`` are expectation-value traces, not raw resonator IQ.  Use
    :func:`project_iq_to_expectation` first when starting from QICK IQ data.
    """

    time_ns = np.asarray(time_ns, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if time_ns.ndim != 1 or x.shape != time_ns.shape or y.shape != time_ns.shape:
        raise ValueError("time_ns, x, and y must be one-dimensional arrays of equal length")
    if time_ns.size < 7 or np.any(np.diff(time_ns) <= 0):
        raise ValueError("time_ns must contain at least 7 strictly increasing points")

    wrapped = np.arctan2(y, x)
    unwrapped = np.unwrap(wrapped)
    window = _odd_window(smooth_window, time_ns.size, polyorder)
    smooth = savgol_filter(unwrapped, window, polyorder, mode="interp")
    detuning_mhz = 1000.0 * np.gradient(smooth, time_ns) / (2 * np.pi)

    tail_points = max(1, min(int(tail_points), detuning_mhz.size))
    steady_detuning = float(np.median(detuning_mhz[-tail_points:]))
    if abs(steady_detuning) < 1e-12:
        raise ValueError("steady detuning is zero; the cryoscope signal is too small to normalize")
    normalized_step = detuning_mhz / steady_detuning

    return CryoscopeTrace(
        time_ns=time_ns,
        x=x,
        y=y,
        wrapped_phase_rad=wrapped,
        unwrapped_phase_rad=unwrapped,
        smooth_phase_rad=smooth,
        detuning_mhz=detuning_mhz,
        normalized_step=normalized_step,
    )


def normalize_step_response(step_response, *, head_points=0, tail_points=10) -> np.ndarray:
    """Scale a step response to one, optionally removing pre-step baseline.

    The default assumes the caller already supplied a zero-referenced response,
    as :func:`trace_from_xy` does.  Set ``head_points`` to the number of known
    pre-step samples when a baseline still needs to be removed.
    """

    step = np.asarray(step_response, dtype=float)
    if step.ndim != 1 or step.size < 4 or not np.all(np.isfinite(step)):
        raise ValueError("step_response must be a finite one-dimensional array")
    head_points = max(0, min(int(head_points), step.size))
    tail_points = max(1, min(int(tail_points), step.size))
    baseline = float(np.median(step[:head_points])) if head_points else 0.0
    settled = float(np.median(step[-tail_points:]))
    scale = settled - baseline
    if abs(scale) < 1e-12:
        raise ValueError("step response has no measurable settled change")
    return (step - baseline) / scale


def impulse_from_step(step_response) -> np.ndarray:
    """Convert a normalized discrete step response into an impulse response."""

    step = normalize_step_response(step_response)
    return np.diff(step, prepend=0.0)


def design_inverse_fir(
    step_response,
    *,
    taps: int = 64,
    delay_samples: int | None = None,
    regularization: float = 1e-3,
    smoothness: float = 1e-2,
    impulse_samples: int | None = None,
) -> InverseFIRDesign:
    """Design a causal regularized FIR inverse by least squares.

    The convolution of the measured line impulse with the returned taps is
    fitted to a delayed unit impulse.  The delay makes a causal approximation
    possible; the two regularizers prevent noisy, alternating large taps.
    """

    taps = int(taps)
    if taps < 2:
        raise ValueError("taps must be at least 2")
    if regularization < 0 or smoothness < 0:
        raise ValueError("regularization and smoothness must be non-negative")

    impulse = impulse_from_step(step_response)
    if impulse_samples is not None:
        impulse = impulse[: int(impulse_samples)]
    if delay_samples is None:
        delay_samples = max(1, taps // 4)
    delay_samples = int(delay_samples)

    output_length = impulse.size + taps - 1
    if not 0 <= delay_samples < output_length:
        raise ValueError("delay_samples lies outside the modeled convolution")

    convolution = np.zeros((output_length, taps))
    for column in range(taps):
        convolution[column : column + impulse.size, column] = impulse
    desired = np.zeros(output_length)
    desired[delay_samples] = 1.0

    difference = np.diff(np.eye(taps), axis=0)
    lhs = convolution.T @ convolution
    lhs += regularization * np.eye(taps)
    lhs += smoothness * (difference.T @ difference)
    rhs = convolution.T @ desired
    inverse_taps = np.linalg.solve(lhs, rhs)

    return InverseFIRDesign(
        taps=inverse_taps,
        line_impulse=impulse,
        delay_samples=delay_samples,
        regularization=float(regularization),
        smoothness=float(smoothness),
    )


def fit_inverse_iir(
    step_response,
    *,
    numerator_order: int = 1,
    denominator_order: int = 2,
    ridge: float = 1e-8,
    allow_unstable: bool = False,
) -> InverseIIRDesign:
    """Fit an ARX line model to a measured step and invert its coefficients.

    Exact IIR inversion is safe only for a minimum-phase fitted line.  When a
    fitted line has a zero on or outside the unit circle, its inverse has an
    unstable pole and this function raises unless ``allow_unstable=True``.
    """

    step = normalize_step_response(step_response)
    na = int(denominator_order)
    nb = int(numerator_order)
    if na < 1 or nb < 0:
        raise ValueError("denominator_order >= 1 and numerator_order >= 0 are required")

    rows = []
    targets = []
    for index in range(step.size):
        past_output = [
            -step[index - lag] if index - lag >= 0 else 0.0
            for lag in range(1, na + 1)
        ]
        step_input = [1.0 if index >= lag else 0.0 for lag in range(nb + 1)]
        rows.append(past_output + step_input)
        targets.append(step[index])
    matrix = np.asarray(rows)
    targets = np.asarray(targets)
    theta = np.linalg.solve(
        matrix.T @ matrix + float(ridge) * np.eye(matrix.shape[1]),
        matrix.T @ targets,
    )
    line_a = np.r_[1.0, theta[:na]]
    line_b = theta[na:]

    # Enforce the measured normalization H(z=1)=1.
    dc_numerator = float(np.sum(line_b))
    if abs(dc_numerator) < 1e-12:
        raise ValueError("identified IIR has zero DC gain and cannot be inverted")
    line_b *= float(np.sum(line_a)) / dc_numerator

    line_poles = np.roots(line_a)
    inverse_poles = np.roots(line_b)
    if inverse_poles.size and np.max(np.abs(inverse_poles)) >= 1 and not allow_unstable:
        raise ValueError(
            "the fitted line is not minimum phase, so its exact IIR inverse is unstable; "
            "use design_inverse_fir() or refit with different orders"
        )

    return InverseIIRDesign(
        line_b=line_b,
        line_a=line_a,
        inverse_b=line_a.copy(),
        inverse_a=line_b.copy(),
        line_poles=line_poles,
        inverse_poles=inverse_poles,
    )


def apply_inverse_fir(target, design: InverseFIRDesign, *, keep_tail=True) -> np.ndarray:
    """Apply a designed FIR inverse to an ideal normalized flux waveform."""

    target = np.asarray(target, dtype=float)
    mode = "full" if keep_tail else "same"
    return np.convolve(target, design.taps, mode=mode)


def apply_inverse_iir(target, design: InverseIIRDesign) -> np.ndarray:
    """Apply a stable exact IIR inverse to an ideal normalized waveform."""

    return lfilter(design.inverse_b, design.inverse_a, np.asarray(target, dtype=float))


def scale_waveform(waveform, *, max_abs: float = 0.95):
    """Scale, never clip, a predistorted waveform to a requested safe peak.

    Returns ``(scaled_waveform, scale_factor)``.  Multiply the intended physical
    flux amplitude by the inverse factor when interpreting the result.
    """

    waveform = np.asarray(waveform, dtype=float)
    if not 0 < max_abs <= 1:
        raise ValueError("max_abs must lie in (0, 1]")
    peak = float(np.max(np.abs(waveform)))
    if peak == 0 or peak <= max_abs:
        return waveform.copy(), 1.0
    factor = max_abs / peak
    return waveform * factor, factor


def predict_corrected_output(predistorted_waveform, step_response) -> np.ndarray:
    """Convolve a candidate waveform with the measured line impulse."""

    impulse = impulse_from_step(step_response)
    return np.convolve(np.asarray(predistorted_waveform, dtype=float), impulse)


__all__ = [
    "CryoscopeTrace",
    "InverseFIRDesign",
    "InverseIIRDesign",
    "as_complex_iq",
    "project_iq_to_expectation",
    "merge_xy_segments",
    "trace_from_xy",
    "normalize_step_response",
    "impulse_from_step",
    "design_inverse_fir",
    "fit_inverse_iir",
    "apply_inverse_fir",
    "apply_inverse_iir",
    "scale_waveform",
    "predict_corrected_output",
]
