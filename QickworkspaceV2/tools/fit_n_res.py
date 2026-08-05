"""Utilities for locating multiple resonators in a complex transmission trace."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter


def _validated_trace(
    freq_hz: np.ndarray,
    s21: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize a frequency-domain transmission trace."""
    frequency = np.asarray(freq_hz, dtype=float)
    transmission = np.asarray(s21, dtype=complex)

    if frequency.ndim != 1 or transmission.ndim != 1:
        raise ValueError("freq_hz and s21 must both be one-dimensional arrays.")
    if frequency.size != transmission.size:
        raise ValueError("freq_hz and s21 must contain the same number of points.")
    if frequency.size < 7:
        raise ValueError("At least seven samples are required to detect resonators.")
    if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(transmission)):
        raise ValueError("freq_hz and s21 must contain only finite values.")
    if not np.all(np.diff(frequency) > 0):
        raise ValueError("freq_hz must be strictly increasing.")

    return frequency, transmission


def _savgol_window(size: int, preferred: int, polyorder: int = 3) -> int:
    """Return a valid odd Savitzky-Golay window length."""
    window = min(preferred, size if size % 2 else size - 1)
    minimum = polyorder + 2 if polyorder % 2 else polyorder + 1
    return max(window, minimum)


def phase_reference_candidates(
    freq_hz: np.ndarray,
    s21: np.ndarray,
    *,
    max_candidates: int | None = 12,
    min_distance_points: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Find resonator candidates from the phase derivative of an S21 trace.

    A linear phase ramp is removed before differentiating the unwrapped phase.
    The slope of that ramp also provides an estimate of the electrical delay.

    Parameters
    ----------
    freq_hz : numpy.ndarray
        Strictly increasing frequency samples in hertz.
    s21 : numpy.ndarray
        Complex transmission values corresponding to ``freq_hz``.
    max_candidates : int or None, optional
        Maximum number of phase candidates to return. If ``None``, retain all
        candidates. The default is 12.
    min_distance_points : int, optional
        Minimum separation between phase candidates in samples. The default is
        20.

    Returns
    -------
    indices : numpy.ndarray
        Integer indices of the phase-reference candidates.
    corrected_phase : numpy.ndarray
        Smoothed, unwrapped phase after removal of the linear phase ramp.
    phase_derivative : numpy.ndarray
        Smoothed absolute phase derivative in radians per hertz.
    electrical_delay_s : float
        Electrical delay estimated from the linear phase ramp, in seconds.

    Raises
    ------
    ValueError
        If the input arrays are invalid or an option is out of range.
    """
    frequency, transmission = _validated_trace(freq_hz, s21)
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be positive or None.")
    if min_distance_points < 1:
        raise ValueError("min_distance_points must be positive.")

    phase = np.unwrap(np.angle(transmission))
    delay_fit = np.polyfit(frequency, phase, 1)
    electrical_delay_s = -float(delay_fit[0]) / (2 * np.pi)
    corrected_phase = phase - np.polyval(delay_fit, frequency)

    window = _savgol_window(corrected_phase.size, preferred=31)
    corrected_phase = savgol_filter(corrected_phase, window, polyorder=3)
    phase_derivative = np.abs(np.gradient(corrected_phase, frequency))
    phase_derivative = savgol_filter(phase_derivative, window, polyorder=3)

    prominence = max(
        float(np.nanstd(phase_derivative) * 0.25),
        float(np.nanpercentile(phase_derivative, 75) * 0.1),
        np.finfo(float).eps,
    )
    peaks, properties = find_peaks(
        phase_derivative,
        distance=min_distance_points,
        prominence=prominence,
    )
    if max_candidates is not None and peaks.size > max_candidates:
        strongest = np.argsort(properties["prominences"])[-max_candidates:]
        peaks = peaks[strongest]

    peaks = peaks[np.argsort(frequency[peaks])]
    return peaks, corrected_phase, phase_derivative, electrical_delay_s


def detect_resonators(
    freq_hz: np.ndarray,
    s21: np.ndarray,
    *,
    count: int = 6,
    min_distance_points: int = 6,
    prominence_sigma: float = 0.08,
    width_penalty: float = 0.5,
    use_phase_reference: bool = True,
    phase_snap_hz: float = 0.015e9,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Detect the strongest resonator dips in a complex transmission trace.

    Candidates are found from the smoothed negative magnitude and ranked by
    prominence with an optional width penalty. Phase-derivative candidates can
    refine the location of broad or asymmetric magnitude dips.

    Parameters
    ----------
    freq_hz : numpy.ndarray
        Strictly increasing frequency samples in hertz.
    s21 : numpy.ndarray
        Complex transmission values corresponding to ``freq_hz``.
    count : int, optional
        Number of resonators to detect. The default is 6.
    min_distance_points : int, optional
        Minimum separation between magnitude candidates in samples. The default
        is 6.
    prominence_sigma : float, optional
        Scale factor applied to the standard deviation of the smoothed negative
        magnitude when setting the prominence threshold. The default is 0.08.
    width_penalty : float, optional
        Exponent used to penalize broad candidates during ranking. Set to zero
        to rank by prominence only. The default is 0.5.
    use_phase_reference : bool, optional
        Whether phase-derivative candidates may refine dip locations. The
        default is ``True``.
    phase_snap_hz : float, optional
        Maximum frequency separation for associating a magnitude candidate with
        a phase candidate. The default is 15 MHz.

    Returns
    -------
    indices : numpy.ndarray
        Sorted integer indices of the detected resonators.
    properties : dict of str to numpy.ndarray
        Candidate metrics and phase-reference diagnostics. Candidate-sized
        arrays refer to every magnitude candidate returned by ``find_peaks``.
    smoothed_negative_magnitude : numpy.ndarray
        Smoothed negative magnitude used for dip detection.

    Raises
    ------
    RuntimeError
        If fewer candidates are found than requested.
    ValueError
        If the inputs or detection options are invalid.
    """
    frequency, transmission = _validated_trace(freq_hz, s21)
    if count < 1:
        raise ValueError("count must be positive.")
    if min_distance_points < 1:
        raise ValueError("min_distance_points must be positive.")
    if prominence_sigma < 0 or width_penalty < 0 or phase_snap_hz < 0:
        raise ValueError(
            "prominence_sigma, width_penalty, and phase_snap_hz must be non-negative."
        )

    magnitude = np.abs(transmission)
    window = _savgol_window(magnitude.size, preferred=11)
    smooth_dips = savgol_filter(-magnitude, window, polyorder=3)
    dynamic_range = float(
        np.nanpercentile(magnitude, 95) - np.nanpercentile(magnitude, 5)
    )
    prominence_floor = max(dynamic_range * 0.01, np.finfo(float).eps)
    prominence = max(
        float(np.nanstd(smooth_dips) * prominence_sigma), prominence_floor
    )

    peaks, properties = find_peaks(
        smooth_dips,
        distance=min_distance_points,
        prominence=prominence,
    )
    if peaks.size < count:
        raise RuntimeError(
            f"Only found {peaks.size} candidate dips; requested {count}. "
            "Try lowering prominence_sigma or min_distance_points."
        )

    widths_result = peak_widths(smooth_dips, peaks, rel_height=0.5)
    widths = widths_result[0]
    left_ips = widths_result[2]
    right_ips = widths_result[3]
    scores = properties["prominences"] / np.maximum(widths, 1.0) ** width_penalty
    ranked_candidates = np.argsort(scores)[::-1]
    selected = peaks[ranked_candidates[:count]].tolist()

    phase_refs = np.array([], dtype=int)
    corrected_phase = np.array([], dtype=float)
    phase_derivative = np.array([], dtype=float)
    electrical_delay_s = np.nan
    if use_phase_reference:
        phase_refs, corrected_phase, phase_derivative, electrical_delay_s = (
            phase_reference_candidates(
                frequency,
                transmission,
                max_candidates=None,
                min_distance_points=min_distance_points,
            )
        )
        refined = []
        for candidate_position in ranked_candidates[:count]:
            candidate = peaks[candidate_position]
            left = max(0, int(np.floor(left_ips[candidate_position])))
            right = min(magnitude.size - 1, int(np.ceil(right_ips[candidate_position])))
            region_refs = phase_refs[(phase_refs >= left) & (phase_refs <= right)]

            if region_refs.size:
                anchor = region_refs[np.argmax(phase_derivative[region_refs])]
                nearby_dips = peaks[
                    (peaks >= left)
                    & (peaks <= right)
                    & (np.abs(frequency[peaks] - frequency[anchor]) <= phase_snap_hz)
                ]
                refined.append(
                    int(nearby_dips[np.argmin(np.abs(nearby_dips - anchor))])
                    if nearby_dips.size
                    else int(anchor)
                )
            else:
                refined.append(int(candidate))

        # Refinement can map two candidates to the same phase feature. Fill any
        # missing slots from the original score ranking so count stays stable.
        selected = list(dict.fromkeys(refined))
        for candidate_position in ranked_candidates:
            candidate = int(peaks[candidate_position])
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) == count:
                break

    selected_array = np.asarray(
        sorted(selected[:count], key=lambda index: frequency[index]), dtype=int
    )
    display_phase_refs = []
    for selected_index in selected_array:
        nearby = phase_refs[
            np.abs(frequency[phase_refs] - frequency[selected_index]) <= 0.025e9
        ]
        if nearby.size:
            display_phase_refs.append(
                int(nearby[np.argmin(np.abs(nearby - selected_index))])
            )

    result_properties = dict(properties)
    result_properties.update(
        {
            "widths": widths,
            "left_ips": left_ips,
            "right_ips": right_ips,
            "scores": scores,
            "phase_refs": np.asarray(sorted(set(display_phase_refs)), dtype=int),
            "all_phase_refs": phase_refs,
            "phase_corr": corrected_phase,
            "dphase": phase_derivative,
            "electrical_delay_s": np.asarray([electrical_delay_s]),
        }
    )
    return selected_array, result_properties, smooth_dips


def fit_n_resonators(
    freq_hz: np.ndarray,
    s21: np.ndarray,
    *,
    count: int,
    **detection_options,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Estimate the frequencies of multiple resonators in an S21 trace.

    Each detected magnitude minimum is refined with a local three-point
    quadratic interpolation. This provides a sub-bin frequency estimate without
    imposing a global multi-resonator line-shape model.

    Parameters
    ----------
    freq_hz : numpy.ndarray
        Strictly increasing frequency samples in hertz.
    s21 : numpy.ndarray
        Complex transmission values corresponding to ``freq_hz``.
    count : int
        Number of resonators to fit.
    **detection_options
        Additional keyword arguments forwarded to :func:`detect_resonators`.

    Returns
    -------
    resonator_freqs_hz : numpy.ndarray
        Sorted sub-bin resonator-frequency estimates in hertz.
    fit_details : dict of str to numpy.ndarray
        Detection diagnostics plus ``indices``, ``sample_freqs_hz``,
        ``fitted_freqs_hz``, and ``dip_magnitudes``.

    Raises
    ------
    RuntimeError
        If fewer candidates are found than requested.
    ValueError
        If the inputs or detection options are invalid.
    """
    frequency, transmission = _validated_trace(freq_hz, s21)
    indices, properties, smooth_dips = detect_resonators(
        frequency,
        transmission,
        count=count,
        **detection_options,
    )
    magnitude = np.abs(transmission)
    fitted_frequencies = []
    for index in indices:
        estimate = float(frequency[index])
        if 0 < index < frequency.size - 1:
            local_frequency = frequency[index - 1 : index + 2]
            local_magnitude = magnitude[index - 1 : index + 2]
            origin = float(local_frequency[1])
            quadratic, linear, _ = np.polyfit(
                local_frequency - origin, local_magnitude, 2
            )
            if quadratic > 0:
                vertex = origin - linear / (2 * quadratic)
                if local_frequency[0] <= vertex <= local_frequency[-1]:
                    estimate = float(vertex)
        fitted_frequencies.append(estimate)

    fitted = np.asarray(fitted_frequencies, dtype=float)
    details = dict(properties)
    details.update(
        {
            "indices": indices,
            "sample_freqs_hz": frequency[indices],
            "fitted_freqs_hz": fitted,
            "dip_magnitudes": magnitude[indices],
            "smooth_dips": smooth_dips,
        }
    )
    return fitted, details


__all__ = [
    "detect_resonators",
    "fit_n_resonators",
    "phase_reference_candidates",
]
