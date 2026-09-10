"""Complex hanger response, including cable delay and coupling asymmetry.

Model convention: Probst et al., https://arxiv.org/abs/1410.3365.
This implementation uses bounded nonlinear least squares, not their circle-fit algorithm.
Frequencies are MHz, cable delay is us. A calibrated transmission notch is required.
"""

import numpy as np
from scipy.optimize import least_squares
from QickworkspaceV2.data.models import FitResult


def notch_response(
    frequency, center, linewidth, depth, asymmetry=0.0, scale=1.0, phase=0.0, delay_us=0.0, reference_mhz=None
):
    f = np.asarray(frequency, float)
    reference = float(np.mean(f)) if reference_mhz is None else reference_mhz
    background = scale * np.exp(1j * phase - 2j * np.pi * (f - reference) * delay_us)
    return background * (1 - depth * np.exp(1j * asymmetry) / (1 + 2j * (f - center) / linewidth))


def fit_resonator(frequency_mhz, iq, *, min_r_squared=0.95):
    """Fit complex S21; return Ql, |Qc| and diameter-corrected Qi with uncertainties."""
    f, z = np.asarray(frequency_mhz, float), np.asarray(iq, complex)
    fail = lambda message: FitResult("complex_notch", False, message=message)
    if f.ndim != 1 or z.shape != f.shape or len(f) < 16:
        return fail("At least 16 complex frequency points required")
    if not np.isfinite(f).all() or not np.isfinite(z).all():
        return fail("Nonfinite resonator data")
    order = np.argsort(f)
    f, z = f[order], z[order]
    if np.any(np.diff(f) <= 0) or np.ptp(abs(z)) < 1e-10:
        return fail("Distinct frequencies and measurable notch contrast required")
    reference, span = float(np.mean(f)), float(np.ptp(f))
    x = (f - reference) / span
    scale = float(np.median(np.r_[abs(z[: len(z) // 8]), abs(z[-len(z) // 8 :])]))
    y = z / scale
    # Initial slope uses both ends where the resonance contribution is small.
    edges = np.r_[np.arange(len(f) // 8), np.arange(len(f) - len(f) // 8, len(f))]
    slope, phase = np.polyfit(x[edges], np.unwrap(np.angle(y))[edges], 1)
    delay = -slope / (2 * np.pi)
    center = float(x[np.argmin(abs(y))])
    lower = [-0.5, 1 / (len(f) * 10), 0.001, -1.5, 0.01, phase - 2 * np.pi, delay - 10]
    upper = [0.5, 2.0, 3.0, 1.5, 10.0, phase + 2 * np.pi, delay + 10]

    def model(p, axis=x):
        c, width, depth, asym, amp, angle, d = p
        return (
            amp
            * np.exp(1j * angle - 2j * np.pi * axis * d)
            * (1 - depth * np.exp(1j * asym) / (1 + 2j * (axis - c) / width))
        )

    def residual(p):
        r = model(p) - y
        return np.r_[r.real, r.imag]

    attempts = []
    for width in (0.03, 0.1, 0.3):
        for asym in (-0.2, 0, 0.2):
            initial = [center, width, max(0.05, 1 - min(abs(y))), asym, 1.0, phase, delay]
            attempts.append(least_squares(residual, initial, bounds=(lower, upper), max_nfev=3000))
    best = min(attempts, key=lambda fit: np.sum(fit.fun**2))
    c, width, depth, asym, amp, angle, d = best.x
    center_mhz, linewidth = reference + c * span, width * span
    ql, inverse_fraction = center_mhz / linewidth, 1 - depth * np.cos(asym)
    r2 = float(1 - np.sum(best.fun**2) / np.sum(abs(y - y.mean()) ** 2))
    reasons = []
    if not best.success or r2 < min_r_squared:
        reasons.append("Complex fit did not pass convergence/contrast quality")
    if min(center_mhz - f[0], f[-1] - center_mhz) < linewidth:
        reasons.append("Notch too close to boundary; include both baseline regions")
    if inverse_fraction <= 0:
        reasons.append("Nonphysical internal loss; check notch model and calibration")
    params = dict(
        center=center_mhz,
        linewidth=linewidth,
        Q_loaded=ql,
        Q_coupling_abs=ql / depth,
        depth=depth,
        asymmetry_rad=asym,
        scale=amp * scale,
        phase_rad=angle,
        delay_us=d / span,
    )
    if inverse_fraction > 0:
        params["Q_internal"] = ql / inverse_fraction
    errors = {}
    if np.linalg.matrix_rank(best.jac) < len(best.x):
        reasons.append("Resonator parameters are not identifiable")
    else:
        covariance = np.linalg.pinv(best.jac.T @ best.jac) * np.sum(best.fun**2) / (2 * len(f) - len(best.x))

        def metrics(p):
            c, w, dep, a, amp, ph, delay = p
            q = (reference + c * span) / (w * span)
            values = [reference + c * span, w * span, q, q / dep, dep, a, amp * scale, ph, delay / span]
            if "Q_internal" in params:
                values.append(q / (1 - dep * np.cos(a)))
            return np.array(values)

        jac = np.empty((len(params), len(best.x)))
        for i in range(len(best.x)):
            step = 1e-6 * max(abs(best.x[i]), 1)
            plus, minus = best.x.copy(), best.x.copy()
            plus[i] += step
            minus[i] -= step
            jac[:, i] = (metrics(plus) - metrics(minus)) / (2 * step)
        std = np.sqrt(np.maximum(np.diag(jac @ covariance @ jac.T), 0))
        errors = dict(zip(params, map(float, std)))
        for key in ("linewidth", "Q_internal"):
            if key in errors and (not np.isfinite(errors[key]) or errors[key] > abs(params[key]) * 0.5):
                reasons.append(f"{key} is poorly constrained")
        errors = {k: v if np.isfinite(v) else None for k, v in errors.items()}
    xf = np.linspace(f[0], f[-1], 400)
    return FitResult(
        "complex_notch",
        not reasons,
        {k: float(v) for k, v in params.items()},
        errors,
        {"center": "MHz", "linewidth": "MHz", "delay_us": "us"},
        r2,
        "; ".join(reasons) or "Complex notch fit passed; Qi assumes the hanger transmission model",
        xf.tolist(),
        (abs(model(best.x, (xf - reference) / span)) * scale).tolist(),
        (abs(z) - abs(model(best.x)) * scale).tolist(),
    )
