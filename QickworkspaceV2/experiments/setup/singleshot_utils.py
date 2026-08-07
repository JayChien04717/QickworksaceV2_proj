"""
Histogram and analysis utilities for SingleShot experiments.

Fitting:  Gaussian Mixture Model (GMM) via scikit-learn.
Fidelity: mean of the confusion-matrix diagonal.
"""

from dataclasses import dataclass
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from scipy.stats import norm as _norm

try:
    from sklearn.mixture import GaussianMixture
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
linestyle_cycle = ["solid", "dashed", "dotted", "dashdot"]
marker_cycle    = ["o", "*", "s", "^"]


@dataclass(frozen=True)
class HistogramAnalysis:
    """Detailed single-shot analysis while preserving the legacy tuple API."""

    legacy_result: list
    fidelity: float
    thresholds: np.ndarray
    rotation_deg: float
    confusion_matrix_pct: np.ndarray
    state_gmms: tuple
    projections: tuple[np.ndarray, ...]
    primary_weights: np.ndarray


def weighted_assignment_score(g_to_e_error, e_to_g_error, e_to_g_weight=3.0):
    """Return assignment fidelity with extra weight on ``|e> -> |g>`` errors.

    The excited-to-ground assignment error is sensitive to T1 decay during
    readout, but ge-only IQ data cannot separate T1 decay from cloud overlap.
    A weight of 1 reproduces ordinary equal-prior threshold fidelity; the
    optimizer defaults to 3 so the excited-state error matters three times as
    much as the ground-state error.
    """
    weight = float(e_to_g_weight)
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("e_to_g_weight must be finite and non-negative")
    g_error = np.asarray(g_to_e_error, dtype=float)
    e_error = np.asarray(e_to_g_error, dtype=float)
    return 1.0 - (g_error + weight * e_error) / (1.0 + weight)


def fast_histogram_metrics(data) -> dict[str, float]:
    """Fast two-state readout metrics without histogram bins or GMM fitting.

    The IQ axis is aligned from robust median centers, then the threshold is
    selected directly from the empirical distributions.  This avoids both
    histogram range/bin sensitivity and repeated Gaussian-mixture fits.
    """
    ig = np.asarray(data["Ig"], dtype=float).ravel()
    qg = np.asarray(data["Qg"], dtype=float).ravel()
    ie = np.asarray(data["Ie"], dtype=float).ravel()
    qe = np.asarray(data["Qe"], dtype=float).ravel()
    g_ok = np.isfinite(ig) & np.isfinite(qg)
    e_ok = np.isfinite(ie) & np.isfinite(qe)
    ig, qg, ie, qe = ig[g_ok], qg[g_ok], ie[e_ok], qe[e_ok]
    if ig.size == 0 or ie.size == 0:
        raise ValueError("Ground and excited shot clouds must be non-empty")

    xg, yg = np.median(ig), np.median(qg)
    xe, ye = np.median(ie), np.median(qe)
    theta = -np.arctan2(ye - yg, xe - xg)
    cosine, sine = np.cos(theta), np.sin(theta)
    g_proj = ig * cosine - qg * sine
    e_proj = ie * cosine - qe * sine
    _, thresholds, confusion = _ordered_threshold_classifier([g_proj, e_proj])
    fidelity = float(np.mean(np.diag(confusion)))
    _, g_core_fraction = _dominant_core(g_proj)
    _, e_core_fraction = _dominant_core(e_proj)
    e_survival = float(1.0 - confusion[1, 0])
    # Useful discrimination x a clean excited-state peak x T1-sensitive
    # excited-state survival. All three factors are measured empirically.
    readout_score = fidelity * e_core_fraction * e_survival
    separation = float(abs(np.median(e_proj) - np.median(g_proj)))
    snr = separation**2 / (
        float(np.var(g_proj)) + float(np.var(e_proj)) + 1e-30
    )
    return {
        "fid": fidelity,
        "soft_fid": fidelity,
        "snr": snr,
        "sep": separation,
        "leakage": float(confusion[1, 0]),
        "thermal": float(confusion[0, 1]),
        "e_to_g_error": float(confusion[1, 0]),
        "g_to_e_error": float(confusion[0, 1]),
        "g_core_fraction": g_core_fraction,
        "e_core_fraction": e_core_fraction,
        "g_tail_fraction": 1.0 - g_core_fraction,
        "e_tail_fraction": 1.0 - e_core_fraction,
        "readout_score": readout_score,
        "threshold": float(thresholds[0]),
        "rotation_deg": float(np.degrees(theta)),
    }


def _dominant_core(values, nsigma=3.0, iterations=3):
    """Return the robust main peak and its retained-shot fraction."""
    values = np.asarray(values, dtype=float).ravel()
    core = values[np.isfinite(values)]
    for _ in range(iterations):
        center = float(np.median(core))
        sigma = 1.4826 * float(np.median(np.abs(core - center)))
        if not np.isfinite(sigma) or sigma <= 1e-12:
            break
        clipped = core[np.abs(core - center) <= nsigma * sigma]
        if clipped.size == 0 or clipped.size == core.size:
            break
        core = clipped
    return core, float(core.size / max(values.size, 1))


def histogram_metrics(details: HistogramAnalysis) -> dict[str, float]:
    """Derive optimizer metrics from an existing GMM fit without refitting.

    Parameters
    ----------
    details : HistogramAnalysis
        Value for ``details``.

    Returns
    -------
    dict[str, float]
        Result of the operation.

    Raises
    ------
    ValueError
        If the operation cannot be completed.
    """
    if len(details.projections) < 2 or len(details.state_gmms) < 2:
        raise ValueError("At least g and e states are required")

    projections = details.projections[:2]
    gmms = details.state_gmms[:2]
    means = [float(np.mean(values)) for values in projections]
    separation = abs(means[1] - means[0])
    snr = separation**2 / (
        float(np.var(projections[0])) + float(np.var(projections[1])) + 1e-30
    )

    soft_accuracies = []
    for state_index, values in enumerate(projections):
        samples = values.reshape(-1, 1)
        log_likelihoods = np.array([gmm.score_samples(samples) for gmm in gmms])
        shifted = log_likelihoods - log_likelihoods.max(axis=0)
        posteriors = np.exp(shifted)
        posteriors /= posteriors.sum(axis=0)
        soft_accuracies.append(float(posteriors[state_index].mean()))

    confusion = np.asarray(details.confusion_matrix_pct, dtype=float) / 100.0
    g_to_e_error = float(confusion[0, 1])
    e_to_g_error = float(confusion[1, 0])
    return {
        "fid": details.fidelity,
        "soft_fid": float(np.mean(soft_accuracies)),
        "snr": snr,
        "sep": separation,
        # Compatibility aliases: ge-only data measures assignment errors, not
        # physical f-level leakage or thermal population.
        "leakage": e_to_g_error,
        "thermal": g_to_e_error,
        "e_to_g_error": e_to_g_error,
        "g_to_e_error": g_to_e_error,
    }


def plot_hist(data, bins, ax=None, xlims=None, color=None, linestyle=None,
              label=None, alpha=None, normalize=True):
    """Plot hist.

    Parameters
    ----------
    data : Any
        Input data to process.
    bins : Any
        Value for ``bins``.
    ax : Any, default: None
        Matplotlib axes on which to draw.
    xlims : Any, default: None
        Value for ``xlims``.
    color : Any, default: None
        Value for ``color``.
    linestyle : Any, default: None
        Value for ``linestyle``.
    label : Any, default: None
        Value for ``label``.
    alpha : Any, default: None
        Value for ``alpha``.
    normalize : Any, default: True
        Value for ``normalize``.

    Returns
    -------
    Any
        Result of the operation.
    """
    if color is None:
        color = next(cycle(default_colors))
    hist_data, bin_edges = np.histogram(data, bins=bins, range=xlims)
    if normalize:
        s = hist_data.sum()
        if s > 0:
            hist_data = hist_data / s

    for i in range(len(hist_data)):
        ax.plot([bin_edges[i], bin_edges[i + 1]], [hist_data[i], hist_data[i]],
                color=color, linestyle=linestyle,
                label=label if i == 0 else None,
                alpha=alpha, linewidth=0.9)
        if i < len(hist_data) - 1:
            ax.plot([bin_edges[i + 1], bin_edges[i + 1]],
                    [hist_data[i], hist_data[i + 1]],
                    color=color, linestyle=linestyle, alpha=alpha, linewidth=0.9)
    ax.relim()
    ax.set_ylim((0, None))
    return hist_data, bin_edges

def _local_density_alpha(x, y, bins=120, alpha_min=0.025,
                         alpha_max=0.95, gamma=0.7):
    """Map each point's local 2D-bin density to a display alpha."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must contain the same number of points")
    if x.size == 0:
        return np.empty(0, dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    alphas = np.full(x.size, alpha_min, dtype=float)
    if not np.any(finite):
        return alphas

    xf, yf = x[finite], y[finite]
    histogram, x_edges, y_edges = np.histogram2d(xf, yf, bins=bins)
    x_idx = np.clip(np.searchsorted(x_edges, xf, side="right") - 1,
                    0, histogram.shape[0] - 1)
    y_idx = np.clip(np.searchsorted(y_edges, yf, side="right") - 1,
                    0, histogram.shape[1] - 1)
    local_counts = histogram[x_idx, y_idx]
    peak = float(np.max(local_counts, initial=0.0))
    if peak > 1.0:
        log_floor = np.log(2.0)  # a one-shot bin maps exactly to alpha_min
        density = (np.log1p(local_counts) - log_floor) / (
            np.log1p(peak) - log_floor
        )
        density = np.clip(density, 0.0, 1.0)
        alphas[finite] = alpha_min + (alpha_max - alpha_min) * density**gamma
    else:
        alphas[finite] = alpha_min
    return alphas


def _density_scatter_interleaved(ax, datasets, bins=120, seed=0):
    """Draw states in one shuffled collection with density-dependent alpha."""
    points, facecolors = [], []
    for x, y, color, label in datasets:
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        alpha = _local_density_alpha(x, y, bins=bins)
        rgba = np.tile(to_rgba(color), (x.size, 1))
        rgba[:, 3] = alpha
        points.append(np.column_stack((x, y)))
        facecolors.append(rgba)
        ax.scatter([], [], color=color, marker=".", label=label, alpha=0.8)

    if not points:
        return None
    points = np.concatenate(points)
    facecolors = np.concatenate(facecolors)
    order = np.random.default_rng(seed).permutation(len(points))
    return ax.scatter(
        points[order, 0], points[order, 1], c=facecolors[order],
        marker=".", s=9, edgecolors="none", linewidths=0, rasterized=True,
    )


def _bic_gmm(X, max_components=2, n_init=5):
    """Return the bic gmm result.

    Parameters
    ----------
    X : Any
        Value for ``X``.
    max_components : Any, default: 2
        Value for ``max_components``.
    n_init : Any, default: 5
        Value for ``n_init``.

    Returns
    -------
    Any
        Result of the operation.
    """
    best_gmm, best_bic = None, np.inf
    for k in range(1, max_components + 1):
        try:
            g = GaussianMixture(
                n_components=k, covariance_type="full",
                n_init=n_init, random_state=0,
            )
            g.fit(X)
            b = g.bic(X)
            if b < best_bic:
                best_bic, best_gmm = b, g
        except Exception:
            pass
    return best_gmm


def _ordered_threshold_classifier(projections):
    """Classify ordered one-dimensional states with contiguous thresholds."""
    values_by_state = [np.asarray(values, dtype=float).ravel() for values in projections]
    n_states = len(values_by_state)
    state_order = np.argsort([np.mean(values) for values in values_by_state])
    ordered = [values_by_state[index] for index in state_order]

    if n_states == 2:
        a, b = ordered
        values = np.unique(np.concatenate((a, b)))
        candidates = np.concatenate((
            [np.nextafter(values[0], -np.inf)],
            (values[:-1] + values[1:]) / 2.0,
            [np.nextafter(values[-1], np.inf)],
        )) if values.size > 1 else values
        cdf_a = np.searchsorted(np.sort(a), candidates, side="right") / a.size
        cdf_b = np.searchsorted(np.sort(b), candidates, side="right") / b.size
        scores = cdf_a - cdf_b
        best = np.flatnonzero(scores == scores.max())
        midpoint = 0.5 * (np.mean(a) + np.mean(b))
        index = best[np.argmin(np.abs(candidates[best] - midpoint))]
        thresholds = [float(candidates[index])]
    elif n_states == 3:
        values = np.unique(np.concatenate(ordered))
        candidates = np.concatenate((
            [np.nextafter(values[0], -np.inf)],
            (values[:-1] + values[1:]) / 2.0,
            [np.nextafter(values[-1], np.inf)],
        )) if values.size > 1 else values
        cdfs = [
            np.searchsorted(np.sort(values), candidates, side="right") / values.size
            for values in ordered
        ]
        left_gain = cdfs[0] - cdfs[1]
        right_gain = cdfs[1] - cdfs[2]
        prefix_best = np.maximum.accumulate(left_gain)
        right_index = int(np.argmax(prefix_best[:-1] + right_gain[1:]) + 1)
        left_index = int(np.argmax(left_gain[:right_index]))
        thresholds = [float(candidates[left_index]), float(candidates[right_index])]
    else:
        means = [float(np.mean(values)) for values in ordered]
        thresholds = [0.5 * (a + b) for a, b in zip(means[:-1], means[1:])]

    ordered_to_state = np.asarray(state_order)
    confusion = np.zeros((n_states, n_states))
    for prepared_state, values in enumerate(values_by_state):
        predictions = ordered_to_state[np.digitize(values, thresholds)]
        for declared_state in range(n_states):
            confusion[prepared_state, declared_state] = np.mean(
                predictions == declared_state
            )
    return state_order, thresholds, confusion


def _optimize_multistate_rotation(iqshots, max_samples_per_state=3000):
    """Jointly optimize the projection angle and ordered state thresholds."""
    rng = np.random.default_rng(0)
    sampled = []
    for I, Q in iqshots:
        cloud = np.asarray(I) + 1j * np.asarray(Q)
        if cloud.size > max_samples_per_state:
            cloud = cloud[rng.choice(cloud.size, max_samples_per_state, replace=False)]
        sampled.append(cloud)

    def score(angle):
        projections = [
            cloud.real * np.cos(angle) - cloud.imag * np.sin(angle)
            for cloud in sampled
        ]
        _, _, confusion = _ordered_threshold_classifier(projections)
        return float(np.mean(np.diag(confusion)))

    coarse = np.linspace(-np.pi / 2, np.pi / 2, 37, endpoint=False)
    coarse_scores = np.array([score(angle) for angle in coarse])
    best = float(coarse[int(np.argmax(coarse_scores))])
    fine = best + np.linspace(-np.deg2rad(5), np.deg2rad(5), 21)
    fine_scores = np.array([score(angle) for angle in fine])
    return float(fine[int(np.argmax(fine_scores))])

def _fit_gmm(I_projs, xlims, n_init=5, max_components=2):
    """Fit gmm.

    Parameters
    ----------
    I_projs : Any
        Value for ``I_projs``.
    xlims : Any
        Value for ``xlims``.
    n_init : Any, default: 5
        Value for ``n_init``.
    max_components : Any, default: 2
        Value for ``max_components``.

    Returns
    -------
    Any
        Result of the operation.
    """
    n_states = len(I_projs)
    state_gmms = []
    for proj in I_projs:
        # Components model the shape of one prepared-state density only. They
        # must never be interpreted as additional qubit populations.
        gmm = _bic_gmm(proj.reshape(-1, 1), max_components, n_init)
        state_gmms.append(gmm)

    primary_means   = np.zeros(n_states)
    primary_stds    = np.zeros(n_states)
    primary_weights = np.zeros(n_states)
    for i, gmm in enumerate(state_gmms):
        idx = int(np.argmax(gmm.weights_))
        primary_means[i]   = float(gmm.means_[idx, 0])
        primary_stds[i]    = float(np.sqrt(gmm.covariances_[idx, 0, 0]))
        primary_weights[i] = float(gmm.weights_[idx])


    state_order, thresholds, conf_matrix = _ordered_threshold_classifier(I_projs)


    return (state_gmms, state_order, conf_matrix, thresholds,
            primary_means, primary_stds, primary_weights)


def general_hist(iqshots, state_labels, g_states, e_states, e_label="e",
                 check_qubit_label=None, numbins=200, amplitude_mode=False,
                 ps_threshold=None, theta=None, plot=True, verbose=True,
                 fid_avg=False, normalize=True, title=None, export=False,
                 return_details=False):
    """Return the general hist result.

    Parameters
    ----------
    iqshots : Any
        Value for ``iqshots``.
    state_labels : Any
        Value for ``state_labels``.
    g_states : Any
        Value for ``g_states``.
    e_states : Any
        Value for ``e_states``.
    e_label : Any, default: 'e'
        Value for ``e_label``.
    check_qubit_label : Any, default: None
        Value for ``check_qubit_label``.
    numbins : Any, default: 200
        Value for ``numbins``.
    amplitude_mode : Any, default: False
        Value for ``amplitude_mode``.
    ps_threshold : Any, default: None
        Value for ``ps_threshold``.
    theta : Any, default: None
        Value for ``theta``.
    plot : Any, default: True
        Value for ``plot``.
    verbose : Any, default: True
        Value for ``verbose``.
    fid_avg : Any, default: False
        Value for ``fid_avg``.
    normalize : Any, default: True
        Value for ``normalize``.
    title : Any, default: None
        Value for ``title``.
    export : Any, default: False
        Value for ``export``.
    return_details : Any, default: False
        Value for ``return_details``.

    Returns
    -------
    Any
        Result of the operation.

    Raises
    ------
    ImportError
        If the operation cannot be completed.
    """
    if not _HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for GMM fitting. "
            "Install with:  pip install scikit-learn"
        )

    if numbins is None:
        numbins = 200

    n_states = len(iqshots)

    if not amplitude_mode:
        if theta is None:
            if n_states >= 3:
                theta_rad = _optimize_multistate_rotation(iqshots)
            else:
                g_c = np.concatenate([
                    iqshots[i][0] + 1j * iqshots[i][1] for i in g_states
                ])
                e_c = np.concatenate([
                    iqshots[i][0] + 1j * iqshots[i][1] for i in e_states
                ])
                theta_rad = -np.arctan2(
                    np.mean(e_c.imag) - np.mean(g_c.imag),
                    np.mean(e_c.real) - np.mean(g_c.real),
                )
        else:
            theta_rad = float(theta) * np.pi / 180.0

        def _rot_I(c):
            """Return the rot I result.

            Parameters
            ----------
            c : Any
                Value for ``c``.

            Returns
            -------
            Any
                Result of the operation.
            """
            return c.real * np.cos(theta_rad) - c.imag * np.sin(theta_rad)

        def _rot_IQ(c):
            """Return the rot IQ result.

            Parameters
            ----------
            c : Any
                Value for ``c``.

            Returns
            -------
            Any
                Result of the operation.
            """
            I = c.real * np.cos(theta_rad) - c.imag * np.sin(theta_rad)
            Q = c.real * np.sin(theta_rad) + c.imag * np.cos(theta_rad)
            return I, Q
    else:
        theta_rad = 0.0
        _rot_I  = lambda c: np.abs(c)
        _rot_IQ = lambda c: (c.real, c.imag)

    all_c    = np.concatenate([I + 1j * Q for I, Q in iqshots])
    proj_all = _rot_I(all_c)
    span     = (proj_all.max() - proj_all.min()) / 2
    mid      = (proj_all.max() + proj_all.min()) / 2
    xlims    = [mid - span, mid + span]

    if plot:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))
        _title = title or (
            "Readout Fidelity"
            + (f" on Q{check_qubit_label}" if check_qubit_label is not None else "")
        )
        fig.suptitle(_title)
        axs[0, 0].set_title("Unrotated", fontsize=13)
        axs[0, 0].set_xlabel("I [ADC levels]", fontsize=11)
        axs[0, 0].set_ylabel("Q [ADC levels]", fontsize=11)
        axs[0, 0].axis("equal")
        axs[0, 1].set_title(
            f"Rotated ($\\theta = {theta_rad * 180 / np.pi:.1f}^\\circ$)", fontsize=13)
        axs[0, 1].set_xlabel("I [ADC levels]", fontsize=11)
        axs[0, 1].axis("equal")
        x_axis_lbl = "Amplitude" if amplitude_mode else "I"
        axs[1, 0].set_xlabel(f"{x_axis_lbl} [ADC levels]", fontsize=11)
        axs[1, 0].set_ylabel("Counts", fontsize=12)
        plt.subplots_adjust(hspace=0.35, wspace=0.15)

    I_projs = []
    bins_dist = None
    unrotated_scatter = []
    rotated_scatter = []
    scatter_centers = []

    for idx, (I, Q) in enumerate(iqshots):
        cmplx        = I + 1j * Q
        I_new, Q_new = _rot_IQ(cmplx)
        proj         = _rot_I(cmplx)
        I_projs.append(proj)
        color     = default_colors[idx % len(default_colors)]
        marker    = marker_cycle[idx % len(marker_cycle)]
        lbl       = state_labels[idx]

        if plot:
            unrotated_scatter.append((I, Q, color, lbl))
            rotated_scatter.append((I_new, Q_new, color, lbl))
            scatter_centers.append((
                np.mean(I), np.mean(Q), np.mean(I_new), np.mean(Q_new),
                color, marker,
            ))
            _, bins_dist = plot_hist(
                proj, bins=numbins, ax=axs[1, 0], xlims=xlims,
                color=color, linestyle=linestyle_cycle[0],
                label=lbl, alpha=0.6, normalize=False,
            )
        else:
            _, bins_dist = np.histogram(proj, bins=numbins, range=xlims)

    if plot:
        _density_scatter_interleaved(axs[0, 0], unrotated_scatter)
        _density_scatter_interleaved(axs[0, 1], rotated_scatter)
        # State centers stay crisp and visible above the shuffled shot cloud.
        for i_mean, q_mean, i_rot_mean, q_rot_mean, color, marker in scatter_centers:
            axs[0, 0].plot(i_mean, q_mean, color="k", marker=marker,
                           markerfacecolor=color, markersize=6, zorder=4)
            axs[0, 1].plot(i_rot_mean, q_rot_mean, color="k", marker=marker,
                           markerfacecolor=color, markersize=6, zorder=4)
    state_gmms, state_order, conf_matrix, thresholds, gmm_means, gmm_stds, gmm_weights = \
        _fit_gmm(I_projs, xlims)

    fid  = float(np.mean(np.diag(conf_matrix)))
    fids = [fid]
    conf_matrix_pct = conf_matrix * 100.0

    if plot:
        x_plot = np.linspace(xlims[0], xlims[1], 500)
        bin_width = bins_dist[1] - bins_dist[0]

        for idx, projection in enumerate(I_projs):
            core, retained = _dominant_core(projection)
            mu = float(np.mean(core))
            sigma = max(float(np.std(core)), 1e-12)
            scale_s = len(projection) * bin_width
            c = default_colors[idx % len(default_colors)]
            core_pdf = _norm.pdf(x_plot, mu, sigma) * retained
            axs[1, 0].plot(
                x_plot,
                core_pdf * scale_s,
                color=c,
                linewidth=1.8,
                label=f"Core {state_labels[idx]} ({100 * retained:.0f}%)",
            )
        for th in thresholds:
            axs[1, 0].axvline(th, color="k", linestyle="--", linewidth=1.2, label="Threshold")

        if ps_threshold is not None:
            axs[1, 0].axvline(ps_threshold, color="gray", linestyle="-.")

        if n_states == 3:
            fid_title = "$F_{\\overline{gef}}$"
        else:
            fid_title = "$F_{\\overline{ge}}$" if fid_avg else "$F_{ge}$"
        axs[1, 0].set_title(f"{fid_title} (threshold): {100 * fid:.2f}%", fontsize=13)
        axs[1, 0].legend(fontsize=8, loc="upper right")
        axs[0, 0].legend(fontsize=8)
        axs[0, 1].legend(fontsize=8)

        cm_labels = [f"|{lbl}⟩" for lbl in state_labels]
        ax_cm = axs[1, 1]
        ax_cm.clear()
        im = ax_cm.imshow(conf_matrix_pct, cmap="Reds", vmin=0, vmax=100)
        ax_cm.set_xticks(np.arange(n_states))
        ax_cm.set_yticks(np.arange(n_states))
        ax_cm.set_xticklabels(cm_labels)
        ax_cm.set_yticklabels(cm_labels)
        ax_cm.set_xlabel("Declared output", fontsize=11)
        ax_cm.set_ylabel("Input state", fontsize=11)
        ax_cm.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
        for i in range(n_states):
            for j in range(n_states):
                val = conf_matrix_pct[i, j]
                ax_cm.text(j, i, f"{val:.1f}%", ha="center", va="center",
                           color="white" if val > 50 else "black", fontsize=12)
        ax_cm.set_title("Confusion Matrix (%)")

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if export:
            plt.savefig("multihist.jpg", dpi=1000)
            print("Exported multihist.jpg")
            plt.close()
        else:
            plt.show()

    if verbose:
        print(f"Rotation angle : {theta_rad * 180 / np.pi:.2f} deg")
        print(f"Threshold Fid. : {100 * fid:.3f}%")
        for lbl, projection in zip(state_labels, I_projs):
            core, retained = _dominant_core(projection)
            print(
                f"  {lbl}: core={100 * retained:.1f}%  "
                f"mean={np.mean(core):.3f}  std={np.std(core):.3f}"
            )
        print(f"Thresholds     : {[f'{t:.3f}' for t in thresholds]}")
        print("Confusion Matrix (%):\n", np.round(conf_matrix_pct, 1))

    rotation_deg = theta_rad * 180 / np.pi
    legacy_result = [fids, thresholds, rotation_deg, conf_matrix_pct]
    details = HistogramAnalysis(
        legacy_result=legacy_result,
        fidelity=fid,
        thresholds=np.asarray(thresholds, dtype=float),
        rotation_deg=float(rotation_deg),
        confusion_matrix_pct=np.asarray(conf_matrix_pct, dtype=float),
        state_gmms=tuple(state_gmms),
        projections=tuple(np.asarray(values) for values in I_projs),
        primary_weights=np.asarray(gmm_weights, dtype=float),
    )
    return details if return_details else legacy_result


def hist(data, amplitude_mode=False, ps_threshold=None, theta=None,
         plot=True, verbose=True, fid_avg=False,
         normalize=True, title=None, export=False, return_details=False):
    """Return the hist result.

    Parameters
    ----------
    data : Any
        Input data to process.
    amplitude_mode : Any, default: False
        Value for ``amplitude_mode``.
    ps_threshold : Any, default: None
        Value for ``ps_threshold``.
    theta : Any, default: None
        Value for ``theta``.
    plot : Any, default: True
        Value for ``plot``.
    verbose : Any, default: True
        Value for ``verbose``.
    fid_avg : Any, default: False
        Value for ``fid_avg``.
    normalize : Any, default: True
        Value for ``normalize``.
    title : Any, default: None
        Value for ``title``.
    export : Any, default: False
        Value for ``export``.
    return_details : Any, default: False
        Value for ``return_details``.

    Returns
    -------
    Any
        Result of the operation.
    """
    iqshots      = [(data["Ig"], data["Qg"]), (data["Ie"], data["Qe"])]
    state_labels = ["g", "e"]
    g_states     = [0]
    e_states     = [1]

    if "If" in data:
        iqshots.append((data["If"], data["Qf"]))
        state_labels.append("f")
        e_states = [2]

    return general_hist(
        iqshots=iqshots, state_labels=state_labels,
        g_states=g_states, e_states=e_states,
        amplitude_mode=amplitude_mode, ps_threshold=ps_threshold,
        theta=theta, plot=plot, verbose=verbose,
        fid_avg=fid_avg, normalize=normalize,
        title=title, export=export, return_details=return_details,
    )


__all__ = [
    "HistogramAnalysis", "histogram_metrics", "plot_hist", "general_hist",
    "hist", "fast_histogram_metrics", "_fit_gmm", "_bic_gmm",
]
