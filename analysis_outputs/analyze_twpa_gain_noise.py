"""Compare TWPA-on/off single-shot histograms using main Gaussian peaks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


FILES = {
    "TWPA off (no pump)": Path(
        r"D:\Labber_Data\Jay\purcell_tmon\withTWPA\2026\08\Data_0807"
        r"\s000_singleshot_ge_Q1_nopump_20260807T015709765200Z-8CPM1YPC15DT9.hdf5"
    ),
    "TWPA on": Path(
        r"D:\Labber_Data\Jay\purcell_tmon\withTWPA\2026\08\Data_0807"
        r"\s000_singleshot_ge_Q1_20260807T012628261453Z-CRWX6JW1D2SF6.hdf5"
    ),
}
OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_outputs"


def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def load_and_rotate(path: Path):
    with h5py.File(path, "r") as handle:
        iq = np.asarray(handle["metagroup/raw/iq"][()])
    if iq.shape[0] != 2:
        raise ValueError(f"Expected g/e state axis of length 2, got {iq.shape}")

    # Median IQ centers make the axis robust against thermal/T1/outlier tails.
    robust_centers = np.median(iq.real, axis=1) + 1j * np.median(iq.imag, axis=1)
    rotation = np.angle(robust_centers[1] - robust_centers[0])
    projected = (iq * np.exp(-1j * rotation)).real
    return iq, projected, float(np.degrees(rotation))


def fit_main_gaussian(values, bin_edges):
    counts, _ = np.histogram(values, bins=bin_edges, density=True)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    median = float(np.median(values))
    sigma0 = 1.4826 * float(np.median(np.abs(values - median)))
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = float(np.std(values))

    # Least-squares fitting to histogram height emphasizes the main peak over
    # sparse non-Gaussian tails.  The two fitted state peaks form the requested
    # double-Gaussian model of the g/e histogram.
    fit_mask = (centers >= median - 4 * sigma0) & (centers <= median + 4 * sigma0)
    params, covariance = curve_fit(
        gaussian,
        centers[fit_mask],
        counts[fit_mask],
        p0=(float(counts.max()), median, sigma0),
        bounds=(
            (0.0, median - 2 * sigma0, 0.05 * sigma0),
            (np.inf, median + 2 * sigma0, 5.0 * sigma0),
        ),
        maxfev=30000,
    )
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return centers, counts, params, errors


def analyze(label: str, path: Path):
    iq, projected, rotation_deg = load_and_rotate(path)
    low, high = np.quantile(projected, [0.002, 0.998])
    edges = np.linspace(float(low), float(high), 181)
    fits = [fit_main_gaussian(projected[index], edges) for index in range(2)]
    centers = np.array([fit[2][1] for fit in fits])
    sigmas = np.array([abs(fit[2][2]) for fit in fits])
    distance = float(abs(centers[1] - centers[0]))
    pooled_sigma = float(np.sqrt(np.mean(sigmas**2)))
    amplitude_snr = distance / pooled_sigma
    return {
        "label": label,
        "path": str(path),
        "iq": iq,
        "projected": projected,
        "rotation_deg": rotation_deg,
        "bin_edges": edges,
        "fits": fits,
        "g_center": float(centers[0]),
        "e_center": float(centers[1]),
        "g_sigma": float(sigmas[0]),
        "e_sigma": float(sigmas[1]),
        "distance": distance,
        "pooled_sigma": pooled_sigma,
        "amplitude_snr": float(amplitude_snr),
    }


def ratio_db(ratio):
    return float(20.0 * np.log10(ratio))


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = {label: analyze(label, path) for label, path in FILES.items()}
    off = results["TWPA off (no pump)"]
    on = results["TWPA on"]

    amplitude_gain = on["distance"] / off["distance"]
    g_noise_gain = on["g_sigma"] / off["g_sigma"]
    e_noise_gain = on["e_sigma"] / off["e_sigma"]
    noise_gain = on["pooled_sigma"] / off["pooled_sigma"]
    effective_gain = amplitude_gain / noise_gain
    comparison = {
        "amplitude_gain": float(amplitude_gain),
        "power_gain": float(amplitude_gain**2),
        "gain_db": ratio_db(amplitude_gain),
        "g_noise_amplitude_gain": float(g_noise_gain),
        "e_noise_amplitude_gain": float(e_noise_gain),
        "pooled_noise_amplitude_gain": float(noise_gain),
        "pooled_noise_power_gain": float(noise_gain**2),
        "noise_gain_db": ratio_db(noise_gain),
        "effective_amplitude_gain": float(effective_gain),
        "effective_power_snr_gain": float(effective_gain**2),
        "effective_gain_db": ratio_db(effective_gain),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    state_colors = ("#2774AE", "#D1495B")
    for ax, result in zip(axes, (off, on)):
        dense_x = np.linspace(result["bin_edges"][0], result["bin_edges"][-1], 1600)
        component_curves = []
        for state_index, state_name in enumerate(("g", "e")):
            centers, counts, params, errors = result["fits"][state_index]
            color = state_colors[state_index]
            ax.step(centers, counts, where="mid", color=color, alpha=0.5,
                    linewidth=1.0, label=f"{state_name} histogram")
            curve = gaussian(dense_x, *params)
            component_curves.append(curve)
            ax.plot(dense_x, curve, color=color, linewidth=2.2,
                    label=(f"{state_name} main Gaussian: "
                           f"mu={params[1]:.4g}, sigma={abs(params[2]):.4g}"))
            ax.axvline(params[1], color=color, linestyle=":", alpha=0.8)
        ax.plot(dense_x, component_curves[0] + component_curves[1], color="black",
                linestyle="--", linewidth=1.4, alpha=0.75,
                label="double-Gaussian sum")
        ax.set_title(
            f"{result['label']}\n"
            f"distance={result['distance']:.4g}, pooled sigma={result['pooled_sigma']:.4g}"
        )
        ax.set_xlabel("Rotated I projection (ADC unit)")
        ax.set_ylabel("Probability density")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=8.6)

    fig.suptitle(
        "Q1 single-shot main-peak double-Gaussian comparison\n"
        f"Gain={amplitude_gain:.3f}x ({comparison['gain_db']:.2f} dB), "
        f"noise={noise_gain:.3f}x ({comparison['noise_gain_db']:.2f} dB), "
        f"effective={effective_gain:.3f}x ({comparison['effective_gain_db']:.2f} dB)",
        fontsize=13,
    )
    figure_path = OUTPUT_DIR / "twpa_gain_noise_hist_fit_20260807.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    serializable = {
        "method": {
            "projection": "median-center g-to-e IQ axis",
            "fit": "two labeled main Gaussian histogram components (g + e)",
            "pooled_sigma": "sqrt((sigma_g^2 + sigma_e^2) / 2)",
            "amplitude_gain": "distance_on / distance_off",
            "noise_gain": "pooled_sigma_on / pooled_sigma_off",
            "effective_gain": "amplitude_gain / noise_gain",
        },
        "datasets": {
            label: {key: value for key, value in result.items()
                    if key not in {"iq", "projected", "bin_edges", "fits"}}
            for label, result in results.items()
        },
        "comparison": comparison,
    }
    json_path = OUTPUT_DIR / "twpa_gain_noise_metrics_20260807.json"
    json_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    csv_path = OUTPUT_DIR / "twpa_gain_noise_metrics_20260807.csv"
    fields = ["label", "g_center", "e_center", "g_sigma", "e_sigma", "distance",
              "pooled_sigma", "amplitude_snr", "rotation_deg", "path"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in (off, on):
            writer.writerow({field: result[field] for field in fields})

    print(json.dumps(serializable, indent=2))
    print(f"FIGURE={figure_path}")
    print(f"JSON={json_path}")
    print(f"CSV={csv_path}")


if __name__ == "__main__":
    main()
