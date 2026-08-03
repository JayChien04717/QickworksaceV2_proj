"""Generate deterministic-looking native HDF5 examples without QICK hardware."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from QickworkspaceV2.core.experiment_data import ExperimentData, QualityFlag


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = HERE / "data"


def _save(result, root, comment, tags):
    return result.save(data_root=root, comment=comment, tags=tags)


def generate(
    data_root=DEFAULT_DATA_ROOT,
    *,
    clean=False,
    seed=20260714,
    days=7,
    repeats=3,
):
    if days < 1 or repeats < 1:
        raise ValueError("days and repeats must both be positive")
    root = Path(data_root).expanduser().resolve()
    if clean and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    base = datetime(2026, 7, 14, 7, 30, tzinfo=timezone.utc)
    paths = []

    delay = np.linspace(0, 100, 101)
    t1_curve = 0.15 + .9 * np.exp(-delay / 31.5)
    t1_iq = t1_curve + .02 * rng.normal(size=delay.size) + 1j * (.08 + .01 * rng.normal(size=delay.size))
    paths.append(_save(ExperimentData(
        experiment_type="s008_T1_ge", timestamp=base, raw_iq=t1_iq,
        axes={"delay_us": {"values": delay, "unit": "us", "label": "Delay"}},
        dataset_dims={"iq": ["delay_us"]},
        analysis_data={"fit_curve": {"values": t1_curve, "dims": ["delay_us"]},
                       "residual": {"values": np.abs(t1_iq) - t1_curve, "dims": ["delay_us"]}},
        fit_result={"T1_us": (31.5, .8)}, quality=QualityFlag.GOOD,
        metadata={"qubit": "Q1"}, plot_id="iq_fit_1d",
    ), root, "冷卻完成後的 T1 baseline", ["demo", "T1", "baseline"]))

    gain = np.linspace(0, 1, 81)
    rabi_curve = .5 + .42 * np.cos(2 * np.pi * gain / .42) * np.exp(-gain / 1.8)
    rabi_iq = rabi_curve + .018 * rng.normal(size=gain.size) + 1j * (.04 * rng.normal(size=gain.size))
    paths.append(_save(ExperimentData(
        experiment_type="s004_power_rabi_ge", timestamp=base + timedelta(minutes=3), raw_iq=rabi_iq,
        axes={"gain": {"values": gain, "unit": "FS", "label": "Normalized gain"}},
        dataset_dims={"iq": ["gain"]},
        analysis_data={"fit_curve": {"values": rabi_curve, "dims": ["gain"]}},
        fit_result={"pi_gain": (.21, .004)}, quality=QualityFlag.GOOD,
        metadata={"qubit": "Q1"}, plot_id="iq_fit_1d",
    ), root, "Rabi pulse gain scan", ["demo", "rabi", "final"]))

    states = np.asarray(["g", "e"])
    shots = np.arange(1500)
    ss_iq = np.vstack([
        rng.normal(-.8, .22, shots.size) + 1j * rng.normal(.05, .19, shots.size),
        rng.normal(.85, .25, shots.size) + 1j * rng.normal(-.04, .21, shots.size),
    ])
    paths.append(_save(ExperimentData(
        experiment_type="s010_single_shot_ge", timestamp=base + timedelta(minutes=6), raw_iq=ss_iq,
        axes={"state": {"values": states}, "shot": {"values": shots}},
        dataset_dims={"iq": ["state", "shot"]},
        analysis_data={"confusion_matrix": np.asarray([[.965, .035], [.042, .958]])},
        fit_result={"fidelity": .9615, "threshold": .01}, quality=QualityFlag.GOOD,
        metadata={"qubit": "Q1"}, plot_id="single_shot_iq",
    ), root, "Single-shot readout optimization", ["demo", "single-shot", "final"]))

    depth = np.asarray([1, 2, 4, 8, 16, 32, 64, 96])
    rb_curve = .49 * (.985 ** depth) + .5
    rb_iq = rb_curve + .008 * rng.normal(size=depth.size) + 1j * .008 * rng.normal(size=depth.size)
    paths.append(_save(ExperimentData(
        experiment_type="s015_RB_ge", timestamp=base + timedelta(minutes=9), raw_iq=rb_iq,
        axes={"depth": {"values": depth, "label": "Clifford depth"}},
        dataset_dims={"iq": ["depth"]},
        analysis_data={"fit_curve": {"values": rb_curve, "dims": ["depth"]}},
        fit_result={"p": (.985, .002), "EPC": (.0075, .001)}, quality=QualityFlag.GOOD,
        metadata={"qubit": "Q1"}, plot_id="rb_decay",
    ), root, "Reference randomized benchmarking", ["demo", "RB"]))

    rho = np.asarray([[.96, .04 - .02j], [.04 + .02j, .04]], dtype=complex)
    tomo_raw = rng.normal(size=(3, 400)) + 1j * rng.normal(size=(3, 400))
    paths.append(_save(ExperimentData(
        experiment_type="s016_state_tomography_ge", timestamp=base + timedelta(minutes=12), raw_iq=tomo_raw,
        axes={"basis": {"values": ["X", "Y", "Z"]}, "shot": {"values": np.arange(400)},
              "bra": {"values": ["g", "e"]}, "ket": {"values": ["g", "e"]}},
        dataset_dims={"iq": ["basis", "shot"]},
        analysis_data={"density_matrix": {"values": rho, "dims": ["bra", "ket"]}, "purity": .926},
        fit_result={"purity": .926}, quality=QualityFlag.WARNING,
        metadata={"qubit": "Q1"}, plot_id="density_matrix",
    ), root, "Tomography demo; intentionally marked warning", ["demo", "tomography"]))

    lengths = np.linspace(.1, .8, 6)
    gains = np.linspace(.1, .9, 7)
    frequencies = np.linspace(6495, 6505, 5)
    opt_shape = (len(lengths), len(gains), len(frequencies), 2, 80)
    opt_iq = rng.normal(size=opt_shape) + 1j * rng.normal(size=opt_shape)
    opt_iq[:, :, :, 1] += gains[None, :, None, None] * 1.6
    paths.append(_save(ExperimentData(
        experiment_type="ssh_readout_optimize", timestamp=base + timedelta(minutes=15), raw_iq=opt_iq,
        axes={"length": {"values": lengths, "unit": "us"}, "gain": {"values": gains},
              "frequency": {"values": frequencies, "unit": "MHz"},
              "state": {"values": states}, "shot": {"values": np.arange(80)}},
        dataset_dims={"iq": ["length", "gain", "frequency", "state", "shot"]},
        analysis_data={"fidelity": {"values": rng.uniform(.85, .99, opt_shape[:3]),
                                     "dims": ["length", "gain", "frequency"]}},
        quality=QualityFlag.GOOD, metadata={"qubit": "Q1"}, plot_id="ssh_optimization",
    ), root, "SSH readout optimization grid", ["demo", "SSH", "optimization"]))

    # Build a small measurement history: multiple dates, repeated runs, two
    # qubits, session linkage, drift, and a few intentional warning results.
    history_start = base - timedelta(days=days - 1) + timedelta(hours=2)
    for day_index in range(days):
        day_time = history_start + timedelta(days=day_index)
        date_tag = day_time.strftime("day-%Y%m%d")
        session_id = day_time.strftime("cooldown-%Y%m%d")

        for repeat in range(repeats):
            qubit = "Q2" if repeat % 3 == 2 else "Q1"
            repeated_tags = ["demo", "repeated", date_tag, qubit]
            timestamp = day_time + timedelta(minutes=repeat * 8)

            delay = np.linspace(0, 120, 121)
            t1_value = 27.5 + 0.85 * day_index + 0.45 * repeat + rng.normal(0, .35)
            t1_curve = .12 + .92 * np.exp(-delay / t1_value)
            t1_iq = (
                t1_curve + rng.normal(0, .018, delay.size)
                + 1j * rng.normal(.06, .012, delay.size)
            )
            t1_quality = (
                QualityFlag.WARNING
                if (day_index, repeat) in {(1, 1), (4, 0)}
                else QualityFlag.GOOD
            )
            paths.append(_save(ExperimentData(
                experiment_type="s008_T1_ge", timestamp=timestamp, raw_iq=t1_iq,
                axes={"delay_us": {"values": delay, "unit": "us", "label": "Delay"}},
                dataset_dims={"iq": ["delay_us"]},
                analysis_data={
                    "fit_curve": {"values": t1_curve, "dims": ["delay_us"]},
                    "residual": {"values": np.abs(t1_iq) - t1_curve, "dims": ["delay_us"]},
                },
                fit_result={"T1_us": (float(t1_value), .65 + .1 * repeat)},
                quality=t1_quality, config={"name": qubit},
                metadata={"qubit": qubit, "repeat": repeat, "day_index": day_index},
                session_id=session_id, plot_id="iq_fit_1d",
            ), root, f"{qubit} T1 重複量測 #{repeat + 1}，日期 {day_time:%Y-%m-%d}",
                [*repeated_tags, "T1"]))

            gain = np.linspace(0, 1, 81)
            period = .39 + .006 * day_index + .004 * repeat
            rabi_curve = .5 + .41 * np.cos(2 * np.pi * gain / period) * np.exp(-gain / 1.9)
            rabi_iq = (
                rabi_curve + rng.normal(0, .017, gain.size)
                + 1j * rng.normal(0, .025, gain.size)
            )
            paths.append(_save(ExperimentData(
                experiment_type="s004_power_rabi_ge", timestamp=timestamp + timedelta(minutes=3),
                raw_iq=rabi_iq,
                axes={"gain": {"values": gain, "unit": "FS", "label": "Normalized gain"}},
                dataset_dims={"iq": ["gain"]},
                analysis_data={"fit_curve": {"values": rabi_curve, "dims": ["gain"]}},
                fit_result={"pi_gain": (period / 2, .004)}, quality=QualityFlag.GOOD,
                config={"name": qubit},
                metadata={"qubit": qubit, "repeat": repeat, "day_index": day_index},
                session_id=session_id, plot_id="iq_fit_1d",
            ), root, f"{qubit} Rabi 重複量測 #{repeat + 1}，日期 {day_time:%Y-%m-%d}",
                [*repeated_tags, "rabi"]))

        for qubit_index, qubit in enumerate(("Q1", "Q2")):
            shots = np.arange(600)
            separation = 1.45 + .025 * day_index - .04 * qubit_index
            ss_iq = np.vstack([
                rng.normal(-separation / 2, .23, shots.size) + 1j * rng.normal(.04, .2, shots.size),
                rng.normal(separation / 2, .25, shots.size) + 1j * rng.normal(-.03, .21, shots.size),
            ])
            fidelity = min(.99, .91 + separation * .035)
            paths.append(_save(ExperimentData(
                experiment_type="s010_single_shot_ge",
                timestamp=day_time + timedelta(minutes=30 + 4 * qubit_index), raw_iq=ss_iq,
                axes={"state": {"values": states}, "shot": {"values": shots}},
                dataset_dims={"iq": ["state", "shot"]},
                analysis_data={"confusion_matrix": np.asarray([
                    [fidelity, 1 - fidelity], [1 - fidelity, fidelity]
                ])},
                fit_result={"fidelity": fidelity, "threshold": 0.0},
                quality=QualityFlag.GOOD, config={"name": qubit},
                metadata={"qubit": qubit, "day_index": day_index},
                session_id=session_id, plot_id="single_shot_iq",
            ), root, f"{qubit} 每日 single-shot check，日期 {day_time:%Y-%m-%d}",
                ["demo", "repeated", "single-shot", date_tag, qubit]))

        depth = np.asarray([1, 2, 4, 8, 16, 32, 64, 96])
        p_value = .977 + .0012 * day_index
        rb_curve = .49 * p_value ** depth + .5
        rb_iq = rb_curve + rng.normal(0, .007, depth.size) + 1j * rng.normal(0, .007, depth.size)
        paths.append(_save(ExperimentData(
            experiment_type="s015_RB_ge", timestamp=day_time + timedelta(minutes=40), raw_iq=rb_iq,
            axes={"depth": {"values": depth, "label": "Clifford depth"}},
            dataset_dims={"iq": ["depth"]},
            analysis_data={"fit_curve": {"values": rb_curve, "dims": ["depth"]}},
            fit_result={"p": (p_value, .002), "EPC": ((1 - p_value) / 2, .001)},
            quality=QualityFlag.GOOD, config={"name": "Q1"},
            metadata={"qubit": "Q1", "day_index": day_index},
            session_id=session_id, plot_id="rb_decay",
        ), root, f"Q1 daily RB，日期 {day_time:%Y-%m-%d}",
            ["demo", "repeated", "RB", date_tag, "Q1"]))

        if day_index % 2 == 0:
            excited = .025 + .004 * day_index
            rho = np.asarray([
                [1 - excited, .025 - .012j], [.025 + .012j, excited]
            ], dtype=complex)
            tomo_raw = rng.normal(size=(3, 200)) + 1j * rng.normal(size=(3, 200))
            paths.append(_save(ExperimentData(
                experiment_type="s016_state_tomography_ge",
                timestamp=day_time + timedelta(minutes=45), raw_iq=tomo_raw,
                axes={"basis": {"values": ["X", "Y", "Z"]},
                      "shot": {"values": np.arange(200)},
                      "bra": {"values": ["g", "e"]}, "ket": {"values": ["g", "e"]}},
                dataset_dims={"iq": ["basis", "shot"]},
                analysis_data={"density_matrix": {"values": rho, "dims": ["bra", "ket"]}},
                fit_result={"purity": float(np.real(np.trace(rho @ rho)))},
                quality=QualityFlag.GOOD, config={"name": "Q1"},
                metadata={"qubit": "Q1", "day_index": day_index},
                session_id=session_id, plot_id="density_matrix",
            ), root, f"Q1 隔日 tomography，日期 {day_time:%Y-%m-%d}",
                ["demo", "repeated", "tomography", date_tag, "Q1"]))

        if day_index in {0, days // 2, days - 1}:
            hist_lengths = np.linspace(.15, .75, 4)
            hist_gains = np.linspace(.2, .8, 5)
            hist_freqs = np.linspace(6497, 6503, 3)
            hist_shape = (4, 5, 3, 2, 40)
            hist_iq = rng.normal(size=hist_shape) + 1j * rng.normal(size=hist_shape)
            hist_iq[:, :, :, 1] += hist_gains[None, :, None, None] * 1.5
            paths.append(_save(ExperimentData(
                experiment_type="ssh_readout_optimize",
                timestamp=day_time + timedelta(minutes=50), raw_iq=hist_iq,
                axes={"length": {"values": hist_lengths, "unit": "us"},
                      "gain": {"values": hist_gains},
                      "frequency": {"values": hist_freqs, "unit": "MHz"},
                      "state": {"values": states}, "shot": {"values": np.arange(40)}},
                dataset_dims={"iq": ["length", "gain", "frequency", "state", "shot"]},
                analysis_data={"fidelity": {
                    "values": rng.uniform(.86, .985, hist_shape[:3]),
                    "dims": ["length", "gain", "frequency"],
                }},
                quality=QualityFlag.GOOD, config={"name": "Q1"},
                metadata={"qubit": "Q1", "day_index": day_index},
                session_id=session_id, plot_id="ssh_optimization",
            ), root, f"Q1 SSH optimization，日期 {day_time:%Y-%m-%d}",
                ["demo", "repeated", "SSH", "optimization", date_tag, "Q1"]))
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    generated = generate(
        args.data_root,
        clean=args.clean,
        days=args.days,
        repeats=args.repeats,
    )
    print(f"Generated {len(generated)} experiments in {args.data_root.resolve()}")
    for path in generated:
        print(path)
