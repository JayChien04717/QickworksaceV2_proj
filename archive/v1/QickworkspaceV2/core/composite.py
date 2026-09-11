"""Sequential and concurrent helpers for groups of experiments."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

from .experiment_data import ExperimentData, QualityFlag


def run_batch(experiments, py_avg: int, *, stop_on_bad=False, **kwargs):
    """Run experiments sequentially and return results keyed by name."""
    from ..tools.hdf5_store import generate_experiment_id

    batch_id = generate_experiment_id()
    results = {}
    items = [
        item
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
        else (None, item)
        for item in experiments
    ]
    for index, (label, experiment) in enumerate(items):
        name = label or experiment.EXPT_NAME or experiment.__class__.__name__
        print(f"\n{'=' * 60}")
        print(f"  Batch [{index + 1}/{len(items)}] — {name}")
        print(f"{'=' * 60}")
        result = experiment.run(py_avg=py_avg, **kwargs)
        result.parent_id = batch_id
        result.session_id = batch_id
        results[name] = result
        if stop_on_bad and result.quality == QualityFlag.BAD:
            print(f"  Stopping: {name} returned BAD ({result.quality_message})")
            break
        sleep(0.5)
    return results


def summarize_results(results) -> str:
    """Return and print a compact quality summary."""
    lines = []
    for name, result in results.items():
        message = f" — {result.quality_message}" if result.quality_message else ""
        lines.append(f"{name:<40s} [{result.quality.value.upper()}]{message}")
    summary = "\n".join(lines)
    print(summary)
    return summary


def run_parallel(experiments, py_avg: int, *, max_workers=None, **kwargs):
    """Run independent experiments concurrently.

    Only use this with genuinely independent hardware sessions. A shared QICK
    proxy is not serialized by this helper.
    """
    from ..tools.hdf5_store import generate_experiment_id

    session_id = generate_experiment_id()
    items = [
        item
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
        else (None, item)
        for item in experiments
    ]
    results = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers or len(items)) as pool:
        for label, experiment in items:
            name = label or experiment.EXPT_NAME or experiment.__class__.__name__
            futures[pool.submit(experiment.run, py_avg, **kwargs)] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                result.parent_id = session_id
                result.session_id = session_id
            except Exception as exc:
                result = ExperimentData(
                    experiment_type=name,
                    quality=QualityFlag.BAD,
                    quality_message=str(exc),
                    parent_id=session_id,
                    session_id=session_id,
                )
            results[name] = result
    return results


__all__ = ["run_batch", "run_parallel", "summarize_results"]
