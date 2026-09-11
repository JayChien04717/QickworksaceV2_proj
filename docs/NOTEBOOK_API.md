# Notebook and experiment ownership

`ChipCalibration.ipynb` is the daily chip-calibration entry. Cells import the SDK, edit parameters and call SDK methods. `NotebookLab` manages acquisition, per-round checkpoints, analysis, final plots and explicit working updates. Existing historical notebook outputs remain intact.

```python
from QickworkspaceV2 import NotebookLab, QickSweep1D
from QickworkspaceV2.notebook import enable_notebook
from QickworkspaceV2.experiments.t1_ge import T1GEProgram

enable_notebook()
lab = NotebookLab("lab/project.yaml", qubit="Q1", calibration_enabled=False)
lab.connect()
qb = lab.config["Q1"]
run_cfg = qb.for_run(steps=81, wait_us=QickSweep1D("waitloop", 0, 100), sigma_ge=0.05)
result = lab.run(T1GEProgram, run_cfg, py_avg=5)
```

Do not wrap this in a Notebook-owned acquisition loop or interrupt handler. LivePlot updates during acquisition. The SDK clears that temporary display when the run completes, is interrupted or fails, leaving the final analysis plot and writing its PNG beside the data. If interrupted after data arrives, it retains that run's partial data and performs analysis and plotting. Before the first returned round there is no result to recover, and the interrupt propagates. `lab.analyze(result)` analyzes saved data after formula/settings edits; `lab.reload_partial(result)` verifies a saved partial result.

## One package per experiment

```text
QickworkspaceV2/
  experiments/
    t1_ge/
      program.py      T1GEProgram and build(ctx, parameters)
      parameters.py   T1GEParameters: node inputs/defaults/validation
      analysis.py     analyze(result), plot(result), updates(result, target)
      __init__.py     public exports, ExperimentSpec, Program.EXPERIMENT
    t1_ef/            independently maintained EF protocol
  analysis/
    fitting.py        equation/fit-function pairs and a generic numerical solver
  notebook.py         NotebookLab execution and presentation
```

`ExperimentSpec` registers one program builder, schema, analysis, plot and update hook. Notebook native dictionaries and catalog jobs use the same program and analysis. `program.py` binds native loops, pulses and readout events. A direct dictionary edit does not require changing a node schema. UI/CLI/Agent node validation uses the experiment's `parameters.py` through the worker catalog. Shared target selection and run options remain common rather than being copied into every experiment.

An experiment-specific final plot lives in its `analysis.py`. Ordinary trace plots use the shared renderer. GE and EF remain separate packages. Public imports use `QickworkspaceV2.experiments.t1_ge`; no old-file forwarding wrappers are retained.

Keep `analyze`, `plot` and `updates` as separate functions in that file: `analyze` returns fits, `plot` returns a Figure, and `updates` returns proposed values. None of them opens hardware or writes calibration. `lab.run()` and `lab.analyze()` orchestrate analysis and final display; the worker can analyze and export a plot without an interactive Notebook display.

## Fitting equations and tuning

Edit `QickworkspaceV2/analysis/fitting.py`. Each formula is followed by its own fitting function, following the earlier equation/fit-function organization:

| Equation | Fitting function |
| --- | --- |
| `exponential()` | `fit_exponential()` |
| `lorentzian()` | `fit_lorentzian()` |
| `oscillation()` | `fit_rabi()` |
| `damped_oscillation()` | `fit_ramsey()` |
| `rb_decay()` | `fit_rb()` |
| `notch_response()` | `fit_resonator()` |

There are also independent fit functions for asymmetric Lorentzian, Gaussian/double Gaussian, Ramsey slope/two-frequency decay, rotation-error and Poisson models. Edit initial estimates, bounds, derived values and physical quality checks inside the relevant `fit_<model>()`. `FIT_OPTIONS` provides optional global overrides. Formula phases use radians; frequencies MHz and times us.

`fit_curve()` takes a formula callable and only performs generic numerical solving and fit diagnostics. It has no model dispatch, Rabi pi-gain conversion, T1 decay acceptance or RB fidelity logic. For example:

```python
from QickworkspaceV2.analysis.fitting import exponential, fit_exponential, fit_curve

values = exponential(delays, offset=0.2, amplitude=1.0, tau=25)
fit = fit_exponential(delays, measured, p0={"tau": 25})
# Generic custom-formula solving requires a full initial guess:
numeric_fit = fit_curve(exponential, delays, measured,
                        p0={"offset": 0.2, "amplitude": 1.0, "tau": 25})
```

Experiment analysis calls a specific function, such as `analyze_traces(result, fit_exponential)`. `FIT_FUNCTIONS` only maps a configured model name to its independent fitter. The former `fit_curve("exponential", ...)` entry is removed.

Per-run settings override model defaults:

```python
run_cfg["fit"] = {
    "p0": {"tau": 25},
    "bounds": {"tau": (0.1, 200)},
    "maxfev": 12000,
    "min_r_squared": 0.9,
}
```

Change `result.metadata["parameters"]["fit"]` only through a new measurement configuration; `lab.analyze(result)` rereads stored parameters and raw data, so editing that in-memory dictionary will not alter the saved input. For reanalysis with different settings without acquiring, pass an analyzer callable to the lower-level `Measurement.analyze`, or edit the central `FIT_OPTIONS` before calling `lab.analyze(result)`.

Named `p0` and `bounds` use formula argument names. Complex notch accepts `center`, `linewidth`, `depth`, `asymmetry_rad`, `scale`, `phase_rad`, `delay_us` in physical units. More specialized/custom formulas require a full starting-parameter mapping. Invalid names and bounds fail visibly. A failed fit preserves the acquisition and its diagnostic message.

## Host procedures and calibration

`lab.scan(Program, run_cfg, "res_freq_ge", frequencies, axis_name="frequency", unit="MHz")` automatically reads the rounded direct/MUX readout frequency. Inner QICK loops infer pulse/time axes. Each host point is saved independently and the parent partial data updates as points complete. Interruption retains completed points and links an interrupted child when one exists.

`lab.readout_grid`, `lab.ramsey_pair`, `lab.drag_scan` and `lab.reset_comparison` orchestrate their loops and display. Candidate selection, comparison plots and update calculations live in the corresponding experiment's `analysis.py`. External `lab.flux_scan` stays disabled unless explicitly enabled. Every procedure stops acquiring after an interrupted result.

Working calibration writes default to disabled. After reviewing real chip measurements, enable `lab.calibration_enabled` and explicitly call `lab.apply_fit(result)`. Incomplete, failed or rejected fits cannot update the working calibration. The working file is separate from the initial hardware profile. The former Notebook-owned fit-to-parameter dictionary is removed.

The local `updates` function declares what an experiment can change. For T1 GE:

```python
# experiments/t1_ge/analysis.py
def updates(result, target):
    fit = accepted_fit(result, target)
    return CalibrationUpdates({"t1_ge_us": fit["tau"]})
```

`CalibrationUpdates.working` contains native working-config keys and values. Optional `device_paths` maps those same keys to strict calibration-store paths; `Session.propose(result)` consumes that local mapping, followed by an explicit `Session.commit(proposal)`. Coherence times remain working metrics because the strict device schema has no coherence fields. Diagnostic-only experiments return an empty update with a review reason.

| Experiment | Working updates |
| --- | --- |
| Resonator / dispersive | `res_freq_ge` from fitted center / best measured SNR frequency |
| Qubit spectroscopy GE / EF | `qb_freq_ge` / `qb_freq_ef` |
| Power Rabi GE / EF | π / π/2 gains and the measured sigma / pulse type |
| T1 / Ramsey / echo | Corresponding transition's coherence time |
| Single shot / best readout grid | Threshold, phase, measured gain and integration / pulse lengths |
| Best DRAG scan | Measured alpha, delta and DRAG pulse shape for that transition |
| Paired GE Ramsey | Signed drive-frequency correction from both accepted acquisitions |

Use `lab.apply_best_readout(rows)`, `lab.apply_best_drag(runs)` and `lab.apply_ramsey_pair(results)` for host procedures. Each uses the experiment's local selection/update rules. Values come from the saved run's `resolved_config`, including readout end margin and DRAG delta, so later Notebook parameter edits cannot change what is applied. GE/EF and selected-target checks remain enforced. Manual timing and candidate-frequency choices stay explicit parameter edits through `lab.update_working(...)`.


### Independent broadband resonator spectrum

`broadband_resonator_spectrum` has its own program identity, local parameter schema, analysis, plot and catalog entry. It reuses only the native one-tone pulse sequence from `ResonatorSpecProgram`. The narrow `resonator_spec` experiment contains no broadband mode switch.

```python
from QickworkspaceV2.experiments.broadband_resonator_spectrum import BroadbandResonatorSpecProgram

run_cfg = qb.for_run(steps=401, res_freq_ge=QickSweep1D("freqloop", 6600, 7000))
run_cfg.update(count=4, y_mode="abs", detection_options={
    "min_distance_points": 6, "prominence_sigma": 0.08, "width_penalty": 0.5,
    "use_phase_reference": True, "phase_snap_hz": 15e6,
})
broadband_result = lab.run(BroadbandResonatorSpecProgram, run_cfg, py_avg=PY_AVG)
```

The original multi-resonator algorithm is maintained separately in `analysis/fit_n_res.py`: Savitzky-Golay smoothing, magnitude-dip prominence/width ranking, optional phase-reference refinement and local three-point quadratic interpolation. The complex IQ and all original detection options are preserved. The helper receives Hz; the SDK converts measured MHz axes to Hz and reports candidates in MHz. `count` is the requested number. Insufficient candidates or a numerically flat trace produce a rejected analysis while retaining acquisition data and its plot. Detection details, sample indices, phase diagnostics and frequencies are saved in each trace's `multi_resonator_fit` metadata. Its local plot supports abs/db/phase and the original frequency markers.

Broadband does not automatically select a readout frequency. Review candidates, then run the separate narrow `ResonatorSpecProgram` experiment before applying calibration. Catalog jobs use the independent `broadband_resonator_spectrum` entry and its host-frequency runner, which supports static MUX readouts. Catalog start/stop are MHz offsets from the configured readout frequency; native Notebook sweeps above use absolute MHz. Native register sweeps require a sweepable readout. The redundant `lab.broadband` method and temporary narrow-experiment broadband option are removed.


### Run settings and explicit channel initialization

`qb.for_run(**overrides)` deep-copies every current qubit setting and then applies the overrides. Omitted values retain the working configuration: `qb_ch`, `nqz_qb`, `qb_mixer`, `res_ch`, `nqz_res`, and readout/pulse settings. EF drive keys are `qb_ch_ef`, `nqz_qb_ef`, and `qb_mixer_ef`. There is no generic `nqz` key in the editable qubit dictionary. Editing a run does not modify `qb` or save the working file.

Each native program declares its qubit drive independently from readout:

```python
def _initialize(self, cfg):
    for qc in cfg["qubits"].values():
        self.setup_qubit_gen(qc, "ge")
        self.setup_standard_gates(qc, "ge")  # Only when this sequence needs gates.
    self.setup_readout(cfg)
```

`setup_readout` groups resonator generators, ADC configuration and readout pulses, including MUX membership. Resonator-only programs do not declare qubit drives. Raw `declare_gen_auto` and QICK declaration methods remain available for custom sequences.


### FPGA readout punch-out

`resonator_punchout` uses two native FPGA loops: `gainloop` outside `freqloop`, matching the archived punch-out sequence. Use `PunchoutProgram` with `g_steps`, `f_steps`, `res_gain_ge=QickSweep1D("gainloop", ...)` and `res_freq_ge=QickSweep1D("freqloop", ...)`, then call `lab.run(...)` once. Do not implement punch-out through `lab.scan` or a Python loop over gain.

Notebook frequencies are absolute MHz. Catalog `freq_start/freq_stop` are offsets from each target's configured resonance; `gain_start/gain_stop` are absolute normalized amplitudes. The result dimensions are `(gain, frequency, readout)` with coordinates read from the compiled pulse parameters. Every software average acquires the entire map. Cancellation retains completed acquisition rounds; it does not interrupt an individual FPGA sweep point. LivePlot is cleared before the final heatmap. Analysis summarizes the map without inferring a physical calibration.

This program requires direct readout with tProc-controlled frequency. Static MUX tones and readouts without tProc frequency control are explicitly rejected; no host-sweep fallback is used.
