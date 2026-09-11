# Cryoscope workflow

This folder keeps the measurement sequence deliberately small.  There are
three hardware experiments and one data-processing module:

1. `const.py` — obtain the ordinary, easy-to-debug part of the step response.
2. `zero_padding.py` — add the first few Gen v6 samples that a `const` pulse
   cannot represent.
3. `filter_design.py` — turn measured X/Y into a step response and design an
   inverse FIR or IIR filter on the host computer.
4. `predistorted.py` — upload the calculated samples as an `arb` envelope and
   repeat the cryoscope measurement.

The QICK programs do not fit filters.  The filter code does not know about the
hardware.  This separation makes it much easier to inspect every intermediate
array before sending a compensated waveform to a DAC.

For a fully executed, step-by-step simulation with plots, open
[`inverse_filter_tutorial.ipynb`](inverse_filter_tutorial.ipynb).

## Common configuration

All experiments use the normal single-qubit keys plus:

```python
cfg.update({
    "flux_ch": 0,          # a Gen v6 channel is preferred
    "nqz_flux": 1,
    "flux_gain": 0.15,    # normalized QICK gain
    "cryoscope_axis": "X",# run_xy() sets X and Y automatically
    "cryoscope_pre_delay": 0.01,   # us, after the first pi/2
    "cryoscope_post_delay": 0.01,  # us, before the analysis pi/2
})
```

The Ramsey sequence is:

```text
x90 ── small guard ── flux pulse ── small guard ── X/Y analysis ── readout
```

`run_xy()` executes the program twice.  Its `X` and `Y` results contain raw
resonator IQ; they are not automatically calibrated Bloch expectation values.

## 1. Baseline with a const pulse

Use this first.  Keep the start length above three generator fabric clocks.

```python
from QickworkspaceV2.experiments.cryoscope import (
    CryoscopeConst,
    configure_const_sweep,
)

const_cfg = configure_const_sweep(
    cfg,
    start_us=0.006,   # choose this from 3 / f_fabric
    stop_us=0.100,
    steps=101,
)

const_experiment = CryoscopeConst(const_cfg)
const_xy = const_experiment.run_xy(py_avg=10)

raw_iq_x = const_xy["X"].raw_iq
raw_iq_y = const_xy["Y"].raw_iq
time_us = const_xy["X"].x_axis
```

For the firmware previously printed in this repository, Gen v6 has
`f_fabric = 599.04 MHz`, so the minimum descriptor is about `5.008 ns`.

## 2. Add the short zero-padded points

Gen v6 exposes 16 envelope samples per fabric clock.  The useful prefix can be
shorter than three clocks, while the complete descriptor is padded to 48
samples.

```python
from QickworkspaceV2.experiments.cryoscope import CryoscopeZeroPadding

short_cfg = dict(cfg)
short_cfg["flux_active_samples"] = 1

short_experiment = CryoscopeZeroPadding(short_cfg)
short_xy = short_experiment.run_sample_sweep(
    active_samples=[0, 1, 2, 4, 8, 16, 24, 32, 40],
    py_avg=10,
)
```

The dictionary key is the number of non-zero Gen v6 samples.  With a
9584.64 MS/s Gen v6, one sample is about `0.1043 ns`.  The zero point uses an
all-zero three-clock envelope, which preserves the same scheduling path while
applying no flux.

## 3. Convert measured X/Y into a step response

First project resonator IQ onto the calibrated ground-excited line.  The exact
array squeeze needed can vary slightly with acquisition shape, so inspect it
once before fitting.

```python
import numpy as np

from QickworkspaceV2.experiments.cryoscope import (
    as_complex_iq,
    merge_xy_segments,
    project_iq_to_expectation,
    trace_from_xy,
)

iq_x = as_complex_iq(const_xy["X"].raw_iq)
iq_y = as_complex_iq(const_xy["Y"].raw_iq)

# iq_ground and iq_excited come from your single-shot/readout calibration.
x_expectation = project_iq_to_expectation(iq_x, iq_ground, iq_excited)
y_expectation = project_iq_to_expectation(iq_y, iq_ground, iq_excited)

trace = trace_from_xy(
    time_ns=1000 * np.asarray(const_xy["X"].x_axis),
    x=x_expectation,
    y=y_expectation,
    smooth_window=15,
    tail_points=10,
)

measured_step = trace.normalized_step
measured_detuning_mhz = trace.detuning_mhz
```

In real analysis, first build X/Y expectation arrays for the zero-padded and
const results, then merge them.  Duplicate boundary points are averaged:

```python
time_ns, x_all, y_all = merge_xy_segments(
    (zero_time_ns, zero_x, zero_y),
    (short_time_ns, short_x, short_y),
    (1000 * np.asarray(const_xy["X"].x_axis), x_expectation, y_expectation),
)

trace = trace_from_xy(time_ns, x_all, y_all)
measured_step = trace.normalized_step
```

Before designing a filter, plot all of these:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(trace.time_ns, trace.x, label="X")
axes[0].plot(trace.time_ns, trace.y, label="Y")
axes[0].legend()
axes[1].plot(trace.time_ns, trace.unwrapped_phase_rad)
axes[1].set_ylabel("phase (rad)")
axes[2].plot(trace.time_ns, trace.normalized_step)
axes[2].axhline(1, color="black", linestyle="--")
axes[2].set_ylabel("normalized response")
axes[2].set_xlabel("time (ns)")
```

If phase unwrap is wrong, stop here.  An inverse filter will amplify that
mistake.

## 4A. Recommended: regularized inverse FIR

```python
from QickworkspaceV2.experiments.cryoscope import (
    apply_inverse_fir,
    design_inverse_fir,
    predict_corrected_output,
    scale_waveform,
)

inverse_fir = design_inverse_fir(
    measured_step,
    taps=64,
    delay_samples=16,
    regularization=1e-3,
    smoothness=1e-2,
)

# Example desired rectangle: 400 active samples followed by room for turn-off.
ideal_flux = np.r_[np.ones(400), np.zeros(200)]
predistorted = apply_inverse_fir(ideal_flux, inverse_fir, keep_tail=True)

# Never silently clip inverse-filter overshoot.
predistorted, scale_factor = scale_waveform(predistorted, max_abs=0.95)
predicted_at_chip = predict_corrected_output(predistorted, measured_step)

print("waveform scale factor:", scale_factor)
print("waveform peak:", np.max(np.abs(predistorted)))
```

`delay_samples` is intentional pre-roll.  The compensated pulse should start
that many samples before the ideal physical step.  Increasing regularization
or smoothness reduces noisy high-frequency correction at the price of a slower
edge.

If `scale_factor < 1`, the waveform plateau was reduced.  You may compensate
by increasing `flux_gain / scale_factor`, but only if the resulting gain stays
within the hardware range and the flux line remains linear.

## 4B. Optional: compact inverse IIR

```python
from QickworkspaceV2.experiments.cryoscope import (
    apply_inverse_iir,
    fit_inverse_iir,
)

inverse_iir = fit_inverse_iir(
    measured_step,
    numerator_order=1,
    denominator_order=2,
)
predistorted_iir = apply_inverse_iir(ideal_flux, inverse_iir)
```

The IIR function refuses a non-minimum-phase fit because its exact inverse
would be unstable.  That is a reason to use the FIR path, not a reason to set
`allow_unstable=True` on hardware.

## 5. Upload and verify the compensated waveform

```python
from QickworkspaceV2.experiments.cryoscope import PredistortedCryoscope

corrected_cfg = dict(cfg)
corrected_cfg["predistorted_waveform"] = predistorted

corrected_experiment = PredistortedCryoscope(corrected_cfg)

# One complete pulse:
corrected_xy = corrected_experiment.run_xy(py_avg=10)

# Or verify prefixes of the waveform with another cryoscope scan:
prefix_xy = corrected_experiment.run_prefix_sweep(
    sample_counts=np.arange(1, len(predistorted) + 1, 8),
    py_avg=10,
)
```

Finally, process the verification X/Y with the same `trace_from_xy` function.
The compensated normalized step should be flatter and closer to one than the
original measurement.  Keep the uncorrected measurement, filter coefficients,
sample interval, scale factor, and verification trace together in the
calibration record.

## Practical cautions

- Use the same Gen v6 channel and sample rate for measurement and correction.
- A filter identified at one flux amplitude may not work at another if the
  qubit spectrum or flux line is nonlinear.
- Phase-reference tracking may be necessary before differentiating data with a
  very large steady detuning.
- FIR/IIR compensation cannot causally undo pure cable delay.  Start the
  waveform earlier and use pre-roll instead.
- Always inspect the waveform peak and predicted output before uploading it.
- Re-run cryoscope after compensation; a good numerical inverse is not yet a
  hardware calibration.
