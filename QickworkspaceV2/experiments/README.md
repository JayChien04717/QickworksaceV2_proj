# Where to edit an experiment

Each experiment has its own directory, such as `t1_ge/`. GE and EF experiments are maintained separately.

| What you want to change | File or location |
|---|---|
| Frequency, gain, sigma, wait time or other settings for this run | Notebook `qb` or `run_cfg` |
| Workflow node defaults and allowed parameter ranges | Local `parameters.py` |
| Qubit channels, resonator/readout setup, pulses and loops | `program.py`: `_initialize()` |
| Pulse order, delays, parallel operations and readout timing | `program.py`: `_body()` |
| Mapping node parameters to QICK settings and sweeps | `program.py`: `build()` or the experiment's named builder |
| Analysis signal and fitting method | `analysis.py`: `analyze()` |
| Final plot appearance, labels and layout | `analysis.py`: `plot()` |
| Calibration values this experiment can update | `analysis.py`: `updates()`; applying them remains explicit |
| Public imports and catalog registration | `__init__.py` |

## Reading parameters.py

For example, the T1 wait-time settings are written as:

```python
# Starting wait after excitation (us).
start: float = Field(
    default=0.0,  # Start at 0 us when omitted.
    ge=0,         # Greater than or equal to 0; zero is allowed.
)

# Sweep point count, separate from reps and py_avg.
points: int = Field(
    default=81,   # Use 81 points when omitted.
    ge=8,         # At least 8 points, including 8.
    le=10001,     # At most 10001 points, including 10001.
    strict=True,  # Require an integer; reject the string "81".
)
```

`ge` means greater than or equal; `le` means less than or equal. `gt` and `lt` exclude the boundary: `gt=0` rejects zero. Here `ge` is unrelated to the qubit g-to-e transition.

`default` specifies the default value. Bounds such as `ge` and `le` validate inputs; they do not clamp invalid values to a boundary. The `model_validator` below the fields checks relationships between parameters, such as `start < stop`. These are software limits, not necessarily the instrument's physical limits.

Identical names can represent different quantities. T1 `start/stop` specify waiting times in us. Resonator spectrum `start/stop` specify offsets from the configured resonator frequency in MHz. TWPA probe `start_mhz/stop_mhz` specify absolute frequencies. Each field documents its meaning and units locally.

## Notebook settings versus workflow parameters

Notebook `qb.for_run(...)` copies all current `qb` settings and then applies the supplied overrides. It does not automatically apply every default from the local `Parameters` model. Daily measurements can directly edit native QICK settings such as `res_freq_ge=QickSweep1D(...)`, `sigma_ge`, `qb_ch` and `nqz_qb`.

Workflow, CLI and Agent node inputs are defined in `parameters.py`. Validated values go to the builder. For T1, the builder maps `points` to `cfg["steps"]` and converts `start/stop` into the `cfg["wait_us"]` sweep. Some programs also validate their specific options directly; check the experiment's `program.py` for those cases.

## Following the T1 implementation

Start with `t1_ge/parameters.py` to choose the wait-time range and point count. In `t1_ge/program.py`, `_initialize()` separately prepares the qubit generator, gates, readout and sweep. `_body()` applies a pi pulse, waits for the selected time and performs readout.

In `t1_ge/analysis.py`, `analyze()` currently uses `fit_exponential`. `plot()` creates the final figure. `updates()` proposes storing the fitted `tau` as the working value `t1_ge_us`; it does not write files itself.

To change the exponential equation, starting guesses or fitting bounds, edit the corresponding equation and `fit_exponential()` in `QickworkspaceV2/analysis/fitting.py`. The generic `fit_curve()` only handles numerical optimization. Broadband multi-resonator detection is maintained separately in `QickworkspaceV2/analysis/fit_n_res.py`.
