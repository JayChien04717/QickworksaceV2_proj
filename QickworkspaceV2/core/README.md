# Core execution workflow

`BaseExperiment.run()` contains the complete ordinary experiment lifecycle in
one place:

```text
validate options → build Program → resolve sweep axes → acquire QICK data
                 → build ExperimentData → run optional Analysis
```

Ordinary experiment classes declare their program and axes:

```python
class PowerRabi(BaseExperiment):
    PROGRAM = PowerRabiProgram
    X_AXIS = SweepAxis.pulse("qb_pulse", "gain")

class Punchout(BaseExperiment):
    PROGRAM = PunchoutProgram
    X_AXIS = SweepAxis.pulse("res_pulse", "freq")
    Y_AXIS = SweepAxis.pulse("res_pulse", "gain")
```

Computed axes may override `_extract_sweep_axis()` or
`_extract_sweep_axis_y()`. Specialized multi-readout experiments may override
`_acquire()` and return `AcquisitionResult`.

## Core types

| Type | Responsibility |
| --- | --- |
| `BaseProgram` | QICK program base plus resonator, qubit-pulse, cooling, measurement, and active-reset helpers |
| `BaseExperiment` | Session, ordinary acquisition lifecycle, plotting, and legacy Labber save |
| `SweepAxis` | Validated, human-readable 1D/2D pulse or time axis declaration |
| `AcquisitionResult` | Raw payload from ordinary or custom acquisition |
| `ExperimentData` | Dimensioned result, fits, metadata, quality, and native HDF5 access |
| `BaseAnalysis` | Shared fit, quality, and rendering behavior |

The previous `ExperimentRuntime`, `SweepDefinition`, `AcquisitionRunner`, and
`ResultBuilder` wrappers were removed. They only forwarded data between stages
and duplicated state already owned by `BaseExperiment`.

## Data contract

- `ExperimentData.raw_iq` is the primary analysis trace.
- `x_axis` and `y_axis` are sweep coordinates only.
- Multiple readouts live in `raw_data["readouts"]` with a leading `readout`
  dimension and are retrieved through `get_readout()`.
- Derived arrays belong in `analysis_data`.
- `dataset_dims` names every stored dataset dimension.

`run_batch()`, `run_parallel()`, and `summarize_results()` are functions rather
than stateful wrapper classes. Never use `run_parallel()` against one shared
QICK board.
