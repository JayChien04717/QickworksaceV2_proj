"""Direct QICK measurement. Edit the cells below, then run: python measure.py.

Importing this file does not connect or acquire. The notebook uses the same API.
Frequencies: MHz; time: us; phase: degrees; gain: normalized QICK units.
"""

# %% Imports / choose the experiment
from qick.asm_v2 import QickSweep1D
from QickworkspaceV2 import Measurement
from QickworkspaceV2.experiments.resonator_spec import ResonatorSpecProgram
from lab.config import CONNECTION, DATA_PATH, make_config

Program = ResonatorSpecProgram
qubit = "Q1"
PY_AVG = 5


# %% Edit every run parameter here, just as in a notebook cell.
def make_run_config():
    config_all = make_config()
    qb = config_all[qubit]
    qb.update(res_gain_ge=0.15, res_sigma=0.01)
    center = qb["res_freq_ge"]
    STEPS, START_FREQ, STOP_FREQ = 101, center - 5, center + 5
    run_cfg = qb.for_run(
        steps=STEPS,
        res_freq_ge=QickSweep1D("freqloop", START_FREQ, STOP_FREQ),
        relax_delay=0,
        reps=100,
    )
    return run_cfg


# %% Connect, measure, fit, save. Only this function contacts the board.
def main():
    run_cfg = make_run_config()
    lab = Measurement.from_pyro4(**CONNECTION, data_path=DATA_PATH)
    result = lab.run(Program, run_cfg, py_avg=PY_AVG)
    figure = result.plot()
    figure.savefig(result.path.parent / "measurement.png", dpi=150)
    print("Data:", result.path)
    print("Fit:", result.metrics)
    print("Analysis:", result.analysis_status, result.analysis_message)
    return result


if __name__ == "__main__":
    main()
