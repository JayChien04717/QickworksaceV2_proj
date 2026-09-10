from __future__ import annotations
from pydantic import Field
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import Parameters, ProgramPlan, ExperimentSpec
from QickworkspaceV2.programs.sweeps import Sweep
from QickworkspaceV2.analysis import analyze_traces

from QickworkspaceV2.data.models import FitResult


class ConditionalRamseyParameters(Parameters):
    target: str = "Q1,Q2"
    start: float = Field(default=0.0, ge=0)
    stop: float = Field(default=30.0, gt=0)
    points: int = Field(default=81, ge=8, le=10001, strict=True)
    detuning_mhz: float = 0.2
    reset_wait_us: float = Field(default=200.0, gt=0)


class ConditionalRamseyProgram(BaseProgram):
    TRANSITION = "ge"

    def _initialize(self, cfg):
        self.setup_device(cfg, gates=True)
        self.add_loop("sweep", cfg["steps"])
        qc = cfg["qubits"][cfg["targets"][1]]
        self.setup_qb_pulse(
            qc, name="analysis90", phase=360 * cfg["detuning_mhz"] * cfg["wait_us"], gain_key="pi2_gain_ge"
        )

    def _body(self, cfg):
        control, target = cfg["targets"]
        qc = cfg["qubits"][target]
        self.delay_auto(cfg["reset_wait_us"])
        for state in (0, 1):
            if state:
                self.delay_auto(cfg["reset_wait_us"])
                self.qubit(control).x()
            self.qubit(target).halfx()
            self.delay_auto(cfg["wait_us"], tag="evolution" if state == 0 else "evolution_e")
            self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, "analysis90"), t=0)
            self.delay_auto(0.05)
            self.measure(cfg)


def build_conditional(ctx, p):
    if len(ctx.targets) != 2:
        raise ValueError("conditional_ramsey needs exactly two GE targets: control,target")
    cfg = ctx.config()
    cfg["steps"] = p.points
    sweep = Sweep("delay", p.start, p.stop, p.points, "us", "sweep", "t", tag="evolution")
    cfg.update(wait_us=sweep.qick(), detuning_mhz=p.detuning_mhz, reset_wait_us=p.reset_wait_us)
    return ProgramPlan(
        ConditionalRamseyProgram,
        cfg,
        (sweep,),
        ("control_g", "control_e"),
    )


def analyze_conditional(result):
    fits_g = analyze_traces(result, "ramsey", signal=result.metadata["iq_process"], event=0)
    fits_e = analyze_traces(result, "ramsey", signal=result.metadata["iq_process"], event=1)
    target = result.metadata["targets"][1]
    g, e = fits_g[target], fits_e[target]
    params, errors = {}, {}
    if "frequency" in g.parameters and "frequency" in e.parameters:
        params = {
            "frequency_g_mhz": g.parameters["frequency"],
            "frequency_e_mhz": e.parameters["frequency"],
            "frequency_difference_mhz": e.parameters["frequency"] - g.parameters["frequency"],
        }
        errors["frequency_difference_mhz"] = (
            (g.errors["frequency"] or 0) ** 2 + (e.errors["frequency"] or 0) ** 2
        ) ** 0.5
    result.metadata["conditional_fits"] = {"ground": g, "excited": e}
    return {
        target: FitResult(
            "conditional_ramsey",
            g.success and e.success,
            params,
            errors,
            {k: "MHz" for k in params},
            message="Difference of positive oscillation frequencies; signed ZZ requires detuning-branch verification",
        )
    }


experiment = ExperimentSpec(
    "conditional_ramsey",
    ConditionalRamseyParameters,
    build_conditional,
    analyze_conditional,
    description="Two-qubit conditional Ramsey: control,target",
)
