"""GE DRAG error amplification. Scan alpha with Session.scan()."""

from pydantic import Field, model_validator
from QickworkspaceV2 import BaseProgram, Parameters, ProgramPlan, ExperimentSpec
from QickworkspaceV2.data.models import FitResult


class DragGEParameters(Parameters):
    alpha: float = 0.0
    delta_mhz: float = Field(
        description="Signed DRAG detuning used by QICK add_DRAG; specify the measured value"
    )
    iterations: int = Field(default=5, ge=1, le=101, strict=True)
    reset_wait_us: float = Field(default=200.0, gt=0)

    @model_validator(mode="after")
    def nonzero_delta(self):
        if self.delta_mhz == 0:
            raise ValueError("DRAG delta_mhz must be nonzero")
        return self


class DragGEProgram(BaseProgram):
    TRANSITION = "ge"

    def _initialize(self, cfg):
        self.setup_device(cfg, gates=True)
        for qc in cfg["qubits"].values():
            self.setup_qb_pulse(
                qc, "ge", name="drag_x", shape="drag", pulse_type="arb", gain_key="pi_gain_ge", phase=0
            )
            self.setup_qb_pulse(
                qc, "ge", name="drag_mx", shape="drag", pulse_type="arb", gain_key="pi_gain_ge", phase=180
            )

    def _body(self, cfg):
        self.delay_auto(cfg["reset_wait_us"])
        self.measure(cfg)  # Ground reference.
        self.delay_auto(cfg["reset_wait_us"])
        with self.parallel():
            for qb in self.qubits.values():
                qb.x()
        self.measure(cfg)  # Excited reference.
        self.delay_auto(cfg["reset_wait_us"])
        for _ in range(cfg["iterations"]):
            for name in ("drag_x", "drag_mx"):
                for qc in cfg["qubits"].values():
                    self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, name), t=0)
                self.delay_auto(0.01)
        self.measure(cfg)


def build(ctx, p):
    cfg = ctx.config("ge")
    cfg.update(alpha=p.alpha, iterations=p.iterations, reset_wait_us=p.reset_wait_us)
    for qc in cfg["qubits"].values():
        qc.update(drag_alpha_ge=p.alpha, drag_delta_ge=p.delta_mhz)
    return ProgramPlan(DragGEProgram, cfg, readout_events=("ground", "excited", "return"))


def analyze(result):
    fits = {}
    for q, trace in result.traces.items():
        g, e, returned = trace.iq
        contrast = abs(e - g)
        if contrast < 1e-10:
            fits[q] = FitResult("drag_return", False, message="No reference-state contrast")
            continue
        projection = float(((returned - g) * (e - g).conjugate()).real / contrast**2)
        fits[q] = FitResult(
            "drag_return",
            -0.1 <= projection <= 1.1,
            {"return_error_signal": projection},
            message="IQ projected on measured reference states; includes leakage and preparation errors",
        )
    return fits


experiment = ExperimentSpec(
    "drag_ge",
    DragGEParameters,
    build,
    analyze,
    description="GE DRAG alpha point with repeated X/-X and reference states",
)
