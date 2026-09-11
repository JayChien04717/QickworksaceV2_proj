"""allxy: program."""

from __future__ import annotations
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import ProgramPlan


ALLXY = [
    ("i", "i"),
    ("x", "x"),
    ("y", "y"),
    ("x", "y"),
    ("y", "x"),
    ("halfx", "i"),
    ("halfy", "i"),
    ("halfx", "halfy"),
    ("halfy", "halfx"),
    ("halfx", "y"),
    ("halfy", "x"),
    ("x", "halfy"),
    ("y", "halfx"),
    ("halfx", "x"),
    ("x", "halfx"),
    ("halfy", "y"),
    ("y", "halfy"),
    ("x", "i"),
    ("y", "i"),
    ("halfx", "halfx"),
    ("halfy", "halfy"),
]


ALLXY_IDEAL = [0.0] * 5 + [0.5] * 12 + [1.0] * 4


class AllXYProgram(BaseProgram):
    TRANSITION = "ge"
    READOUT_EVENTS = tuple(f"{a},{b}" for a, b in ALLXY)

    def _initialize(self, cfg):
        cfg.setdefault("sequences", [{q: gates for q in cfg["targets"]} for gates in ALLXY])
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, cfg["transition"])
            self.setup_standard_gates(qc, cfg["transition"])
        self.setup_readout(cfg)

    def _body(self, cfg):
        self.delay_auto(cfg["reset_wait_us"])
        for index, sequences in enumerate(cfg["sequences"]):
            if index:
                self.delay_auto(cfg["reset_wait_us"])
            for layer in range(max(map(len, sequences.values()))):
                with self.parallel():
                    for q, sequence in sequences.items():
                        if layer < len(sequence) and sequence[layer] != "i":
                            getattr(self.qubit(q), sequence[layer])()
            self.measure(cfg)


def build_allxy(ctx, p):
    cfg = ctx.config()
    cfg["reset_wait_us"] = p.reset_wait_us
    cfg["sequences"] = [{q: gates for q in ctx.targets} for gates in ALLXY]
    cfg["ideal_population"] = ALLXY_IDEAL
    return ProgramPlan(
        AllXYProgram,
        cfg,
        readout_events=tuple(f"{a},{b}" for a, b in ALLXY),
        metadata={"ideal_population": ALLXY_IDEAL},
    )
