"""randomized_benchmarking: program."""

from __future__ import annotations
import numpy as np
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import ProgramPlan
from QickworkspaceV2.programs.cliffords import randomized_sequence


class RBProgram(BaseProgram):
    TRANSITION = "ge"

    def _initialize(self, cfg):
        self.READOUT_EVENTS = prepare_sequences(cfg)
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


def prepare_sequences(cfg):
    """Build the same seeded protocol for native Notebook and catalog jobs."""
    depths = cfg.get("depths", [0, 1, 2, 4, 8, 12, 16])
    if (
        len(depths) < 5
        or any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) for v in depths)
        or len(set(depths)) != len(depths)
        or min(depths) < 0
        or max(depths) > 1000
    ):
        raise ValueError("RB needs at least five distinct integer depths between 0 and 1000")
    repeats = cfg.get("sequences_per_depth", 2)
    if not isinstance(repeats, int) or isinstance(repeats, bool) or not 2 <= repeats <= 20:
        raise ValueError("RB sequences_per_depth must be an integer from 2 to 20")
    interleaved = cfg.get("interleaved", "none")
    if interleaved not in ("none", "x", "y", "halfx", "halfy", "mhalfx", "mhalfy"):
        raise ValueError("Unknown interleaved gate")
    rng = np.random.default_rng(cfg.get("seed", 42))
    cfg["sequences"] = [
        {
            q: randomized_sequence(depth, rng, None if interleaved == "none" else interleaved)
            for q in cfg["targets"]
        }
        for depth in depths
        for _ in range(repeats)
    ]
    cfg["rb_depths"] = list(np.repeat(depths, repeats).astype(int))
    if sum(max(map(len, seq.values())) for seq in cfg["sequences"]) > 2000:
        raise ValueError("RB program is too large; reduce depths/sequences or partition the acquisition")
    return tuple(f"m{depth}_s{repeat}" for depth in depths for repeat in range(repeats))


def build_rb(ctx, p):
    depths = [int(x.strip()) for x in p.depths.split(",")]
    cfg = ctx.config()
    cfg.update(
        depths=depths,
        sequences_per_depth=p.sequences_per_depth,
        seed=p.seed,
        interleaved=p.interleaved,
        reset_wait_us=p.reset_wait_us,
    )
    events = prepare_sequences(cfg)
    return ProgramPlan(
        RBProgram,
        cfg,
        readout_events=events,
        metadata={
            "rb_depths": depths,
            "sequences_per_depth": p.sequences_per_depth,
            "seed": p.seed,
            "interleaved": p.interleaved,
        },
    )
