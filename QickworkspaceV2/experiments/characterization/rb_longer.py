"""Long-depth single-qubit randomized benchmarking for tProc v2.

Unlike :mod:`.rb`, this module never stores or compile-time-unrolls the RB
sequence.  A deterministic xorshift32 generator produces Clifford indices in
the tProc, and a fixed 24 x 24 Clifford multiplication table tracks the net
Clifford so the recovery operation can also be selected in hardware.

The program-memory and data-memory footprints are therefore independent of
the requested Clifford depth.  DMEM contains only the 576-entry group table
and the 24-entry inverse table (600 int32 words total).

This implementation targets QICK 0.2.406/tProc v2.  It intentionally uses a
small amount of low-level ASM through ``Macro`` because the public Python API
does not expose XOR and logical-shift register writes.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from tqdm.auto import tqdm

from qick.asm_v2 import AsmInst, AsmV2, Macro

from ...analysis.rb import RBAnalysis
from ...core.base_program import BaseProgram
from ...core.experiment_data import ExperimentData, QualityFlag
from ...tools.rb_generator import (
    INTERLEAVE_GATES,
    clifford_decompositions,
    clifford_matrices,
    find_clifford_index,
    inverse_table,
)
from .rb import RandomizedBenchmarking


CLIFFORD_COUNT = 24
MAX_TPROC_LOOP_COUNT = 2**23
INVERSE_TABLE_OFFSET = CLIFFORD_COUNT * CLIFFORD_COUNT
LONG_RB_DMEM_WORDS = INVERSE_TABLE_OFFSET + CLIFFORD_COUNT


# ---------------------------------------------------------------------------
# Small, regular lookup tables shared by Python and the tProc
# ---------------------------------------------------------------------------

def _build_clifford_tables() -> tuple[np.ndarray, np.ndarray]:
    """Build tables for ``new_state = C_gate @ C_state`` and inversion."""
    multiplication = np.empty((CLIFFORD_COUNT, CLIFFORD_COUNT), dtype=np.int32)
    for gate_index, gate in enumerate(clifford_matrices):
        for state_index, state in enumerate(clifford_matrices):
            multiplication[gate_index, state_index] = find_clifford_index(
                gate @ state
            )
    inverses = np.asarray(
        [inverse_table[index] for index in range(CLIFFORD_COUNT)],
        dtype=np.int32,
    )
    return multiplication, inverses


CLIFFORD_MULTIPLICATION_TABLE, CLIFFORD_INVERSE_TABLE = _build_clifford_tables()


def _interleaved_clifford_index(label: str | None) -> int | None:
    if label is None:
        return None
    if label not in INTERLEAVE_GATES:
        raise ValueError(
            f"interleaved_gate {label!r} not in {list(INTERLEAVE_GATES.keys())}"
        )
    return find_clifford_index(INTERLEAVE_GATES[label][0])


def _xorshift32_step(state: int) -> int:
    """Python reference for the exact PRNG implemented by the tProc macros."""
    value = int(state) & 0xFFFFFFFF
    value ^= (value << 13) & 0xFFFFFFFF
    value ^= value >> 17
    value ^= (value << 5) & 0xFFFFFFFF
    return value & 0xFFFFFFFF


def _iter_clifford_indices(seed: int, depth: int):
    if depth < 0:
        raise ValueError("depth must be non-negative")
    state = int(seed) & 0xFFFFFFFF
    if state == 0:
        raise ValueError("xorshift32 seed must be non-zero")
    accepted = 0
    while accepted < depth:
        state = _xorshift32_step(state)
        candidate = state & 0x1F
        if candidate < CLIFFORD_COUNT:
            yield candidate
            accepted += 1


def clifford_indices_from_seed(seed: int, depth: int) -> list[int]:
    """Reproduce the tProc Clifford stream for validation and metadata.

    Five low PRNG bits are used with rejection sampling, making all indices
    0..23 equiprobable (up to xorshift32's excluded all-zero state).
    """
    return list(_iter_clifford_indices(seed, depth))


def verify_long_rb_sequence(
    seed: int,
    depth: int,
    interleaved_gate: str | None = None,
) -> bool:
    """Verify in Python that the hardware table selects a valid recovery."""
    state = 0
    interleaved_index = _interleaved_clifford_index(interleaved_gate)
    for clifford_index in _iter_clifford_indices(seed, depth):
        state = int(CLIFFORD_MULTIPLICATION_TABLE[clifford_index, state])
        if interleaved_index is not None:
            state = int(CLIFFORD_MULTIPLICATION_TABLE[interleaved_index, state])
    recovery = int(CLIFFORD_INVERSE_TABLE[state])
    return int(CLIFFORD_MULTIPLICATION_TABLE[recovery, state]) == 0


# ---------------------------------------------------------------------------
# The only low-level helper we need from the tProc assembler
# ---------------------------------------------------------------------------

class _RegAlu(Macro):
    """Write ``dst = lhs OP rhs`` using a full tProc-v2 ALU operation."""

    def expand(self, prog):
        dst = prog._get_reg(self.dst)
        lhs = prog._get_reg(self.lhs)
        if isinstance(self.rhs, Integral):
            rhs = f"#{int(self.rhs)}"
        elif isinstance(self.rhs, str):
            rhs = prog._get_reg(self.rhs)
        else:
            raise RuntimeError(f"invalid ALU operand: {self.rhs!r}")
        return [
            AsmInst(
                inst={
                    "CMD": "REG_WR",
                    "DST": dst,
                    "SRC": "op",
                    "OP": f"{lhs} {self.op} {rhs}",
                },
                addr_inc=1,
            )
        ]


# ---------------------------------------------------------------------------
# Hardware program
# ---------------------------------------------------------------------------

class LongerRBProgram(BaseProgram):
    """Constant-memory RB program using an on-tProc deterministic PRNG."""

    def _initialize(self, cfg):
        depth = int(cfg["rb_depth"])
        seed = int(cfg["rb_seed"])
        if depth < 1:
            raise ValueError("rb_depth must be at least 1")
        if depth > MAX_TPROC_LOOP_COUNT:
            raise ValueError(
                f"rb_depth exceeds the tProc 24-bit loop limit "
                f"({MAX_TPROC_LOOP_COUNT})"
            )
        if seed <= 0 or seed > 0x7FFFFFFF:
            raise ValueError("rb_seed must be in the range 1..2**31-1")

        tproc_cfg = self.soccfg["tprocs"][0]
        if int(tproc_cfg["dmem_size"]) < LONG_RB_DMEM_WORDS:
            raise RuntimeError(
                f"LongerRBProgram needs {LONG_RB_DMEM_WORDS} DMEM words, but "
                f"this tProc only has {tproc_cfg['dmem_size']}"
            )
        if int(tproc_cfg.get("call_depth", 0)) < 1:
            raise RuntimeError("LongerRBProgram requires a tProc CALL stack")

        prefix = cfg.get("prefix", "ge")
        self.setup_resonator(cfg, prefix=prefix)
        self.setup_qubit_gen(cfg, prefix=prefix)
        self.setup_standard_gates(cfg, prefix=prefix)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)

        for name in (
            "rb_rng",
            "rb_tmp",
            "rb_candidate",
            "rb_group_state",
            "rb_dmem_addr",
            "rb_dispatch_index",
        ):
            self.add_reg(name)

        self._add_dispatch_subroutine(cfg)

    def compile_datamem(self):
        return np.concatenate(
            (
                CLIFFORD_MULTIPLICATION_TABLE.reshape(-1),
                CLIFFORD_INVERSE_TABLE,
            )
        ).astype(np.int32, copy=False)

    def _reg_alu(self, dst: str, lhs: str, op: str, rhs: int | str):
        self.append_macro(_RegAlu(dst=dst, lhs=lhs, op=op, rhs=rhs))

    def _xorshift32(self):
        # x ^= x << 13
        self.write_reg("rb_tmp", "rb_rng")
        self._reg_alu("rb_tmp", "rb_tmp", "SL", 13)
        self._reg_alu("rb_rng", "rb_rng", "XOR", "rb_tmp")

        # x ^= x >> 17.  tProc immediate shifts are limited to 15 bits,
        # therefore perform the logical shift as 15 + 2.
        self.write_reg("rb_tmp", "rb_rng")
        self._reg_alu("rb_tmp", "rb_tmp", "SR", 15)
        self._reg_alu("rb_tmp", "rb_tmp", "SR", 2)
        self._reg_alu("rb_rng", "rb_rng", "XOR", "rb_tmp")

        # x ^= x << 5
        self.write_reg("rb_tmp", "rb_rng")
        self._reg_alu("rb_tmp", "rb_tmp", "SL", 5)
        self._reg_alu("rb_rng", "rb_rng", "XOR", "rb_tmp")

    def _generate_clifford_index(self):
        self.label("RB_RANDOM_RETRY")
        self._xorshift32()
        self._reg_alu("rb_candidate", "rb_rng", "AND", 0x1F)
        self.cond_jump(
            "RB_RANDOM_ACCEPT",
            "rb_candidate",
            "S",
            op="-",
            arg2=CLIFFORD_COUNT,
        )
        self.jump("RB_RANDOM_RETRY")
        self.label("RB_RANDOM_ACCEPT")

    def _update_group_state(self, gate_index: int | str):
        if isinstance(gate_index, Integral):
            self.write_reg("rb_candidate", int(gate_index))
            gate_reg = "rb_candidate"
        else:
            gate_reg = gate_index

        # address = gate_index * 24 + group_state
        self.write_reg("rb_tmp", gate_reg)
        self._reg_alu("rb_tmp", "rb_tmp", "SL", 4)
        self.write_reg("rb_dmem_addr", "rb_tmp")
        self.write_reg("rb_tmp", gate_reg)
        self._reg_alu("rb_tmp", "rb_tmp", "SL", 3)
        self.inc_reg("rb_dmem_addr", "rb_tmp")
        self.inc_reg("rb_dmem_addr", "rb_group_state")
        self.read_dmem("rb_group_state", "rb_dmem_addr")

    def _dispatch(self, index: int | str):
        if isinstance(index, Integral):
            self.write_reg("rb_dispatch_index", int(index))
        else:
            self.write_reg("rb_dispatch_index", index)
        self.call("RB_DISPATCH")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.cooling_body(cfg)

        self.write_reg("rb_rng", int(cfg["rb_seed"]))
        self.write_reg("rb_group_state", 0)

        self.open_loop(int(cfg["rb_depth"]), name="rb_depth_loop")
        self._generate_clifford_index()
        self._dispatch("rb_candidate")
        self._update_group_state("rb_candidate")

        interleaved_index = cfg.get("rb_interleaved_index")
        if interleaved_index is not None:
            self._dispatch(int(interleaved_index))
            self._update_group_state(int(interleaved_index))
        self.close_loop()

        self.write_reg("rb_dmem_addr", "rb_group_state")
        self.inc_reg("rb_dmem_addr", INVERSE_TABLE_OFFSET)
        self.read_dmem("rb_dispatch_index", "rb_dmem_addr")
        self.call("RB_DISPATCH")

        self.delay_auto(0.05)
        self.measure(cfg)

    def _add_dispatch_subroutine(self, cfg):
        dispatch = AsmV2()
        self._emit_dispatch_tree(dispatch, tuple(range(CLIFFORD_COUNT)), "RB_D")

        prefix = cfg.get("prefix", "ge")
        channel = cfg["qb_ch"]
        pulse_names = {
            "X": f"x180_{prefix}",
            "Y": f"y180_{prefix}",
            "X/2": f"x90_{prefix}",
            "-X/2": f"x90m_{prefix}",
            "Y/2": f"y90_{prefix}",
            "-Y/2": f"y90m_{prefix}",
        }
        primitive_slot = max(
            float(self.pulses[name].get_length()) for name in pulse_names.values()
        ) + float(cfg.get("rb_gate_gap", 0.01))
        for index, decomposition in enumerate(clifford_decompositions):
            dispatch.label(f"RB_CLIFFORD_{index}")
            for gate in decomposition:
                if gate != "I":
                    dispatch.pulse(ch=channel, name=pulse_names[gate], t=0)
                dispatch.delay(primitive_slot)
            dispatch.jump("RB_DISPATCH_RETURN")
        dispatch.label("RB_DISPATCH_RETURN")
        self.add_subroutine("RB_DISPATCH", dispatch)

    @classmethod
    def _emit_dispatch_tree(
        cls,
        dispatch: AsmV2,
        indices: tuple[int, ...],
        label: str,
    ):
        dispatch.label(label)
        if len(indices) == 1:
            dispatch.jump(f"RB_CLIFFORD_{indices[0]}")
            return
        split = len(indices) // 2
        left = indices[:split]
        right = indices[split:]
        threshold = right[0]
        left_label = f"{label}_L"
        right_label = f"{label}_R"
        dispatch.cond_jump(
            left_label,
            "rb_dispatch_index",
            "S",
            op="-",
            arg2=threshold,
        )
        dispatch.jump(right_label)
        cls._emit_dispatch_tree(dispatch, left, left_label)
        cls._emit_dispatch_tree(dispatch, right, right_label)


# ---------------------------------------------------------------------------
# Notebook-facing experiment
# ---------------------------------------------------------------------------

class RandomizedBenchmarkingLonger(RandomizedBenchmarking):
    """Single-qubit RB/IRB whose tProc memory use is independent of depth.

    Plotting and legacy saving are inherited from the regular RB experiment;
    only sequence generation and acquisition are different here.
    """

    EXPT_NAME = "s015_RB_longer"
    Analysis = RBAnalysis

    def run(
        self,
        py_avg: int,
        max_circuit_depth: int,
        delta_clifford: int,
        number_sample: int,
        interleaved_gate: str | None = None,
        seed: int | None = None,
        prefix: str = "ge",
        iq_process: str = "abs",
        randomize_depth_order: bool = False,
    ) -> ExperimentData:
        if py_avg < 1 or number_sample < 1:
            raise ValueError("py_avg and number_sample must be at least 1")
        if delta_clifford < 1 or max_circuit_depth <= 1:
            raise ValueError("invalid RB depth range")
        if iq_process not in {"abs", "real"}:
            raise ValueError("iq_process must be 'abs' or 'real'")

        interleaved_index = _interleaved_clifford_index(interleaved_gate)
        self._iq_process = iq_process
        self.x = np.arange(1, max_circuit_depth, delta_clifford, dtype=int)
        self._number_sample = number_sample
        self._interleaved = interleaved_gate
        self._prefix = prefix

        rng = np.random.default_rng(seed)
        seeds_matrix = rng.integers(
            1,
            2**31,
            size=(len(self.x), number_sample),
            dtype=np.int64,
        )
        depth_indices = np.arange(len(self.x))
        if randomize_depth_order:
            rng.shuffle(depth_indices)

        rb_result = [[None] * number_sample for _ in range(len(self.x))]
        memory_usage = None
        for depth_index in tqdm(depth_indices, desc="Long RB"):
            depth = int(self.x[depth_index])
            for sample_index in tqdm(
                range(number_sample), desc="Samples", leave=False
            ):
                circuit_seed = int(seeds_matrix[depth_index, sample_index])
                if not verify_long_rb_sequence(
                    circuit_seed, depth, interleaved_gate
                ):
                    raise RuntimeError("internal RB recovery-table validation failed")

                program_cfg = dict(self.cfg)
                program_cfg.update(
                    rb_depth=depth,
                    rb_seed=circuit_seed,
                    rb_interleaved_index=interleaved_index,
                    prefix=prefix,
                )
                program = LongerRBProgram(
                    self.soccfg,
                    reps=program_cfg["reps"],
                    final_delay=program_cfg["relax_delay"],
                    cfg=program_cfg,
                )
                if memory_usage is None:
                    memory_usage = {
                        name: 0 if value is None else len(value)
                        for name, value in program.binprog.items()
                    }
                acquired = program.acquire(
                    self.soc, rounds=py_avg, progress=False
                )
                rb_result[depth_index][sample_index] = acquired[0][0].dot(
                    [1, 1j]
                )

        self.rb_result = rb_result
        raw_iq = np.asarray(rb_result)
        process = np.real if iq_process == "real" else np.abs
        processed = process(raw_iq)
        average = processed.reshape(len(self.x), -1).mean(axis=1)

        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=raw_iq,
            x_axis=self.x.astype(float),
            y_axis=average,
            metadata={
                "qubit": self.cfg.get("name"),
                "iq_process": iq_process,
                "number_sample": number_sample,
                "interleaved_gate": interleaved_gate,
                "prefix": prefix,
                "seeds": seeds_matrix.tolist(),
                "randomized_depth_order": self.x[depth_indices].tolist(),
                "tproc_memory_words": memory_usage,
                "sequence_generator": "xorshift32-rejection-v1",
            },
            axes={
                "depth": {
                    "values": self.x.astype(float),
                    "label": "Circuit depth",
                    "unit": "# Cliffords",
                },
                "sample": {
                    "values": np.arange(number_sample),
                    "unit": "#",
                },
            },
            dataset_dims={"iq": ["depth", "sample"]},
            analysis_data={
                "mean_signal": {"values": average, "dims": ["depth"]}
            },
            data_kind="rb",
            analysis_id="rb",
            plot_id="rb_decay",
            avg_count=py_avg,
            quality=QualityFlag.NO_INFORMATION,
        )
        if self.Analysis is not None:
            result = self.Analysis().run(result)
        self.result = result
        return result


__all__ = [
    "LongerRBProgram",
    "RandomizedBenchmarkingLonger",
    "clifford_indices_from_seed",
    "verify_long_rb_sequence",
    "CLIFFORD_MULTIPLICATION_TABLE",
    "CLIFFORD_INVERSE_TABLE",
    "LONG_RB_DMEM_WORDS",
]
