from functools import lru_cache
import numpy as np


@lru_cache
def cliffords():
    """Generate the 24 single-qubit Cliffords and their native pulse decompositions."""
    eye = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], complex)
    y = np.array([[0, -1j], [1j, 0]], complex)
    gates = {"halfx": (eye - 1j * x) / np.sqrt(2), "halfy": (eye - 1j * y) / np.sqrt(2)}
    library = [(eye, ())]
    cursor = 0
    while cursor < len(library):
        unitary, sequence = library[cursor]
        for name, gate in gates.items():
            candidate = gate @ unitary
            if not any(abs(np.trace(old.conj().T @ candidate)) > 2 - 1e-8 for old, _ in library):
                library.append((candidate, (*sequence, name)))
        cursor += 1
    if len(library) != 24:
        raise RuntimeError("Clifford closure failed")
    return library


def randomized_sequence(length, rng):
    library = cliffords()
    total, sequence = np.eye(2, dtype=complex), []
    for index in rng.integers(0, 24, size=length):
        unitary, gates = library[index]
        total = unitary @ total
        sequence.extend(gates)
    inverse = next(gates for unitary, gates in library if abs(np.trace(unitary @ total)) > 2 - 1e-8)
    sequence.extend(inverse)
    return tuple(sequence)
