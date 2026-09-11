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


def randomized_sequence(length, rng, interleaved=None):
    library = cliffords()
    x = np.array([[0, 1], [1, 0]], complex)
    y = np.array([[0, -1j], [1j, 0]], complex)
    eye = np.eye(2, dtype=complex)
    gates_by_name = {"x": -1j*x, "y": -1j*y,
                     "halfx": (eye-1j*x)/np.sqrt(2), "mhalfx": (eye+1j*x)/np.sqrt(2),
                     "halfy": (eye-1j*y)/np.sqrt(2), "mhalfy": (eye+1j*y)/np.sqrt(2)}
    if interleaved is not None and interleaved not in gates_by_name:
        raise ValueError("Select a native single-qubit Clifford gate for interleaving")
    total, sequence = np.eye(2, dtype=complex), []
    for index in rng.integers(0, 24, size=length):
        unitary, gates = library[index]
        total = unitary @ total
        sequence.extend(gates)
        if interleaved is not None:
            total = gates_by_name[interleaved] @ total
            sequence.append(interleaved)
    inverse = next(gates for unitary, gates in library if abs(np.trace(unitary @ total)) > 2 - 1e-8)
    sequence.extend(inverse)
    return tuple(sequence)
