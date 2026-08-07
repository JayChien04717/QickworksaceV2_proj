"""Shared QICK acquisition call and output decoding helpers."""

import numpy as np


def decode_acquisition(acquired, *, threshold: bool, scalar_readout=False):
    """Decode one QICK readout as complex IQ or real threshold population.

    Parameters
    ----------
    acquired : Any
        Value for ``acquired``.
    threshold : bool
        Value for ``threshold``.

    Returns
    -------
    Any
        Result of the operation.
    """
    try:
        data = acquired[0][0]
    except (IndexError, TypeError):
        data = acquired

    values = np.asarray(data)
    if threshold:
        # QICK keeps the trailing [I, Q] axis in threshold mode. The first
        # component is the thresholded population; retaining Q would turn an
        # N-point sweep into 2N plotted values.
        if values.ndim >= 2 and values.shape[-1] == 2:
            values = values[..., 0]
        elif scalar_readout and values.ndim == 1 and values.size == 2:
            values = values[0]
        elif np.iscomplexobj(values):
            values = np.real(values)
        return values.astype(float, copy=False).squeeze()

    if values.ndim > 0 and values.shape[-1] == 2:
        return values.dot([1, 1j])
    return values.astype(complex, copy=False)


def acquire_values(prog, soc, *, rounds: int, progress: bool, threshold=None,
                   scalar_readout=False):
    """Run ``prog.acquire`` and return values in the selected readout mode.

    Parameters
    ----------
    prog : Any
        Value for ``prog``.
    soc : Any
        Value for ``soc``.
    rounds : int
        Value for ``rounds``.
    progress : bool
        Value for ``progress``.
    threshold : Any, default: None
        Value for ``threshold``.

    Returns
    -------
    Any
        Result of the operation.
    """
    call_kwargs = {"rounds": rounds, "progress": progress}
    if threshold is not None:
        call_kwargs["threshold"] = threshold
    acquired = prog.acquire(soc, **call_kwargs)
    return decode_acquisition(
        acquired,
        threshold=threshold is not None,
        scalar_readout=scalar_readout,
    )


__all__ = ["acquire_values", "decode_acquisition"]
