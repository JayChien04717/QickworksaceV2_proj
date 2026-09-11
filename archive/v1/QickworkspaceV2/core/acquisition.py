"""Decode accumulated QICK acquisition results.

QICK returns one array per readout channel.  Each channel starts with a
``readout`` axis and ends with an ``[I, Q]`` axis.  Most experiments use the
first channel and first readout; active-reset experiments keep both readouts.
"""

import numpy as np


def _decode_values(values, *, threshold: bool, scalar_readout: bool):
    """Convert a single readout's final QICK ``[I, Q]`` axis."""
    values = np.asarray(values)
    if threshold:
        # In threshold mode QICK retains a trailing Q placeholder.  A
        # one-dimensional two-point sweep is ambiguous, so callers must mark
        # scalar readouts explicitly before that axis is removed.
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


def decode_acquisition(
    acquired,
    *,
    threshold: bool,
    scalar_readout: bool = False,
    channel: int = 0,
    readout: int = 0,
):
    """Decode one channel/readout as complex IQ or threshold population.

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
        data = acquired[channel][readout]
    except (IndexError, TypeError):
        data = acquired
    return _decode_values(
        data,
        threshold=threshold,
        scalar_readout=scalar_readout,
    )


def decode_readouts(
    acquired,
    *,
    threshold: bool,
    scalar_readout: bool = False,
    channel: int = 0,
):
    """Decode every readout for one channel and retain the readout axis.

    The returned shape is ``(n_readouts, *sweep_shape)``.  This function is
    intended for programs that deliberately trigger the same ADC more than
    once per experiment point, such as measurement-based active reset.
    """
    try:
        channel_data = acquired[channel]
    except (IndexError, TypeError):
        channel_data = acquired

    channel_values = np.asarray(channel_data)
    if channel_values.ndim == 0:
        channel_values = channel_values.reshape(1)

    decoded = [
        _decode_values(
            readout_values,
            threshold=threshold,
            scalar_readout=scalar_readout,
        )
        for readout_values in channel_values
    ]
    try:
        return np.stack(decoded)
    except ValueError as exc:
        shapes = [np.shape(values) for values in decoded]
        raise ValueError(f"QICK readouts have inconsistent shapes: {shapes}") from exc


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


__all__ = ["acquire_values", "decode_acquisition", "decode_readouts"]
