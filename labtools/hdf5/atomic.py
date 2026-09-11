"""Atomic replacement with bounded recovery from Windows sharing violations."""

import os
import time


def replace_file(source, destination):
    """Retry a temporarily locked destination; propagate persistent failures.

    Windows readers/indexers may briefly prevent rename even after our own HDF5
    handle has closed. Keep the previous checkpoint intact until rename succeeds.
    The ordinary path performs one rename without any delay.
    """
    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 7:
                raise
            time.sleep(0.01 * 2**attempt)
