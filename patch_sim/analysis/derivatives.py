"""Derivative utilities for patch-clamp simulation results."""

import numpy as np
import numpy.typing as npt


def compute_dvdt(
    time: npt.ArrayLike,
    voltage: npt.ArrayLike,
) -> np.ndarray:
    """Compute dV/dt from time and voltage arrays.

    Uses ``np.gradient`` (central differences) so the output length matches
    the input length, avoiding the off-by-one issue of ``np.diff``.

    Args:
        time: Time array in ms (length N).
        voltage: Membrane voltage array in mV (length N).

    Raises:
        ValueError: If ``time`` and ``voltage`` have different lengths.

    Returns:
        ``dvdt`` array in mV/ms with the same length as ``voltage``.
        If ``len(time) < 2``, an empty array is returned.
    """
    time = np.asarray(time, dtype=float)
    voltage = np.asarray(voltage, dtype=float)
    if len(time) != len(voltage):
        raise ValueError(
            f"time and voltage must have the same length; "
            f"got {len(time)} and {len(voltage)}"
        )
    if len(time) < 2:
        return np.empty(0)
    return np.gradient(voltage, time)
