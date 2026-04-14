"""Derivative utilities for patch-clamp simulation results."""

import numpy as np
import numpy.typing as npt


def compute_dvdt(
    time: npt.ArrayLike,
    voltage: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute dV/dt from time and voltage arrays.

    Uses ``np.gradient`` (central differences) so the output length matches
    the input length, avoiding the off-by-one issue of ``np.diff``.

    Args:
        time: Time array in ms (length N).
        voltage: Membrane voltage array in mV (length N).

    Returns:
        Tuple of ``(voltage, dvdt)`` arrays, both length N.  ``voltage`` is
        the input converted to an ``np.ndarray``, returned for convenience
        when building phase-plane plots.  ``dvdt`` is in mV/ms.  If
        ``len(time) < 2``, both returned arrays are empty.
    """
    time = np.asarray(time, dtype=float)
    voltage = np.asarray(voltage, dtype=float)
    if len(time) < 2:
        return np.empty(0), np.empty(0)
    dvdt = np.gradient(voltage, time)
    return voltage, dvdt
