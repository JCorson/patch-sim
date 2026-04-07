"""I-V curve analysis for voltage clamp multi-sweep simulations.

Provides functions to extract peak (inward and outward) and steady-state
currents from each voltage step of a multi-sweep voltage clamp experiment
and assemble the results into an :class:`IVAnalysisResult`.
"""

import numpy as np

from .results import IVAnalysisResult, IVPoint


def compute_iv_point(
    time: np.ndarray,
    current: np.ndarray,
    voltage_step: float,
    stim_start_ms: float,
    stim_end_ms: float,
) -> IVPoint:
    """Compute I-V metrics for a single voltage step sweep.

    The stimulus window is identified using the provided timing boundaries.
    Peak inward current is the minimum (most negative) value in the window;
    peak outward current is the maximum (most positive) value.  Steady-state
    current is the mean of the last 10% of the stimulus window.

    Args:
        time: Time axis array in ms.
        current: Total ionic current array in µA/cm², same length as ``time``.
        voltage_step: Command voltage applied during the step (mV).
        stim_start_ms: Start of the stimulus window in ms.
        stim_end_ms: End of the stimulus window in ms.

    Returns:
        An :class:`IVPoint` with peak inward, peak outward, and steady-state
        current measurements for this voltage step.
    """
    i_start = int(np.searchsorted(time, stim_start_ms))
    i_end = int(np.searchsorted(time, stim_end_ms))
    window = current[i_start:i_end]

    if len(window) == 0:
        return IVPoint(
            voltage_step=voltage_step,
            peak_inward_current=0.0,
            peak_outward_current=0.0,
            steady_state_current=0.0,
        )

    peak_inward = float(np.min(window))
    peak_outward = float(np.max(window))

    n_ss = max(1, len(window) // 10)
    steady_state = float(np.mean(window[-n_ss:]))

    return IVPoint(
        voltage_step=voltage_step,
        peak_inward_current=peak_inward,
        peak_outward_current=peak_outward,
        steady_state_current=steady_state,
    )


def analyze_iv(
    time: np.ndarray,
    currents: list[np.ndarray],
    voltage_steps: list[float],
    stim_start_ms: float,
    stim_end_ms: float,
) -> IVAnalysisResult:
    """Compute I-V curve from a multi-sweep voltage clamp simulation.

    Calls :func:`compute_iv_point` for each sweep and assembles the results,
    sorted in ascending order of voltage step.

    Args:
        time: Shared time axis in ms (same for all sweeps).
        currents: List of total ionic current arrays (µA/cm²), one per sweep.
        voltage_steps: Command voltage (mV) for each sweep, parallel to
            ``currents``.
        stim_start_ms: Start of the stimulus window in ms.
        stim_end_ms: End of the stimulus window in ms.

    Returns:
        An :class:`IVAnalysisResult` with per-step measurements sorted by
        voltage.
    """
    points = [
        compute_iv_point(time, current, v, stim_start_ms, stim_end_ms)
        for current, v in zip(currents, voltage_steps)
    ]
    points.sort(key=lambda p: p.voltage_step)
    return IVAnalysisResult(points=points)
