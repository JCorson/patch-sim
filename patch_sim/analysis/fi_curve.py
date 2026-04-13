"""F-I curve analysis for current clamp multi-sweep simulations.

Provides functions to detect spikes and compute firing rates from each current
step of a multi-sweep current clamp experiment and assemble the results into
an :class:`FIAnalysisResult`.

Data classes:
    FIPoint: Per-step firing rate measurement record.
    FIAnalysisResult: Aggregated F-I analysis output.
"""

import dataclasses

import numpy as np

from patch_sim.analysis.ap_metrics import APAnalysisResult, analyze_aps


@dataclasses.dataclass
class FIPoint:
    """Firing rate measurements for a single current step in an F-I protocol.

    Attributes:
        current_step: Injected current amplitude during the step (µA/cm²).
        spike_count: Number of spikes detected within the stimulus window.
        mean_firing_rate: Mean firing rate in Hz (``1000 / mean_ISI``), or
            ``None`` when fewer than two spikes were detected.
        initial_firing_rate: Initial firing rate in Hz (``1000 / first_ISI``),
            or ``None`` when fewer than two spikes were detected.
        steady_state_firing_rate: Steady-state firing rate in Hz
            (``1000 / last_ISI``), or ``None`` when fewer than two spikes
            were detected.
    """

    current_step: float
    spike_count: int
    mean_firing_rate: float | None
    initial_firing_rate: float | None
    steady_state_firing_rate: float | None


@dataclasses.dataclass
class FIAnalysisResult:
    """Complete F-I analysis from a multi-sweep current clamp simulation.

    Points are stored in ascending order of current step.  The convenience
    properties ``current_steps``, ``mean_firing_rates``,
    ``initial_firing_rates``, and ``steady_state_firing_rates`` extract the
    corresponding field from each point on demand.

    Attributes:
        points: Per-step measurements, one entry per current step, sorted by
            ascending current amplitude.
    """

    points: list[FIPoint]

    @property
    def current_steps(self) -> list[float]:
        """Injected current amplitudes in µA/cm², sorted ascending."""
        return [p.current_step for p in self.points]

    @property
    def mean_firing_rates(self) -> list[float | None]:
        """Mean firing rate at each step (Hz), or ``None`` for silent steps."""
        return [p.mean_firing_rate for p in self.points]

    @property
    def initial_firing_rates(self) -> list[float | None]:
        """Initial firing rate at each step (Hz), or ``None`` for silent steps."""
        return [p.initial_firing_rate for p in self.points]

    @property
    def steady_state_firing_rates(self) -> list[float | None]:
        """Steady-state firing rate at each step (Hz), or ``None`` for silent steps."""
        return [p.steady_state_firing_rate for p in self.points]


def _fi_point_from_ap_result(
    ap_result: APAnalysisResult,
    current_step: float,
    stim_start_ms: float,
    stim_end_ms: float,
) -> FIPoint:
    """Build an FIPoint from a pre-computed APAnalysisResult.

    Restricts the analysis to spikes whose threshold time falls within the
    stimulus window ``[stim_start_ms, stim_end_ms]`` and derives firing rates
    from inter-spike intervals (ISIs) between consecutive spike peak times:

    - Mean firing rate: ``1000 / mean(ISIs)``
    - Initial firing rate: ``1000 / ISIs[0]`` (first ISI)
    - Steady-state firing rate: ``1000 / ISIs[-1]`` (last ISI)

    All three rates are ``None`` when fewer than two in-window spikes are
    detected.

    Args:
        ap_result: Pre-computed AP analysis result for this sweep.
        current_step: Injected current amplitude during the step (µA/cm²).
        stim_start_ms: Start of the stimulus window in ms.
        stim_end_ms: End of the stimulus window in ms.

    Returns:
        An :class:`FIPoint` with spike count and firing rate measurements for
        this current step.
    """
    in_window = [
        s for s in ap_result.spikes if stim_start_ms <= s.threshold_time <= stim_end_ms
    ]
    spike_count = len(in_window)

    if spike_count < 2:
        return FIPoint(
            current_step=current_step,
            spike_count=spike_count,
            mean_firing_rate=None,
            initial_firing_rate=None,
            steady_state_firing_rate=None,
        )

    peak_times = [s.peak_time for s in in_window]
    isis = [peak_times[i + 1] - peak_times[i] for i in range(len(peak_times) - 1)]

    return FIPoint(
        current_step=current_step,
        spike_count=spike_count,
        mean_firing_rate=1000.0 / float(np.mean(isis)),
        initial_firing_rate=1000.0 / isis[0],
        steady_state_firing_rate=1000.0 / isis[-1],
    )


def compute_fi_point(
    time: np.ndarray,
    voltage: np.ndarray,
    current_step: float,
    stim_start_ms: float,
    stim_end_ms: float,
) -> FIPoint:
    """Compute F-I metrics for a single current step sweep.

    Detects spikes using :func:`~patch_sim.analysis.ap_metrics.analyze_aps`
    and delegates to :func:`_fi_point_from_ap_result` for the rate computation.

    Args:
        time: Time axis array in ms.
        voltage: Membrane voltage array in mV, same length as ``time``.
        current_step: Injected current amplitude during the step (µA/cm²).
        stim_start_ms: Start of the stimulus window in ms.
        stim_end_ms: End of the stimulus window in ms.

    Returns:
        An :class:`FIPoint` with spike count and firing rate measurements for
        this current step.
    """
    return _fi_point_from_ap_result(
        analyze_aps(time, voltage), current_step, stim_start_ms, stim_end_ms
    )


def estimate_rheobase(fi_result: FIAnalysisResult) -> float | None:
    """Estimate the rheobase from F-I analysis results.

    The rheobase is the minimum injected current that elicits at least one
    action potential.  It is identified as the lowest current step with a
    non-zero spike count.  A single spike qualifies as suprathreshold even
    though it produces no ISI-derived firing rate.

    Args:
        fi_result: F-I analysis result.  Points may be in any order.

    Returns:
        The rheobase current in µA/cm² (the ``current_step`` of the first
        :class:`FIPoint` with ``spike_count >= 1``), or ``None`` when no
        sweep produced a spike.
    """
    spiking = [p.current_step for p in fi_result.points if p.spike_count >= 1]
    return min(spiking) if spiking else None


def analyze_fi(
    time: np.ndarray,
    voltages: list[np.ndarray],
    current_steps: list[float],
    stim_start_ms: float,
    stim_end_ms: float,
) -> FIAnalysisResult:
    """Compute F-I curve from a multi-sweep current clamp simulation.

    Calls :func:`compute_fi_point` for each sweep and assembles the results,
    sorted in ascending order of current step.

    Args:
        time: Shared time axis in ms (same for all sweeps).
        voltages: List of membrane voltage arrays (mV), one per sweep.
        current_steps: Injected current amplitude (µA/cm²) for each sweep,
            parallel to ``voltages``.
        stim_start_ms: Start of the stimulus window in ms.
        stim_end_ms: End of the stimulus window in ms.

    Returns:
        An :class:`FIAnalysisResult` with per-step measurements sorted by
        current amplitude.
    """
    points = [
        compute_fi_point(time, voltage, i_step, stim_start_ms, stim_end_ms)
        for voltage, i_step in zip(voltages, current_steps)
    ]
    points.sort(key=lambda p: p.current_step)
    return FIAnalysisResult(points=points)
