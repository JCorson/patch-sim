"""Sag and rebound analysis for hyperpolarization multi-sweep simulations.

Provides functions to measure voltage sag (Ih-driven depolarizing drift during a
negative current step) and count rebound spikes at step offset from each sweep of
a multi-sweep current-clamp experiment.

Voltage sag is driven by Ih (HCN channels) and is present only in neurons that
express those channels (cortical pyramidal, thalamic relay, CA1, STN, dopaminergic).

Rebound spikes arise from several distinct biophysical mechanisms depending on the
neuron model:

- **ICaT de-inactivation** (thalamic relay, TRN, STN, Purkinje): sustained
  hyperpolarization removes T-type Ca²⁺ channel inactivation; on release the
  low-threshold ICaT activates and drives a post-inhibitory burst.
- **Ih-driven overshoot** (dopaminergic, cortical pyramidal): Ih activated during
  the step continues to conduct after release, transiently depolarizing the
  membrane above threshold in high-excitability cells.
- **HH anode-break excitation** (squid giant axon, cortical pyramidal): deep
  hyperpolarization fully de-inactivates the Na⁺ h-gate and deactivates the K⁺
  n-gate; on release m activates before h re-inactivates, triggering an action
  potential (Hodgkin & Huxley, 1952).

``rebound_spike_count`` is a mechanism-agnostic counter — it records spikes that
fall within ``rebound_window_ms`` of step offset regardless of which of the above
mechanisms produced them.

Data classes:
    SagPoint: Per-step sag and rebound measurement record.
    HyperpolarizationAnalysisResult: Aggregated analysis output.
"""

import dataclasses

import numpy as np

from patch_sim.analysis.ap_metrics import APAnalysisResult, analyze_aps


@dataclasses.dataclass
class SagPoint:
    """Sag and rebound measurements for a single hyperpolarizing current step.

    Attributes:
        current_step: Injected current amplitude during the step (µA/cm²,
            negative).
        peak_voltage: Most negative membrane voltage reached during the step
            (mV).
        steady_state_voltage: Mean membrane voltage over the last 50 ms of the
            step (mV).
        sag_amplitude: Depolarizing drift during the step in mV
            (``steady_state_voltage − peak_voltage``; ≥ 0 when sag is
            present).
        rebound_spike_count: Number of action potentials detected in the
            ``rebound_window_ms`` immediately following step offset.  This is
            mechanism-agnostic: spikes from ICaT de-inactivation, Ih overshoot,
            or HH anode-break excitation are all counted.
    """

    current_step: float
    peak_voltage: float
    steady_state_voltage: float
    sag_amplitude: float
    rebound_spike_count: int


@dataclasses.dataclass
class HyperpolarizationAnalysisResult:
    """Complete hyperpolarization analysis from a multi-sweep simulation.

    Points are stored in ascending order of current step (most negative first).
    Convenience properties extract each field across all points.

    Attributes:
        points: Per-step measurements, one entry per sweep, sorted by
            ascending (most negative) current amplitude.
    """

    points: list[SagPoint]

    @property
    def current_steps(self) -> list[float]:
        """Injected current amplitudes (µA/cm²), most negative first."""
        return [p.current_step for p in self.points]

    @property
    def sag_amplitudes(self) -> list[float]:
        """Sag amplitude (mV) at each step, sorted with most negative step first."""
        return [p.sag_amplitude for p in self.points]

    @property
    def rebound_spike_counts(self) -> list[int]:
        """Rebound spike count at each step, sorted with most negative step first."""
        return [p.rebound_spike_count for p in self.points]


def _sag_point_from_ap_result(
    time: np.ndarray,
    voltage: np.ndarray,
    ap_result: APAnalysisResult,
    current_step: float,
    stim_start_ms: float,
    stim_end_ms: float,
    rebound_window_ms: float = 50.0,
    steady_state_fraction: float = 0.15,
) -> SagPoint:
    """Build a SagPoint from a pre-computed APAnalysisResult.

    Computes peak and steady-state voltage from the raw voltage array, and
    counts rebound spikes from the already-detected spikes in ``ap_result``,
    avoiding a redundant :func:`analyze_aps` call.

    Args:
        time: Time axis array in ms.
        voltage: Membrane voltage array in mV.
        ap_result: Pre-computed AP analysis result for this sweep.
        current_step: Injected current amplitude during the step (µA/cm²).
        stim_start_ms: Start of the current step in ms.
        stim_end_ms: End of the current step in ms.
        rebound_window_ms: Duration of the post-step window for rebound spike
            counting (ms).
        steady_state_fraction: Trailing fraction of the step used to compute
            the steady-state voltage.

    Returns:
        A :class:`SagPoint` with sag and rebound measurements.
    """
    step_mask = (time >= stim_start_ms) & (time < stim_end_ms)
    step_voltage = voltage[step_mask]
    peak_voltage = float(step_voltage.min())

    ss_duration = (stim_end_ms - stim_start_ms) * steady_state_fraction
    ss_start = stim_end_ms - ss_duration
    ss_mask = (time >= ss_start) & (time < stim_end_ms)
    steady_state_voltage = float(voltage[ss_mask].mean())

    sag_amplitude = max(0.0, steady_state_voltage - peak_voltage)

    rebound_end_ms = stim_end_ms + rebound_window_ms
    rebound_spikes = [
        s for s in ap_result.spikes if stim_end_ms <= s.peak_time <= rebound_end_ms
    ]

    return SagPoint(
        current_step=current_step,
        peak_voltage=peak_voltage,
        steady_state_voltage=steady_state_voltage,
        sag_amplitude=sag_amplitude,
        rebound_spike_count=len(rebound_spikes),
    )


def compute_sag_point(
    time: np.ndarray,
    voltage: np.ndarray,
    current_step: float,
    stim_start_ms: float,
    stim_end_ms: float,
    rebound_window_ms: float = 50.0,
    steady_state_fraction: float = 0.15,
) -> SagPoint:
    """Compute sag and rebound metrics for a single hyperpolarizing sweep.

    Peak voltage is the minimum of the voltage trace during the step.
    Steady-state voltage is the mean over the last ``steady_state_fraction`` of
    the step duration.  Rebound spikes are action potentials whose peak time
    falls within ``[stim_end_ms, stim_end_ms + rebound_window_ms]``.  The count
    is mechanism-agnostic: ICaT de-inactivation, Ih overshoot, and HH anode-break
    excitation all contribute equally.

    Args:
        time: Time axis array in ms.
        voltage: Membrane voltage array in mV, same length as ``time``.
        current_step: Injected current amplitude during the step (µA/cm²).
        stim_start_ms: Start of the current step in ms.
        stim_end_ms: End of the current step in ms.
        rebound_window_ms: Duration of the post-step window in which rebound
            spikes are counted (ms).
        steady_state_fraction: Fraction of the step duration used to compute
            the steady-state voltage (trailing portion of the step).

    Returns:
        A :class:`SagPoint` with sag and rebound measurements for this sweep.
    """
    return _sag_point_from_ap_result(
        time,
        voltage,
        analyze_aps(time, voltage),
        current_step,
        stim_start_ms,
        stim_end_ms,
        rebound_window_ms,
        steady_state_fraction,
    )


def analyze_hyperpolarization(
    time: np.ndarray,
    voltages: list[np.ndarray],
    current_steps: list[float],
    stim_start_ms: float,
    stim_end_ms: float,
    rebound_window_ms: float = 50.0,
) -> HyperpolarizationAnalysisResult:
    """Compute sag and rebound metrics from a multi-sweep hyperpolarization run.

    Calls :func:`compute_sag_point` for each sweep and assembles the results,
    sorted in ascending order of current step (most negative first).  Rebound
    spike counts are mechanism-agnostic; see the module docstring for the
    mechanisms that contribute in different neuron models.

    Args:
        time: Shared time axis in ms (same for all sweeps).
        voltages: List of membrane voltage arrays (mV), one per sweep.
        current_steps: Injected current amplitude (µA/cm²) for each sweep,
            parallel to ``voltages``.
        stim_start_ms: Start of the current step in ms.
        stim_end_ms: End of the current step in ms.
        rebound_window_ms: Duration of the post-step window in which rebound
            spikes are counted (ms).

    Returns:
        A :class:`HyperpolarizationAnalysisResult` with per-step measurements
        sorted by current amplitude (most negative first).
    """
    points = [
        compute_sag_point(
            time, voltage, i_step, stim_start_ms, stim_end_ms, rebound_window_ms
        )
        for voltage, i_step in zip(voltages, current_steps)
    ]
    points.sort(key=lambda p: p.current_step)
    return HyperpolarizationAnalysisResult(points=points)
