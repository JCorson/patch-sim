"""Unit tests for patch_sim.analysis.fi_curve.

Covers compute_fi_point, analyze_fi, and estimate_rheobase with synthetic
voltage traces and edge cases.
Integration tests against real HH simulations live in
tests/integration/test_fi_curve_simulation.py.
"""

import numpy as np
import pytest

from patch_sim.analysis.fi_curve import (
    FIAnalysisResult,
    FIPoint,
    analyze_fi,
    compute_fi_point,
    estimate_rheobase,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DT = 0.025  # ms — matches SIM_SAMPLING_FREQ = 40 kHz


def _make_time(duration_ms: float) -> np.ndarray:
    """Create a time array from 0 to duration_ms with dt = 0.025 ms.

    Args:
        duration_ms: Total duration in ms.

    Returns:
        1-D float array of time points in ms.
    """
    return np.arange(0.0, duration_ms, _DT)


def _make_spike_voltage(
    time: np.ndarray,
    spike_times_ms: list[float],
    baseline: float = -65.0,
    spike_height: float = 80.0,
    spike_width_ms: float = 1.0,
) -> np.ndarray:
    """Create a synthetic voltage trace with rectangular spikes.

    Outside spike windows the voltage is held at ``baseline``.  Each spike is
    a rectangular bump from ``baseline`` to ``baseline + spike_height`` lasting
    ``spike_width_ms``.  The default ``spike_height`` of 80 mV places the peak
    at 15 mV, comfortably above the ``analyze_aps`` rejection threshold of
    -20 mV.

    Args:
        time: Time axis in ms.
        spike_times_ms: Peak times for each spike in ms.
        baseline: Resting membrane potential in mV.
        spike_height: Height of each rectangular spike above baseline in mV.
        spike_width_ms: Duration of each rectangular spike in ms.

    Returns:
        1-D voltage array in mV.
    """
    voltage = np.full_like(time, baseline)
    for t_peak in spike_times_ms:
        mask = (time >= t_peak - spike_width_ms / 2) & (
            time < t_peak + spike_width_ms / 2
        )
        voltage[mask] = baseline + spike_height
    return voltage


def _make_fi_result(
    spike_counts: list[int], current_steps: list[float]
) -> FIAnalysisResult:
    """Build a synthetic FIAnalysisResult with given spike counts and current steps.

    Firing rate fields are left as None throughout — they are not needed for
    rheobase estimation.

    Args:
        spike_counts: Number of spikes for each point, parallel to current_steps.
        current_steps: Injected current amplitude for each point (µA/cm²).

    Returns:
        An :class:`FIAnalysisResult` with one :class:`FIPoint` per entry.
    """
    points = [
        FIPoint(
            current_step=step,
            spike_count=count,
            mean_firing_rate=None,
            initial_firing_rate=None,
            steady_state_firing_rate=None,
        )
        for step, count in zip(current_steps, spike_counts)
    ]
    return FIAnalysisResult(points=points)


# ---------------------------------------------------------------------------
# compute_fi_point — basic attributes
# ---------------------------------------------------------------------------


def test_compute_fi_point_returns_fipoint_instance():
    """compute_fi_point returns an FIPoint dataclass instance."""
    time = _make_time(50.0)
    voltage = np.full_like(time, -65.0)
    point = compute_fi_point(time, voltage, 5.0, 10.0, 40.0)
    assert isinstance(point, FIPoint)


def test_compute_fi_point_current_step_stored():
    """The supplied current_step is stored on the returned FIPoint."""
    time = _make_time(30.0)
    voltage = np.full_like(time, -65.0)
    point = compute_fi_point(time, voltage, 12.5, 5.0, 25.0)
    assert point.current_step == pytest.approx(12.5)


# ---------------------------------------------------------------------------
# compute_fi_point — no spikes
# ---------------------------------------------------------------------------


def test_compute_fi_point_no_spikes_zero_count():
    """A silent trace produces spike_count of 0."""
    time = _make_time(50.0)
    voltage = np.full_like(time, -65.0)
    point = compute_fi_point(time, voltage, 0.0, 10.0, 40.0)
    assert point.spike_count == 0


def test_compute_fi_point_no_spikes_rates_none():
    """A silent trace has all firing rate fields set to None."""
    time = _make_time(50.0)
    voltage = np.full_like(time, -65.0)
    point = compute_fi_point(time, voltage, 0.0, 10.0, 40.0)
    assert point.mean_firing_rate is None
    assert point.initial_firing_rate is None
    assert point.steady_state_firing_rate is None


# ---------------------------------------------------------------------------
# compute_fi_point — single spike
# ---------------------------------------------------------------------------


def test_compute_fi_point_single_spike_count():
    """A trace with one in-window spike produces spike_count of 1."""
    time = _make_time(50.0)
    voltage = _make_spike_voltage(time, [20.0])
    point = compute_fi_point(time, voltage, 5.0, 10.0, 40.0)
    assert point.spike_count == 1


def test_compute_fi_point_single_spike_rates_none():
    """A single in-window spike produces all firing rate fields as None."""
    time = _make_time(50.0)
    voltage = _make_spike_voltage(time, [20.0])
    point = compute_fi_point(time, voltage, 5.0, 10.0, 40.0)
    assert point.mean_firing_rate is None
    assert point.initial_firing_rate is None
    assert point.steady_state_firing_rate is None


# ---------------------------------------------------------------------------
# compute_fi_point — two spikes (initial == steady-state)
# ---------------------------------------------------------------------------


def test_compute_fi_point_two_spikes_count():
    """A trace with two in-window spikes produces spike_count of 2."""
    time = _make_time(100.0)
    voltage = _make_spike_voltage(time, [20.0, 40.0])
    point = compute_fi_point(time, voltage, 5.0, 10.0, 80.0)
    assert point.spike_count == 2


def test_compute_fi_point_two_spikes_mean_rate():
    """Mean firing rate for two spikes equals 1000 / ISI."""
    time = _make_time(100.0)
    # Peaks at 20 ms and 40 ms → ISI = 20 ms → rate = 50 Hz
    voltage = _make_spike_voltage(time, [20.0, 40.0])
    point = compute_fi_point(time, voltage, 5.0, 10.0, 80.0)
    assert point.mean_firing_rate == pytest.approx(50.0, rel=0.05)


def test_compute_fi_point_two_spikes_initial_equals_steady():
    """With two spikes there is one ISI so initial and steady-state rates are equal."""
    time = _make_time(100.0)
    voltage = _make_spike_voltage(time, [20.0, 40.0])
    point = compute_fi_point(time, voltage, 5.0, 10.0, 80.0)
    assert point.initial_firing_rate == pytest.approx(point.steady_state_firing_rate)


# ---------------------------------------------------------------------------
# compute_fi_point — three spikes (distinct initial and steady-state)
# ---------------------------------------------------------------------------


def test_compute_fi_point_three_spikes_initial_from_first_isi():
    """Initial firing rate is derived from the first ISI."""
    time = _make_time(150.0)
    # Peaks at 20, 30, 60 ms → ISIs = [10, 30] ms
    # Initial rate = 1000/10 = 100 Hz
    voltage = _make_spike_voltage(time, [20.0, 30.0, 60.0])
    point = compute_fi_point(time, voltage, 5.0, 10.0, 120.0)
    assert point.initial_firing_rate == pytest.approx(100.0, rel=0.05)


def test_compute_fi_point_three_spikes_steady_from_last_isi():
    """Steady-state firing rate is derived from the last ISI."""
    time = _make_time(150.0)
    # ISIs = [10, 30] ms → steady-state rate = 1000/30 ≈ 33.3 Hz
    voltage = _make_spike_voltage(time, [20.0, 30.0, 60.0])
    point = compute_fi_point(time, voltage, 5.0, 10.0, 120.0)
    assert point.steady_state_firing_rate == pytest.approx(1000.0 / 30.0, rel=0.05)


# ---------------------------------------------------------------------------
# compute_fi_point — window filtering
# ---------------------------------------------------------------------------


def test_compute_fi_point_spike_outside_window_excluded():
    """Spikes whose threshold time falls outside the stimulus window are excluded."""
    time = _make_time(100.0)
    # Spike at 5 ms is before stim_start=10; spike at 50 ms is inside window
    voltage = _make_spike_voltage(time, [5.0, 50.0])
    point = compute_fi_point(time, voltage, 5.0, 10.0, 80.0)
    # Only the in-window spike at 50 ms counts
    assert point.spike_count == 1


# ---------------------------------------------------------------------------
# analyze_fi
# ---------------------------------------------------------------------------


def test_analyze_fi_returns_fianalysisresult():
    """analyze_fi returns an FIAnalysisResult instance."""
    time = _make_time(50.0)
    voltages = [np.full_like(time, -65.0), np.full_like(time, -65.0)]
    result = analyze_fi(time, voltages, [0.0, 5.0], 10.0, 40.0)
    assert isinstance(result, FIAnalysisResult)


def test_analyze_fi_point_count_matches_sweeps():
    """analyze_fi produces one FIPoint per sweep."""
    time = _make_time(50.0)
    n = 4
    voltages = [np.full_like(time, -65.0)] * n
    current_steps = [float(i) for i in range(n)]
    result = analyze_fi(time, voltages, current_steps, 10.0, 40.0)
    assert len(result.points) == n


def test_analyze_fi_sorted_by_current():
    """analyze_fi sorts points by ascending current step regardless of input order."""
    time = _make_time(50.0)
    voltages = [np.full_like(time, -65.0)] * 3
    # Supply in reverse order
    result = analyze_fi(time, voltages, [10.0, 5.0, 0.0], 10.0, 40.0)
    steps = result.current_steps
    assert steps == sorted(steps)


def test_analyze_fi_convenience_properties_length():
    """Convenience list properties have length equal to the number of points."""
    time = _make_time(50.0)
    n = 3
    voltages = [np.full_like(time, -65.0)] * n
    result = analyze_fi(time, voltages, [0.0, 5.0, 10.0], 10.0, 40.0)
    assert len(result.mean_firing_rates) == n
    assert len(result.initial_firing_rates) == n
    assert len(result.steady_state_firing_rates) == n


# ---------------------------------------------------------------------------
# estimate_rheobase — unit tests
# ---------------------------------------------------------------------------


def test_estimate_rheobase_returns_first_spiking_step():
    """estimate_rheobase returns the current step of the first spiking point.

    With two silent steps followed by one spiking step, the rheobase equals
    the current step of that first spiking point.
    """
    fi = _make_fi_result(spike_counts=[0, 0, 3], current_steps=[5.0, 10.0, 15.0])
    assert estimate_rheobase(fi) == pytest.approx(15.0)


def test_estimate_rheobase_all_silent_returns_none():
    """estimate_rheobase returns None when no sweep produced a spike."""
    fi = _make_fi_result(spike_counts=[0, 0, 0], current_steps=[1.0, 2.0, 3.0])
    assert estimate_rheobase(fi) is None


def test_estimate_rheobase_all_spiking_returns_lowest():
    """estimate_rheobase returns the lowest current step when all sweeps spike."""
    fi = _make_fi_result(spike_counts=[2, 4, 6], current_steps=[5.0, 10.0, 20.0])
    assert estimate_rheobase(fi) == pytest.approx(5.0)


def test_estimate_rheobase_single_spike_sufficient():
    """A sweep with exactly one spike qualifies as suprathreshold.

    spike_count == 1 means no ISI-derived firing rate, but the spike still
    constitutes rheobase-level excitation.
    """
    fi = _make_fi_result(spike_counts=[0, 1, 5], current_steps=[5.0, 10.0, 15.0])
    result = estimate_rheobase(fi)
    assert result == pytest.approx(10.0)


def test_estimate_rheobase_empty_points_returns_none():
    """estimate_rheobase returns None for an FIAnalysisResult with no points."""
    fi = FIAnalysisResult(points=[])
    assert estimate_rheobase(fi) is None
