"""Tests for patch_sim.analysis.iv_curve.

Covers compute_iv_point and analyze_iv with synthetic data,
edge cases, and an integration test against a real multi-sweep simulation.
"""

import numpy as np
import pytest

import patch_sim
from patch_sim.analysis.iv_curve import analyze_iv, compute_iv_point
from patch_sim.analysis.results import IVAnalysisResult, IVPoint

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


def _current_trace(
    time: np.ndarray,
    stim_start_ms: float,
    stim_end_ms: float,
    peak_value: float,
    steady_value: float,
) -> np.ndarray:
    """Create a piecewise constant current trace for testing.

    Outside the stimulus window the current is zero.  During the stimulus it
    is set to ``peak_value`` for the first 90% of the window and
    ``steady_value`` for the last 10%.

    Args:
        time: Time axis in ms.
        stim_start_ms: Start of stimulus window (ms).
        stim_end_ms: End of stimulus window (ms).
        peak_value: Current value during the first 90% of the window.
        steady_value: Current value during the last 10% of the window.

    Returns:
        1-D current array in µA/cm².
    """
    current = np.zeros_like(time)
    i_start = int(np.searchsorted(time, stim_start_ms))
    i_end = int(np.searchsorted(time, stim_end_ms))
    n = i_end - i_start
    split = max(1, n - max(1, n // 10))
    current[i_start : i_start + split] = peak_value
    current[i_start + split : i_end] = steady_value
    return current


# ---------------------------------------------------------------------------
# compute_iv_point
# ---------------------------------------------------------------------------


def test_compute_iv_point_peak_inward():
    """Peak inward current is the minimum value within the stimulus window."""
    time = _make_time(50.0)
    current = _current_trace(time, 10.0, 40.0, -5.0, -1.0)
    point = compute_iv_point(time, current, 0.0, 10.0, 40.0)
    assert point.peak_inward_current == pytest.approx(-5.0)


def test_compute_iv_point_peak_outward():
    """Peak outward current is the maximum value within the stimulus window."""
    time = _make_time(50.0)
    current = _current_trace(time, 10.0, 40.0, 3.0, 1.0)
    point = compute_iv_point(time, current, 0.0, 10.0, 40.0)
    assert point.peak_outward_current == pytest.approx(3.0)


def test_compute_iv_point_steady_state():
    """Steady-state current is the mean of the last 10% of the stimulus window."""
    time = _make_time(50.0)
    current = _current_trace(time, 10.0, 40.0, -5.0, -1.5)
    point = compute_iv_point(time, current, 0.0, 10.0, 40.0)
    assert point.steady_state_current == pytest.approx(-1.5, abs=0.1)


def test_compute_iv_point_voltage_step_stored():
    """The supplied voltage_step is stored on the returned IVPoint."""
    time = _make_time(30.0)
    current = np.zeros_like(time)
    point = compute_iv_point(time, current, -40.0, 5.0, 25.0)
    assert point.voltage_step == pytest.approx(-40.0)


def test_compute_iv_point_returns_ivpoint_instance():
    """compute_iv_point returns an IVPoint dataclass instance."""
    time = _make_time(20.0)
    current = np.full_like(time, 2.0)
    point = compute_iv_point(time, current, 10.0, 5.0, 15.0)
    assert isinstance(point, IVPoint)


def test_compute_iv_point_flat_trace_inward_equals_outward():
    """A constant current trace has equal inward and outward peaks."""
    time = _make_time(30.0)
    current = np.full_like(time, -3.0)
    point = compute_iv_point(time, current, -3.0, 5.0, 25.0)
    assert point.peak_inward_current == pytest.approx(-3.0)
    assert point.peak_outward_current == pytest.approx(-3.0)


def test_compute_iv_point_outside_window_ignored():
    """Current outside the stimulus window does not affect computed metrics."""
    time = _make_time(50.0)
    # Put a large spike outside the window
    current = np.zeros_like(time)
    i_outside = int(np.searchsorted(time, 45.0))
    current[i_outside] = -100.0
    # Stimulus from 10 to 40 ms — all zeros there
    point = compute_iv_point(time, current, 0.0, 10.0, 40.0)
    assert point.peak_inward_current == pytest.approx(0.0)
    assert point.peak_outward_current == pytest.approx(0.0)


def test_compute_iv_point_empty_window_returns_zeros():
    """An impossible stimulus window (start >= end) returns zero metrics."""
    time = _make_time(20.0)
    current = np.full_like(time, -5.0)
    point = compute_iv_point(time, current, 0.0, 15.0, 15.0)
    assert point.peak_inward_current == pytest.approx(0.0)
    assert point.peak_outward_current == pytest.approx(0.0)
    assert point.steady_state_current == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# analyze_iv
# ---------------------------------------------------------------------------


def test_analyze_iv_returns_ivanalysisresult():
    """analyze_iv returns an IVAnalysisResult instance."""
    time = _make_time(30.0)
    currents = [np.zeros_like(time), np.zeros_like(time)]
    result = analyze_iv(time, currents, [-20.0, 20.0], 5.0, 25.0)
    assert isinstance(result, IVAnalysisResult)


def test_analyze_iv_sorted_by_voltage():
    """Results are sorted by ascending voltage regardless of input order."""
    time = _make_time(30.0)
    currents = [
        _current_trace(time, 5.0, 25.0, -1.0, -0.5),
        _current_trace(time, 5.0, 25.0, -3.0, -1.5),
        _current_trace(time, 5.0, 25.0, -2.0, -1.0),
    ]
    voltage_steps = [20.0, -40.0, 0.0]  # Deliberately unsorted
    result = analyze_iv(time, currents, voltage_steps, 5.0, 25.0)
    assert result.voltage_steps == [-40.0, 0.0, 20.0]


def test_analyze_iv_correct_peak_per_voltage():
    """Peak inward current is correctly associated with each voltage step."""
    time = _make_time(30.0)
    # Step at -40 mV: small inward; step at +20 mV: large inward
    c_minus40 = _current_trace(time, 5.0, 25.0, -1.0, -0.5)
    c_plus20 = _current_trace(time, 5.0, 25.0, -5.0, -1.0)
    result = analyze_iv(time, [c_minus40, c_plus20], [-40.0, 20.0], 5.0, 25.0)
    assert result.peak_inward_currents[0] == pytest.approx(-1.0)
    assert result.peak_inward_currents[1] == pytest.approx(-5.0)


def test_analyze_iv_convenience_lists_match_points():
    """Convenience lists on IVAnalysisResult match the points list."""
    time = _make_time(30.0)
    currents = [_current_trace(time, 5.0, 25.0, v, v * 0.1) for v in [-3.0, -1.0, 2.0]]
    result = analyze_iv(time, currents, [-20.0, 0.0, 40.0], 5.0, 25.0)
    for i, point in enumerate(result.points):
        assert result.voltage_steps[i] == point.voltage_step
        assert result.peak_inward_currents[i] == point.peak_inward_current
        assert result.peak_outward_currents[i] == point.peak_outward_current
        assert result.steady_state_currents[i] == point.steady_state_current


def test_analyze_iv_single_sweep():
    """analyze_iv handles a single-sweep input gracefully."""
    time = _make_time(30.0)
    current = _current_trace(time, 5.0, 25.0, -2.0, -0.5)
    result = analyze_iv(time, [current], [0.0], 5.0, 25.0)
    assert len(result.points) == 1
    assert result.voltage_steps[0] == pytest.approx(0.0)
    assert result.peak_inward_currents[0] == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_iv_curve_integration_hh_neuron(hh_model):
    """I-V curve from a real HH simulation has inward current at intermediate voltages.

    The HH model should produce a region of inward (negative) sodium current
    at intermediate depolarised voltage steps (e.g. -10 to +20 mV), and
    predominantly outward potassium current at strong depolarisations.
    """
    pre_ms, stim_ms, post_ms = 5.0, 30.0, 5.0
    min_v, max_v, step_v = -80.0, 60.0, 20.0
    protocol_2d = patch_sim.build_voltage_protocol(
        "Step",
        sampling_frequency=40_000,
        pre_stimulus_duration=pre_ms,
        stimulus_duration=stim_ms,
        post_stimulus_duration=post_ms,
        holding_voltage=-65.0,
        min_stimulus=min_v,
        max_stimulus=max_v,
        stimulus_step=step_v,
    )
    n_steps = round((max_v - min_v) / step_v) + 1
    voltage_steps = list(np.linspace(min_v, max_v, n_steps))
    protocols = [protocol_2d[i] for i in range(protocol_2d.shape[0])]

    results = list(
        patch_sim.simulate_batch(hh_model, protocols, patch_sim.simulate_voltage_clamp)
    )
    time = results[0]["time"]
    currents = [r["Itotal"] for r in results]
    iv = analyze_iv(time, currents, voltage_steps, pre_ms, pre_ms + stim_ms)

    # Should have one entry per voltage step, sorted
    assert len(iv.points) == len(voltage_steps)
    assert iv.voltage_steps == sorted(voltage_steps)

    # At moderately depolarised steps, total peak should have an inward component
    # (the sodium current drives total inward in the -40 to +10 mV range for HH)
    any_inward = any(p < 0 for p in iv.peak_inward_currents)
    assert any_inward, "Expected at least one inward peak current in the HH I-V curve"
