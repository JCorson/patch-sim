"""Unit tests for the patch_sim.analysis.ap_metrics module.

Verifies spike detection and per-spike metric extraction using synthetic traces:
no spikes, single spike, parameter sensitivity. Tests that drive a real HH
simulation live in tests/integration/test_ap_metrics_simulation.py.
"""

import numpy as np
import pytest

from patch_sim.analysis.ap_metrics import (
    APAnalysisResult,
    analyze_aps,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DT = 0.025  # ms, matches SIM_SAMPLING_FREQ = 40 kHz


def _make_time(duration_ms: float) -> np.ndarray:
    """Create a time array from 0 to duration_ms with dt = 0.025 ms.

    Args:
        duration_ms: Duration of the trace in ms.

    Returns:
        1-D float array of time points.
    """
    return np.arange(0.0, duration_ms, _DT)


def _flat_trace(
    duration_ms: float = 100.0, v: float = -65.0
) -> tuple[np.ndarray, np.ndarray]:
    """Create a flat voltage trace at a constant voltage.

    Args:
        duration_ms: Duration of the trace in ms.
        v: Voltage level in mV.

    Returns:
        Tuple of (time, voltage) arrays.
    """
    t = _make_time(duration_ms)
    return t, np.full_like(t, v)


def _triangular_spike(
    baseline: float = -65.0,
    peak: float = 40.0,
    onset_ms: float = 10.0,
    rise_ms: float = 0.5,
    fall_ms: float = 0.5,
    total_ms: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a voltage trace containing one perfectly triangular spike.

    The spike rises linearly from ``baseline`` to ``peak`` in ``rise_ms`` ms,
    then falls linearly back to ``baseline`` in ``fall_ms`` ms.

    Args:
        baseline: Resting voltage in mV.
        peak: Peak voltage of the spike in mV.
        onset_ms: Time of spike onset (threshold crossing) in ms.
        rise_ms: Duration of the rising phase in ms.
        fall_ms: Duration of the falling phase in ms.
        total_ms: Total trace duration in ms.

    Returns:
        Tuple of (time, voltage) arrays.
    """
    t = _make_time(total_ms)
    v = np.full_like(t, baseline)
    rise_end = onset_ms + rise_ms
    fall_end = rise_end + fall_ms
    # Rising phase
    mask_rise = (t >= onset_ms) & (t < rise_end)
    v[mask_rise] = baseline + (peak - baseline) * (t[mask_rise] - onset_ms) / rise_ms
    # Peak sample
    mask_peak = t == rise_end
    v[mask_peak] = peak
    # Falling phase
    mask_fall = (t > rise_end) & (t <= fall_end)
    v[mask_fall] = peak - (peak - baseline) * (t[mask_fall] - rise_end) / fall_ms
    return t, v


# ---------------------------------------------------------------------------
# No spikes
# ---------------------------------------------------------------------------


def test_no_spikes_flat_trace():
    """analyze_aps returns zero spikes for a flat trace at resting potential."""
    t, v = _flat_trace()
    result = analyze_aps(t, v)

    assert isinstance(result, APAnalysisResult)
    assert result.spike_count == 0
    assert result.spikes == []
    assert result.isis == []
    assert result.mean_threshold_voltage is None
    assert result.mean_peak_voltage is None
    assert result.mean_rise_time is None
    assert result.mean_half_width is None
    assert result.mean_ahp_depth is None
    assert result.mean_isi is None
    assert result.firing_rate is None


def test_no_spikes_subthreshold_oscillation():
    """analyze_aps rejects events that never reach min_spike_height."""
    t = _make_time(50.0)
    # Oscillation that peaks at -30 mV — below default min_spike_height of -20 mV
    v = -50.0 + 20.0 * np.sin(2 * np.pi * 0.1 * t)  # peak ≈ -30 mV
    result = analyze_aps(t, v, min_spike_height=-20.0)

    assert result.spike_count == 0


def test_empty_trace():
    """analyze_aps handles a trace with fewer than 2 samples gracefully."""
    result = analyze_aps(np.array([0.0]), np.array([-65.0]))
    assert result.spike_count == 0


# ---------------------------------------------------------------------------
# Single spike
# ---------------------------------------------------------------------------


def test_single_spike_detection():
    """analyze_aps detects exactly one spike in a trace with one AP."""
    t, v = _triangular_spike()
    result = analyze_aps(t, v)

    assert result.spike_count == 1
    assert len(result.spikes) == 1
    assert result.isis == []
    assert result.mean_isi is None
    assert result.firing_rate is None


def test_single_spike_peak_voltage():
    """Peak voltage is extracted near the true spike apex."""
    t, v = _triangular_spike(peak=40.0)
    result = analyze_aps(t, v)

    assert result.spikes[0].peak_voltage == pytest.approx(40.0, abs=1.0)
    assert result.mean_peak_voltage == pytest.approx(40.0, abs=1.0)


def test_single_spike_rise_time():
    """Rise time is approximately equal to the constructed rise duration."""
    rise_ms = 0.5
    t, v = _triangular_spike(rise_ms=rise_ms)
    result = analyze_aps(t, v)

    # Allow ±2 samples of tolerance (~0.05 ms)
    assert result.spikes[0].rise_time == pytest.approx(rise_ms, abs=0.05)
    assert result.mean_rise_time == pytest.approx(rise_ms, abs=0.05)


def test_single_spike_ahp_depth():
    """AHP depth equals the post-spike minimum (baseline) for a single spike."""
    baseline = -65.0
    t, v = _triangular_spike(baseline=baseline)
    result = analyze_aps(t, v)

    # After the spike the trace returns to baseline
    assert result.spikes[0].ahp_depth == pytest.approx(baseline, abs=1.0)
    assert result.mean_ahp_depth == pytest.approx(baseline, abs=1.0)


def test_single_spike_half_width_geometry():
    """Half-width of a symmetric triangular spike equals half the total spike duration.

    For a symmetric triangle (rise_ms == fall_ms), the half-amplitude level
    bisects each slope at the midpoint, so the half-width should be half the
    total (rise + fall) duration.
    """
    rise_ms = 0.5
    fall_ms = 0.5
    t, v = _triangular_spike(rise_ms=rise_ms, fall_ms=fall_ms)
    result = analyze_aps(t, v)

    expected_hw = (rise_ms + fall_ms) / 2.0
    assert result.spikes[0].half_width == pytest.approx(expected_hw, abs=0.06)
    assert result.mean_half_width == pytest.approx(expected_hw, abs=0.06)


# ---------------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------------


def test_min_spike_height_filters_events():
    """Events below min_spike_height are excluded from results."""
    t, v = _triangular_spike(peak=5.0)  # peak at +5 mV

    # With default min_spike_height=-20 mV the +5 mV peak passes
    result_permissive = analyze_aps(t, v, min_spike_height=-20.0)
    # With min_spike_height=10 mV the +5 mV peak is rejected
    result_strict = analyze_aps(t, v, min_spike_height=10.0)

    assert result_permissive.spike_count == 1
    assert result_strict.spike_count == 0
