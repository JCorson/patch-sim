"""Unit tests for patch_sim.analysis.hyperpolarization with synthetic traces.

All tests use hand-crafted voltage and time arrays to verify the pure
computation logic in isolation from the ODE solver.
"""

import numpy as np
import pytest

from patch_sim.analysis.hyperpolarization import (
    analyze_hyperpolarization,
    compute_sag_point,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DT_MS = 0.025  # 40 kHz sampling


def _make_time(duration_ms: float) -> np.ndarray:
    """Return a uniformly spaced time array in ms.

    Args:
        duration_ms: Total duration of the trace in ms.

    Returns:
        1-D array of time values from 0 to ``duration_ms`` (exclusive) at
        40 kHz resolution.
    """
    return np.arange(0.0, duration_ms, _DT_MS)


def _flat_step_voltage(
    time: np.ndarray,
    v_rest: float,
    v_step: float,
    stim_start_ms: float,
    stim_end_ms: float,
) -> np.ndarray:
    """Build a voltage trace that instantly steps to v_step and stays flat.

    No sag, no rebound — ideal passive response.

    Args:
        time: Time axis in ms.
        v_rest: Resting (pre- and post-step) voltage in mV.
        v_step: Voltage during the step window in mV.
        stim_start_ms: Step onset in ms.
        stim_end_ms: Step offset in ms.

    Returns:
        Voltage array in mV the same length as ``time``.
    """
    v = np.full_like(time, v_rest)
    mask = (time >= stim_start_ms) & (time < stim_end_ms)
    v[mask] = v_step
    return v


def _sag_voltage(
    time: np.ndarray,
    v_rest: float,
    v_peak: float,
    v_steady: float,
    stim_start_ms: float,
    stim_end_ms: float,
    tau_ms: float = 30.0,
) -> np.ndarray:
    """Build a voltage trace with exponential sag from peak toward steady-state.

    Within the step window the voltage starts at ``v_peak`` and decays
    exponentially toward ``v_steady`` with time constant ``tau_ms``.

    Args:
        time: Time axis in ms.
        v_rest: Resting voltage before and after the step (mV).
        v_peak: Initial hyperpolarization peak (most negative voltage, mV).
        v_steady: Steady-state voltage during the step (mV, less negative than
            peak).
        stim_start_ms: Step onset in ms.
        stim_end_ms: Step offset in ms.
        tau_ms: Time constant of the exponential recovery in ms.

    Returns:
        Voltage array in mV the same length as ``time``.
    """
    v = np.full_like(time, v_rest)
    mask = (time >= stim_start_ms) & (time < stim_end_ms)
    t_step = time[mask] - stim_start_ms
    v[mask] = v_steady + (v_peak - v_steady) * np.exp(-t_step / tau_ms)
    return v


def _add_spike(
    voltage: np.ndarray,
    time: np.ndarray,
    spike_time_ms: float,
    spike_width_ms: float = 1.0,
    spike_peak_mv: float = 30.0,
    threshold_mv: float = -40.0,
) -> np.ndarray:
    """Insert a synthetic action potential into a voltage trace.

    Inserts a narrow Gaussian-shaped spike centered at ``spike_time_ms``.
    The spike is placed on top of the existing trace, so the baseline
    reflects the surrounding voltage.

    Args:
        voltage: Voltage array to modify (copy is returned, original unchanged).
        time: Time axis in ms, same length as ``voltage``.
        spike_time_ms: Center of the spike in ms.
        spike_width_ms: Standard deviation of the Gaussian in ms.
        spike_peak_mv: Target peak of the spike in mV.
        threshold_mv: Threshold voltage used to set spike amplitude.

    Returns:
        New voltage array with the spike inserted.
    """
    v = voltage.copy()
    gaussian = np.exp(-0.5 * ((time - spike_time_ms) / spike_width_ms) ** 2)
    v += (spike_peak_mv - threshold_mv) * gaussian
    return v


# ---------------------------------------------------------------------------
# compute_sag_point
# ---------------------------------------------------------------------------


def test_compute_sag_point_flat_no_sag() -> None:
    """Flat hyperpolarizing step with no Ih produces zero sag amplitude.

    The steady-state voltage equals the peak voltage for a perfectly flat
    step (constant displacement, no drift).
    """
    time = _make_time(200.0)
    voltage = _flat_step_voltage(time, -65.0, -85.0, 50.0, 150.0)

    pt = compute_sag_point(time, voltage, -5.0, 50.0, 150.0)

    assert pt.current_step == pytest.approx(-5.0)
    assert pt.peak_voltage == pytest.approx(-85.0, abs=0.1)
    assert pt.steady_state_voltage == pytest.approx(-85.0, abs=0.1)
    assert pt.sag_amplitude == pytest.approx(0.0, abs=0.2)
    assert pt.rebound_spike_count == 0


def test_compute_sag_point_with_sag() -> None:
    """Exponential recovery toward steady-state produces the correct sag amplitude.

    With v_peak=−90 mV and v_steady=−75 mV, the sag amplitude should be 15 mV
    (steady_state − peak = −75 − (−90) = 15 mV).
    """
    time = _make_time(300.0)
    voltage = _sag_voltage(time, -65.0, -90.0, -75.0, 50.0, 250.0, tau_ms=20.0)

    pt = compute_sag_point(time, voltage, -10.0, 50.0, 250.0)

    assert pt.sag_amplitude == pytest.approx(15.0, abs=0.5)
    assert pt.peak_voltage == pytest.approx(-90.0, abs=0.2)
    assert pt.steady_state_voltage == pytest.approx(-75.0, abs=0.5)
    assert pt.rebound_spike_count == 0


def test_compute_sag_point_rebound_spike_in_window() -> None:
    """A spike placed within the rebound window is counted as a rebound spike.

    The default rebound window is 50 ms post step-offset.  A spike at
    stim_end + 10 ms should be detected.
    """
    stim_start, stim_end = 50.0, 250.0
    time = _make_time(400.0)
    voltage = _flat_step_voltage(time, -65.0, -85.0, stim_start, stim_end)
    voltage = _add_spike(voltage, time, spike_time_ms=stim_end + 10.0)

    pt = compute_sag_point(time, voltage, -5.0, stim_start, stim_end)

    assert pt.rebound_spike_count == 1


def test_compute_sag_point_spike_outside_rebound_window_not_counted() -> None:
    """A spike beyond the 50 ms rebound window is not counted as a rebound spike."""
    stim_start, stim_end = 50.0, 250.0
    time = _make_time(500.0)
    voltage = _flat_step_voltage(time, -65.0, -85.0, stim_start, stim_end)
    voltage = _add_spike(voltage, time, spike_time_ms=stim_end + 60.0)

    pt = compute_sag_point(time, voltage, -5.0, stim_start, stim_end)

    assert pt.rebound_spike_count == 0


def test_compute_sag_point_spike_during_step_not_counted_as_rebound() -> None:
    """A spike occurring during the step is not counted as a rebound spike."""
    stim_start, stim_end = 50.0, 250.0
    time = _make_time(400.0)
    voltage = _flat_step_voltage(time, -65.0, -85.0, stim_start, stim_end)
    voltage = _add_spike(voltage, time, spike_time_ms=stim_start + 20.0)

    pt = compute_sag_point(time, voltage, -5.0, stim_start, stim_end)

    assert pt.rebound_spike_count == 0


def test_compute_sag_point_multiple_rebound_spikes() -> None:
    """Multiple spikes within the rebound window are all counted."""
    stim_start, stim_end = 50.0, 250.0
    time = _make_time(500.0)
    voltage = _flat_step_voltage(time, -65.0, -85.0, stim_start, stim_end)
    for spike_t in [stim_end + 5.0, stim_end + 15.0, stim_end + 30.0]:
        voltage = _add_spike(voltage, time, spike_time_ms=spike_t)

    pt = compute_sag_point(time, voltage, -5.0, stim_start, stim_end)

    assert pt.rebound_spike_count == 3


def test_compute_sag_point_custom_rebound_window() -> None:
    """A narrow custom rebound window excludes spikes outside it.

    With rebound_window_ms=5 only the spike at step_end+3 ms is inside the
    window; the spike at step_end+10 ms is excluded.
    """
    stim_start, stim_end = 50.0, 250.0
    time = _make_time(500.0)
    voltage = _flat_step_voltage(time, -65.0, -85.0, stim_start, stim_end)
    voltage = _add_spike(voltage, time, spike_time_ms=stim_end + 3.0)
    voltage = _add_spike(voltage, time, spike_time_ms=stim_end + 10.0)

    pt = compute_sag_point(
        time, voltage, -5.0, stim_start, stim_end, rebound_window_ms=5.0
    )

    assert pt.rebound_spike_count == 1


def test_compute_sag_point_sag_amplitude_is_non_negative() -> None:
    """sag_amplitude is always ≥ 0 (steady-state cannot be more negative than peak).

    For a true hyperpolarizing step the peak is the minimum, so
    steady_state ≥ peak and sag_amplitude ≥ 0.
    """
    time = _make_time(200.0)
    voltage = _flat_step_voltage(time, -65.0, -80.0, 50.0, 150.0)

    pt = compute_sag_point(time, voltage, -3.0, 50.0, 150.0)

    assert pt.sag_amplitude >= 0.0


# ---------------------------------------------------------------------------
# analyze_hyperpolarization
# ---------------------------------------------------------------------------


def test_analyze_hyperpolarization_sorts_by_current() -> None:
    """Points are returned in ascending (most negative first) current order.

    Regardless of the order the sweeps are passed, the result should be sorted
    by current_step from most negative to least negative.
    """
    time = _make_time(200.0)
    sweeps = [
        _flat_step_voltage(time, -65.0, -65.0 + amp * 3, 50.0, 150.0)
        for amp in [-2.0, -5.0, -8.0]
    ]
    current_steps = [-2.0, -5.0, -8.0]

    result = analyze_hyperpolarization(time, sweeps, current_steps, 50.0, 150.0)

    steps = result.current_steps
    assert steps == sorted(steps), f"Expected sorted steps, got {steps}"
    assert steps[0] == pytest.approx(-8.0)
    assert steps[-1] == pytest.approx(-2.0)


def test_analyze_hyperpolarization_properties_match_points() -> None:
    """Convenience properties extract the correct fields from each point."""
    time = _make_time(300.0)
    current_steps = [-3.0, -6.0]
    sweeps = [
        _sag_voltage(time, -65.0, -70.0, -68.0, 50.0, 250.0),
        _sag_voltage(time, -65.0, -80.0, -73.0, 50.0, 250.0),
    ]

    result = analyze_hyperpolarization(time, sweeps, current_steps, 50.0, 250.0)

    assert result.current_steps == [pt.current_step for pt in result.points]
    assert result.sag_amplitudes == [pt.sag_amplitude for pt in result.points]
    assert result.rebound_spike_counts == [
        pt.rebound_spike_count for pt in result.points
    ]


def test_analyze_hyperpolarization_single_sweep() -> None:
    """A single-sweep hyperpolarization run returns one SagPoint."""
    time = _make_time(200.0)
    voltage = _flat_step_voltage(time, -65.0, -80.0, 50.0, 150.0)

    result = analyze_hyperpolarization(time, [voltage], [-5.0], 50.0, 150.0)

    assert len(result.points) == 1
    assert result.points[0].current_step == pytest.approx(-5.0)
