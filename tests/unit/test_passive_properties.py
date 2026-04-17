"""Unit tests for patch_sim.analysis.passive_properties.

Covers subthreshold detection on flat traces, guard conditions with zero/short
stimuli, and exponential fitting on synthetic step responses. Tests that drive
a real HH simulation live in tests/integration/test_passive_properties_simulation.py.
"""

import numpy as np
import pytest

from patch_sim.analysis.passive_properties import (
    analyze_passive_properties,
    is_subthreshold,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DT = 0.025  # ms — matches SIM_SAMPLING_FREQ = 40 kHz
_PRE_MS = 50.0
_STIM_MS = 200.0
_POST_MS = 50.0


def _make_time(duration_ms: float) -> np.ndarray:
    """Create a time array from 0 to duration_ms with dt = 0.025 ms.

    Args:
        duration_ms: Total duration in ms.

    Returns:
        1-D float array of time points in ms.
    """
    return np.arange(0.0, duration_ms, _DT)


def _flat_trace(
    duration_ms: float = 100.0, v: float = -65.0
) -> tuple[np.ndarray, np.ndarray]:
    """Create a constant-voltage trace.

    Args:
        duration_ms: Total duration in ms.
        v: Voltage level in mV.

    Returns:
        Tuple of (time, voltage) arrays.
    """
    t = _make_time(duration_ms)
    return t, np.full_like(t, v)


def _synthetic_step_trace(
    v_baseline: float,
    v_ss: float,
    tau_ms: float,
    pre_ms: float = _PRE_MS,
    stim_ms: float = _STIM_MS,
    post_ms: float = _POST_MS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic voltage trace with an ideal exponential step response.

    The voltage is held at ``v_baseline`` before the step, follows
    ``V(t) = v_ss + (v_baseline - v_ss) * exp(-t / tau_ms)`` during the step,
    and is held at ``v_ss`` after the step.

    Args:
        v_baseline: Baseline voltage before the step (mV).
        v_ss: Steady-state voltage during the step (mV).
        tau_ms: True membrane time constant (ms).
        pre_ms: Duration of the pre-stimulus period (ms).
        stim_ms: Duration of the current step (ms).
        post_ms: Duration of the post-stimulus period (ms).

    Returns:
        Tuple of (time, voltage) arrays.
    """
    total_ms = pre_ms + stim_ms + post_ms
    time = _make_time(total_ms)
    voltage = np.empty_like(time)

    pre_mask = time < pre_ms
    stim_mask = (time >= pre_ms) & (time < pre_ms + stim_ms)
    post_mask = time >= pre_ms + stim_ms

    voltage[pre_mask] = v_baseline
    t_rel = time[stim_mask] - pre_ms
    voltage[stim_mask] = v_ss + (v_baseline - v_ss) * np.exp(-t_rel / tau_ms)
    voltage[post_mask] = v_ss

    return time, voltage


# ---------------------------------------------------------------------------
# is_subthreshold
# ---------------------------------------------------------------------------


def test_is_subthreshold_flat_trace() -> None:
    """Flat constant-voltage trace contains no spikes and is subthreshold."""
    time, voltage = _flat_trace(duration_ms=100.0, v=-65.0)
    assert is_subthreshold(time, voltage) is True


# ---------------------------------------------------------------------------
# analyze_passive_properties — guard conditions
# ---------------------------------------------------------------------------


def test_returns_none_for_zero_current() -> None:
    """analyze_passive_properties returns None when current amplitude is zero."""
    time, voltage = _flat_trace(duration_ms=200.0)
    props = analyze_passive_properties(
        time,
        voltage,
        current_amplitude=0.0,
        stim_start_ms=50.0,
        stim_end_ms=150.0,
    )
    assert props is None


def test_returns_none_for_short_stimulus() -> None:
    """analyze_passive_properties returns None when the stimulus is too short."""
    time, voltage = _flat_trace(duration_ms=200.0)
    props = analyze_passive_properties(
        time,
        voltage,
        current_amplitude=-1.0,
        stim_start_ms=50.0,
        stim_end_ms=51.0,  # only 1 ms — below _MIN_STIM_DURATION_MS
    )
    assert props is None


# ---------------------------------------------------------------------------
# analyze_passive_properties — synthetic exponential trace
# ---------------------------------------------------------------------------


def test_synthetic_exponential_recovers_tau() -> None:
    """Exponential fit recovers the true time constant from a synthetic trace."""
    v_baseline = -65.0
    v_ss = -75.0  # hyperpolarising step
    true_tau = 10.0  # ms
    time, voltage = _synthetic_step_trace(v_baseline, v_ss, true_tau)

    delta_v = v_ss - v_baseline  # -10 mV
    current_amplitude = -2.0  # µA/cm²; R_in = 5 kΩ·cm²
    props = analyze_passive_properties(
        time,
        voltage,
        current_amplitude=current_amplitude,
        stim_start_ms=_PRE_MS,
        stim_end_ms=_PRE_MS + _STIM_MS,
    )
    assert props is not None
    assert props.fit_converged is True
    assert props.time_constant == pytest.approx(true_tau, rel=0.05)
    expected_rin = delta_v / current_amplitude  # 5.0 kΩ·cm²
    assert props.input_resistance == pytest.approx(expected_rin, rel=0.05)


def test_synthetic_exponential_membrane_capacitance() -> None:
    """Derived Cₘ = τ / R_in matches the analytically expected value."""
    v_baseline = -65.0
    v_ss = -75.0
    true_tau = 8.0
    current_amplitude = -2.0
    time, voltage = _synthetic_step_trace(v_baseline, v_ss, true_tau)

    props = analyze_passive_properties(
        time,
        voltage,
        current_amplitude=current_amplitude,
        stim_start_ms=_PRE_MS,
        stim_end_ms=_PRE_MS + _STIM_MS,
    )
    assert props is not None
    assert props.membrane_capacitance is not None
    # R_in = (v_ss - v_baseline) / current_amplitude = (-10) / (-2) = 5 kΩ·cm²
    # C_m = tau / R_in = 8 / 5 = 1.6 µF/cm²
    expected_rin = (v_ss - v_baseline) / current_amplitude  # 5.0
    expected_cm = true_tau / expected_rin  # 1.6
    assert props.membrane_capacitance == pytest.approx(expected_cm, rel=0.05)


def test_depolarising_step_gives_positive_rin() -> None:
    """Depolarising subthreshold step yields a positive R_in."""
    v_baseline = -65.0
    v_ss = -60.0  # depolarising (+5 mV)
    true_tau = 5.0
    current_amplitude = 1.0  # +1 µA/cm²
    time, voltage = _synthetic_step_trace(v_baseline, v_ss, true_tau)

    props = analyze_passive_properties(
        time,
        voltage,
        current_amplitude=current_amplitude,
        stim_start_ms=_PRE_MS,
        stim_end_ms=_PRE_MS + _STIM_MS,
    )
    assert props is not None
    assert props.input_resistance > 0.0


def test_hyperpolarising_step_gives_positive_rin() -> None:
    """Hyperpolarising step (negative current) also yields a positive R_in."""
    v_baseline = -65.0
    v_ss = -75.0  # hyperpolarising (-10 mV)
    true_tau = 5.0
    current_amplitude = -2.0  # -2 µA/cm²; R_in = 5.0 (positive)
    time, voltage = _synthetic_step_trace(v_baseline, v_ss, true_tau)

    props = analyze_passive_properties(
        time,
        voltage,
        current_amplitude=current_amplitude,
        stim_start_ms=_PRE_MS,
        stim_end_ms=_PRE_MS + _STIM_MS,
    )
    assert props is not None
    assert props.input_resistance > 0.0
