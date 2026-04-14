"""Tests for patch_sim.analysis.passive_properties.

Covers subthreshold detection, input resistance computation, time constant
fitting, membrane capacitance derivation, edge cases, and integration with
a real HH simulation.
"""

import numpy as np
import pytest

import patch_sim
from patch_sim.analysis.passive_properties import (
    PassiveProperties,
    analyze_passive_from_result,
    analyze_passive_properties,
    is_subthreshold,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DT = 0.025  # ms — matches SIM_SAMPLING_FREQ = 40 kHz
_PRE_MS = 50.0  # pre-stimulus duration used in integration tests
_STIM_MS = 200.0  # stimulus duration used in integration tests
_POST_MS = 50.0  # post-stimulus duration


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


def _run_subthreshold_sim(
    hh_model: patch_sim.Neuron,
    current_amplitude: float = -1.0,
) -> tuple[patch_sim.SimulationResult, float, float]:
    """Run a short subthreshold current clamp simulation.

    Args:
        hh_model: Neuron instance to simulate.
        current_amplitude: Injected current in µA/cm² (should be subthreshold).

    Returns:
        Tuple of (result, stim_start_ms, stim_end_ms).
    """
    stim_start = _PRE_MS
    stim_end = _PRE_MS + _STIM_MS
    stimulus = patch_sim.step_current(
        duration=_PRE_MS + _STIM_MS + _POST_MS,
        current_amplitude=current_amplitude,
        step_start=stim_start,
        step_duration=_STIM_MS,
    )
    result = patch_sim.simulate_current_clamp(hh_model, stimulus)
    return result, stim_start, stim_end


# ---------------------------------------------------------------------------
# is_subthreshold
# ---------------------------------------------------------------------------


def test_is_subthreshold_flat_trace() -> None:
    """Flat constant-voltage trace contains no spikes and is subthreshold."""
    time, voltage = _flat_trace(duration_ms=100.0, v=-65.0)
    assert is_subthreshold(time, voltage) is True


def test_is_subthreshold_rejects_spiking(hh_model: patch_sim.Neuron) -> None:
    """Suprathreshold stimulus produces spikes; is_subthreshold returns False."""
    stimulus = patch_sim.step_current(
        duration=_PRE_MS + _STIM_MS + _POST_MS,
        current_amplitude=20.0,  # well above threshold
        step_start=_PRE_MS,
        step_duration=_STIM_MS,
    )
    result = patch_sim.simulate_current_clamp(hh_model, stimulus)
    assert is_subthreshold(result["time"], result["voltage"]) is False


# ---------------------------------------------------------------------------
# analyze_passive_properties — guard conditions
# ---------------------------------------------------------------------------


def test_returns_none_for_suprathreshold(hh_model: patch_sim.Neuron) -> None:
    """analyze_passive_properties returns None for a spiking trace."""
    stimulus = patch_sim.step_current(
        duration=_PRE_MS + _STIM_MS + _POST_MS,
        current_amplitude=20.0,
        step_start=_PRE_MS,
        step_duration=_STIM_MS,
    )
    result = patch_sim.simulate_current_clamp(hh_model, stimulus)
    props = analyze_passive_properties(
        result["time"],
        result["voltage"],
        current_amplitude=20.0,
        stim_start_ms=_PRE_MS,
        stim_end_ms=_PRE_MS + _STIM_MS,
    )
    assert props is None


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


# ---------------------------------------------------------------------------
# analyze_passive_properties — HH model integration
# ---------------------------------------------------------------------------


def test_input_resistance_hh_model(hh_model: patch_sim.Neuron) -> None:
    """R_in from a real HH simulation is positive and in a plausible range.

    The HH model at rest has g_L = 0.3 mS/cm² plus active conductances.
    Total conductance is typically 0.5–2.0 mS/cm², giving R_in in the range
    0.5–2.0 kΩ·cm².  This test verifies the sign and order of magnitude only,
    since the exact value depends on gating variables at the new steady state.
    """
    result, stim_start, stim_end = _run_subthreshold_sim(hh_model, -1.0)
    props = analyze_passive_from_result(
        result,
        current_amplitude=-1.0,
        stim_start_ms=stim_start,
        stim_end_ms=stim_end,
    )
    assert props is not None
    assert props.input_resistance > 0.0
    assert props.input_resistance < 5.0  # must be below 1/g_L even with tolerance


def test_time_constant_hh_model(hh_model: patch_sim.Neuron) -> None:
    """τₘ from a real HH simulation is positive and within a plausible range.

    Standard HH: C_m = 1.0 µF/cm², total g ≈ 0.5–1.5 mS/cm², so τₘ is
    expected to be between 0.5 and 20 ms.  The active conductances at rest
    mean the effective τₘ is considerably shorter than the passive C_m / g_L
    ≈ 3.33 ms estimate, so this test verifies the order of magnitude only.
    """
    result, stim_start, stim_end = _run_subthreshold_sim(hh_model, -1.0)
    props = analyze_passive_from_result(
        result,
        current_amplitude=-1.0,
        stim_start_ms=stim_start,
        stim_end_ms=stim_end,
    )
    assert props is not None
    assert props.time_constant > 0.0
    assert props.time_constant < 20.0


def test_membrane_capacitance_hh_model(hh_model: patch_sim.Neuron) -> None:
    """Derived Cₘ from the HH model is positive and in a plausible range.

    Cₘ = τₘ / R_in.  With τₘ < 20 ms and R_in ≈ 0.5–5 kΩ·cm², the derived
    Cₘ should be in the range 0.05–10 µF/cm².  This test verifies sign and
    rough order of magnitude only.
    """
    result, stim_start, stim_end = _run_subthreshold_sim(hh_model, -1.0)
    props = analyze_passive_from_result(
        result,
        current_amplitude=-1.0,
        stim_start_ms=stim_start,
        stim_end_ms=stim_end,
    )
    assert props is not None
    assert props.membrane_capacitance is not None
    assert props.membrane_capacitance > 0.0
    assert props.membrane_capacitance < 10.0


def test_fit_converged_flag_hh_model(hh_model: patch_sim.Neuron) -> None:
    """Exponential fit converges for a clean subthreshold HH sweep."""
    result, stim_start, stim_end = _run_subthreshold_sim(hh_model, -1.0)
    props = analyze_passive_from_result(
        result,
        current_amplitude=-1.0,
        stim_start_ms=stim_start,
        stim_end_ms=stim_end,
    )
    assert props is not None
    assert props.fit_converged is True


def test_analyze_passive_from_result_matches_raw(hh_model: patch_sim.Neuron) -> None:
    """analyze_passive_from_result matches calling the raw function with arrays."""
    result, stim_start, stim_end = _run_subthreshold_sim(hh_model, -1.0)

    from_result = analyze_passive_from_result(
        result,
        current_amplitude=-1.0,
        stim_start_ms=stim_start,
        stim_end_ms=stim_end,
    )
    from_arrays = analyze_passive_properties(
        result["time"],
        result["voltage"],
        current_amplitude=-1.0,
        stim_start_ms=stim_start,
        stim_end_ms=stim_end,
    )

    assert from_result is not None
    assert from_arrays is not None
    assert from_result.input_resistance == pytest.approx(from_arrays.input_resistance)
    assert from_result.time_constant == pytest.approx(from_arrays.time_constant)
    assert from_result.fit_converged == from_arrays.fit_converged
