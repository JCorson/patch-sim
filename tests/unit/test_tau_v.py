"""Unit tests for patch_sim.analysis.tau_v.

Covers the exponential model functions, single/double-exp selection logic,
and per-sweep / multi-sweep entry points using synthetic data.  Integration
tests against real HH simulations live in
tests/integration/test_tau_v_simulation.py.
"""

import numpy as np
import pytest

from patch_sim.analysis.tau_v import (
    analyze_tau_v,
    compute_tau_v_point,
    double_exp_decay,
    single_exp_decay,
    single_exp_rise,
)

# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------


def test_single_exp_rise_at_zero_returns_offset():
    """single_exp_rise(0, A, tau, C) equals C for any A, tau."""
    assert single_exp_rise(0.0, A=1.0, tau=5.0, C=-2.0) == pytest.approx(-2.0)


def test_single_exp_rise_at_infinity_approaches_a_plus_offset():
    """single_exp_rise at large t approaches A + C."""
    val = single_exp_rise(1000.0, A=3.0, tau=5.0, C=1.0)
    assert val == pytest.approx(4.0, abs=1e-6)


def test_single_exp_decay_at_zero_returns_amplitude_plus_offset():
    """single_exp_decay(0, A, tau, C) equals A + C."""
    assert single_exp_decay(0.0, A=2.0, tau=5.0, C=1.0) == pytest.approx(3.0)


def test_single_exp_decay_at_infinity_approaches_offset():
    """single_exp_decay at large t approaches C."""
    val = single_exp_decay(1000.0, A=2.0, tau=5.0, C=1.0)
    assert val == pytest.approx(1.0, abs=1e-6)


def test_double_exp_decay_at_zero_returns_sum_plus_offset():
    """double_exp_decay(0, A1, tau1, A2, tau2, C) equals A1 + A2 + C."""
    val = double_exp_decay(0.0, A1=1.0, tau1=2.0, A2=0.5, tau2=20.0, C=0.1)
    assert val == pytest.approx(1.6)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synth_sweep(
    rise_amp: float = -10.0,
    rise_tau: float = 0.8,
    decay_tau: float = 8.0,
    baseline: float = 0.0,
    asymptote: float | None = None,
    stim_start: float = 5.0,
    stim_end: float = 105.0,
    pre_post_padding: float = 5.0,
    sample_freq_khz: float = 50.0,
    rng_seed: int = 0,
    noise_std: float = 0.0,
    rise_window_ms: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic voltage-clamp sweep with controllable rise + decay.

    The sweep is flat at ``baseline`` before ``stim_start``, rises with
    ``single_exp_rise(rise_amp, rise_tau)`` over the first ``rise_window_ms``
    of the step, then decays with ``single_exp_decay`` toward ``asymptote``
    until ``stim_end``, then flat at ``asymptote`` afterwards.

    Args:
        rise_amp: Amplitude of the rising phase (signed).
        rise_tau: τ_activation of the rising phase (ms).
        decay_tau: τ_inactivation of the decaying phase (ms).
        baseline: Pre-step current (µA/cm²).
        asymptote: Steady-state current at the end of the decay; defaults to
            ``baseline``.
        stim_start: Start of the step window (ms).
        stim_end: End of the step window (ms).
        pre_post_padding: Padding before / after the step (ms).
        sample_freq_khz: Sampling frequency (kHz).
        rng_seed: Seed for additive Gaussian noise.
        noise_std: Standard deviation of additive Gaussian noise.
        rise_window_ms: Duration of the rising phase before decay begins.

    Returns:
        A 2-tuple ``(time, current)`` of NumPy arrays.
    """
    if asymptote is None:
        asymptote = baseline
    t_total_start = stim_start - pre_post_padding
    t_total_end = stim_end + pre_post_padding
    dt = 1.0 / sample_freq_khz
    n = int(round((t_total_end - t_total_start) / dt)) + 1
    t = np.linspace(t_total_start, t_total_end, n)
    i = np.full_like(t, baseline, dtype=float)
    rise_end = stim_start + rise_window_ms
    rise_mask = (t >= stim_start) & (t < rise_end)
    i[rise_mask] = single_exp_rise(
        t[rise_mask] - stim_start, rise_amp, rise_tau, baseline
    )
    decay_mask = (t >= rise_end) & (t <= stim_end)
    peak_value = float(
        single_exp_rise(rise_end - stim_start, rise_amp, rise_tau, baseline)
    )
    decay_amp_used = peak_value - asymptote
    i[decay_mask] = single_exp_decay(
        t[decay_mask] - rise_end, decay_amp_used, decay_tau, asymptote
    )
    post_mask = t > stim_end
    i[post_mask] = asymptote
    if noise_std > 0.0:
        rng = np.random.default_rng(rng_seed)
        i = i + rng.normal(0.0, noise_std, size=i.size)
    return t, i


# ---------------------------------------------------------------------------
# compute_tau_v_point — recovery of known parameters
# ---------------------------------------------------------------------------


def test_compute_tau_v_point_recovers_known_activation_tau():
    """A clean rise with τ=0.5 ms is recovered to within 5%."""
    rise_tau = 0.5
    t, i = _make_synth_sweep(rise_amp=-10.0, rise_tau=rise_tau, decay_tau=20.0)
    pt = compute_tau_v_point(
        time=t,
        current=i,
        voltage_step=-20.0,
        stim_start_ms=5.0,
        stim_end_ms=105.0,
    )
    assert pt.activation_converged is True
    assert pt.tau_activation_ms is not None
    assert pt.tau_activation_ms == pytest.approx(rise_tau, rel=0.05)


def test_compute_tau_v_point_recovers_known_inactivation_tau():
    """A clean decay with τ=8 ms is recovered to within 5%."""
    decay_tau = 8.0
    t, i = _make_synth_sweep(
        rise_amp=-12.0,
        rise_tau=0.5,
        decay_tau=decay_tau,
        rise_window_ms=2.0,
    )
    pt = compute_tau_v_point(
        time=t,
        current=i,
        voltage_step=-20.0,
        stim_start_ms=5.0,
        stim_end_ms=105.0,
    )
    assert pt.inactivation_converged is True
    assert pt.tau_inactivation_ms is not None
    assert pt.tau_inactivation_ms == pytest.approx(decay_tau, rel=0.05)
    assert pt.inactivation_is_double is False


def test_compute_tau_v_point_voltage_step_preserved():
    """The voltage_step argument passes through to the returned point."""
    t, i = _make_synth_sweep()
    pt = compute_tau_v_point(t, i, 42.0, 5.0, 105.0)
    assert pt.voltage_step == 42.0


# ---------------------------------------------------------------------------
# compute_tau_v_point — edge cases
# ---------------------------------------------------------------------------


def test_compute_tau_v_point_no_inactivation_returns_none():
    """Sustained current (peak ≈ end value) yields tau_inactivation_ms is None."""
    # decay_amp=0 makes the post-peak trace flat → no inactivation phase
    t, i = _make_synth_sweep(
        rise_amp=-10.0,
        rise_tau=0.5,
        decay_tau=8.0,
        asymptote=-10.0,  # asymptote equals plateau → no decay
        rise_window_ms=5.0,
    )
    pt = compute_tau_v_point(t, i, -20.0, 5.0, 105.0)
    assert pt.tau_inactivation_ms is None
    assert pt.inactivation_converged is False


def test_compute_tau_v_point_window_too_short_returns_none_taus():
    """A stimulus window with fewer than _MIN_FIT_POINTS samples yields None taus."""
    # 50 kHz sampling gives 0.02 ms per sample; a 0.05 ms window has ~3 samples
    t = np.linspace(0.0, 1.0, 51)  # dt = 0.02 ms
    i = np.zeros_like(t)
    pt = compute_tau_v_point(
        time=t,
        current=i,
        voltage_step=-20.0,
        stim_start_ms=0.50,
        stim_end_ms=0.55,
    )
    assert pt.tau_activation_ms is None
    assert pt.tau_inactivation_ms is None
    assert pt.activation_converged is False
    assert pt.inactivation_converged is False


def test_compute_tau_v_point_handles_constant_current_gracefully():
    """A flat current trace yields None taus without raising."""
    t = np.linspace(0.0, 100.0, 5001)
    i = np.full_like(t, -5.0)
    pt = compute_tau_v_point(t, i, -20.0, 5.0, 95.0)
    # No activation rise, no inactivation drop → both None
    assert pt.tau_activation_ms is None
    assert pt.activation_converged is False
    assert pt.tau_inactivation_ms is None
    assert pt.inactivation_converged is False


def test_compute_tau_v_point_fit_inactivation_false_skips_decay():
    """fit_inactivation=False reports tau_inactivation_ms as None for clean decay."""
    t, i = _make_synth_sweep(rise_tau=0.5, decay_tau=8.0)
    pt = compute_tau_v_point(
        time=t,
        current=i,
        voltage_step=-20.0,
        stim_start_ms=5.0,
        stim_end_ms=105.0,
        fit_inactivation=False,
    )
    assert pt.tau_inactivation_ms is None
    assert pt.inactivation_converged is False
    assert pt.activation_converged is True


# ---------------------------------------------------------------------------
# compute_tau_v_point — double-exponential acceptance
# ---------------------------------------------------------------------------


def _make_biexp_sweep(
    a_fast: float,
    tau_fast: float,
    a_slow: float,
    tau_slow: float,
    baseline: float = 0.0,
    asymptote: float = 0.0,
    stim_start: float = 5.0,
    stim_end: float = 205.0,
    rise_window_ms: float = 1.0,
    rise_tau: float = 0.3,
    sample_freq_khz: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a sweep whose decay is the sum of two exponentials.

    Args:
        a_fast: Fast component amplitude.
        tau_fast: Fast time constant (ms).
        a_slow: Slow component amplitude.
        tau_slow: Slow time constant (ms).
        baseline: Pre-step current.
        asymptote: Asymptote of the decay.
        stim_start: Start of the step (ms).
        stim_end: End of the step (ms).
        rise_window_ms: Activation rise window (ms).
        rise_tau: τ for the rising phase (ms).
        sample_freq_khz: Sampling frequency (kHz).

    Returns:
        A 2-tuple ``(time, current)`` of NumPy arrays.
    """
    pre_post_padding = 5.0
    dt = 1.0 / sample_freq_khz
    n = (
        int(round((stim_end + pre_post_padding - (stim_start - pre_post_padding)) / dt))
        + 1
    )
    t = np.linspace(stim_start - pre_post_padding, stim_end + pre_post_padding, n)
    i = np.full_like(t, baseline, dtype=float)
    peak_value = a_fast + a_slow + asymptote
    rise_end = stim_start + rise_window_ms
    rise_mask = (t >= stim_start) & (t < rise_end)
    i[rise_mask] = single_exp_rise(
        t[rise_mask] - stim_start, peak_value - baseline, rise_tau, baseline
    )
    decay_mask = (t >= rise_end) & (t <= stim_end)
    i[decay_mask] = double_exp_decay(
        t[decay_mask] - rise_end,
        a_fast,
        tau_fast,
        a_slow,
        tau_slow,
        asymptote,
    )
    post_mask = t > stim_end
    i[post_mask] = asymptote
    return t, i


def test_compute_tau_v_point_double_exp_accepted_when_clearly_biexponential():
    """A clean A1*exp(-t/2) + A2*exp(-t/30) decay is recovered as double-exp."""
    t, i = _make_biexp_sweep(
        a_fast=-7.0,
        tau_fast=2.0,
        a_slow=-3.0,
        tau_slow=30.0,
        stim_end=305.0,
    )
    pt = compute_tau_v_point(t, i, -20.0, 5.0, 305.0)
    assert pt.inactivation_converged is True
    assert pt.inactivation_is_double is True
    assert pt.tau_inactivation_ms is not None
    assert pt.tau_inactivation_slow_ms is not None
    assert pt.tau_inactivation_ms == pytest.approx(2.0, rel=0.10)
    assert pt.tau_inactivation_slow_ms == pytest.approx(30.0, rel=0.10)


def test_compute_tau_v_point_double_exp_rejected_when_marginal():
    """A pure single-exponential decay is not promoted to a double-exp fit."""
    t, i = _make_synth_sweep(
        rise_amp=-10.0,
        rise_tau=0.5,
        decay_tau=8.0,
        rise_window_ms=2.0,
    )
    pt = compute_tau_v_point(t, i, -20.0, 5.0, 105.0)
    assert pt.inactivation_converged is True
    assert pt.inactivation_is_double is False
    assert pt.tau_inactivation_slow_ms is None


# ---------------------------------------------------------------------------
# analyze_tau_v
# ---------------------------------------------------------------------------


def test_analyze_tau_v_sorts_points_by_voltage():
    """analyze_tau_v sorts its output ascending by voltage_step."""
    t, i = _make_synth_sweep()
    voltages = [40.0, -20.0, 0.0]
    currents = [i, i, i]
    result = analyze_tau_v(t, currents, voltages, 5.0, 105.0)
    assert result.voltage_steps == [-20.0, 0.0, 40.0]


def test_analyze_tau_v_handles_mixed_success_and_failure():
    """One sweep with failed activation does not affect successful sweeps."""
    t_good, i_good = _make_synth_sweep(rise_tau=0.5, decay_tau=8.0)
    # Sweep with a flat trace (no rise, no decay) → both taus None
    t_bad = t_good
    i_bad = np.full_like(t_bad, -2.0)
    voltages = [-40.0, -20.0]
    result = analyze_tau_v(t_good, [i_bad, i_good], voltages, 5.0, 105.0)
    points_by_v = {p.voltage_step: p for p in result.points}
    assert points_by_v[-20.0].tau_activation_ms is not None
    assert points_by_v[-40.0].tau_inactivation_ms is None


def test_analyze_tau_v_empty_currents_returns_empty_result():
    """analyze_tau_v with no sweeps returns an empty result without raising."""
    t = np.linspace(0.0, 10.0, 101)
    result = analyze_tau_v(t, [], [], 0.0, 10.0)
    assert result.points == []
    assert result.voltage_steps == []


def test_tau_v_result_property_lists_match_points():
    """The convenience properties on TauVAnalysisResult agree with .points."""
    t, i = _make_synth_sweep(rise_tau=0.5, decay_tau=8.0)
    voltages = [-40.0, -20.0]
    result = analyze_tau_v(t, [i, i], voltages, 5.0, 105.0)
    assert len(result.tau_activation_values) == 2
    assert len(result.tau_inactivation_values) == 2
    assert len(result.tau_inactivation_slow_values) == 2
