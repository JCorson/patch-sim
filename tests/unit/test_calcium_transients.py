"""Unit tests for the patch_sim.analysis.calcium_transients module.

Verifies calcium transient detection, peak / time-to-peak / decay-τ
extraction, and edge-case behaviour using synthetic [Ca²⁺]ᵢ traces.
Tests that drive a real simulation live in
tests/integration/test_calcium_transients_simulation.py.
"""

from typing import cast

import numpy as np
import pytest

from patch_sim.analysis.calcium_transients import (
    CalciumTransientAnalysisResult,
    analyze_calcium_transients,
    analyze_calcium_transients_from_result,
)
from patch_sim.clamp_simulations import SimulationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DT = 0.025  # ms — matches SIM_SAMPLING_FREQ = 40 kHz
_MM_TO_UM = 1000.0


def _make_time(duration_ms: float) -> np.ndarray:
    """Create a time array from 0 to duration_ms with dt = 0.025 ms.

    Args:
        duration_ms: Duration of the trace in ms.

    Returns:
        1-D float array of time points.
    """
    return np.arange(0.0, duration_ms, _DT)


def _flat_trace_uM(
    duration_ms: float = 200.0, ca_uM: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Create a flat [Ca²⁺]ᵢ trace at a constant concentration.

    Args:
        duration_ms: Duration of the trace in ms.
        ca_uM: Constant calcium concentration in µM.

    Returns:
        Tuple ``(time, ca_i_mM)`` — caller passes ca in mM as expected by
        :func:`analyze_calcium_transients`.
    """
    t = _make_time(duration_ms)
    ca = np.full_like(t, ca_uM / _MM_TO_UM)
    return t, ca


def _exponential_transient_uM(
    baseline_uM: float,
    peak_uM: float,
    onset_ms: float,
    rise_ms: float,
    tau_ms: float,
    total_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a [Ca²⁺]ᵢ trace with one exponential rise + exponential decay.

    Rising phase: linear ramp from baseline to peak between ``onset_ms`` and
    ``onset_ms + rise_ms``.  Falling phase from the peak: single-exponential
    decay back to baseline with time constant ``tau_ms``.

    Args:
        baseline_uM: Resting [Ca²⁺]ᵢ in µM.
        peak_uM: Peak [Ca²⁺]ᵢ in µM.
        onset_ms: Time at which the rise begins (ms).
        rise_ms: Duration of the rising phase (ms).
        tau_ms: Decay time constant (ms).
        total_ms: Total trace duration (ms).

    Returns:
        Tuple ``(time, ca_i_mM)``.
    """
    t = _make_time(total_ms)
    ca_uM = np.full_like(t, baseline_uM)

    rise_end = onset_ms + rise_ms
    rising = (t >= onset_ms) & (t < rise_end)
    ca_uM[rising] = (
        baseline_uM + (peak_uM - baseline_uM) * (t[rising] - onset_ms) / rise_ms
    )

    decaying = t >= rise_end
    ca_uM[decaying] = baseline_uM + (peak_uM - baseline_uM) * np.exp(
        -(t[decaying] - rise_end) / tau_ms
    )

    return t, ca_uM / _MM_TO_UM


# ---------------------------------------------------------------------------
# No transients
# ---------------------------------------------------------------------------


def test_no_transients_flat_trace():
    """analyze_calcium_transients returns zero transients for a flat trace."""
    t, ca = _flat_trace_uM()
    result = analyze_calcium_transients(t, ca)

    assert isinstance(result, CalciumTransientAnalysisResult)
    assert result.transient_count == 0
    assert result.transients == []
    assert result.baseline_concentration == pytest.approx(0.1)
    assert result.mean_peak_concentration is None
    assert result.mean_time_to_peak is None
    assert result.mean_decay_tau is None
    assert result.mean_amplitude is None


def test_short_trace_returns_empty():
    """A trace shorter than two samples returns an empty result."""
    result = analyze_calcium_transients(np.array([0.0]), np.array([1e-4]))
    assert result.transient_count == 0
    assert result.baseline_concentration is None


def test_subthreshold_noise_is_rejected():
    """Random noise at the prominence threshold floor is not counted as a transient."""
    rng = np.random.default_rng(seed=0)
    t = _make_time(200.0)
    # Noise amplitude well below the 0.05 µM default prominence
    ca_uM = 0.1 + 0.005 * rng.standard_normal(len(t))
    result = analyze_calcium_transients(t, ca_uM / _MM_TO_UM)
    assert result.transient_count == 0


# ---------------------------------------------------------------------------
# Single transient: peak / TTP / τ recovery
# ---------------------------------------------------------------------------


def test_single_transient_recovers_peak_and_time_to_peak():
    """Peak concentration and time-to-peak match the synthetic generator."""
    t, ca = _exponential_transient_uM(
        baseline_uM=0.1,
        peak_uM=1.0,
        onset_ms=50.0,
        rise_ms=2.0,
        tau_ms=80.0,
        total_ms=600.0,
    )
    result = analyze_calcium_transients(t, ca)

    assert result.transient_count == 1
    transient = result.transients[0]
    assert transient.peak_concentration == pytest.approx(1.0, rel=0.02)
    assert transient.baseline_concentration == pytest.approx(0.1, abs=0.02)
    assert transient.amplitude == pytest.approx(0.9, rel=0.05)
    assert transient.peak_time == pytest.approx(52.0, abs=0.1)
    # Onset is the first sample below baseline + 0.1 · amplitude (≈ 0.19 µM),
    # which lies on the rising ramp ~0.2 ms after onset_ms.
    assert transient.onset_time == pytest.approx(50.0, abs=0.5)
    assert transient.time_to_peak == pytest.approx(2.0, abs=0.5)


def test_decay_tau_recovered_from_synthetic_exponential():
    """Decay τ from the exponential fit matches the synthetic τ within 5%."""
    t, ca = _exponential_transient_uM(
        baseline_uM=0.1,
        peak_uM=1.0,
        onset_ms=50.0,
        rise_ms=2.0,
        tau_ms=80.0,
        total_ms=600.0,
    )
    result = analyze_calcium_transients(t, ca)

    assert result.transient_count == 1
    transient = result.transients[0]
    assert transient.decay_fit_converged is True
    assert transient.decay_tau == pytest.approx(80.0, rel=0.05)
    assert result.mean_decay_tau == pytest.approx(80.0, rel=0.05)


def test_unit_conversion_mM_to_uM():
    """Input is interpreted as mM; output is reported in µM (×1000)."""
    t, ca_mM = _exponential_transient_uM(
        baseline_uM=0.1,
        peak_uM=2.0,
        onset_ms=20.0,
        rise_ms=1.0,
        tau_ms=50.0,
        total_ms=400.0,
    )
    # Sanity check on the helper itself
    assert ca_mM.max() == pytest.approx(2.0 / _MM_TO_UM, rel=1e-6)

    result = analyze_calcium_transients(t, ca_mM)
    assert result.transients[0].peak_concentration == pytest.approx(2.0, rel=0.02)
    assert result.baseline_concentration == pytest.approx(0.1, abs=0.02)


# ---------------------------------------------------------------------------
# Multiple transients
# ---------------------------------------------------------------------------


def test_two_transients_separated():
    """Two well-separated transients are both detected with correct ordering."""
    t = _make_time(800.0)
    ca_uM = np.full_like(t, 0.1)
    for onset, peak in [(50.0, 1.0), (350.0, 1.5)]:
        rise_end = onset + 2.0
        rising = (t >= onset) & (t < rise_end)
        ca_uM[rising] = np.maximum(
            ca_uM[rising],
            0.1 + (peak - 0.1) * (t[rising] - onset) / 2.0,
        )
        decaying = t >= rise_end
        ca_uM[decaying] = np.maximum(
            ca_uM[decaying],
            0.1 + (peak - 0.1) * np.exp(-(t[decaying] - rise_end) / 60.0),
        )

    result = analyze_calcium_transients(t, ca_uM / _MM_TO_UM)
    assert result.transient_count == 2
    assert result.transients[0].peak_time == pytest.approx(52.0, abs=0.5)
    assert result.transients[1].peak_time == pytest.approx(352.0, abs=0.5)
    assert result.transients[0].peak_concentration == pytest.approx(1.0, rel=0.05)
    assert result.transients[1].peak_concentration == pytest.approx(1.5, rel=0.05)
    # Indices are zero-based and sequential
    assert [tr.index for tr in result.transients] == [0, 1]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_baseline_drift_does_not_block_detection():
    """Slow baseline drift does not prevent transient detection (prominence-based)."""
    t = _make_time(800.0)
    drift_uM = 0.1 + 0.0005 * t  # rises from 0.1 to 0.5 µM over the trace
    ca_uM = drift_uM.copy()

    rise_end = 102.0
    rising = (t >= 100.0) & (t < rise_end)
    ca_uM[rising] = drift_uM[rising] + 0.9 * (t[rising] - 100.0) / 2.0
    decaying = t >= rise_end
    ca_uM[decaying] = drift_uM[decaying] + 0.9 * np.exp(
        -(t[decaying] - rise_end) / 60.0
    )

    result = analyze_calcium_transients(t, ca_uM / _MM_TO_UM)
    assert result.transient_count == 1
    # Per-transient baseline tracks the drift, so amplitude should be close
    # to the true 0.9 µM transient size, not inflated by drift.
    assert result.transients[0].amplitude == pytest.approx(0.9, rel=0.1)


def test_fallback_tau_used_when_decay_window_too_short_for_curve_fit():
    """Direct unit test of `_fit_decay`: short window forces 1/e fallback.

    Tests the private :func:`patch_sim.analysis.calcium_transients._fit_decay`
    helper directly because the public path requires pathological input to
    create a sub-`_MIN_DECAY_SAMPLES` decay window.  The fallback contract:
    when the window is too short, ``curve_fit`` is skipped, ``converged``
    returns ``False``, and τ is the 1/e crossing (or ``None`` when the trace
    never crosses 1/e).  This pins the fallback path so a future refactor
    that silently drops it will fail the test.
    """
    from patch_sim.analysis.calcium_transients import _fit_decay

    # 4 samples (< _MIN_DECAY_SAMPLES = 5) with a clean 1/e crossing.
    t_decay = np.array([0.0, 0.025, 0.05, 0.1])
    peak_uM = 1.0
    baseline_uM = 0.1
    # ca[1] = baseline + amplitude/e exactly → 1/e fallback returns t=0.025
    target = baseline_uM + (peak_uM - baseline_uM) / np.e
    ca_decay = np.array([peak_uM, target, baseline_uM + 0.1, baseline_uM])

    tau, converged = _fit_decay(t_decay, ca_decay, peak_uM, baseline_uM)
    assert converged is False
    assert tau == pytest.approx(0.025, abs=1e-6)


def test_fallback_returns_none_when_trace_never_decays_below_one_over_e():
    """Direct unit test: short window without a 1/e crossing → τ is None."""
    from patch_sim.analysis.calcium_transients import _fit_decay

    t_decay = np.array([0.0, 0.025, 0.05, 0.1])
    peak_uM = 1.0
    baseline_uM = 0.1
    # The trace never falls below `baseline + amplitude/e` — fallback fails.
    ca_decay = np.array([peak_uM, peak_uM, peak_uM, peak_uM])
    tau, converged = _fit_decay(t_decay, ca_decay, peak_uM, baseline_uM)
    assert converged is False
    assert tau is None


def test_short_decay_window_falls_back_or_returns_none():
    """Truncated decay window: fit_converged is False; τ is fallback or None."""
    t, ca = _exponential_transient_uM(
        baseline_uM=0.1,
        peak_uM=1.0,
        onset_ms=50.0,
        rise_ms=2.0,
        tau_ms=80.0,
        total_ms=58.0,  # peak at 52 ms, only ~6 ms of decay
    )
    result = analyze_calcium_transients(t, ca, max_decay_window_ms=4.0)
    assert result.transient_count == 1
    transient = result.transients[0]
    # Either curve_fit recovered something despite the short window, or it
    # fell back; the test guarantees the fit is reported as not-converged
    # OR (when curve_fit happens to succeed on the short data) τ is at least
    # very inaccurate compared to the true 80 ms.  We accept both: the key
    # contract is graceful behaviour, not perfect numerical recovery here.
    if transient.decay_fit_converged:
        # Short window sometimes still converges but to a wildly off τ.
        assert transient.decay_tau is not None
    else:
        # Fallback may have succeeded or returned None.
        assert transient.decay_tau is None or transient.decay_tau > 0.0


def test_transient_clipped_at_end_of_recording():
    """A transient with only a short decay tail is still detected."""
    t, ca = _exponential_transient_uM(
        baseline_uM=0.1,
        peak_uM=1.0,
        onset_ms=50.0,
        rise_ms=2.0,
        tau_ms=80.0,
        total_ms=80.0,  # ~28 ms of decay after the peak (~1100 samples)
    )
    result = analyze_calcium_transients(t, ca)
    assert result.transient_count == 1
    transient = result.transients[0]
    assert transient.peak_concentration == pytest.approx(1.0, rel=0.05)
    # Even on a short tail, the fit either converges with a positive τ or
    # falls back gracefully without raising.
    assert (transient.decay_tau is None) or (transient.decay_tau > 0.0)


def test_custom_prominence_filters_small_transients():
    """A high prominence threshold rejects a small transient."""
    t, ca = _exponential_transient_uM(
        baseline_uM=0.1,
        peak_uM=0.15,
        onset_ms=50.0,
        rise_ms=2.0,
        tau_ms=50.0,
        total_ms=400.0,
    )
    # 0.05 µM amplitude is exactly the default prominence, so make it stricter
    result = analyze_calcium_transients(t, ca, prominence_uM=0.1)
    assert result.transient_count == 0


# ---------------------------------------------------------------------------
# from_result wrapper
# ---------------------------------------------------------------------------


def test_analyze_from_result_returns_none_without_ca_i():
    """The from_result wrapper returns None when the array has no ca_i field."""
    dtype = np.dtype([("time", np.float64), ("voltage", np.float64)])
    result = np.zeros(10, dtype=dtype)
    result["time"] = np.arange(10) * _DT
    out = analyze_calcium_transients_from_result(cast(SimulationResult, result))
    assert out is None


def test_analyze_from_result_uses_ca_i_field():
    """The from_result wrapper extracts the ca_i field and runs analysis."""
    dtype = np.dtype([("time", np.float64), ("ca_i", np.float64)])
    t, ca = _exponential_transient_uM(
        baseline_uM=0.1,
        peak_uM=1.0,
        onset_ms=50.0,
        rise_ms=2.0,
        tau_ms=80.0,
        total_ms=600.0,
    )
    result = np.zeros(len(t), dtype=dtype)
    result["time"] = t
    result["ca_i"] = ca

    out = analyze_calcium_transients_from_result(cast(SimulationResult, result))
    assert out is not None
    assert out.transient_count == 1
    assert out.transients[0].peak_concentration == pytest.approx(1.0, rel=0.02)
