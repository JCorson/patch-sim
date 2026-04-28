"""Unit tests for the patch_sim.analysis.burst_metrics module.

Verifies burst detection, threshold estimation, per-burst and aggregate
metric computation, and edge-case behaviour using synthetic
:class:`APAnalysisResult` instances.  Tests that drive a real simulation
live in tests/integration/test_burst_metrics_simulation.py.
"""

from typing import cast

import numpy as np
import pytest

from patch_sim.analysis.ap_metrics import APAnalysisResult, SpikeMetrics
from patch_sim.analysis.burst_metrics import (
    BurstAnalysisResult,
    analyze_bursts,
    analyze_bursts_from_result,
)
from patch_sim.clamp_simulations import SimulationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLD_MS = 100.0


def _make_ap_result(peak_times: list[float]) -> APAnalysisResult:
    """Build a synthetic :class:`APAnalysisResult` from a list of peak times.

    The :class:`SpikeMetrics` records carry only the fields needed by the
    burst analyser (``peak_time``).  Other fields are populated with
    plausible placeholder values so the dataclass remains valid.

    Args:
        peak_times: Spike peak times (ms) in chronological order.

    Returns:
        A populated :class:`APAnalysisResult` whose ``isis`` are computed
        from consecutive ``peak_times``.
    """
    spikes = [
        SpikeMetrics(
            index=i,
            threshold_voltage=-50.0,
            threshold_time=t - 0.5,
            peak_voltage=30.0,
            peak_time=float(t),
            rise_time=0.5,
            half_width=1.0,
            ahp_depth=-70.0,
        )
        for i, t in enumerate(peak_times)
    ]
    isis = [peak_times[i + 1] - peak_times[i] for i in range(len(peak_times) - 1)]
    return APAnalysisResult(
        spike_count=len(spikes),
        spikes=spikes,
        isis=isis,
        mean_threshold_voltage=-50.0 if spikes else None,
        mean_peak_voltage=30.0 if spikes else None,
        mean_rise_time=0.5 if spikes else None,
        mean_half_width=1.0 if spikes else None,
        mean_ahp_depth=-70.0 if spikes else None,
        mean_isi=float(np.mean(isis)) if isis else None,
        firing_rate=(1000.0 / float(np.mean(isis))) if isis else None,
    )


def _triangular_voltage(time: np.ndarray, peak_times: list[float]) -> np.ndarray:
    """Build a synthetic voltage trace with triangular spikes at ``peak_times``.

    Each spike is a 2 ms-wide triangular peak rising from -70 mV to +30 mV
    so :func:`analyze_aps` will detect it via dV/dt.

    Args:
        time: Time array in ms (uniformly sampled).
        peak_times: Locations of spike peaks (ms).

    Returns:
        Voltage array in mV with the same shape as ``time``.
    """
    voltage = np.full_like(time, -70.0)
    half_width_ms = 0.5
    for pt in peak_times:
        mask = np.abs(time - pt) <= half_width_ms
        if not np.any(mask):
            continue
        # Linear ramp peak.
        local = 1.0 - np.abs(time[mask] - pt) / half_width_ms
        voltage[mask] = -70.0 + 100.0 * local
    return voltage


# ---------------------------------------------------------------------------
# Edge cases: empty / single-spike / single-burst
# ---------------------------------------------------------------------------


def test_no_spikes_returns_empty_result() -> None:
    """Zero spikes should yield zero bursts and the default fixed threshold."""
    ap = _make_ap_result([])
    result = analyze_bursts(ap, total_duration_ms=1000.0)
    assert result.burst_count == 0
    assert result.bursts == []
    assert result.unburst_spike_count == 0
    assert result.mean_spikes_per_burst is None
    assert result.mean_intra_burst_frequency is None
    assert result.mean_inter_burst_interval is None
    assert result.duty_cycle is None
    assert result.isi_threshold_ms == pytest.approx(_DEFAULT_THRESHOLD_MS)
    assert result.threshold_method == "default-fixed"


def test_single_spike_returns_zero_bursts_and_one_unburst_spike() -> None:
    """A single spike with default min_spikes_per_burst=2 is unburst, not a burst."""
    ap = _make_ap_result([100.0])
    result = analyze_bursts(ap, total_duration_ms=500.0)
    assert result.burst_count == 0
    assert result.unburst_spike_count == 1
    assert result.threshold_method == "default-fixed"


def test_single_spike_with_min_one_returns_one_burst() -> None:
    """A single spike with min_spikes_per_burst=1 forms a single-spike burst."""
    ap = _make_ap_result([100.0])
    result = analyze_bursts(ap, total_duration_ms=500.0, min_spikes_per_burst=1)
    assert result.burst_count == 1
    assert result.bursts[0].spike_count == 1
    assert result.bursts[0].duration == pytest.approx(0.0)
    assert result.bursts[0].intra_burst_frequency is None
    assert result.unburst_spike_count == 0


def test_single_burst_no_inter_burst_interval() -> None:
    """A clean cluster with all ISIs below threshold yields one burst with no IBI."""
    peak_times = [10.0, 15.0, 20.0, 25.0, 30.0]  # 5 spikes, 5 ms ISIs
    ap = _make_ap_result(peak_times)
    result = analyze_bursts(ap, total_duration_ms=500.0)
    assert result.burst_count == 1
    burst = result.bursts[0]
    assert burst.spike_count == 5
    assert burst.start_time == pytest.approx(10.0)
    assert burst.end_time == pytest.approx(30.0)
    assert burst.duration == pytest.approx(20.0)
    assert burst.intra_burst_frequency == pytest.approx(200.0)
    assert burst.mean_intra_burst_isi == pytest.approx(5.0)
    assert result.mean_inter_burst_interval is None
    assert result.duty_cycle == pytest.approx(20.0 / 500.0)


# ---------------------------------------------------------------------------
# Burst counting and frequency recovery
# ---------------------------------------------------------------------------


def test_two_bursts_with_clean_gap() -> None:
    """Two well-separated clusters should produce two bursts with the expected IBI."""
    # Burst A: spikes at 10, 15, 20, 25 (3 ISIs of 5 ms).
    # Burst B: spikes at 200, 205, 210, 215 (3 ISIs of 5 ms).
    # Gap between burst A end (25) and burst B start (200) = 175 ms.
    peak_times = [10.0, 15.0, 20.0, 25.0, 200.0, 205.0, 210.0, 215.0]
    ap = _make_ap_result(peak_times)
    result = analyze_bursts(ap, total_duration_ms=1000.0, isi_threshold_ms=50.0)
    assert result.burst_count == 2
    assert result.threshold_method == "user"
    assert result.mean_spikes_per_burst == pytest.approx(4.0)
    assert result.mean_intra_burst_frequency == pytest.approx(200.0)
    assert result.mean_inter_burst_interval == pytest.approx(175.0)
    assert result.duty_cycle == pytest.approx((15.0 + 15.0) / 1000.0)


def test_intra_burst_frequency_recovery() -> None:
    """A 5-spike burst with 10 ms ISIs should report ≈ 100 Hz."""
    peak_times = [10.0, 20.0, 30.0, 40.0, 50.0]
    ap = _make_ap_result(peak_times)
    result = analyze_bursts(ap, total_duration_ms=1000.0, isi_threshold_ms=50.0)
    assert result.burst_count == 1
    assert result.bursts[0].intra_burst_frequency == pytest.approx(100.0)


def test_duty_cycle_computation() -> None:
    """Duty cycle should equal sum-of-burst-durations / total duration."""
    peak_times = [100.0, 110.0, 120.0, 130.0, 200.0, 210.0, 220.0, 230.0]
    ap = _make_ap_result(peak_times)
    # Burst A duration = 30, Burst B duration = 30, total in burst = 60 ms
    # IBI gap = 200 - 130 = 70 ms (above 50 ms threshold) → two bursts.
    result = analyze_bursts(ap, total_duration_ms=1000.0, isi_threshold_ms=50.0)
    assert result.burst_count == 2
    assert result.duty_cycle == pytest.approx(60.0 / 1000.0)


# ---------------------------------------------------------------------------
# Threshold method selection
# ---------------------------------------------------------------------------


def test_identical_isis_fall_back_to_default_threshold() -> None:
    """An ISI distribution with zero spread should fall back to the default."""
    # All ISIs identical at 5 ms → log10 spread is 0; the np.ptp early
    # exit must engage and the analyser must fall back to 100 ms.
    peak_times = list(np.cumsum([10.0, *([5.0] * 8)]))
    ap = _make_ap_result(peak_times)
    result = analyze_bursts(ap, total_duration_ms=200.0)
    assert result.threshold_method == "default-fixed"
    assert result.isi_threshold_ms == pytest.approx(_DEFAULT_THRESHOLD_MS)


def test_exactly_min_isis_for_histogram_attempts_auto() -> None:
    """At the boundary of the histogram threshold, auto-detect should be tried."""
    # 4 ISIs is the minimum supported; with a clear bimodal split the
    # histogram path should still fire.
    peak_times = list(np.cumsum([10.0, 5.0, 5.0, 200.0, 5.0]))
    ap = _make_ap_result(peak_times)
    result = analyze_bursts(ap, total_duration_ms=500.0)
    # Boundary triggers histogram; clear bimodality may or may not be
    # detected at this small sample size, but the method must not be "user".
    assert result.threshold_method in {"auto-histogram", "default-fixed"}


def test_default_threshold_used_when_few_isis() -> None:
    """Fewer than 4 ISIs should fall back to the fixed default threshold."""
    peak_times = [10.0, 20.0, 30.0]  # 2 ISIs only
    ap = _make_ap_result(peak_times)
    result = analyze_bursts(ap, total_duration_ms=200.0)
    assert result.isi_threshold_ms == pytest.approx(_DEFAULT_THRESHOLD_MS)
    assert result.threshold_method == "default-fixed"


def test_auto_histogram_threshold_detects_bimodal_distribution() -> None:
    """A clearly bimodal ISI distribution should trigger auto-histogram detection."""
    # Build five 5 ms ISIs (intra-burst) and four 200 ms ISIs (inter-burst).
    # Spike peak times constructed by cumulative-summing the chosen ISIs.
    isi_pattern = [5.0, 5.0, 5.0, 5.0, 5.0, 200.0, 200.0, 200.0, 200.0]
    peak_times = list(np.cumsum([10.0, *isi_pattern]))
    ap = _make_ap_result(peak_times)
    result = analyze_bursts(ap, total_duration_ms=2000.0)
    assert result.threshold_method == "auto-histogram"
    assert 5.0 < result.isi_threshold_ms < 200.0


def test_unimodal_isis_fall_back_to_default() -> None:
    """A unimodal ISI distribution should fall back to the fixed default."""
    rng = np.random.default_rng(seed=0)
    isis = list(50.0 + rng.normal(0.0, 1.0, size=12))  # tightly clustered around 50
    peak_times = list(np.cumsum([10.0, *isis]))
    ap = _make_ap_result(peak_times)
    result = analyze_bursts(ap, total_duration_ms=2000.0)
    assert result.threshold_method == "default-fixed"
    assert result.isi_threshold_ms == pytest.approx(_DEFAULT_THRESHOLD_MS)


def test_user_supplied_threshold_overrides_auto() -> None:
    """An explicit ``isi_threshold_ms`` should bypass auto-detection."""
    peak_times = [10.0, 15.0, 20.0, 100.0, 105.0, 110.0, 200.0]
    ap = _make_ap_result(peak_times)
    result = analyze_bursts(ap, total_duration_ms=500.0, isi_threshold_ms=30.0)
    assert result.threshold_method == "user"
    assert result.isi_threshold_ms == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# min_spikes_per_burst filter and unburst counting
# ---------------------------------------------------------------------------


def test_min_spikes_per_burst_filter() -> None:
    """A short cluster shorter than ``min_spikes_per_burst`` should be unburst."""
    # Cluster A: spikes at 10, 15 (1 ISI of 5 ms) → only 2 spikes.
    # Long gap.
    # Cluster B: spikes at 200, 205, 210, 215 (3 ISIs of 5 ms) → 4 spikes.
    peak_times = [10.0, 15.0, 200.0, 205.0, 210.0, 215.0]
    ap = _make_ap_result(peak_times)
    # With min_spikes_per_burst=3, only the second cluster qualifies.
    result = analyze_bursts(
        ap, total_duration_ms=500.0, isi_threshold_ms=50.0, min_spikes_per_burst=3
    )
    assert result.burst_count == 1
    assert result.unburst_spike_count == 2
    assert result.bursts[0].spike_count == 4


def test_all_isis_above_threshold_yields_zero_bursts() -> None:
    """When every ISI exceeds the threshold no group reaches min_spikes_per_burst."""
    peak_times = [10.0, 200.0, 400.0, 600.0, 800.0]
    ap = _make_ap_result(peak_times)
    result = analyze_bursts(ap, total_duration_ms=1000.0, isi_threshold_ms=50.0)
    assert result.burst_count == 0
    # 5 isolated single-spike "groups", each rejected by min_spikes_per_burst=2.
    assert result.unburst_spike_count == 5
    assert result.duty_cycle is None


# ---------------------------------------------------------------------------
# from_result wrapper
# ---------------------------------------------------------------------------


def test_analyze_bursts_from_result_consumes_simulation_result() -> None:
    """``analyze_bursts_from_result`` should match an explicit two-step pipeline."""
    dt = 0.025
    duration_ms = 1000.0
    time = np.arange(0.0, duration_ms, dt)
    peak_times = [100.0, 105.0, 110.0, 115.0, 500.0, 505.0, 510.0, 515.0]
    voltage = _triangular_voltage(time, peak_times)

    dtype = np.dtype([("time", np.float64), ("voltage", np.float64)])
    arr = np.empty(len(time), dtype=dtype)
    arr["time"] = time
    arr["voltage"] = voltage
    result = cast(SimulationResult, arr)

    burst_result = analyze_bursts_from_result(result, isi_threshold_ms=50.0)

    assert isinstance(burst_result, BurstAnalysisResult)
    assert burst_result.burst_count == 2
    assert burst_result.threshold_method == "user"
    assert burst_result.mean_inter_burst_interval == pytest.approx(385.0, rel=0.05)
