"""Tests for the spike-frequency adaptation (SFA) analysis module."""

import pytest

from patch_sim.analysis.ap_metrics import APAnalysisResult
from patch_sim.analysis.sfa import SFAAnalysisResult, SFACurve, analyze_sfa, compute_sfa


def _make_ap_result(isis: list[float]) -> APAnalysisResult:
    """Build a minimal APAnalysisResult with only the isis field populated.

    Args:
        isis: Inter-spike intervals in ms.

    Returns:
        An APAnalysisResult whose isis list matches the argument and all
        other optional fields are None.
    """
    spike_count = len(isis) + 1 if isis else 0
    return APAnalysisResult(
        spike_count=spike_count,
        spikes=[],
        isis=isis,
        mean_threshold_voltage=None,
        mean_peak_voltage=None,
        mean_rise_time=None,
        mean_half_width=None,
        mean_ahp_depth=None,
        mean_isi=None,
        firing_rate=None,
    )


def test_compute_sfa_no_spikes() -> None:
    """compute_sfa returns None when there are zero spikes (empty isis)."""
    ap = _make_ap_result([])
    assert compute_sfa(ap) is None


def test_compute_sfa_one_spike() -> None:
    """compute_sfa returns None when only one spike is detected (no ISIs)."""
    ap = _make_ap_result([])
    ap.spike_count = 1
    assert compute_sfa(ap) is None


def test_compute_sfa_two_spikes() -> None:
    """compute_sfa produces a single-point curve with adaptation_index 0 for two spikes."""
    ap = _make_ap_result([10.0])
    result = compute_sfa(ap)
    assert result is not None
    assert len(result.instantaneous_frequencies) == 1
    assert result.spike_indices == [0]
    assert result.instantaneous_frequencies[0] == pytest.approx(100.0)
    assert result.adaptation_index == pytest.approx(0.0)


def test_compute_sfa_equal_isis() -> None:
    """compute_sfa reports near-zero adaptation index for uniform spike timing."""
    ap = _make_ap_result([10.0, 10.0, 10.0])
    result = compute_sfa(ap)
    assert result is not None
    assert len(result.instantaneous_frequencies) == 3
    assert all(f == pytest.approx(100.0) for f in result.instantaneous_frequencies)
    assert result.adaptation_index == pytest.approx(0.0)


def test_compute_sfa_adapting_train() -> None:
    """compute_sfa correctly computes frequencies and positive adaptation index."""
    # ISIs increase: 10, 15, 20 ms → freqs: 100, 66.67, 50 Hz
    ap = _make_ap_result([10.0, 15.0, 20.0])
    result = compute_sfa(ap)
    assert result is not None
    assert result.spike_indices == [0, 1, 2]
    assert result.instantaneous_frequencies[0] == pytest.approx(100.0)
    assert result.instantaneous_frequencies[1] == pytest.approx(1000.0 / 15.0)
    assert result.instantaneous_frequencies[2] == pytest.approx(50.0)
    # adaptation_index = (100 - 50) / 100 = 0.5
    assert result.adaptation_index == pytest.approx(0.5)


def test_compute_sfa_accelerating_train() -> None:
    """compute_sfa reports a negative adaptation index when firing accelerates."""
    # ISIs decrease: 20, 10 ms → freqs: 50, 100 Hz
    ap = _make_ap_result([20.0, 10.0])
    result = compute_sfa(ap)
    assert result is not None
    # adaptation_index = (50 - 100) / 50 = -1.0
    assert result.adaptation_index == pytest.approx(-1.0)


def test_compute_sfa_label_passthrough() -> None:
    """compute_sfa stores the provided label on the returned SFACurve."""
    ap = _make_ap_result([10.0, 12.0])
    result = compute_sfa(ap, label="5.0 µA/cm²")
    assert result is not None
    assert result.label == "5.0 µA/cm²"


def test_compute_sfa_default_label_empty() -> None:
    """compute_sfa uses an empty string label when no label is provided."""
    ap = _make_ap_result([10.0])
    result = compute_sfa(ap)
    assert result is not None
    assert result.label == ""


def test_analyze_sfa_filters_insufficient_sweeps() -> None:
    """analyze_sfa excludes sweeps with fewer than two spikes."""
    ap_no_spikes = _make_ap_result([])
    ap_two_spikes = _make_ap_result([10.0])
    ap_three_spikes = _make_ap_result([10.0, 12.0])

    result = analyze_sfa([ap_no_spikes, ap_two_spikes, ap_three_spikes])
    assert isinstance(result, SFAAnalysisResult)
    assert len(result.curves) == 2


def test_analyze_sfa_labels_preserved() -> None:
    """analyze_sfa assigns the correct label to each curve."""
    ap_a = _make_ap_result([10.0])
    ap_b = _make_ap_result([8.0, 12.0])
    result = analyze_sfa([ap_a, ap_b], labels=["sweep A", "sweep B"])
    assert result.curves[0].label == "sweep A"
    assert result.curves[1].label == "sweep B"


def test_analyze_sfa_default_labels_empty() -> None:
    """analyze_sfa uses empty string labels when no labels argument is provided."""
    ap = _make_ap_result([10.0])
    result = analyze_sfa([ap])
    assert result.curves[0].label == ""


def test_analyze_sfa_all_insufficient() -> None:
    """analyze_sfa returns an empty curves list when no sweep has enough spikes."""
    result = analyze_sfa([_make_ap_result([]), _make_ap_result([])])
    assert result.curves == []
