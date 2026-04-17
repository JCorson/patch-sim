"""Integration tests for patch_sim.analysis.ap_metrics with real HH simulations.

Verifies spike detection, ISI statistics, and parameter sensitivity when applied
to voltage traces produced by simulate_current_clamp.
Unit tests with synthetic traces live in tests/unit/test_ap_metrics.py.
"""

import numpy as np
import pytest

import patch_sim
from patch_sim.analysis.ap_metrics import (
    analyze_aps,
    analyze_aps_from_result,
)

_DT = 0.025  # ms, matches SIM_SAMPLING_FREQ = 40 kHz


def _stimulus(duration_ms: float, amp: float, start_ms: float, stop_ms: float):
    """Build a step-current stimulus array.

    Args:
        duration_ms: Total duration in ms.
        amp: Current amplitude (µA/cm²).
        start_ms: Step start time in ms.
        stop_ms: Step end time in ms.

    Returns:
        1-D numpy array of stimulus values.
    """
    s = np.zeros(int(duration_ms / _DT))
    s[int(start_ms / _DT) : int(stop_ms / _DT)] = amp
    return s


# ---------------------------------------------------------------------------
# Multiple spikes
# ---------------------------------------------------------------------------


def test_multiple_spikes_hh_model(hh_model):
    """Suprathreshold current on the HH model produces multiple detected spikes."""
    stimulus = _stimulus(100.0, 20.0, 10.0, 90.0)
    result_sim = patch_sim.simulate_current_clamp(hh_model, stimulus)
    analysis = analyze_aps_from_result(result_sim)

    assert analysis.spike_count > 1
    assert len(analysis.spikes) == analysis.spike_count
    assert len(analysis.isis) == analysis.spike_count - 1
    assert analysis.mean_isi is not None
    assert analysis.firing_rate is not None
    assert analysis.firing_rate == pytest.approx(1000.0 / analysis.mean_isi, rel=1e-6)


def test_isi_length_consistent_with_spike_count(hh_model):
    """ISI list has exactly spike_count - 1 entries."""
    stimulus = _stimulus(100.0, 20.0, 10.0, 90.0)
    result_sim = patch_sim.simulate_current_clamp(hh_model, stimulus)
    analysis = analyze_aps_from_result(result_sim)

    assert len(analysis.isis) == max(0, analysis.spike_count - 1)


def test_firing_rate_from_isi(hh_model):
    """Firing rate equals 1000 / mean_isi when multiple spikes are present."""
    stimulus = _stimulus(150.0, 15.0, 10.0, 140.0)
    result_sim = patch_sim.simulate_current_clamp(hh_model, stimulus)
    analysis = analyze_aps_from_result(result_sim)

    assert analysis.spike_count >= 2
    assert analysis.mean_isi is not None
    assert analysis.firing_rate == pytest.approx(1000.0 / analysis.mean_isi, rel=1e-6)


# ---------------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------------


def test_high_dvdt_threshold_reduces_detections(hh_model):
    """A higher dvdt_threshold detects fewer or equal spikes than a lower one."""
    stimulus = _stimulus(100.0, 20.0, 10.0, 90.0)
    result_sim = patch_sim.simulate_current_clamp(hh_model, stimulus)

    low = analyze_aps_from_result(result_sim, dvdt_threshold=10.0)
    high = analyze_aps_from_result(result_sim, dvdt_threshold=100.0)

    assert high.spike_count <= low.spike_count


# ---------------------------------------------------------------------------
# analyze_aps_from_result
# ---------------------------------------------------------------------------


def test_analyze_aps_from_result_matches_direct_call(hh_model):
    """analyze_aps_from_result matches calling analyze_aps with extracted arrays."""
    stimulus = _stimulus(80.0, 20.0, 5.0, 75.0)
    result_sim = patch_sim.simulate_current_clamp(hh_model, stimulus)

    via_result = analyze_aps_from_result(result_sim)
    direct = analyze_aps(result_sim["time"], result_sim["voltage"])

    assert via_result.spike_count == direct.spike_count
    assert via_result.mean_peak_voltage == pytest.approx(
        direct.mean_peak_voltage or 0.0, abs=1e-9
    )
    assert via_result.firing_rate == pytest.approx(direct.firing_rate or 0.0, abs=1e-9)
