"""Unit tests for patch_sim.analysis.inactivation.

Covers compute_inactivation() with synthetic I-V data and edge cases.
Integration tests against real HH simulations live in
tests/integration/test_inactivation_simulation.py.
"""

import numpy as np
import pytest

from patch_sim.analysis.gv_curve import boltzmann
from patch_sim.analysis.inactivation import compute_inactivation
from patch_sim.analysis.iv_curve import IVAnalysisResult, IVPoint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_iv_result(
    prepulse_voltages: list[float],
    peak_inward_currents: list[float],
) -> IVAnalysisResult:
    """Build a synthetic IVAnalysisResult for testing compute_inactivation.

    Steady-state and peak-outward values are set to zero since they are not
    used by compute_inactivation.

    Args:
        prepulse_voltages: Conditioning prepulse voltages in mV.
        peak_inward_currents: Corresponding test-pulse peak inward currents
            (µA/cm²).

    Returns:
        An IVAnalysisResult with one IVPoint per prepulse/current pair.
    """
    points = [
        IVPoint(
            voltage_step=v,
            peak_inward_current=i,
            peak_outward_current=0.0,
            steady_state_current=0.0,
        )
        for v, i in zip(prepulse_voltages, peak_inward_currents)
    ]
    return IVAnalysisResult(points=points)


# ---------------------------------------------------------------------------
# compute_inactivation() — unit tests
# ---------------------------------------------------------------------------


def test_compute_inactivation_normalization():
    """h∞ values are I_peak / I_peak_max (normalized by the most-negative peak)."""
    prepulses = [-100.0, -80.0, -60.0, -40.0]
    peaks = [-100.0, -100.0, -50.0, -10.0]
    result = compute_inactivation(_make_iv_result(prepulses, peaks))
    assert result.h_normalized_values == pytest.approx([1.0, 1.0, 0.5, 0.1])


def test_compute_inactivation_sorted_by_prepulse():
    """Output points are sorted by ascending prepulse voltage."""
    # Supplied out of order; -100 has the largest inward current.
    prepulses = [-40.0, -100.0, -60.0]
    peaks = [-10.0, -100.0, -50.0]
    result = compute_inactivation(_make_iv_result(prepulses, peaks))
    assert result.prepulse_voltages == [-100.0, -60.0, -40.0]
    assert result.h_normalized_values == pytest.approx([1.0, 0.5, 0.1])


def test_compute_inactivation_clamps_to_unit_range():
    """A non-negative test-pulse peak (full inactivation) maps to h∞ = 0.0."""
    prepulses = [-100.0, -60.0, -20.0]
    # Last sweep is net outward at the test pulse → ratio would be negative.
    peaks = [-200.0, -80.0, 5.0]
    result = compute_inactivation(_make_iv_result(prepulses, peaks))
    h = result.h_normalized_values
    assert h[0] == pytest.approx(1.0)
    assert h[1] == pytest.approx(0.4)
    assert h[2] == 0.0
    for value in h:
        assert 0.0 <= value <= 1.0


def test_compute_inactivation_boltzmann_recovers_params():
    """The decreasing-Boltzmann fit recovers v_half and k from a known sigmoid."""
    v_half_true = -65.0
    k_true = 8.0
    i_max = -120.0  # µA/cm² at full availability
    prepulses = np.linspace(-120.0, -20.0, 11)
    # I_peak(V) = i_max * h∞(V), with h∞(V) = boltzmann(V, v_half, -k).
    peaks = [float(i_max * boltzmann(v, v_half_true, -k_true)) for v in prepulses]
    result = compute_inactivation(_make_iv_result(prepulses.tolist(), peaks))
    assert result.boltzmann.converged is True
    assert result.boltzmann.v_half == pytest.approx(v_half_true, abs=1.5)
    assert result.boltzmann.k == pytest.approx(k_true, abs=1.5)


def test_compute_inactivation_curve_is_decreasing():
    """For a monotone synthetic input the h∞ values are non-increasing."""
    prepulses = np.linspace(-120.0, -20.0, 11)
    peaks = [float(-100.0 * boltzmann(v, -60.0, -7.0)) for v in prepulses]
    result = compute_inactivation(_make_iv_result(prepulses.tolist(), peaks))
    h = result.h_normalized_values
    assert all(h[i + 1] <= h[i] + 1e-9 for i in range(len(h) - 1))


def test_compute_inactivation_empty_iv_result():
    """An empty IVAnalysisResult produces an empty result with no fit."""
    result = compute_inactivation(IVAnalysisResult(points=[]))
    assert result.points == []
    assert result.boltzmann.converged is False


def test_compute_inactivation_single_point():
    """A single point yields one record but no Boltzmann fit (< 2 points)."""
    result = compute_inactivation(_make_iv_result([-80.0], [-50.0]))
    assert len(result.points) == 1
    assert result.h_normalized_values == pytest.approx([1.0])
    assert result.boltzmann.converged is False


def test_compute_inactivation_all_non_negative_peaks():
    """No inward current at any prepulse → all h∞ = 0.0 and no fit."""
    prepulses = [-100.0, -60.0, -20.0]
    peaks = [0.0, 0.0, 1.0]
    result = compute_inactivation(_make_iv_result(prepulses, peaks))
    assert result.h_normalized_values == [0.0, 0.0, 0.0]
    assert result.boltzmann.converged is False


def test_compute_inactivation_convenience_properties_match_points():
    """Convenience list properties match the underlying points list."""
    prepulses = [-100.0, -60.0, -20.0]
    peaks = [-100.0, -40.0, -10.0]
    result = compute_inactivation(_make_iv_result(prepulses, peaks))
    for i, pt in enumerate(result.points):
        assert result.prepulse_voltages[i] == pt.prepulse_voltage
        assert result.peak_inward_currents[i] == pt.peak_inward_current
        assert result.h_normalized_values[i] == pt.h_normalized
