"""Unit tests for patch_sim.analysis.gv_curve.

Covers boltzmann() and compute_gv() with synthetic I-V data and edge cases.
Integration tests against real HH simulations live in
tests/integration/test_gv_curve_simulation.py.
"""

import numpy as np
import pytest

from patch_sim.analysis.gv_curve import (
    boltzmann,
    compute_gv,
)
from patch_sim.analysis.iv_curve import IVAnalysisResult, IVPoint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_iv_result(
    voltages: list[float],
    peak_inward_currents: list[float],
) -> IVAnalysisResult:
    """Build a synthetic IVAnalysisResult for testing.

    Steady-state and peak outward values are set to zero since they are not
    used by compute_gv.

    Args:
        voltages: Voltage steps in mV.
        peak_inward_currents: Corresponding peak inward currents (µA/cm²).

    Returns:
        An IVAnalysisResult with one IVPoint per voltage/current pair.
    """
    points = [
        IVPoint(
            voltage_step=v,
            peak_inward_current=i,
            peak_outward_current=0.0,
            steady_state_current=0.0,
        )
        for v, i in zip(voltages, peak_inward_currents)
    ]
    return IVAnalysisResult(points=points)


# ---------------------------------------------------------------------------
# boltzmann()
# ---------------------------------------------------------------------------


def test_boltzmann_at_vhalf_returns_half():
    """boltzmann(v_half, v_half, k) equals 0.5 for any k."""
    for k in [2.0, 6.0, 15.0]:
        assert boltzmann(-20.0, -20.0, k) == pytest.approx(0.5)


def test_boltzmann_approaches_zero_at_minus_infinity():
    """Boltzmann at very negative V approaches 0."""
    assert boltzmann(-1000.0, -20.0, 6.0) == pytest.approx(0.0, abs=1e-6)


def test_boltzmann_approaches_one_at_plus_infinity():
    """Boltzmann at very positive V approaches 1."""
    assert boltzmann(1000.0, -20.0, 6.0) == pytest.approx(1.0, abs=1e-6)


def test_boltzmann_slope_factor_effect():
    """Larger k produces a shallower sigmoid."""
    v = -10.0
    v_half = -20.0
    val_steep = boltzmann(v, v_half, 3.0)
    val_shallow = boltzmann(v, v_half, 12.0)
    # Both should be above 0.5 (V > v_half), but steep rises faster
    assert val_steep > val_shallow


# ---------------------------------------------------------------------------
# compute_gv() — unit tests
# ---------------------------------------------------------------------------


def test_compute_gv_basic_conductances():
    """Chord conductances are correctly computed from I = g * (V - E_rev)."""
    # E_rev = +60 mV; voltage steps = -20, 0, +20 mV (all below E_rev)
    # Driving force = V - E_rev = -80, -60, -40
    # Choose inward currents = g * df with g = 1, 2, 3 mS/cm²
    e_rev = 60.0
    voltages = [-20.0, 0.0, 20.0]
    # I = g * (V - E_rev): with df = -80, -60, -40
    # and g = 1, 2, 3 → I = -80, -120, -120 ... let's use nicer numbers
    # g = 2, 2, 2 → I = -160, -120, -80
    conductances_expected = [2.0, 2.0, 2.0]
    peak_inward = [g * (v - e_rev) for g, v in zip(conductances_expected, voltages)]
    iv = _make_iv_result(voltages, peak_inward)
    result = compute_gv(iv, reversal_potential=e_rev)
    assert len(result.points) == 3
    for pt, g_exp in zip(result.points, conductances_expected):
        assert pt.conductance == pytest.approx(g_exp, rel=1e-6)


def test_compute_gv_normalization():
    """G/Gmax values are normalized so the maximum is 1.0."""
    e_rev = 60.0
    voltages = [-40.0, -20.0, 0.0]
    # conductances 1, 2, 4 → I = g * (V - 60)
    gs = [1.0, 2.0, 4.0]
    peak_inward = [g * (v - e_rev) for g, v in zip(gs, voltages)]
    iv = _make_iv_result(voltages, peak_inward)
    result = compute_gv(iv, reversal_potential=e_rev)
    assert max(result.g_normalized_values) == pytest.approx(1.0)
    assert result.g_normalized_values[0] == pytest.approx(0.25, rel=1e-5)
    assert result.g_normalized_values[1] == pytest.approx(0.5, rel=1e-5)


def test_compute_gv_excludes_near_reversal():
    """Points within driving_force_threshold of E_rev are excluded."""
    e_rev = 50.0
    # voltage step at 47 mV is 3 mV from E_rev — within default threshold of 5 mV
    voltages = [-20.0, 47.0, 0.0]
    peak_inward = [-140.0, -0.1, -100.0]  # arbitrary values
    iv = _make_iv_result(voltages, peak_inward)
    result = compute_gv(iv, reversal_potential=e_rev, driving_force_threshold=5.0)
    included_voltages = result.voltage_steps
    assert 47.0 not in included_voltages


def test_compute_gv_excludes_non_positive_conductance():
    """Points producing non-positive conductance are excluded."""
    e_rev = 60.0
    # At v = +80 mV, driving force = +20 mV (positive)
    # Inward current is negative → g = I/df < 0 — should be excluded.
    # Supply three points so two remain after exclusion (avoids the < 2 guard).
    voltages = [-40.0, -20.0, 80.0]
    # First two give positive conductance; third gives negative conductance.
    peak_inward = [-200.0, -80.0, -5.0]
    iv = _make_iv_result(voltages, peak_inward)
    result = compute_gv(iv, reversal_potential=e_rev)
    included_voltages = result.voltage_steps
    assert 80.0 not in included_voltages
    assert len(result.points) == 2


def test_compute_gv_empty_iv_result():
    """Empty IVAnalysisResult produces empty GVAnalysisResult with no fit."""
    iv = IVAnalysisResult(points=[])
    result = compute_gv(iv, reversal_potential=55.0)
    assert result.points == []
    assert result.boltzmann.converged is False


def test_compute_gv_all_points_excluded():
    """All points near E_rev produces empty result with no fit."""
    e_rev = 55.0
    voltages = [53.0, 56.0]  # both within 5 mV of 55 mV
    peak_inward = [-1.0, -1.0]
    iv = _make_iv_result(voltages, peak_inward)
    result = compute_gv(iv, reversal_potential=e_rev)
    assert result.points == []
    assert result.boltzmann.converged is False


def test_compute_gv_single_valid_point():
    """A single valid point after filtering produces empty result (< 2 points)."""
    e_rev = 60.0
    voltages = [-20.0, 58.0]  # second within 5 mV
    peak_inward = [-80.0, -0.1]
    iv = _make_iv_result(voltages, peak_inward)
    result = compute_gv(iv, reversal_potential=e_rev)
    assert result.points == []
    assert result.boltzmann.converged is False


def test_compute_gv_sorted_by_voltage():
    """Output points are sorted by ascending voltage step."""
    e_rev = 60.0
    voltages = [0.0, -40.0, -20.0]
    gs = [3.0, 1.0, 2.0]
    peak_inward = [g * (v - e_rev) for g, v in zip(gs, voltages)]
    iv = _make_iv_result(voltages, peak_inward)
    result = compute_gv(iv, reversal_potential=e_rev)
    assert result.voltage_steps == sorted(result.voltage_steps)


def test_compute_gv_boltzmann_fit_converges_on_sigmoid():
    """Boltzmann fit recovers v_half and k from a known sigmoid input."""
    v_half_true = -25.0
    k_true = 7.0
    e_rev = 60.0
    voltages = np.linspace(-70.0, 10.0, 17).tolist()
    # g proportional to boltzmann, with driving force = V - E_rev (all negative)
    # I = g * (V - E_rev); g = g_max * boltzmann(V, v_half, k)
    g_max = 5.0
    peak_inward = [
        g_max * boltzmann(v, v_half_true, k_true) * (v - e_rev) for v in voltages
    ]
    iv = _make_iv_result(voltages, peak_inward)
    result = compute_gv(iv, reversal_potential=e_rev)
    assert result.boltzmann.converged is True
    assert result.boltzmann.v_half == pytest.approx(v_half_true, abs=1.0)
    assert result.boltzmann.k == pytest.approx(k_true, abs=1.0)


def test_compute_gv_reversal_potential_stored():
    """GVAnalysisResult stores the reversal potential used for computation."""
    e_rev = 57.3
    voltages = [-20.0, -10.0, 0.0]
    gs = [1.0, 2.0, 3.0]
    peak_inward = [g * (v - e_rev) for g, v in zip(gs, voltages)]
    iv = _make_iv_result(voltages, peak_inward)
    result = compute_gv(iv, reversal_potential=e_rev)
    assert result.reversal_potential == pytest.approx(e_rev)


def test_compute_gv_convenience_properties_match_points():
    """Convenience list properties on GVAnalysisResult match the points list."""
    e_rev = 60.0
    voltages = [-30.0, -10.0, 10.0]
    gs = [1.0, 2.0, 3.0]
    peak_inward = [g * (v - e_rev) for g, v in zip(gs, voltages)]
    iv = _make_iv_result(voltages, peak_inward)
    result = compute_gv(iv, reversal_potential=e_rev)
    for i, pt in enumerate(result.points):
        assert result.voltage_steps[i] == pt.voltage_step
        assert result.conductances[i] == pt.conductance
        assert result.g_normalized_values[i] == pt.g_normalized
