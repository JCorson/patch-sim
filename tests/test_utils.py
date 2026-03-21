"""Tests for the utilities module."""

import math

import numpy as np
import pytest

from patch_sim.utils import boltzmann_cosh_rates, safe_exp


class TestSafeExp:
    """Test the safe_exp function."""

    def test_safe_exp_scalar_normal_range(self):
        """Test safe_exp with scalar input in normal range."""
        assert safe_exp(0.0) == pytest.approx(1.0)
        assert safe_exp(1.0) == pytest.approx(np.exp(1.0))
        assert safe_exp(-1.0) == pytest.approx(np.exp(-1.0))

    def test_safe_exp_scalar_extreme_values(self):
        """Test safe_exp with scalar input at extreme values."""
        assert safe_exp(200.0) == pytest.approx(np.exp(100.0))
        assert safe_exp(-200.0) == pytest.approx(np.exp(-100.0))

    def test_safe_exp_boundary_values(self):
        """Test safe_exp at the clipping boundaries."""
        assert safe_exp(100.0) == pytest.approx(np.exp(100.0))
        assert safe_exp(-100.0) == pytest.approx(np.exp(-100.0))

    def test_safe_exp_prevents_overflow(self):
        """Test that safe_exp prevents overflow for very large inputs."""
        result = safe_exp(1000.0)
        assert np.isfinite(result)
        assert result == pytest.approx(np.exp(100.0))

    def test_safe_exp_prevents_underflow(self):
        """Test that safe_exp prevents underflow for very small inputs."""
        result = safe_exp(-1000.0)
        assert result == pytest.approx(np.exp(-100.0))

    def test_safe_exp_returns_float(self):
        """Test that safe_exp always returns a Python float."""
        assert isinstance(safe_exp(0.0), float)
        assert isinstance(safe_exp(1.0), float)
        assert isinstance(safe_exp(-1.0), float)

    def test_safe_exp_comparison_with_regular_exp(self):
        """Test that safe_exp matches regular exp for normal values."""
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            assert safe_exp(x) == pytest.approx(np.exp(x))

    def test_safe_exp_nan_input_returns_nan(self):
        """NaN input should propagate through (clip is a no-op for NaN)."""
        result = safe_exp(float("nan"))
        assert np.isnan(result)

    def test_safe_exp_positive_inf_clips_to_exp_100(self):
        """Positive Inf is clipped to 100 before exp, yielding exp(100)."""
        assert safe_exp(float("inf")) == pytest.approx(np.exp(100.0))

    def test_safe_exp_negative_inf_clips_to_exp_neg_100(self):
        """Negative Inf is clipped to -100 before exp, yielding exp(-100)."""
        assert safe_exp(float("-inf")) == pytest.approx(np.exp(-100.0))

    def test_safe_exp_integer_input(self):
        """Integer input should work identically to float input."""
        assert safe_exp(0) == pytest.approx(1.0)
        assert safe_exp(1) == pytest.approx(np.exp(1.0))
        assert safe_exp(-1) == pytest.approx(np.exp(-1.0))

    def test_safe_exp_boolean_input(self):
        """Boolean input (True=1, False=0) should be handled without error."""
        assert safe_exp(True) == pytest.approx(np.exp(1.0))
        assert safe_exp(False) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# boltzmann_cosh_rates
# ---------------------------------------------------------------------------


def test_boltzmann_cosh_rates_alpha_plus_beta_equals_one_over_tau():
    """alpha(half) + beta(half) equals 1/tau(half) for a standard gate."""
    half, slope, tau_scale, tau_floor = -52.6, 4.6, 6.0, 0.1
    alpha, beta = boltzmann_cosh_rates(half, slope, tau_scale, tau_floor)
    tau_at_half = tau_scale / math.cosh(0.0)  # cosh(0) = 1
    expected = 1.0 / max(tau_at_half, tau_floor)
    assert alpha(half, 0.0) + beta(half, 0.0) == pytest.approx(expected, rel=1e-9)


def test_boltzmann_cosh_rates_steady_state_is_half_at_half_voltage():
    """alpha/(alpha+beta) = 0.5 exactly at V=half for a standard gate."""
    alpha, beta = boltzmann_cosh_rates(
        half=-52.6, slope=4.6, tau_scale=6.0, tau_floor=0.1
    )
    V = -52.6
    ss = alpha(V, 0.0) / (alpha(V, 0.0) + beta(V, 0.0))
    assert ss == pytest.approx(0.5, rel=1e-9)


def test_boltzmann_cosh_rates_inverted_steady_state_is_half_at_half_voltage():
    """alpha/(alpha+beta) = 0.5 exactly at V=half for an inverted gate."""
    alpha, beta = boltzmann_cosh_rates(
        half=-80.0, slope=12.0, tau_scale=10.0, tau_floor=0.5, inverted=True
    )
    V = -80.0
    ss = alpha(V, 0.0) / (alpha(V, 0.0) + beta(V, 0.0))
    assert ss == pytest.approx(0.5, rel=1e-9)


def test_boltzmann_cosh_rates_tau_floor_respected():
    """Time constant does not drop below tau_floor at extreme voltages."""
    tau_floor = 0.5
    alpha, beta = boltzmann_cosh_rates(
        half=-80.0, slope=12.0, tau_scale=10.0, tau_floor=tau_floor
    )
    for V in (-200.0, 200.0):
        total = alpha(V, 0.0) + beta(V, 0.0)  # = 1 / tau
        tau = 1.0 / total
        assert tau >= tau_floor - 1e-12


def test_boltzmann_cosh_rates_tau_rate_scales_correctly():
    """Doubling tau_rate halves the time constant."""
    half, slope, tau_scale, tau_floor = -35.0, 10.0, 1000.0, 10.0
    alpha1, beta1 = boltzmann_cosh_rates(
        half, slope, tau_scale, tau_floor, tau_rate=1.0
    )
    alpha2, beta2 = boltzmann_cosh_rates(
        half, slope, tau_scale, tau_floor, tau_rate=2.0
    )
    V = -35.0
    tau1 = 1.0 / (alpha1(V, 0.0) + beta1(V, 0.0))
    tau2 = 1.0 / (alpha2(V, 0.0) + beta2(V, 0.0))
    assert tau1 == pytest.approx(2.0 * tau2, rel=1e-9)


def test_boltzmann_cosh_rates_inverted_flips_boltzmann():
    """inverted=True gives high steady state at hyperpolarised voltages."""
    alpha_std, beta_std = boltzmann_cosh_rates(
        half=-52.6, slope=4.6, tau_scale=6.0, tau_floor=0.1
    )
    alpha_inv, beta_inv = boltzmann_cosh_rates(
        half=-52.6, slope=4.6, tau_scale=6.0, tau_floor=0.1, inverted=True
    )
    V_hyper = -100.0
    ss_std = alpha_std(V_hyper, 0.0) / (
        alpha_std(V_hyper, 0.0) + beta_std(V_hyper, 0.0)
    )
    ss_inv = alpha_inv(V_hyper, 0.0) / (
        alpha_inv(V_hyper, 0.0) + beta_inv(V_hyper, 0.0)
    )
    assert ss_std < 0.1  # standard gate is mostly closed at hyperpolarisation
    assert ss_inv > 0.9  # inverted gate is mostly open at hyperpolarisation
