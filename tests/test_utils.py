"""Tests for the utilities module."""

import numpy as np
import pytest

from patch_sim.utils import safe_exp


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
