"""
Tests for the utilities module.
"""

import pytest
import numpy as np
import pandas as pd
from ap_sim.utils import safe_exp


class TestSafeExp:
    """Test the safe_exp function."""

    def test_safe_exp_scalar_normal_range(self):
        """Test safe_exp with scalar input in normal range."""
        # Test normal values
        result = safe_exp(0.0)
        assert result == pytest.approx(1.0)

        result = safe_exp(1.0)
        assert result == pytest.approx(np.exp(1.0))

        result = safe_exp(-1.0)
        assert result == pytest.approx(np.exp(-1.0))

    def test_safe_exp_scalar_extreme_values(self):
        """Test safe_exp with scalar input at extreme values."""
        # Test large positive value (should be clipped)
        result = safe_exp(200.0)
        expected = np.exp(100.0)  # Should be clipped to 100
        assert result == pytest.approx(expected)

        # Test large negative value (should be clipped)
        result = safe_exp(-200.0)
        expected = np.exp(-100.0)  # Should be clipped to -100
        assert result == pytest.approx(expected)

    def test_safe_exp_boundary_values(self):
        """Test safe_exp at the clipping boundaries."""
        # Test exactly at boundaries
        result = safe_exp(100.0)
        expected = np.exp(100.0)
        assert result == pytest.approx(expected)

        result = safe_exp(-100.0)
        expected = np.exp(-100.0)
        assert result == pytest.approx(expected)

    def test_safe_exp_numpy_array(self):
        """Test safe_exp with numpy array input."""
        # Test array with mixed values
        input_array = np.array([-200.0, -1.0, 0.0, 1.0, 200.0])
        result = safe_exp(input_array)

        expected = np.array(
            [
                np.exp(-100.0),  # clipped from -200
                np.exp(-1.0),  # unchanged
                np.exp(0.0),  # unchanged
                np.exp(1.0),  # unchanged
                np.exp(100.0),  # clipped from 200
            ]
        )

        assert isinstance(result, np.ndarray)
        assert len(result) == len(input_array)
        np.testing.assert_array_almost_equal(result, expected)

    def test_safe_exp_pandas_series(self):
        """Test safe_exp with pandas Series input."""
        # Test pandas Series
        input_series = pd.Series([-150.0, -2.0, 0.0, 2.0, 150.0])
        result = safe_exp(input_series)

        expected = pd.Series(
            [
                np.exp(-100.0),  # clipped from -150
                np.exp(-2.0),  # unchanged
                np.exp(0.0),  # unchanged
                np.exp(2.0),  # unchanged
                np.exp(100.0),  # clipped from 150
            ]
        )

        assert isinstance(result, (np.ndarray, pd.Series))
        assert len(result) == len(input_series)
        pd.testing.assert_series_equal(pd.Series(result), expected)

    def test_safe_exp_empty_array(self):
        """Test safe_exp with empty array."""
        empty_array = np.array([])
        result = safe_exp(empty_array)

        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_safe_exp_single_element_array(self):
        """Test safe_exp with single element array."""
        single_array = np.array([5.0])
        result = safe_exp(single_array)

        expected = np.array([np.exp(5.0)])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, expected)

    def test_safe_exp_prevents_overflow(self):
        """Test that safe_exp prevents overflow for very large inputs."""
        # Test value that would normally cause overflow
        large_value = 1000.0  # This would cause overflow with regular exp
        result = safe_exp(large_value)

        # Should not be inf
        assert np.isfinite(result)
        # Should equal exp(100) since it's clipped
        assert result == pytest.approx(np.exp(100.0))

    def test_safe_exp_prevents_underflow(self):
        """Test that safe_exp prevents underflow for very small inputs."""
        # Test value that would normally cause underflow
        small_value = -1000.0  # This would cause underflow with regular exp
        result = safe_exp(small_value)

        # Should not be exactly zero (unless exp(-100) is effectively zero)
        # Should equal exp(-100) since it's clipped
        assert result == pytest.approx(np.exp(-100.0))

    def test_safe_exp_array_mixed_extreme_values(self):
        """Test safe_exp with array containing extreme positive and negative values."""
        # Array with values that would cause both overflow and underflow
        extreme_array = np.array([-1000.0, -500.0, 0.0, 500.0, 1000.0])
        result = safe_exp(extreme_array)

        expected = np.array(
            [
                np.exp(-100.0),  # clipped from -1000
                np.exp(-100.0),  # clipped from -500
                np.exp(0.0),  # unchanged
                np.exp(100.0),  # clipped from 500
                np.exp(100.0),  # clipped from 1000
            ]
        )

        # Check that all results are finite
        assert np.all(np.isfinite(result))
        np.testing.assert_array_almost_equal(result, expected)

    def test_safe_exp_preserves_dtype(self):
        """Test that safe_exp preserves input data types appropriately."""
        # Test with float32 array
        float32_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = safe_exp(float32_array)

        # Result should be numpy array (dtype may be promoted)
        assert isinstance(result, np.ndarray)

        # Test with float64 array
        float64_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = safe_exp(float64_array)

        assert isinstance(result, np.ndarray)

    def test_safe_exp_comparison_with_regular_exp(self):
        """Test that safe_exp matches regular exp for normal values."""
        # Test range where both should give same results
        normal_values = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])

        safe_result = safe_exp(normal_values)
        regular_result = np.exp(normal_values)

        np.testing.assert_array_almost_equal(safe_result, regular_result)

    def test_safe_exp_nan_input_returns_nan(self):
        """NaN input should propagate through (clip is a no-op for NaN)."""
        result = safe_exp(float("nan"))
        assert np.isnan(result)

    def test_safe_exp_positive_inf_clips_to_exp_100(self):
        """Positive Inf is clipped to 100 before exp, yielding exp(100)."""
        result = safe_exp(float("inf"))
        assert result == pytest.approx(np.exp(100.0))

    def test_safe_exp_negative_inf_clips_to_exp_neg_100(self):
        """Negative Inf is clipped to -100 before exp, yielding exp(-100)."""
        result = safe_exp(float("-inf"))
        assert result == pytest.approx(np.exp(-100.0))

    def test_safe_exp_integer_input(self):
        """Integer input should work identically to float input."""
        assert safe_exp(0) == pytest.approx(1.0)
        assert safe_exp(1) == pytest.approx(np.exp(1.0))
        assert safe_exp(-1) == pytest.approx(np.exp(-1.0))

    def test_safe_exp_boolean_input(self):
        """Boolean input (True=1, False=0) should be handled without error."""
        assert safe_exp(True) == pytest.approx(np.exp(1.0))
        assert safe_exp(False) == pytest.approx(1.0)
