"""
Tests for the Hodgkin-Huxley model implementation.
"""

import pytest
import pandas as pd
from src.hodgkin_huxley import HodgkinHuxley


@pytest.fixture
def hh_model():
    """Fixture to create a HodgkinHuxley model instance for testing."""
    return HodgkinHuxley()


def test_initialization(hh_model):
    """Test that the model is initialized with correct parameters."""
    assert hh_model.C_m == pytest.approx(1.0)
    assert hh_model.g_Na == pytest.approx(120.0)
    assert hh_model.g_K == pytest.approx(36.0)
    assert hh_model.g_L == pytest.approx(0.3)

    # Test that reversal potentials are within expected ranges
    assert 40.0 < hh_model.E_Na < 60.0
    assert -90.0 < hh_model.E_K < -65.0
    assert -70.0 < hh_model.E_L < -40.0


def test_compute_returns_dataframe(hh_model):
    """Test that compute method returns a pandas DataFrame with correct structure."""
    result = hh_model.compute(simulation_time=10, time_step=0.1)

    # Check result type
    assert isinstance(result, pd.DataFrame)

    # Check that DataFrame has expected index name
    assert result.index.name == "time"

    # Check that DataFrame has expected columns
    expected_columns = [
        "voltage",
        "potassium_activation",
        "sodium_activation",
        "sodium_inactivation",
    ]
    for col in expected_columns:
        assert col in result.columns

    # Check DataFrame length matches expected number of time steps
    expected_length = int(10 / 0.1) + 1  # (simulation_time/time_step) + 1
    assert len(result) == expected_length


def test_rate_constants(hh_model):
    """Test the rate constant calculation methods."""
    voltage = -65.0

    # Test each rate constant at resting potential
    assert hh_model.alpha_n(voltage) > 0
    assert hh_model.beta_n(voltage) > 0
    assert hh_model.alpha_m(voltage) > 0
    assert hh_model.beta_m(voltage) > 0
    assert hh_model.alpha_h(voltage) > 0
    assert hh_model.beta_h(voltage) > 0


def test_simulation_dynamics(hh_model):
    """Test that the simulation shows expected dynamics."""
    result = hh_model.compute(simulation_time=50, time_step=0.05)

    # Voltage should change from initial value
    initial_voltage = result["voltage"].iloc[0]
    max_voltage = result["voltage"].max()
    assert initial_voltage != max_voltage

    # Should observe an action potential (voltage exceeding threshold)
    threshold = 0  # mV, typical AP threshold
    assert any(result["voltage"] > threshold)

    # Sodium activation should increase during depolarization
    assert result["sodium_activation"].max() > result["sodium_activation"].iloc[0]
