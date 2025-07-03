"""
Tests for the core Hodgkin-Huxley model functionality.
"""

import pytest
from ap_sim.hodgkin_huxley import HodgkinHuxley


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
    assert hh_model.time_step == pytest.approx(0.01)

    # Test that reversal potentials are within expected ranges
    assert 60.0 < hh_model.E_Na < 65.0
    assert -90.0 < hh_model.E_K < -65.0
    assert -70.0 < hh_model.E_L < -40.0


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


def test_steady_state_values(hh_model):
    """Test that steady-state gating variable values are calculated correctly."""
    voltage = -65.0

    # Calculate steady-state values
    alpha_n = hh_model.alpha_n(voltage)
    beta_n = hh_model.beta_n(voltage)
    n_inf = alpha_n / (alpha_n + beta_n)

    alpha_m = hh_model.alpha_m(voltage)
    beta_m = hh_model.beta_m(voltage)
    m_inf = alpha_m / (alpha_m + beta_m)

    alpha_h = hh_model.alpha_h(voltage)
    beta_h = hh_model.beta_h(voltage)
    h_inf = alpha_h / (alpha_h + beta_h)

    # Steady-state values should be between 0 and 1
    assert 0 <= n_inf <= 1
    assert 0 <= m_inf <= 1
    assert 0 <= h_inf <= 1

    # At resting potential, h should be high, m and n should be low
    assert h_inf > 0.5  # Sodium inactivation high at rest
    assert m_inf < 0.5  # Sodium activation low at rest
    assert n_inf < 0.5  # Potassium activation low at rest


def test_time_constants(hh_model):
    """Test that time constants are positive and reasonable."""
    voltage = -65.0

    # Calculate time constants
    tau_n = 1.0 / (hh_model.alpha_n(voltage) + hh_model.beta_n(voltage))
    tau_m = 1.0 / (hh_model.alpha_m(voltage) + hh_model.beta_m(voltage))
    tau_h = 1.0 / (hh_model.alpha_h(voltage) + hh_model.beta_h(voltage))

    # Time constants should be positive
    assert tau_n > 0
    assert tau_m > 0
    assert tau_h > 0

    # Time constants should be in reasonable physiological range (ms)
    assert 0.1 <= tau_n <= 10
    assert 0.01 <= tau_m <= 1
    assert 1 <= tau_h <= 10


def test_custom_initialization():
    """Test that the model can be initialized with custom parameters."""
    custom_time_step = 0.05
    custom_model = HodgkinHuxley(time_step=custom_time_step)

    assert custom_model.time_step == pytest.approx(custom_time_step)

    # Other parameters should still have default values
    assert custom_model.C_m == pytest.approx(1.0)
    assert custom_model.g_Na == pytest.approx(120.0)
    assert custom_model.g_K == pytest.approx(36.0)
    assert custom_model.g_L == pytest.approx(0.3)
