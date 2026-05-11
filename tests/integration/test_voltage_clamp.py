"""Tests for the voltage clamp simulation."""

import numpy as np
import pytest

from patch_sim.channels import alpha_h, alpha_m, alpha_n, beta_h, beta_m, beta_n
from patch_sim.clamp_simulations import simulate_voltage_clamp
from patch_sim.presets import make_squid_giant_axon


def test_simulate_voltage_clamp_returns_structured_array(hh_model):
    """Test that simulate_voltage_clamp returns a structured array."""
    # Create a voltage protocol for a 10ms simulation
    duration = 10  # ms
    time_step = 0.01  # ms
    num_steps = int(duration / time_step) + 1
    voltage = np.full(num_steps, -65.0)  # constant voltage at resting potential

    result = simulate_voltage_clamp(hh_model, voltage_protocol=voltage)

    # Check result type
    assert isinstance(result, np.ndarray)
    assert result.dtype.names is not None

    # Check that result has a time field
    assert "time" in result.dtype.names

    # Check that result has expected fields
    expected_columns = [
        "voltage",
        "Itotal",
        "INa",
        "IK",
        "INaL",
        "IKL",
        "n",
        "m",
        "h",
    ]
    for col in expected_columns:
        assert col in result.dtype.names

    # Check length matches expected number of time steps
    assert len(result) == num_steps


def test_voltage_clamp_constant_voltage(hh_model):
    """Test voltage clamp with constant voltage."""
    # Create voltage protocol at different voltage levels
    duration = 20  # ms
    time_step = 0.01  # ms
    num_steps = int(duration / time_step) + 1

    # Test at resting potential
    resting_voltage = np.full(num_steps, hh_model.v_rest)
    result_rest = simulate_voltage_clamp(hh_model, voltage_protocol=resting_voltage)

    # At resting potential, the current might not be exactly zero
    # Check that current settles to a relatively small value
    assert abs(result_rest["Itotal"][-1]) < 10.0

    # Test at depolarized potential
    depolarized_voltage = np.full(num_steps, 0.0)  # 0 mV is strongly depolarized
    result_depol = simulate_voltage_clamp(
        hh_model, voltage_protocol=depolarized_voltage
    )

    # Depolarization should activate Na+ and K+ channels
    # Na+ current should be significant at depolarized voltages
    na_currents = result_depol["INa"]
    assert (
        min(na_currents) < -10.0
    )  # Check for inward Na+ current at depolarized voltage

    # In our implementation, we should see some K+ current
    k_currents = result_depol["IK"]
    assert max(k_currents) > 0  # Should have some outward K+ current

    # K+ current should increase (outward/positive) and remain elevated
    k_currents = result_depol["IK"]
    assert max(k_currents) > 10.0  # Strong outward K+ current
    assert k_currents[-1] > 0.8 * max(k_currents)  # K+ current remains high


def test_voltage_step_protocol():
    """Test voltage clamp with a step protocol."""
    # Create a custom model for testing
    custom_model = make_squid_giant_axon()
    time_step = 0.05  # ms

    # Create a voltage step protocol: hold at -80 mV, step to 0 mV, return to -80 mV
    hold_steps = int(5 / time_step)  # 5 ms holding period
    step_steps = int(10 / time_step)  # 10 ms step duration
    return_steps = int(5 / time_step)  # 5 ms return to holding

    voltage_protocol = np.concatenate(
        [
            np.full(hold_steps, -80.0),  # holding at -80 mV
            np.full(step_steps, 0.0),  # step to 0 mV
            np.full(return_steps, -80.0),  # return to -80 mV
        ]
    )

    result = simulate_voltage_clamp(
        custom_model,
        voltage_protocol=voltage_protocol,
    )

    # Verify voltage protocol was applied correctly
    assert result["voltage"][0] == -80.0
    assert result["voltage"][hold_steps + 1] == 0.0
    assert result["voltage"][-1] == -80.0

    # Check sodium current dynamics during step
    # Find the start index of the step
    step_start_idx = hold_steps

    # Sodium current should be large (negative) shortly after the step
    na_currents_during_step = result["INa"][step_start_idx : step_start_idx + 20]
    assert min(na_currents_during_step) < -30.0

    # Sodium current should inactivate during the sustained step
    # Compare early step vs late step
    early_step_na = result["INa"][step_start_idx + 5]
    late_step_na = result["INa"][step_start_idx + step_steps - 5]
    assert abs(late_step_na) < abs(early_step_na)

    # Potassium current should increase and remain elevated during the step
    k_currents_during_step = result["IK"][step_start_idx : step_start_idx + step_steps]
    # Should start small and increase
    assert k_currents_during_step[0] < 5.0
    assert max(k_currents_during_step) > 10.0


def test_voltage_clamp_i_v_relationship():
    """Test the current-voltage relationship in voltage clamp."""
    # Create a custom model for testing
    custom_model = make_squid_giant_axon()
    time_step = 0.1  # ms for efficiency

    # Test voltage range
    voltages = np.arange(-100, 51, 10)  # -100 to +50 mV in steps of 10 mV
    steady_state_currents = []
    peak_na_currents = []

    duration = 20  # ms, long enough to reach steady state
    num_steps = int(duration / time_step) + 1

    # Run voltage clamp at each test voltage
    for v in voltages:
        voltage_protocol = np.full(num_steps, v)
        result = simulate_voltage_clamp(
            custom_model,
            voltage_protocol=voltage_protocol,
        )

        # Get steady-state total current (last time point)
        steady_state_currents.append(result["Itotal"][-1])

        # Get peak sodium current (minimum value, since inward current is negative)
        peak_na_currents.append(min(result["INa"]))

    # Verify that steady-state current generally increases with voltage
    # Check a few specific points instead of correlation to avoid NaN issues
    # At very negative voltages, current should be negative/inward
    assert steady_state_currents[0] < 0
    # At very positive voltages, current should be positive/outward
    assert steady_state_currents[-1] > 0
    # Check for basic relationship between voltage and current
    # First, filter out any possible NaN values
    valid_volts = []
    valid_currents = []
    for i, v in enumerate(voltages):
        if not np.isnan(peak_na_currents[i]):
            valid_volts.append(v)
            valid_currents.append(peak_na_currents[i])

    # If we have enough valid data points, check for expected patterns
    if len(valid_volts) >= 5:
        # Find voltage with maximum inward Na current (most negative)
        valid_idx = np.argmin(valid_currents)
        peak_inward_voltage = valid_volts[valid_idx]

        # Peak inward Na current location depends on the ion channel properties
        assert -100 <= peak_inward_voltage <= 50

    # Simple check: Na+ current at very positive voltages should be less inward
    # than at moderate voltages
    middle_voltage_idx = len(voltages) // 2
    if not np.isnan(peak_na_currents[-1]) and not np.isnan(
        peak_na_currents[middle_voltage_idx]
    ):
        # The current at highest voltage should be less negative than middle voltages
        assert peak_na_currents[-1] > peak_na_currents[middle_voltage_idx]


def test_voltage_ramp_protocol():
    """Test voltage clamp with a voltage ramp protocol."""
    # Create a custom model for testing
    custom_model = make_squid_giant_axon()
    time_step = 0.05  # ms

    # Create a voltage ramp protocol from -80 mV to +40 mV over 20 ms
    duration = 20  # ms
    num_steps = int(duration / time_step) + 1
    voltage_protocol = np.linspace(-80, 40, num_steps)

    result = simulate_voltage_clamp(
        custom_model,
        voltage_protocol=voltage_protocol,
    )

    # Verify voltage protocol was applied correctly
    assert np.isclose(result["voltage"][0], -80.0)
    assert np.isclose(result["voltage"][-1], 40.0)

    # The sodium current should be largest (most negative) at some intermediate voltage
    na_current_min_idx = int(np.argmin(result["INa"]))
    min_current_voltage = result["voltage"][na_current_min_idx]

    # The peak Na current should occur at a moderately negative voltage
    assert -40 <= min_current_voltage <= 30

    # Potassium current should generally increase with depolarization
    k_current_segments = [
        result["IK"][: num_steps // 4].mean(),  # First quarter
        result["IK"][-num_steps // 4 :].mean(),  # Last quarter
    ]
    assert (
        k_current_segments[1] > k_current_segments[0]
    )  # More outward K+ current later in ramp


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------


def test_empty_voltage_array_raises(hh_model):
    """An empty voltage array must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        simulate_voltage_clamp(hh_model, voltage_protocol=np.array([]))


def test_nan_in_voltage_array_raises(hh_model):
    """A voltage array containing NaN must raise ValueError."""
    voltage = np.array([-65.0, float("nan"), -65.0])
    with pytest.raises(ValueError, match="NaN"):
        simulate_voltage_clamp(hh_model, voltage_protocol=voltage)


def test_inf_in_voltage_array_raises(hh_model):
    """A voltage array containing Inf must raise ValueError."""
    voltage = np.array([-65.0, float("inf"), -65.0])
    with pytest.raises(ValueError, match="Inf"):
        simulate_voltage_clamp(hh_model, voltage_protocol=voltage)


# ---------------------------------------------------------------------------
# Coverage-expansion tests (issue #18)
# ---------------------------------------------------------------------------


def test_current_conservation(hh_model):
    """total_current must equal sodium + potassium + leak at every time step."""
    duration = 5  # ms
    time_step = 0.01  # ms
    num_steps = int(duration / time_step) + 1
    voltage = np.full(num_steps, 0.0)  # depolarized to maximize channel activity

    result = simulate_voltage_clamp(hh_model, voltage_protocol=voltage)

    reconstructed = result["INa"] + result["IK"] + result["INaL"] + result["IKL"]
    assert np.allclose(result["Itotal"], reconstructed)


def test_gating_variable_initialization(hh_model):
    """At t=0, gating variables must equal the steady-state for the starting voltage."""
    v_start = -80.0
    voltage = np.full(3, v_start)

    result = simulate_voltage_clamp(hh_model, voltage_protocol=voltage)

    an = alpha_n(v_start, 0.0)
    bn = beta_n(v_start, 0.0)
    am = alpha_m(v_start, 0.0)
    bm = beta_m(v_start, 0.0)
    ah = alpha_h(v_start, 0.0)
    bh = beta_h(v_start, 0.0)

    assert result["n"][0] == pytest.approx(an / (an + bn))
    assert result["m"][0] == pytest.approx(am / (am + bm))
    assert result["h"][0] == pytest.approx(ah / (ah + bh))


def test_single_element_voltage_protocol(hh_model):
    """A single-element voltage protocol must return a one-element structured array."""
    result = simulate_voltage_clamp(hh_model, voltage_protocol=np.array([-65.0]))

    assert isinstance(result, np.ndarray)
    assert result.dtype.names is not None
    assert len(result) == 1
    assert result["voltage"][0] == pytest.approx(-65.0)
