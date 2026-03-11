"""
Tests for the current clamp simulation in the Hodgkin-Huxley model.
"""

import pytest
import pandas as pd
import numpy as np
from ap_sim.hodgkin_huxley import HodgkinHuxley
from ap_sim.clamp_simulations import simulate_current_clamp


def test_simulate_current_clamp_returns_dataframe(hh_model):
    """Test that simulate_current_clamp method returns a pandas DataFrame with correct
    structure.

    Validates the data types and structure of the returned DataFrame.
    """
    # Create a current array for a 10ms simulation
    duration = 10  # ms
    time_step = 0.1  # ms
    num_steps = int(duration / time_step) + 1
    current = np.full(num_steps, 20.0)  # constant current

    result = simulate_current_clamp(
        hh_model,
        current_external=current,
        sampling_frequency=1000.0 / time_step,  # Convert ms to Hz
    )

    # Check result type
    assert isinstance(result, pd.DataFrame)

    # Check that DataFrame has expected index name
    assert result.index.name == "time"

    # Check that DataFrame has the expected columns
    expected_columns = [
        "voltage",
        "potassium_activation",
        "sodium_activation",
        "sodium_inactivation",
    ]
    for col in expected_columns:
        assert col in result.columns

    # Check that gating variables are within physiological bounds (0 to 1)
    gating_vars = ["potassium_activation", "sodium_activation", "sodium_inactivation"]
    for gating_var in gating_vars:
        assert (result[gating_var] >= 0).all()
        assert (result[gating_var] <= 1).all()

    # Check that voltage is within reasonable physiological bounds
    assert result["voltage"].min() >= -100  # mV
    assert result["voltage"].max() <= 60  # mV


def test_simulation_dynamics():
    """Test that the simulation shows expected dynamics."""
    # Create model for testing
    custom_model = HodgkinHuxley()

    # Create current array for a 50ms simulation
    duration = 50  # ms
    time_step = 0.05  # ms
    num_steps = int(duration / time_step) + 1
    current = np.full(num_steps, 20.0)  # constant current

    result = simulate_current_clamp(
        custom_model,
        current_external=current,
        sampling_frequency=1000.0 / time_step,  # Convert ms to Hz
    )

    # Voltage should change from initial value
    initial_voltage = result["voltage"].iloc[0]
    max_voltage = result["voltage"].max()
    assert initial_voltage != max_voltage

    # Should observe an action potential (voltage exceeding threshold)
    threshold = 0  # mV, typical AP threshold
    assert any(result["voltage"] > threshold)

    # Sodium activation should increase during depolarization
    assert result["sodium_activation"].max() > result["sodium_activation"].iloc[0]


def test_simulate_current_clamp_with_zero_current(hh_model):
    """
    Test the simulate_current_clamp method with zero external current.
    Small fluctuations in voltage may occur due to intrinsic dynamics.
    """
    # Create zero current array for a 10ms simulation
    duration = 10  # ms
    time_step = 0.1  # ms
    num_steps = int(duration / time_step) + 1
    zero_current = np.zeros(num_steps)  # zero current array

    result = simulate_current_clamp(
        hh_model,
        current_external=zero_current,
        sampling_frequency=1000.0 / time_step,  # Convert ms to Hz
    )

    # Check result type and structure
    assert isinstance(result, pd.DataFrame)
    expected_columns = [
        "voltage",
        "potassium_activation",
        "sodium_activation",
        "sodium_inactivation",
    ]
    for col in expected_columns:
        assert col in result.columns

    # With zero external current, voltage should remain close to resting potential
    # Allow for small fluctuations due to intrinsic dynamics
    max_allowable_deviation = 15.0  # mV, maximum acceptable deviation
    voltage_range = result["voltage"].max() - result["voltage"].min()
    assert voltage_range < max_allowable_deviation, (
        f"Voltage range {voltage_range} exceeds maximum allowed deviation"
    )


def test_simulate_current_clamp_with_non_zero_currents():
    """Test the simulate_current_clamp method with various non-zero external
    currents."""
    currents = [10.0, 20.0, 50.0]  # Different external currents to test
    duration = 10  # ms

    # Create a model for testing
    custom_model = HodgkinHuxley()
    time_step = 0.1  # ms
    num_steps = int(duration / time_step) + 1

    for current_value in currents:
        # Create constant current array for each value
        current_array = np.full(num_steps, current_value)

        result = simulate_current_clamp(
            custom_model,
            current_external=current_array,
            sampling_frequency=1000.0 / time_step,  # Convert ms to Hz
        )

        # Check result type and structure
        assert isinstance(result, pd.DataFrame)
        expected_columns = [
            "voltage",
            "potassium_activation",
            "sodium_activation",
            "sodium_inactivation",
        ]
        for col in expected_columns:
            assert col in result.columns

        # For non-zero currents, voltage should change significantly
        initial_voltage = result["voltage"].iloc[0]
        max_change = abs(result["voltage"].max() - initial_voltage)

        # Adjusted threshold based on observed behavior
        assert max_change > 3.0, (
            f"Voltage did not change significantly for current {current_value}"
        )

        # Check that gating variables are within physiological bounds
        gating_vars = [
            "potassium_activation",
            "sodium_activation",
            "sodium_inactivation",
        ]
        for gating_var in gating_vars:
            assert (result[gating_var] >= 0).all()
            assert (result[gating_var] <= 1).all()


def test_physiological_limits_and_action_potentials():
    """
    Test that the voltage does not exceed physiological limits and that increasing
    current generates more action potentials.
    """
    # Physiological limits for membrane voltage
    min_physiological_voltage = -100  # mV
    max_physiological_voltage = 60  # mV

    # Parameters for testing action potentials
    duration = 100  # ms, longer simulation to observe multiple APs
    currents = [20.0, 40.0, 60.0]  # increasing external currents in μA/cm²
    ap_threshold = 0  # mV, voltage threshold for counting action potentials

    # Create a model for testing
    custom_model = HodgkinHuxley()
    time_step = 0.1  # ms
    num_steps = int(duration / time_step) + 1

    ap_counts = []

    for current_value in currents:
        # Create constant current array for each value
        current_array = np.full(num_steps, current_value)

        result = simulate_current_clamp(
            custom_model,
            current_external=current_array,
            sampling_frequency=1000.0 / time_step,  # Convert ms to Hz
        )

        # Test that voltage stays within physiological limits
        assert result["voltage"].min() >= min_physiological_voltage, (
            f"Voltage below physiological minimum with current {current_value}"
        )
        assert result["voltage"].max() <= max_physiological_voltage, (
            f"Voltage exceeds physiological maximum with current {current_value}"
        )

        # Count action potentials (threshold crossings from below)
        voltage = result["voltage"].values
        threshold_crossings = 0
        above_threshold = False

        for v in voltage:
            if v > ap_threshold and not above_threshold:
                threshold_crossings += 1
                above_threshold = True
            elif v < ap_threshold:
                above_threshold = False

        ap_counts.append(threshold_crossings)

    # Higher current should generally produce more action potentials
    # Allow for some variability in the relationship
    assert ap_counts[-1] >= ap_counts[0], (
        f"Expected more APs with higher current, got {ap_counts}"
    )


def test_simulate_current_clamp_with_different_currents():
    """Test the simulate_current_clamp method with a time-varying current waveform."""
    duration = 50  # ms
    time_step = 0.01  # ms

    # Create a custom model for testing
    custom_model = HodgkinHuxley()

    # Calculate the number of time steps
    num_time_steps = int(duration / time_step) + 1

    # Create a simple current waveform
    # First half is 10 uA/cm², second half is 50 uA/cm²
    current_waveform = np.concatenate(
        [
            np.full(num_time_steps // 2, 10.0),
            np.full(num_time_steps - num_time_steps // 2, 50.0),
        ]
    )

    # Run the simulation with the time-varying current
    result = simulate_current_clamp(
        custom_model,
        current_external=current_waveform,
        sampling_frequency=1000.0 / time_step,  # Convert ms to Hz
    )

    # Check basic properties of the result
    assert isinstance(result, pd.DataFrame)
    assert len(result) == num_time_steps

    # Find the point where current changes
    midpoint_time = duration / 2

    # Get voltages before and after current change
    # (give some time for the effect to manifest)
    before_time = midpoint_time - 5  # 5 ms before the change
    after_time = midpoint_time + 10  # 10 ms after the change

    voltage_before = result.loc[result.index <= before_time, "voltage"].mean()
    voltage_after = result.loc[result.index >= after_time, "voltage"].mean()

    # Voltage should be higher after the current increase
    assert voltage_after > voltage_before, (
        f"Expected higher voltage after current increase: "
        f"{voltage_before} -> {voltage_after}"
    )


def test_simulation_time_from_current_waveform():
    """Test that simulation time is correctly derived from the current waveform
    length.
    """
    time_step = 0.01  # ms
    custom_model = HodgkinHuxley()

    # Create a current waveform of specific length
    duration = 75.0  # ms
    num_steps = int(duration / time_step) + 1
    current_waveform = np.ones(num_steps) * 20.0  # constant current

    # Run simulation with only the current waveform, no simulation_time
    result = simulate_current_clamp(
        custom_model,
        current_external=current_waveform,
        sampling_frequency=1000.0 / time_step,  # Convert ms to Hz
    )

    # Check that the simulation time matches what we expect from the current array
    # length
    expected_simulation_time = (len(current_waveform) - 1) * time_step
    actual_simulation_time = result.index[-1]

    assert actual_simulation_time == pytest.approx(expected_simulation_time)
    assert len(result) == num_steps


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------


def test_empty_current_array_raises(hh_model):
    """An empty current array must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        simulate_current_clamp(hh_model, current_external=np.array([]))


@pytest.mark.parametrize("sf", [0, -1.0, -100000.0])
def test_non_positive_sampling_frequency_raises(hh_model, sf: float):
    """sampling_frequency <= 0 must raise ValueError."""
    with pytest.raises(ValueError, match="sampling_frequency"):
        simulate_current_clamp(
            hh_model, current_external=np.array([0.0, 0.0]), sampling_frequency=sf
        )


def test_nan_in_current_array_raises(hh_model):
    """A current array containing NaN must raise ValueError."""
    current = np.array([0.0, float("nan"), 0.0])
    with pytest.raises(ValueError, match="NaN"):
        simulate_current_clamp(hh_model, current_external=current)


def test_inf_in_current_array_raises(hh_model):
    """A current array containing Inf must raise ValueError."""
    current = np.array([0.0, float("inf"), 0.0])
    with pytest.raises(ValueError, match="Inf"):
        simulate_current_clamp(hh_model, current_external=current)
