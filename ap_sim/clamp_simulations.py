"""
Clamp simulation functions for the Hodgkin-Huxley model.

This module contains functions for voltage clamp and current clamp experiments
that can be performed on Hodgkin-Huxley neuron objects.
"""

import numpy as np
import pandas as pd
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .hodgkin_huxley import HodgkinHuxley


def simulate_voltage_clamp(
    neuron: "HodgkinHuxley",
    voltage_protocol: Union[np.ndarray, list],
    sampling_frequency: float = 100000.0,  # Hz (100 kHz default)
) -> pd.DataFrame:
    """
    Simulate a voltage clamp experiment using the Hodgkin-Huxley model.

    In a voltage clamp experiment, the membrane potential is held at specified
    values and the current required to maintain those voltages is measured. This
    function simulates this process by computing the ionic currents that would flow
    at each voltage step in the protocol.

    Parameters:
        neuron (HodgkinHuxley): The Hodgkin-Huxley neuron object to simulate.
        voltage_protocol (Union[np.ndarray, list]): Voltage values in mV to clamp
            the membrane at for each time step. Must be an array/list for a
            time-varying voltage protocol. The length of the array determines the
            simulation duration.
        sampling_frequency (float): Sampling frequency in Hz for the simulation.
            Default is 100 kHz (0.01 ms time steps). Higher frequencies give
            finer temporal resolution but increase computation time.

    Returns:
        pd.DataFrame: DataFrame with time points and corresponding current values
            including total_current, sodium_current, potassium_current,
            leak_current, as well as gating variables potassium_activation,
            sodium_activation, and sodium_inactivation.
    """
    # Convert voltage_protocol to numpy array if it's a list
    voltage_array = np.asarray(voltage_protocol)
    num_time_steps = len(voltage_array)

    # Calculate time step from sampling frequency
    time_step = 1.0 / sampling_frequency * 1000.0  # Convert Hz to milliseconds

    # Calculate the actual simulation time
    actual_simulation_time = (num_time_steps - 1) * time_step

    # Create time array for the entire simulation
    time_array = np.round(
        np.arange(0, actual_simulation_time + time_step, time_step), 10
    )

    # Ensure the time array matches the voltage array length
    if len(time_array) != len(voltage_array):
        time_array = np.linspace(0, actual_simulation_time, num_time_steps)

    # Create the DataFrame at the start of the method
    results = pd.DataFrame(
        index=time_array,
        columns=[
            "voltage",
            "total_current",
            "sodium_current",
            "potassium_current",
            "leak_current",
            "potassium_activation",
            "sodium_activation",
            "sodium_inactivation",
        ],
        dtype=np.float64,  # Set float dtype for all columns
    )
    results.index.name = "time"

    # Initialize the first row of the DataFrame
    initial_voltage = voltage_array[0]
    results.loc[0, "voltage"] = initial_voltage
    results.loc[0, "potassium_activation"] = neuron.alpha_n(initial_voltage) / (
        neuron.alpha_n(initial_voltage) + neuron.beta_n(initial_voltage)
    )
    results.loc[0, "sodium_activation"] = neuron.alpha_m(initial_voltage) / (
        neuron.alpha_m(initial_voltage) + neuron.beta_m(initial_voltage)
    )
    results.loc[0, "sodium_inactivation"] = neuron.alpha_h(initial_voltage) / (
        neuron.alpha_h(initial_voltage) + neuron.beta_h(initial_voltage)
    )

    # Calculate initial currents
    sodium_activation = results.loc[0, "sodium_activation"]
    sodium_inactivation = results.loc[0, "sodium_inactivation"]
    potassium_activation = results.loc[0, "potassium_activation"]

    # Calculate conductances and currents
    conductance_Na = neuron.g_Na * (sodium_activation**3) * sodium_inactivation
    conductance_K = neuron.g_K * (potassium_activation**4)
    conductance_leak = neuron.g_L

    # Calculate individual ionic currents (outward current is positive)
    sodium_current = conductance_Na * (initial_voltage - neuron.E_Na)
    potassium_current = conductance_K * (initial_voltage - neuron.E_K)
    leak_current = conductance_leak * (initial_voltage - neuron.E_L)

    # Total current is the sum of all ionic currents
    total_current = sodium_current + potassium_current + leak_current

    results.loc[0, "sodium_current"] = sodium_current
    results.loc[0, "potassium_current"] = potassium_current
    results.loc[0, "leak_current"] = leak_current
    results.loc[0, "total_current"] = total_current

    # Iterate over the time index
    for i, t in enumerate(results.index[1:], start=1):
        previous_idx = i - 1
        previous_time = results.index[previous_idx]

        # In voltage clamp, voltage is controlled externally
        voltage = voltage_array[i]
        results.loc[t, "voltage"] = voltage

        # Get previous state variables
        potassium_activation = results.loc[previous_time, "potassium_activation"]
        sodium_activation = results.loc[previous_time, "sodium_activation"]
        sodium_inactivation = results.loc[previous_time, "sodium_inactivation"]

        # Calculate rate of change for gating variables
        dn = (
            neuron.alpha_n(voltage) * (1 - potassium_activation)
            - neuron.beta_n(voltage) * potassium_activation
        )
        dm = (
            neuron.alpha_m(voltage) * (1 - sodium_activation)
            - neuron.beta_m(voltage) * sodium_activation
        )
        dh = (
            neuron.alpha_h(voltage) * (1 - sodium_inactivation)
            - neuron.beta_h(voltage) * sodium_inactivation
        )

        # Update gating variables
        new_potassium_activation = potassium_activation + dn * time_step
        new_sodium_activation = sodium_activation + dm * time_step
        new_sodium_inactivation = sodium_inactivation + dh * time_step

        # Ensure gating variables remain within physiological bounds (0 to 1)
        results.loc[t, "potassium_activation"] = np.clip(new_potassium_activation, 0, 1)
        results.loc[t, "sodium_activation"] = np.clip(new_sodium_activation, 0, 1)
        results.loc[t, "sodium_inactivation"] = np.clip(new_sodium_inactivation, 0, 1)

        # Calculate conductances and currents with updated gating variables
        conductance_Na = (
            neuron.g_Na * (new_sodium_activation**3) * new_sodium_inactivation
        )
        conductance_K = neuron.g_K * (new_potassium_activation**4)
        conductance_leak = neuron.g_L

        # Calculate individual ionic currents (outward current is positive)
        sodium_current = conductance_Na * (voltage - neuron.E_Na)
        potassium_current = conductance_K * (voltage - neuron.E_K)
        leak_current = conductance_leak * (voltage - neuron.E_L)

        # Total current is the sum of all ionic currents
        total_current = sodium_current + potassium_current + leak_current

        results.loc[t, "sodium_current"] = sodium_current
        results.loc[t, "potassium_current"] = potassium_current
        results.loc[t, "leak_current"] = leak_current
        results.loc[t, "total_current"] = total_current

    return results


def simulate_current_clamp(
    neuron: "HodgkinHuxley",
    current_external: Union[np.ndarray, list],
    sampling_frequency: float = 100000.0,  # Hz (100 kHz default)
) -> pd.DataFrame:
    """
    Simulate a current clamp experiment using the Hodgkin-Huxley model.

    In a current clamp experiment, current is injected into the cell membrane and
    the resulting voltage changes are recorded. This function simulates this process
    by computing the membrane voltage over time in response to the specified
    external current.

    Parameters:
        neuron (HodgkinHuxley): The Hodgkin-Huxley neuron object to simulate.
        current_external (Union[np.ndarray, list]): External current in uA/cm^2.
            Must be an array/list for a time-varying current waveform.
            The length of the array determines the simulation duration.
        sampling_frequency (float): Sampling frequency in Hz for the simulation.
            Default is 100 kHz (0.01 ms time steps). Higher frequencies give
            finer temporal resolution but increase computation time.

    Returns:
        pd.DataFrame: DataFrame with time points and corresponding voltage values,
            as well as gating variables potassium_activation, sodium_activation,
            and sodium_inactivation.
    """
    # Convert current_external to numpy array if it's a list
    current_array = np.asarray(current_external)
    num_time_steps = len(current_array)

    # Calculate time step from sampling frequency
    time_step = 1.0 / sampling_frequency * 1000.0  # Convert Hz to milliseconds

    # Calculate the actual simulation time
    actual_simulation_time = (num_time_steps - 1) * time_step

    # Create time array for the entire simulation
    time_array = np.round(
        np.arange(0, actual_simulation_time + time_step, time_step), 10
    )

    # Ensure the time array matches the current array length
    if len(time_array) != len(current_array):
        time_array = np.linspace(0, actual_simulation_time, num_time_steps)

    # Create the DataFrame at the start of the method
    results = pd.DataFrame(
        index=time_array,
        columns=[
            "voltage",
            "potassium_activation",
            "sodium_activation",
            "sodium_inactivation",
        ],
        dtype=np.float64,  # Set float dtype for all columns
    )
    results.index.name = "time"

    # Initialize the first row of the DataFrame
    results.loc[0, "voltage"] = neuron.v_rest
    results.loc[0, "potassium_activation"] = neuron.alpha_n(neuron.v_rest) / (
        neuron.alpha_n(neuron.v_rest) + neuron.beta_n(neuron.v_rest)
    )
    results.loc[0, "sodium_activation"] = neuron.alpha_m(neuron.v_rest) / (
        neuron.alpha_m(neuron.v_rest) + neuron.beta_m(neuron.v_rest)
    )
    results.loc[0, "sodium_inactivation"] = neuron.alpha_h(neuron.v_rest) / (
        neuron.alpha_h(neuron.v_rest) + neuron.beta_h(neuron.v_rest)
    )

    # Define physiological limits for membrane voltage
    min_voltage = -100.0  # mV
    max_voltage = 60.0  # mV

    # Iterate over the time index
    for i, t in enumerate(results.index[1:], start=1):
        previous_idx = i - 1
        previous_time = results.index[previous_idx]
        voltage = results.loc[previous_time, "voltage"]
        potassium_activation = results.loc[previous_time, "potassium_activation"]
        sodium_activation = results.loc[previous_time, "sodium_activation"]
        sodium_inactivation = results.loc[previous_time, "sodium_inactivation"]

        conductance_Na = neuron.g_Na * (sodium_activation**3) * sodium_inactivation
        conductance_K = neuron.g_K * (potassium_activation**4)
        conductance_leak = neuron.g_L

        current_Na = conductance_Na * (voltage - neuron.E_Na)
        current_K = conductance_K * (voltage - neuron.E_K)
        current_leak = conductance_leak * (voltage - neuron.E_L)

        # Use the current from the current_array at this time step
        current_at_time = current_array[i]

        dV = (current_at_time - current_Na - current_K - current_leak) / neuron.C_m
        dn = (
            neuron.alpha_n(voltage) * (1 - potassium_activation)
            - neuron.beta_n(voltage) * potassium_activation
        )
        dm = (
            neuron.alpha_m(voltage) * (1 - sodium_activation)
            - neuron.beta_m(voltage) * sodium_activation
        )
        dh = (
            neuron.alpha_h(voltage) * (1 - sodium_inactivation)
            - neuron.beta_h(voltage) * sodium_inactivation
        )

        # Calculate new values
        new_voltage = voltage + dV * time_step
        new_potassium_activation = potassium_activation + dn * time_step
        new_sodium_activation = sodium_activation + dm * time_step
        new_sodium_inactivation = sodium_inactivation + dh * time_step

        # Ensure values remain within physiological bounds
        results.at[t, "voltage"] = float(np.clip(new_voltage, min_voltage, max_voltage))
        results.at[t, "potassium_activation"] = float(
            np.clip(new_potassium_activation, 0, 1)
        )
        results.at[t, "sodium_activation"] = float(np.clip(new_sodium_activation, 0, 1))
        results.at[t, "sodium_inactivation"] = float(
            np.clip(new_sodium_inactivation, 0, 1)
        )

    return results
