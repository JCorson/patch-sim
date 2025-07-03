"""
This module implements the Hodgkin-Huxley model for simulating action potentials.
The model includes equations for ion channel dynamics and membrane voltage.
"""

import numpy as np
import pandas as pd
from typing import Union
from .nernst_neuron import nernst_potential
from .utils import safe_exp, FloatOrArray


class HodgkinHuxley:
    """
    Simulates the Hodgkin-Huxley model of action potentials.

    Attributes:
        C_m (float): Membrane capacitance in uF/cm^2.
        g_Na (float): Maximum sodium conductance in mS/cm^2.
        g_K (float): Maximum potassium conductance in mS/cm^2.
        g_L (float): Leak conductance in mS/cm^2.
        E_Na (float): Sodium reversal potential in mV.
        E_K (float): Potassium reversal potential in mV.
        E_L (float): Leak reversal potential in mV.
        v_rest (float): Resting potential in mV.
        time_step (float): Time step for simulation in ms.
    """

    def __init__(
        self,
        g_Na: float = 120.0,
        g_K: float = 36.0,
        g_L: float = 0.3,
        v_rest: float = -65.0,
        time_step: float = 0.01,
    ) -> None:
        """
        Initialize the Hodgkin-Huxley model with default or user-defined parameters.

        Args:
            g_Na (float): Sodium conductance in mS/cm^2.
            g_K (float): Potassium conductance in mS/cm^2.
            g_L (float): Leak conductance in mS/cm^2.
            v_rest (float): Resting membrane potential in mV.
            time_step (float): Time step for simulation in ms.
        """
        self.C_m: float = 1.0
        self.g_Na: float = g_Na
        self.g_K: float = g_K
        self.g_L: float = g_L
        self.v_rest: float = v_rest
        self.time_step: float = time_step

        # Ion concentrations (in mM)
        Na_out: float = 145.0  # Extracellular sodium
        Na_in: float = 15.0  # Intracellular sodium
        K_out: float = 5.0  # Extracellular potassium
        K_in: float = 140.0  # Intracellular potassium
        Cl_out: float = 120.0  # Extracellular chloride
        Cl_in: float = 10.0  # Intracellular chloride

        # Calculate reversal potentials using the Nernst equation
        # Temperature in Kelvin set to 310.15 K (37°C), which is typical for mammalian
        # cells.
        T: float = 310.15
        # Sodium reversal potential
        self.E_Na: float = nernst_potential(1, T, Na_out, Na_in)
        # Potassium reversal potential
        self.E_K: float = nernst_potential(1, T, K_out, K_in)
        # Leak reversal potential
        self.E_L: float = nernst_potential(-1, T, Cl_out, Cl_in)

    def alpha_n(self, V: FloatOrArray) -> FloatOrArray:
        """
        Calculate the rate constant alpha_n for potassium channel activation.

        Parameters:
            V (FloatOrArray): Membrane voltage in mV.

        Returns:
            FloatOrArray: The rate constant alpha_n.
        """
        # Handle array input
        if isinstance(V, (np.ndarray, pd.Series)):
            # Create a copy to avoid modifying the original array
            result = np.zeros_like(V, dtype=float)

            # Calculate values for non-edge cases
            mask = np.abs(V + 55) > 1e-6  # Points not at the singularity
            safe_V = V[mask]
            denominator = 1 - safe_exp(-(safe_V + 55) / 10)
            result[mask] = 0.01 * (safe_V + 55) / denominator

            # Handle near-singularity cases using L'Hôpital's rule approximation
            # When V ≈ -55, the function approaches 0.1
            # This value is derived from the limit as V approaches -55
            result[~mask] = 0.1

            return result
        else:
            # Handle scalar input
            if abs(V + 55) < 1e-6:
                # Handle near-singularity case
                # This is the limit as V approaches -55
                return 0.1
            else:
                denominator = 1 - safe_exp(-(V + 55) / 10)
                return 0.01 * (V + 55) / denominator

    def beta_n(self, V: FloatOrArray) -> FloatOrArray:
        """
        Calculate the rate constant beta_n for potassium channel deactivation.

        Parameters:
            V (FloatOrArray): Membrane voltage in mV.

        Returns:
            FloatOrArray: The rate constant beta_n.
        """
        return 0.125 * safe_exp(-(V + 65) / 80)

    def alpha_m(self, V: FloatOrArray) -> FloatOrArray:
        """
        Calculate the rate constant alpha_m for sodium channel activation.

        Parameters:
            V (FloatOrArray): Membrane voltage in mV.

        Returns:
            FloatOrArray: The rate constant alpha_m.
        """
        # Handle array input
        if isinstance(V, (np.ndarray, pd.Series)):
            # Create a copy to avoid modifying the original array
            result = np.zeros_like(V, dtype=float)

            # Calculate values for non-edge cases
            mask = np.abs(V + 40) > 1e-6  # Points not at the singularity
            safe_V = V[mask]
            denominator = 1 - safe_exp(-(safe_V + 40) / 10)
            result[mask] = 0.1 * (safe_V + 40) / denominator

            # Handle near-singularity cases using L'Hôpital's rule approximation
            # When V ≈ -40, the function approaches 1.0
            # This value is derived from the limit as V approaches -40
            result[~mask] = 1.0

            return result
        else:
            # Handle scalar input
            if abs(V + 40) < 1e-6:
                # Handle near-singularity case
                # This is the limit as V approaches -40
                return 1.0
            else:
                denominator = 1 - safe_exp(-(V + 40) / 10)
                return 0.1 * (V + 40) / denominator

    def beta_m(self, V: FloatOrArray) -> FloatOrArray:
        """
        Calculate the rate constant beta_m for sodium channel deactivation.

        Parameters:
            V (FloatOrArray): Membrane voltage in mV.

        Returns:
            FloatOrArray: The rate constant beta_m.
        """
        return 4.0 * safe_exp(-(V + 65) / 18)

    def alpha_h(self, V: FloatOrArray) -> FloatOrArray:
        """
        Calculate the rate constant alpha_h for sodium channel inactivation.

        Parameters:
            V (FloatOrArray): Membrane voltage in mV.

        Returns:
            FloatOrArray: The rate constant alpha_h.
        """
        return 0.07 * safe_exp(-(V + 65) / 20)

    def beta_h(self, V: FloatOrArray) -> FloatOrArray:
        """
        Calculate the rate constant beta_h for sodium channel reactivation.

        Parameters:
            V (FloatOrArray): Membrane voltage in mV.

        Returns:
            FloatOrArray: The rate constant beta_h.
        """
        return 1 / (1 + safe_exp(-(V + 35) / 10))


def simulate_voltage_clamp(
    neuron: HodgkinHuxley,
    voltage_protocol: Union[np.ndarray, list],
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

    Returns:
        pd.DataFrame: DataFrame with time points and corresponding current values
            including total_current, sodium_current, potassium_current,
            leak_current, as well as gating variables potassium_activation,
            sodium_activation, and sodium_inactivation.
    """
    # Convert voltage_protocol to numpy array if it's a list
    voltage_array = np.asarray(voltage_protocol)
    num_time_steps = len(voltage_array)

    # Calculate the actual simulation time
    actual_simulation_time = (num_time_steps - 1) * neuron.time_step

    # Create time array for the entire simulation
    time_array = np.round(
        np.arange(0, actual_simulation_time + neuron.time_step, neuron.time_step), 10
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
        new_potassium_activation = potassium_activation + dn * neuron.time_step
        new_sodium_activation = sodium_activation + dm * neuron.time_step
        new_sodium_inactivation = sodium_inactivation + dh * neuron.time_step

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
    neuron: HodgkinHuxley,
    current_external: Union[np.ndarray, list],
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

    Returns:
        pd.DataFrame: DataFrame with time points and corresponding voltage values,
            as well as gating variables potassium_activation, sodium_activation,
            and sodium_inactivation.
    """
    # Convert current_external to numpy array if it's a list
    current_array = np.asarray(current_external)
    num_time_steps = len(current_array)

    # Calculate the actual simulation time
    actual_simulation_time = (num_time_steps - 1) * neuron.time_step

    # Create time array for the entire simulation
    time_array = np.round(
        np.arange(0, actual_simulation_time + neuron.time_step, neuron.time_step), 10
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
        new_voltage = voltage + dV * neuron.time_step
        new_potassium_activation = potassium_activation + dn * neuron.time_step
        new_sodium_activation = sodium_activation + dm * neuron.time_step
        new_sodium_inactivation = sodium_inactivation + dh * neuron.time_step

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
