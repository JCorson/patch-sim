"""
This module implements the Hodgkin-Huxley model for simulating action potentials.
The model includes equations for ion channel dynamics and membrane voltage.
"""

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Union
from .nernst_neuron import nernst_potential

# Type aliases for better readability
FloatOrArray = Union[float, NDArray[np.float64], pd.Series]


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
            g_Na (float): Maximum sodium conductance in mS/cm^2.
            g_K (float): Maximum potassium conductance in mS/cm^2.
            g_L (float): Leak conductance in mS/cm^2.
            v_rest (float): Resting potential in mV.
            time_step (float): Time step for simulation in ms.
        """
        self.C_m: float = 1.0  # Membrane capacitance, in uF/cm^2
        self.g_Na: float = g_Na  # Maximum conductances, in mS/cm^2
        self.g_K: float = g_K
        self.g_L: float = g_L
        self.v_rest: float = v_rest  # Resting potential
        self.time_step: float = time_step  # Time step for simulation

        # Ion concentrations (in mM)
        Na_out: float = 145.0  # Extracellular sodium
        Na_in: float = 15.0  # Intracellular sodium
        K_out: float = 5.0  # Extracellular potassium
        K_in: float = 140.0  # Intracellular potassium
        Cl_out: float = 120.0  # Extracellular chloride
        Cl_in: float = 10.0  # Intracellular chloride

        # Calculate reversal potentials using the Nernst equation
        T: float = 310.15  # Temperature in Kelvin (37°C)
        # Sodium reversal potential
        self.E_Na: float = nernst_potential(1, T, Na_out, Na_in)
        # Potassium reversal potential
        self.E_K: float = nernst_potential(1, T, K_out, K_in)
        # Leak reversal potential
        self.E_L: float = nernst_potential(-1, T, Cl_out, Cl_in)

    @staticmethod
    def safe_exp(x: FloatOrArray) -> FloatOrArray:
        """
        Safely compute the exponential to avoid overflow.

        Parameters:
            x (FloatOrArray): The input value or array.

        Returns:
            FloatOrArray: The computed exponential value, capped to prevent overflow.
        """
        return np.exp(np.clip(x, -100, 100))

    def alpha_n(self, V: FloatOrArray) -> FloatOrArray:
        """
        Calculate the rate constant alpha_n for potassium channel activation.

        Parameters:
            V (FloatOrArray): Membrane voltage in mV.

        Returns:
            FloatOrArray: The rate constant alpha_n.
        """
        return 0.01 * (V + 55) / (1 - self.safe_exp(-(V + 55) / 10))

    def beta_n(self, V: FloatOrArray) -> FloatOrArray:
        """
        Calculate the rate constant beta_n for potassium channel deactivation.

        Parameters:
            V (FloatOrArray): Membrane voltage in mV.

        Returns:
            FloatOrArray: The rate constant beta_n.
        """
        return 0.125 * self.safe_exp(-(V + 65) / 80)

    def alpha_m(self, V: FloatOrArray) -> FloatOrArray:
        """
        Calculate the rate constant alpha_m for sodium channel activation.

        Parameters:
            V (FloatOrArray): Membrane voltage in mV.

        Returns:
            FloatOrArray: The rate constant alpha_m.
        """
        return 0.1 * (V + 40) / (1 - self.safe_exp(-(V + 40) / 10))

    def beta_m(self, V: FloatOrArray) -> FloatOrArray:
        """
        Calculate the rate constant beta_m for sodium channel deactivation.

        Parameters:
            V (FloatOrArray): Membrane voltage in mV.

        Returns:
            FloatOrArray: The rate constant beta_m.
        """
        return 4.0 * self.safe_exp(-(V + 65) / 18)

    def alpha_h(self, V: FloatOrArray) -> FloatOrArray:
        """
        Calculate the rate constant alpha_h for sodium channel inactivation.

        Parameters:
            V (FloatOrArray): Membrane voltage in mV.

        Returns:
            FloatOrArray: The rate constant alpha_h.
        """
        return 0.07 * self.safe_exp(-(V + 65) / 20)

    def beta_h(self, V: FloatOrArray) -> FloatOrArray:
        """
        Calculate the rate constant beta_h for sodium channel reactivation.

        Parameters:
            V (FloatOrArray): Membrane voltage in mV.

        Returns:
            FloatOrArray: The rate constant beta_h.
        """
        return 1 / (1 + self.safe_exp(-(V + 35) / 10))

    def compute(
        self,
        current_external: Union[np.ndarray, list],
    ) -> pd.DataFrame:
        """
        Simulate the membrane voltage over time using the Hodgkin-Huxley model.

        Parameters:
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
        actual_simulation_time = (num_time_steps - 1) * self.time_step

        # Create time array for the entire simulation
        time_array = np.round(
            np.arange(0, actual_simulation_time + self.time_step, self.time_step), 10
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
        results.loc[0, "voltage"] = self.v_rest
        results.loc[0, "potassium_activation"] = self.alpha_n(self.v_rest) / (
            self.alpha_n(self.v_rest) + self.beta_n(self.v_rest)
        )
        results.loc[0, "sodium_activation"] = self.alpha_m(self.v_rest) / (
            self.alpha_m(self.v_rest) + self.beta_m(self.v_rest)
        )
        results.loc[0, "sodium_inactivation"] = self.alpha_h(self.v_rest) / (
            self.alpha_h(self.v_rest) + self.beta_h(self.v_rest)
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

            conductance_Na = self.g_Na * (sodium_activation**3) * sodium_inactivation
            conductance_K = self.g_K * (potassium_activation**4)
            conductance_leak = self.g_L

            current_Na = conductance_Na * (voltage - self.E_Na)
            current_K = conductance_K * (voltage - self.E_K)
            current_leak = conductance_leak * (voltage - self.E_L)

            # Use the current from the current_array at this time step
            current_at_time = current_array[i]

            dV = (current_at_time - current_Na - current_K - current_leak) / self.C_m
            dn = (
                self.alpha_n(voltage) * (1 - potassium_activation)
                - self.beta_n(voltage) * potassium_activation
            )
            dm = (
                self.alpha_m(voltage) * (1 - sodium_activation)
                - self.beta_m(voltage) * sodium_activation
            )
            dh = (
                self.alpha_h(voltage) * (1 - sodium_inactivation)
                - self.beta_h(voltage) * sodium_inactivation
            )

            # Calculate new values
            new_voltage = voltage + dV * self.time_step
            new_potassium_activation = potassium_activation + dn * self.time_step
            new_sodium_activation = sodium_activation + dm * self.time_step
            new_sodium_inactivation = sodium_inactivation + dh * self.time_step

            # Ensure values remain within physiological bounds
            results.at[t, "voltage"] = float(
                np.clip(new_voltage, min_voltage, max_voltage)
            )
            results.at[t, "potassium_activation"] = float(
                np.clip(new_potassium_activation, 0, 1)
            )
            results.at[t, "sodium_activation"] = float(
                np.clip(new_sodium_activation, 0, 1)
            )
            results.at[t, "sodium_inactivation"] = float(
                np.clip(new_sodium_inactivation, 0, 1)
            )

        return results
