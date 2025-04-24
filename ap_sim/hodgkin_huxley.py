"""
This module implements the Hodgkin-Huxley model for simulating action potentials.
The model includes equations for ion channel dynamics and membrane voltage.
"""

import numpy as np
import pandas as pd
from .nernst_neuron import nernst_potential


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
    """

    def __init__(self, g_Na=120.0, g_K=36.0, g_L=0.3, v_rest=-65.0):
        """
        Initialize the Hodgkin-Huxley model with default or user-defined parameters.

        Args:
            g_Na (float): Maximum sodium conductance in mS/cm^2.
            g_K (float): Maximum potassium conductance in mS/cm^2.
            g_L (float): Leak conductance in mS/cm^2.
            v_rest (float): Resting potential in mV.
        """
        self.C_m = 1.0  # Membrane capacitance, in uF/cm^2
        self.g_Na = g_Na  # Maximum conductances, in mS/cm^2
        self.g_K = g_K
        self.g_L = g_L
        self.v_rest = v_rest  # Resting potential

        # Ion concentrations (in mM)
        Na_out = 145.0  # Extracellular sodium
        Na_in = 15.0  # Intracellular sodium
        K_out = 5.0  # Extracellular potassium
        K_in = 140.0  # Intracellular potassium
        Cl_out = 120.0  # Extracellular chloride
        Cl_in = 10.0  # Intracellular chloride

        # Calculate reversal potentials using the Nernst equation
        T = 310.15  # Temperature in Kelvin (37°C)
        self.E_Na = nernst_potential(1, T, Na_out, Na_in)  # Sodium reversal potential
        self.E_K = nernst_potential(1, T, K_out, K_in)  # Potassium reversal potential
        self.E_L = nernst_potential(-1, T, Cl_out, Cl_in)  # Leak reversal potential

    @staticmethod
    def safe_exp(x: float) -> float:
        """
        Safely compute the exponential to avoid overflow.

        Parameters:
            x (float): The input value.

        Returns:
            float: The computed exponential value, capped to prevent overflow.
        """
        return np.exp(np.clip(x, -100, 100))

    def alpha_n(self, V: float) -> float:
        """
        Calculate the rate constant alpha_n for potassium channel activation.

        Parameters:
            V (float): Membrane voltage in mV.

        Returns:
            float: The rate constant alpha_n.
        """
        return 0.01 * (V + 55) / (1 - self.safe_exp(-(V + 55) / 10))

    def beta_n(self, V: float) -> float:
        """
        Calculate the rate constant beta_n for potassium channel deactivation.

        Parameters:
            V (float): Membrane voltage in mV.

        Returns:
            float: The rate constant beta_n.
        """
        return 0.125 * self.safe_exp(-(V + 65) / 80)

    def alpha_m(self, V: float) -> float:
        """
        Calculate the rate constant alpha_m for sodium channel activation.

        Parameters:
            V (float): Membrane voltage in mV.

        Returns:
            float: The rate constant alpha_m.
        """
        return 0.1 * (V + 40) / (1 - self.safe_exp(-(V + 40) / 10))

    def beta_m(self, V: float) -> float:
        """
        Calculate the rate constant beta_m for sodium channel deactivation.

        Parameters:
            V (float): Membrane voltage in mV.

        Returns:
            float: The rate constant beta_m.
        """
        return 4.0 * self.safe_exp(-(V + 65) / 18)

    def alpha_h(self, V: float) -> float:
        """
        Calculate the rate constant alpha_h for sodium channel inactivation.

        Parameters:
            V (float): Membrane voltage in mV.

        Returns:
            float: The rate constant alpha_h.
        """
        return 0.07 * self.safe_exp(-(V + 65) / 20)

    def beta_h(self, V: float) -> float:
        """
        Calculate the rate constant beta_h for sodium channel reactivation.

        Parameters:
            V (float): Membrane voltage in mV.

        Returns:
            float: The rate constant beta_h.
        """
        return 1 / (1 + self.safe_exp(-(V + 35) / 10))

    def compute(
        self, simulation_time: float = 50, time_step: float = 0.01
    ) -> pd.DataFrame:
        """
        Simulate the membrane voltage over time using the Hodgkin-Huxley model.

        Parameters:
            simulation_time (float): Total simulation time in ms.
            time_step (float): Time step in ms.

        Returns:
            pd.DataFrame: DataFrame containing time points and corresponding voltage values,
                         as well as gating variables potassium_activation, sodium_activation, and sodium_inactivation.
        """
        # Create the DataFrame at the start of the method
        results = pd.DataFrame(
            index=np.arange(0, simulation_time + time_step, time_step),
            columns=["voltage", "potassium_activation", "sodium_activation", "sodium_inactivation"],
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

        current_external = 20.0  # Increased external current, in uA/cm^2

        # Ensure consistent precision for time index calculations
        results.index = results.index.round(10)

        # Iterate over the time index
        for t in results.index[1:]:
            previous_time = results.index[results.index.get_loc(t) - 1]
            voltage = results.loc[previous_time, "voltage"]
            potassium_activation = results.loc[previous_time, "potassium_activation"]
            sodium_activation = results.loc[previous_time, "sodium_activation"]
            sodium_inactivation = results.loc[previous_time, "sodium_inactivation"]

            conductance_Na = self.g_Na * (sodium_activation ** 3) * sodium_inactivation
            conductance_K = self.g_K * (potassium_activation ** 4)
            conductance_leak = self.g_L

            current_Na = conductance_Na * (voltage - self.E_Na)
            current_K = conductance_K * (voltage - self.E_K)
            current_leak = conductance_leak * (voltage - self.E_L)

            dV = (current_external - current_Na - current_K - current_leak) / self.C_m
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

            # Ensure gating variables and voltage remain within physiological bounds
            voltage = np.clip(voltage, -100, 100)
            potassium_activation = np.clip(potassium_activation, 0, 1)
            sodium_activation = np.clip(sodium_activation, 0, 1)
            sodium_inactivation = np.clip(sodium_inactivation, 0, 1)

            # Update results with bounded values
            results.loc[t, "voltage"] = voltage + dV * time_step
            results.loc[t, "potassium_activation"] = np.clip(potassium_activation + dn * time_step, 0, 1)
            results.loc[t, "sodium_activation"] = np.clip(sodium_activation + dm * time_step, 0, 1)
            results.loc[t, "sodium_inactivation"] = np.clip(sodium_inactivation + dh * time_step, 0, 1)

        return results


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create an instance of the HodgkinHuxley model
    hh_model = HodgkinHuxley()

    # Run the simulation
    simulation_time = 50  # in ms
    time_step = 0.01  # in ms
    results = hh_model.compute(simulation_time=simulation_time, time_step=time_step)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(results.index, results["voltage"], label="Membrane Voltage (mV)")
    plt.title("Hodgkin-Huxley Model Simulation")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.grid()
    plt.show()
