"""
This module implements the Hodgkin-Huxley model for simulating action potentials.
The model includes equations for ion channel dynamics and membrane voltage.
"""

import numpy as np
import pandas as pd
from nernst_neuron import nernst_potential


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
    """

    def __init__(self):
        """
        Initialize the Hodgkin-Huxley model with default parameters.
        """
        self.C_m = 1.0  # Membrane capacitance, in uF/cm^2
        self.g_Na = 120.0  # Maximum conductances, in mS/cm^2
        self.g_K = 36.0
        self.g_L = 0.3

        # Ion concentrations (in mM)
        Na_out = 145.0  # Extracellular sodium
        Na_in = 15.0  # Intracellular sodium
        K_out = 5.0  # Extracellular potassium
        K_in = 140.0  # Intracellular potassium
        Cl_out = 120.0  # Extracellular chloride
        Cl_in = 10.0  # Intracellular chloride

        # Calculate reversal potentials using the Nernst equation
        self.E_Na = nernst_potential(Na_out, Na_in, z=1)  # Sodium reversal potential
        self.E_K = nernst_potential(K_out, K_in, z=1)  # Potassium reversal potential
        self.E_L = nernst_potential(Cl_out, Cl_in, z=-1)  # Leak reversal potential

    def alpha_n(self, V: float) -> float:
        """
        Calculate the rate constant alpha_n for potassium channel activation.

        Parameters:
            V (float): Membrane voltage in mV.

        Returns:
            float: The rate constant alpha_n.
        """
        return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

    def beta_n(self, V: float) -> float:
        """
        Calculate the rate constant beta_n for potassium channel deactivation.

        Parameters:
            V (float): Membrane voltage in mV.

        Returns:
            float: The rate constant beta_n.
        """
        return 0.125 * np.exp(-(V + 65) / 80)

    def alpha_m(self, V: float) -> float:
        """
        Calculate the rate constant alpha_m for sodium channel activation.

        Parameters:
            V (float): Membrane voltage in mV.

        Returns:
            float: The rate constant alpha_m.
        """
        return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

    def beta_m(self, V: float) -> float:
        """
        Calculate the rate constant beta_m for sodium channel deactivation.

        Parameters:
            V (float): Membrane voltage in mV.

        Returns:
            float: The rate constant beta_m.
        """
        return 4.0 * np.exp(-(V + 65) / 18)

    def alpha_h(self, V: float) -> float:
        """
        Calculate the rate constant alpha_h for sodium channel inactivation.

        Parameters:
            V (float): Membrane voltage in mV.

        Returns:
            float: The rate constant alpha_h.
        """
        return 0.07 * np.exp(-(V + 65) / 20)

    def beta_h(self, V: float) -> float:
        """
        Calculate the rate constant beta_h for sodium channel reactivation.

        Parameters:
            V (float): Membrane voltage in mV.

        Returns:
            float: The rate constant beta_h.
        """
        return 1 / (1 + np.exp(-(V + 35) / 10))

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
        time = np.arange(0, simulation_time + time_step, time_step)
        voltage = np.zeros(len(time))
        potassium_activation = np.zeros(len(time))  # n
        sodium_activation = np.zeros(len(time))  # m
        sodium_inactivation = np.zeros(len(time))  # h

        voltage[0] = -65.0
        potassium_activation[0] = self.alpha_n(voltage[0]) / (
            self.alpha_n(voltage[0]) + self.beta_n(voltage[0])
        )
        sodium_activation[0] = self.alpha_m(voltage[0]) / (
            self.alpha_m(voltage[0]) + self.beta_m(voltage[0])
        )
        sodium_inactivation[0] = self.alpha_h(voltage[0]) / (
            self.alpha_h(voltage[0]) + self.beta_h(voltage[0])
        )

        current_external = 10.0  # External current, in uA/cm^2

        for i in range(1, len(time)):
            conductance_Na = (
                self.g_Na * (sodium_activation[i - 1] ** 3) * sodium_inactivation[i - 1]
            )
            conductance_K = self.g_K * (potassium_activation[i - 1] ** 4)
            conductance_leak = self.g_L

            current_Na = conductance_Na * (voltage[i - 1] - self.E_Na)
            current_K = conductance_K * (voltage[i - 1] - self.E_K)
            current_leak = conductance_leak * (voltage[i - 1] - self.E_L)

            dV = (current_external - current_Na - current_K - current_leak) / self.C_m
            dn = (
                self.alpha_n(voltage[i - 1]) * (1 - potassium_activation[i - 1])
                - self.beta_n(voltage[i - 1]) * potassium_activation[i - 1]
            )
            dm = (
                self.alpha_m(voltage[i - 1]) * (1 - sodium_activation[i - 1])
                - self.beta_m(voltage[i - 1]) * sodium_activation[i - 1]
            )
            dh = (
                self.alpha_h(voltage[i - 1]) * (1 - sodium_inactivation[i - 1])
                - self.beta_h(voltage[i - 1]) * sodium_inactivation[i - 1]
            )

            voltage[i] = voltage[i - 1] + dV * time_step
            potassium_activation[i] = potassium_activation[i - 1] + dn * time_step
            sodium_activation[i] = sodium_activation[i - 1] + dm * time_step
            sodium_inactivation[i] = sodium_inactivation[i - 1] + dh * time_step

        # Create and return pandas DataFrame
        results = pd.DataFrame(
            {
                "voltage": voltage,
                "potassium_activation": potassium_activation,
                "sodium_activation": sodium_activation,
                "sodium_inactivation": sodium_inactivation,
            },
            index=time,
        )
        results.index.name = "time"

        return results
