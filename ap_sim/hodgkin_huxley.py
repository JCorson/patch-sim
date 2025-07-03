"""
This module implements the Hodgkin-Huxley model for simulating action potentials.
The model includes equations for ion channel dynamics and membrane voltage.
"""

import numpy as np
import pandas as pd
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
