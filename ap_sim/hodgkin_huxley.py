"""
This module implements the Hodgkin-Huxley model for simulating action potentials.
The model includes equations for ion channel dynamics and membrane voltage.
"""

from dataclasses import dataclass
from functools import cached_property
import numpy as np
import pandas as pd
from .nernst_neuron import nernst_potential
from .utils import safe_exp, FloatOrArray


@dataclass
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
        Na_out (float): Extracellular sodium concentration in mM.
        Na_in (float): Intracellular sodium concentration in mM.
        K_out (float): Extracellular potassium concentration in mM.
        K_in (float): Intracellular potassium concentration in mM.
        Cl_out (float): Extracellular chloride concentration in mM.
        Cl_in (float): Intracellular chloride concentration in mM.
        T (float): Temperature in Kelvin.
    """

    # Membrane properties
    g_Na: float = 120.0
    g_K: float = 36.0
    g_L: float = 0.3
    C_m: float = 1.0
    v_rest: float = -65.0

    # Ion concentrations (in mM)
    Na_out: float = 145.0
    Na_in: float = 15.0
    K_out: float = 5.0
    K_in: float = 140.0
    Cl_out: float = 120.0
    Cl_in: float = 10.0

    # Temperature in Kelvin (37°C for mammalian cells)
    T: float = 310.15

    @cached_property
    def E_Na(self) -> float:
        """Sodium reversal potential in mV."""
        return nernst_potential(1, self.T, self.Na_out, self.Na_in)

    @cached_property
    def E_K(self) -> float:
        """Potassium reversal potential in mV."""
        return nernst_potential(1, self.T, self.K_out, self.K_in)

    @cached_property
    def E_L(self) -> float:
        """Leak reversal potential in mV."""
        return nernst_potential(-1, self.T, self.Cl_out, self.Cl_in)

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
