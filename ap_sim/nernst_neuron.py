"""
This module provides functionality to compute the Nernst potential for ions.

It includes:
- A function to calculate the Nernst potential for a single ion based on its valence,
  temperature, and concentration gradients.
"""

import numpy as np

# Constants
R = 8.314  # Universal gas constant, J/(mol·K)
F = 96485  # Faraday's constant, C/mol


def nernst_potential(
    z: int, T: float, ion_concentration_out: float, ion_concentration_in: float
) -> float:
    """
    Calculate the Nernst potential for a given ion.

    Parameters:
        z (int): Valence of the ion (e.g., +1 for K+, +2 for Ca2+).
        T (float): Temperature in Kelvin.
        ion_concentration_out (float): Extracellular ion concentration (mM).
        ion_concentration_in (float): Intracellular ion concentration (mM).

    Returns:
        float: Nernst potential in millivolts (mV).
    """
    if z == 0:
        raise ValueError("Valence (z) must not be zero.")
    if T <= 0:
        raise ValueError("Temperature (T) must be positive (in Kelvin).")
    if ion_concentration_out <= 0:
        raise ValueError("Extracellular ion concentration must be positive.")
    if ion_concentration_in <= 0:
        raise ValueError("Intracellular ion concentration must be positive.")

    # Convert to millivolts
    return (
        (R * T / (z * F)) * 1000 * np.log(ion_concentration_out / ion_concentration_in)
    )
