"""Electrochemical potential calculations for ions.

This module provides:

- :func:`nernst_potential` -- Nernst equation for a single ion species.
- :func:`goldman_potential` -- Goldman-Hodgkin-Katz voltage equation for a
  mixture of monovalent cations (and optionally anions) with different
  membrane permeabilities.
"""

from collections.abc import Sequence

import numpy as np

# Constants
R = 8.314  # Universal gas constant, J/(mol·K)
F = 96485  # Faraday's constant, C/mol


def nernst_potential(
    z: int, T: float, ion_concentration_out: float, ion_concentration_in: float
) -> float:
    """Calculate the Nernst potential for a given ion.

    Args:
        z: Valence of the ion (e.g., +1 for K+, +2 for Ca2+).
        T: Temperature in Kelvin.
        ion_concentration_out: Extracellular ion concentration (mM).
        ion_concentration_in: Intracellular ion concentration (mM).

    Returns:
        Nernst potential in millivolts (mV).
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


def goldman_potential(
    T: float,
    cation_terms: Sequence[tuple[float, float, float]],
    anion_terms: Sequence[tuple[float, float, float]] = (),
) -> float:
    """Calculate the Goldman-Hodgkin-Katz (GHK) reversal potential.

    Implements the GHK voltage equation for a mixture of monovalent ions:

        E = (RT/F) * ln(
                (sum_i P_i^cat * C_i_out + sum_j P_j^an * C_j_in)
              / (sum_i P_i^cat * C_i_in  + sum_j P_j^an * C_j_out)
            )

    where cations contribute their extracellular concentration to the
    numerator and intracellular to the denominator; anions are reversed.

    All ion species passed must be monovalent (|z| = 1).  For divalent ions
    such as Ca²⁺ use :func:`nernst_potential` directly.

    Args:
        T: Temperature in Kelvin.
        cation_terms: Sequence of ``(P, C_out, C_in)`` tuples for each
            monovalent cation, where *P* is the relative permeability
            (dimensionless), *C_out* the extracellular concentration (mM),
            and *C_in* the intracellular concentration (mM).
        anion_terms: Sequence of ``(P, C_out, C_in)`` tuples for each
            monovalent anion.  Defaults to an empty sequence (cations only).

    Returns:
        GHK reversal potential in millivolts (mV).

    Raises:
        ValueError: If *T* is not positive, if both *cation_terms* and
            *anion_terms* are empty, if any permeability is negative, or if
            any concentration is non-positive.
    """
    if T <= 0:
        raise ValueError("Temperature (T) must be positive (in Kelvin).")
    if not cation_terms and not anion_terms:
        raise ValueError("At least one ion term (cation or anion) must be provided.")

    all_terms = list(cation_terms) + list(anion_terms)
    for P, c_out, c_in in all_terms:
        if P < 0:
            raise ValueError(f"Permeability must be non-negative, got {P}.")
        if c_out <= 0:
            raise ValueError(
                f"Extracellular concentration must be positive, got {c_out}."
            )
        if c_in <= 0:
            raise ValueError(
                f"Intracellular concentration must be positive, got {c_in}."
            )

    # Numerator: cation C_out + anion C_in
    numerator = sum(P * c_out for P, c_out, _ in cation_terms) + sum(
        P * c_in for P, _, c_in in anion_terms
    )
    # Denominator: cation C_in + anion C_out
    denominator = sum(P * c_in for P, _, c_in in cation_terms) + sum(
        P * c_out for P, c_out, _ in anion_terms
    )

    if numerator <= 0 or denominator <= 0:
        raise ValueError(
            "GHK numerator and denominator must both be positive. "
            "Check that at least one permeability is non-zero."
        )

    return (R * T / F) * 1000 * np.log(numerator / denominator)
