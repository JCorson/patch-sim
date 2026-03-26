"""Electrochemical potential calculations for ions.

This module provides:

- :func:`nernst_potential` -- Nernst equation for a single ion species.
- :func:`goldman_potential` -- Goldman-Hodgkin-Katz voltage equation for a
  mixture of monovalent cations (and optionally anions) with different
  membrane permeabilities.
- :class:`BoltzmannCoshRate` -- picklable callable for Boltzmann/cosh rate
  functions used in gating variable kinetics.
- :func:`boltzmann_cosh_rates` -- factory returning (alpha, beta) rate
  callable pairs from Boltzmann/cosh kinetic parameters.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .utils import safe_exp

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


@dataclass(frozen=True)
class BoltzmannCoshRate:
    """Picklable callable implementing a single Boltzmann/cosh rate function.

    Stores all kinetic parameters and computes either the alpha or beta rate
    on each call.  Being a frozen dataclass with only plain numeric fields,
    instances are fully picklable and safe to pass to worker processes via
    :class:`concurrent.futures.ProcessPoolExecutor`.

    Attributes:
        half: Half-activation voltage in mV.
        slope: Activation slope in mV.
        tau_scale: Numerator of the time-constant expression in ms.
        tau_floor: Minimum allowed time constant in ms.
        inverted: If True the Boltzmann curve is inverted (gate open at
            hyperpolarised potentials).
        cosh_scale: Voltage scale in the cosh denominator in mV.
        tau_rate: Extra multiplier in the tau denominator.
        is_alpha: If True, computes alpha = inf / tau; otherwise beta =
            (1 - inf) / tau.
    """

    half: float
    slope: float
    tau_scale: float
    tau_floor: float
    inverted: bool
    cosh_scale: float
    tau_rate: float
    is_alpha: bool

    def __call__(self, V: float, ca_i: float) -> float:
        """Compute the rate at voltage V.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular calcium concentration in mM (ignored).

        Returns:
            The rate (alpha or beta) in 1/ms.
        """
        if self.inverted:
            inf = 1.0 / (1.0 + safe_exp((V - self.half) / self.slope))
        else:
            inf = 1.0 / (1.0 + safe_exp(-(V - self.half) / self.slope))
        tau = self.tau_scale / (
            self.tau_rate * math.cosh((V - self.half) / self.cosh_scale)
        )
        tau = max(tau, self.tau_floor)
        if self.is_alpha:
            return inf / tau
        return (1.0 - inf) / tau


def boltzmann_cosh_rates(
    half: float,
    slope: float,
    tau_scale: float,
    tau_floor: float,
    *,
    inverted: bool = False,
    tau_cosh_scale: float | None = None,
    tau_rate: float = 1.0,
) -> tuple[Callable[[float, float], float], Callable[[float, float], float]]:
    """Return (alpha, beta) callables from Boltzmann/cosh kinetic parameters.

    Constructs a pair of rate functions for a gating variable whose steady
    state follows a Boltzmann curve and whose time constant follows a
    cosh-based expression with a floor:

        inf(V) = 1 / (1 + exp(±(V - half) / slope))
        tau(V) = tau_scale / (tau_rate * cosh((V - half) / cosh_scale))
        alpha  = inf / tau
        beta   = (1 - inf) / tau

    The returned callables accept ``(V, ca_i)`` — the ``ca_i`` argument is
    accepted but ignored, keeping the signature consistent with
    :class:`~patch_sim.channels.GatingVariable` rate functions.

    The returned objects are :class:`BoltzmannCoshRate` instances, which are
    fully picklable and safe to use with
    :func:`~patch_sim.clamp_simulations.simulate_batch`.

    Args:
        half: Half-activation voltage in mV.
        slope: Activation slope in mV (positive).
        tau_scale: Numerator of the time-constant expression in ms.
        tau_floor: Minimum allowed time constant in ms.
        inverted: If True, use exp(+(V-half)/slope) so the gate is maximally
            open at hyperpolarised potentials (e.g. IKir).
        tau_cosh_scale: Voltage scale in the cosh denominator in mV.
            Defaults to ``2 * slope``.
        tau_rate: Extra multiplier in the tau denominator.  Use this to
            absorb constant prefactors (e.g. ``tau_rate=6.6`` for IM).

    Returns:
        Tuple of (alpha, beta) where each is a callable
        ``(V: float, ca_i: float) -> float`` returning the corresponding
        rate in 1/ms.
    """
    cosh_scale: float = tau_cosh_scale if tau_cosh_scale is not None else 2.0 * slope
    return (
        BoltzmannCoshRate(
            half=half,
            slope=slope,
            tau_scale=tau_scale,
            tau_floor=tau_floor,
            inverted=inverted,
            cosh_scale=cosh_scale,
            tau_rate=tau_rate,
            is_alpha=True,
        ),
        BoltzmannCoshRate(
            half=half,
            slope=slope,
            tau_scale=tau_scale,
            tau_floor=tau_floor,
            inverted=inverted,
            cosh_scale=cosh_scale,
            tau_rate=tau_rate,
            is_alpha=False,
        ),
    )
