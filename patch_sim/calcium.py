"""Intracellular calcium dynamics for the Hodgkin-Huxley model.

Provides the CalciumDynamics dataclass which encapsulates the ODE for
tracking intracellular Ca2+ concentration ([Ca2+]_i).
"""

from dataclasses import dataclass

from .constants import DEFAULT_ALPHA_CA, DEFAULT_CA_REST, DEFAULT_TAU_CA


@dataclass(frozen=True)
class CalciumDynamics:
    """Parameters and ODE for intracellular Ca2+ concentration dynamics.

    Models the change in intracellular Ca2+ concentration as::

        d[Ca2+]/dt = -alpha_ca * I_Ca - ([Ca2+] - ca_rest) / tau_ca

    Sign convention: positive current is outward (HH convention), so an
    inward Ca2+ current is negative, making ``-alpha_ca * I_Ca`` positive
    (increasing [Ca2+]_i).

    Attributes:
        alpha_ca: Scaling factor converting Ca2+ current to concentration
            change in mM / (µA/cm² · ms).
        tau_ca: Time constant for Ca2+ removal/buffering in ms.
        ca_rest: Resting intracellular Ca2+ concentration in mM.

    Raises:
        ValueError: If ``alpha_ca <= 0``, ``tau_ca <= 0``, or ``ca_rest < 0``.
    """

    alpha_ca: float = DEFAULT_ALPHA_CA
    tau_ca: float = DEFAULT_TAU_CA
    ca_rest: float = DEFAULT_CA_REST

    def __post_init__(self) -> None:
        """Validate calcium dynamics parameters on construction."""
        if self.alpha_ca <= 0:
            raise ValueError(f"alpha_ca must be positive, got {self.alpha_ca}.")
        if self.tau_ca <= 0:
            raise ValueError(f"tau_ca must be positive, got {self.tau_ca}.")
        if self.ca_rest < 0:
            raise ValueError(f"ca_rest must be non-negative, got {self.ca_rest}.")

    def derivative(self, I_Ca: float, ca_i: float) -> float:
        """Compute the rate of change of intracellular Ca2+ concentration.

        Implements: d[Ca2+]/dt = -alpha_ca * I_Ca - (ca_i - ca_rest) / tau_ca

        Args:
            I_Ca: Total calcium current in µA/cm² (positive = outward).
            ca_i: Current intracellular Ca2+ concentration in mM.

        Returns:
            d[Ca2+]/dt in mM/ms.
        """
        return -self.alpha_ca * I_Ca - (ca_i - self.ca_rest) / self.tau_ca
