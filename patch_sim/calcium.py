"""Intracellular calcium dynamics for conductance-based neuron models.

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
        ca_rest: Resting intracellular Ca2+ concentration in mM.  This is
            the ODE equilibrium target (the concentration the pumps/buffers
            maintain in the absence of Ca2+ current) and is NOT necessarily
            equal to the true simulation starting concentration when window
            currents are present at rest.  See *ca_init* below.
        ca_init: Initial intracellular Ca2+ concentration in mM used at the
            start of :func:`~patch_sim.simulate_current_clamp`.  When
            ``None`` (default), *ca_rest* is used.  Set this explicitly when
            the true resting Ca2+ concentration differs from *ca_rest* due to
            persistent window currents at the resting potential (i.e. the
            coupled (V, Ca2+) equilibrium has ca_i > ca_rest).  Use
            :func:`~patch_sim.find_coupled_equilibrium` to compute the
            correct value.

    Raises:
        ValueError: If ``alpha_ca <= 0``, ``tau_ca <= 0``, ``ca_rest < 0``,
            or ``ca_init < 0``.
    """

    alpha_ca: float = DEFAULT_ALPHA_CA
    tau_ca: float = DEFAULT_TAU_CA
    ca_rest: float = DEFAULT_CA_REST
    ca_init: float | None = None

    def __post_init__(self) -> None:
        """Validate calcium dynamics parameters on construction."""
        if self.alpha_ca <= 0:
            raise ValueError(f"alpha_ca must be positive, got {self.alpha_ca}.")
        if self.tau_ca <= 0:
            raise ValueError(f"tau_ca must be positive, got {self.tau_ca}.")
        if self.ca_rest < 0:
            raise ValueError(f"ca_rest must be non-negative, got {self.ca_rest}.")
        if self.ca_init is not None and self.ca_init < 0:
            raise ValueError(f"ca_init must be non-negative, got {self.ca_init}.")

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
