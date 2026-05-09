"""Conductance-based neuron model for simulating action potentials.

The model includes equations for ion channel dynamics and membrane voltage.
"""

import logging
from dataclasses import dataclass, field
from functools import cached_property

from .calcium import CalciumDynamics
from .channels import (
    GatingVariable,
    IonChannel,
    IonSpecies,
)
from .constants import (
    DEFAULT_C_M,
    DEFAULT_CA_IN,
    DEFAULT_CA_OUT,
    DEFAULT_K_IN,
    DEFAULT_K_OUT,
    DEFAULT_NA_IN,
    DEFAULT_NA_OUT,
    DEFAULT_Q10,
    DEFAULT_T,
    DEFAULT_T_REF,
    DEFAULT_V_REST,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Neuron:
    """Conductance-based neuron model for simulating action potentials.

    All parameters are fixed at construction time. The class is immutable
    (frozen) to prevent accidental mutation of parameters after the reversal
    potentials have been computed.

    Attributes:
        channels: Tuple of ion channels carried by the membrane. Each channel
            is fully configured with its own ``g_max`` and kinetics; there is
            no separate "core" Na/K slot.  Channel names must be unique.
        C_m: Membrane capacitance in uF/cm^2.
        v_rest: Resting potential in mV.
        Na_out: Extracellular sodium concentration in mM.
        Na_in: Intracellular sodium concentration in mM.
        K_out: Extracellular potassium concentration in mM.
        K_in: Intracellular potassium concentration in mM.
        Ca_out: Extracellular calcium concentration in mM.
        Ca_in: Intracellular calcium concentration in mM.
        T: Temperature in Kelvin.
        Q10: Q10 temperature coefficient for gating kinetics (dimensionless).
            Gating rate constants are scaled by ``Q10^((T - T_ref) / 10)``.
            A value of 1.0 disables temperature scaling entirely.
        T_ref: Reference temperature in Kelvin at which the gating rate
            constants were measured.  Defaults to 295.15 K (22 °C), matching
            the original HH52 squid axon experimental conditions.
        calcium_dynamics: Optional calcium dynamics model.
        area_cm2: Total membrane surface area in cm².  Optional physical
            attribute of the cell that is **not** read by the ODE solver —
            HH dynamics in this single-compartment model are scale-invariant
            in the per-area units (mS/cm², µF/cm², µA/cm²) used everywhere
            else.  Surface area is consumed by the analysis layer to convert
            the per-area passive properties (R_in in kΩ·cm², C_m in µF/cm²)
            into absolute MΩ / pF for display.  ``None`` means absolute units
            are not available for this neuron.

    Cached properties (built on first access):
        all_gating_variables: Flat tuple of every gating variable across all
            channels, in channel-declaration order.
        q10_factor: Dimensionless scaling factor ``Q10^((T - T_ref) / 10)``
            applied to all gating rate constants at simulation time.
        reversal_potentials: Dict mapping each non-Ca²⁺ channel name to its
            reversal potential in mV, computed once on first access.
    """

    # Ion channels — empty by default; presets supply the full list.
    channels: tuple[IonChannel, ...] = ()

    # Membrane properties
    C_m: float = DEFAULT_C_M
    v_rest: float = DEFAULT_V_REST

    # Ion concentrations (in mM)
    Na_out: float = DEFAULT_NA_OUT
    Na_in: float = DEFAULT_NA_IN
    K_out: float = DEFAULT_K_OUT
    K_in: float = DEFAULT_K_IN
    Ca_out: float = DEFAULT_CA_OUT
    Ca_in: float = DEFAULT_CA_IN

    # Temperature in Kelvin (37°C for mammalian cells)
    T: float = DEFAULT_T

    # Q10 temperature scaling for gating kinetics
    Q10: float = DEFAULT_Q10
    T_ref: float = DEFAULT_T_REF

    # Calcium dynamics — None by default for backward compatibility
    calcium_dynamics: CalciumDynamics | None = field(default=None)

    # Total membrane surface area in cm² — analysis-only metadata, not read
    # by the ODE solver.  Used by the passive-property analysis layer to
    # report absolute MΩ / pF instead of per-area density units.
    area_cm2: float | None = None

    def __post_init__(self) -> None:
        """Validate parameter values on construction."""
        if self.C_m <= 0:
            raise ValueError("Membrane capacitance (C_m) must be positive.")
        if self.T <= 0:
            raise ValueError("Temperature (T) must be positive (in Kelvin).")
        if self.Q10 <= 0:
            raise ValueError("Q10 must be positive.")
        if self.T_ref <= 0:
            raise ValueError("T_ref must be positive (in Kelvin).")
        if self.area_cm2 is not None and self.area_cm2 <= 0:
            raise ValueError("area_cm2 must be positive when provided.")
        for name, value in [
            ("Na_out", self.Na_out),
            ("Na_in", self.Na_in),
            ("K_out", self.K_out),
            ("K_in", self.K_in),
            ("Ca_out", self.Ca_out),
            ("Ca_in", self.Ca_in),
        ]:
            if value <= 0:
                raise ValueError(f"Ion concentration ({name}) must be positive.")
        ch_names = [ch.name for ch in self.channels]
        if len(ch_names) != len(set(ch_names)):
            raise ValueError(f"Channel names must be unique, got {ch_names}.")
        logger.debug(
            "Neuron: C_m=%.2f T=%.1f K Q10=%.1f T_ref=%.1f K channels=%s calcium=%s",
            self.C_m,
            self.T,
            self.Q10,
            self.T_ref,
            ch_names if ch_names else "none",
            "enabled" if self.calcium_dynamics is not None else "disabled",
        )

    @cached_property
    def q10_factor(self) -> float:
        """Return the Q10 temperature scaling factor for gating rate constants.

        Computes ``Q10^((T - T_ref) / 10)`` where both temperatures are in
        Kelvin.  A value of 1.0 (obtained when ``T == T_ref`` or ``Q10 == 1``)
        means no scaling is applied.

        Returns:
            Dimensionless multiplicative factor applied to all gating alpha/beta
            rate constants during simulation.
        """
        return self.Q10 ** ((self.T - self.T_ref) / 10.0)

    @cached_property
    def all_gating_variables(self) -> tuple[GatingVariable, ...]:
        """Return a flat tuple of all gating variables across all channels.

        Returns:
            Tuple of GatingVariable objects from all channels, in the order
            the channels and their gating variables are declared.
        """
        result: list[GatingVariable] = []
        for ch in self.channels:
            result.extend(ch.gating_variables)
        return tuple(result)

    @cached_property
    def reversal_potentials(self) -> dict[str, float]:
        """Return constant reversal potentials for all non-Ca²⁺ channels.

        Because this dataclass is frozen, ion concentrations and temperature
        are constant over the neuron's lifetime, so non-Ca²⁺ channels always
        have the same reversal potential.  Building the map lazily here
        eliminates a ``numpy.log`` call per channel per RK4 substep.

        Ca²⁺-carrying channels are excluded because their reversal potential
        depends on the live intracellular Ca²⁺ concentration, which changes
        each substep.  :meth:`~patch_sim.IonChannel.compute_current` recomputes
        E_Ca from live ``ca_i`` for those channels.

        Returns:
            Dict mapping each non-Ca²⁺ channel name to its reversal potential
            in mV.
        """
        return {
            ch.name: ch.reversal_potential(self)
            for ch in self.channels
            if not ch.carries_calcium
        }

    def ion_concentrations(self, species: IonSpecies) -> tuple[float, float]:
        """Return the extracellular and intracellular concentrations for an ion.

        Used by :class:`~patch_sim.channels.IonChannel` to look up the
        concentrations needed for dynamic reversal potential computation.

        Args:
            species: The ion species to look up.

        Returns:
            A ``(C_out, C_in)`` tuple of concentrations in mM.

        Raises:
            ValueError: If *species* is not recognised.
        """
        if species is IonSpecies.SODIUM:
            return self.Na_out, self.Na_in
        if species is IonSpecies.POTASSIUM:
            return self.K_out, self.K_in
        if species is IonSpecies.CALCIUM:
            return self.Ca_out, self.Ca_in
        raise ValueError(f"Unknown ion species: {species!r}")  # pragma: no cover
