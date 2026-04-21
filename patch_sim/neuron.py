"""Conductance-based neuron model for simulating action potentials.

The model includes equations for ion channel dynamics and membrane voltage.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property

from .calcium import CalciumDynamics
from .channels import GatingVariable, IonChannel, IonSpecies
from .constants import (
    DEFAULT_C_M,
    DEFAULT_CA_IN,
    DEFAULT_CA_OUT,
    DEFAULT_G_K,
    DEFAULT_G_KL,
    DEFAULT_G_NA,
    DEFAULT_G_NAL,
    DEFAULT_K_IN,
    DEFAULT_K_OUT,
    DEFAULT_NA_IN,
    DEFAULT_NA_OUT,
    DEFAULT_Q10,
    DEFAULT_T,
    DEFAULT_T_REF,
    DEFAULT_V_REST,
)
from .core_channels import (
    make_k_channel,
    make_k_leak_channel,
    make_na_channel,
    make_na_leak_channel,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Neuron:
    """Conductance-based neuron model for simulating action potentials.

    All parameters are fixed at construction time. The class is immutable
    (frozen) to prevent accidental mutation of parameters after the reversal
    potentials have been computed.

    Attributes:
        C_m: Membrane capacitance in uF/cm^2.
        g_Na: Maximum sodium conductance in mS/cm^2.
        g_K: Maximum potassium conductance in mS/cm^2.
        g_NaL: Na⁺ leak conductance in mS/cm². Together with g_KL this
            replaces the single chloride-based leak, providing a biophysically
            realistic mixed-cation background conductance.
        g_KL: K⁺ leak conductance in mS/cm².
        v_rest: Resting potential in mV.
        Na_out: Extracellular sodium concentration in mM.
        Na_in: Intracellular sodium concentration in mM.
        K_out: Extracellular potassium concentration in mM.
        K_in: Intracellular potassium concentration in mM.
        T: Temperature in Kelvin.
        Q10: Q10 temperature coefficient for gating kinetics (dimensionless).
            Gating rate constants are scaled by ``Q10^((T - T_ref) / 10)``.
            A value of 1.0 disables temperature scaling entirely.
        T_ref: Reference temperature in Kelvin at which the gating rate
            constants were measured.  Defaults to 295.15 K (22 °C), matching
            the original HH52 squid axon experimental conditions.
        na_channel_factory: Factory function that builds the Na⁺ core channel
            given a maximum conductance. Defaults to the HH52 squid axon kinetics.
        k_channel_factory: Factory function that builds the K⁺ core channel
            given a maximum conductance. Defaults to the HH52 squid axon kinetics.
        na_leak_channel_factory: Factory function that builds the Na⁺ leak channel
            given a maximum conductance. Defaults to make_na_leak_channel.
        k_leak_channel_factory: Factory function that builds the K⁺ leak channel
            given a maximum conductance. Defaults to make_k_leak_channel.
        additional_channels: Tuple of additional ion channels added on top of
            the classic Na/K/NaL/KL quartet.  Defaults to an empty tuple so
            that all existing code is unaffected.
        calcium_dynamics: Optional calcium dynamics model.

    Cached properties (built on first access):
        core_channels: Tuple of four IonChannel objects (Na, K, NaL, KL) built
            from the constructor conductances and factory functions.
        all_channels: All channels — core_channels + additional_channels.
        all_gating_variables: Flat tuple of every gating variable across all
            channels, in channel-declaration order.
        q10_factor: Dimensionless scaling factor ``Q10^((T - T_ref) / 10)``
            applied to all gating rate constants at simulation time.
    """

    # Membrane properties
    g_Na: float = DEFAULT_G_NA
    g_K: float = DEFAULT_G_K
    g_NaL: float = DEFAULT_G_NAL
    g_KL: float = DEFAULT_G_KL
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

    # Core channel factories — override to use non-HH52 kinetics
    na_channel_factory: Callable[[float], IonChannel] = field(default=make_na_channel)
    k_channel_factory: Callable[[float], IonChannel] = field(default=make_k_channel)
    na_leak_channel_factory: Callable[[float], IonChannel] = field(
        default=make_na_leak_channel
    )
    k_leak_channel_factory: Callable[[float], IonChannel] = field(
        default=make_k_leak_channel
    )

    # Additional extra channels — empty by default so existing code is unaffected
    additional_channels: tuple[IonChannel, ...] = field(default_factory=tuple)

    # Calcium dynamics — None by default for backward compatibility
    calcium_dynamics: CalciumDynamics | None = None

    def __post_init__(self) -> None:
        """Validate parameter values on construction."""
        if self.g_Na < 0:
            raise ValueError("Sodium conductance (g_Na) must be non-negative.")
        if self.g_K < 0:
            raise ValueError("Potassium conductance (g_K) must be non-negative.")
        if self.g_NaL < 0:
            raise ValueError("Na leak conductance (g_NaL) must be non-negative.")
        if self.g_KL < 0:
            raise ValueError("K leak conductance (g_KL) must be non-negative.")
        if self.C_m <= 0:
            raise ValueError("Membrane capacitance (C_m) must be positive.")
        if self.T <= 0:
            raise ValueError("Temperature (T) must be positive (in Kelvin).")
        if self.Q10 <= 0:
            raise ValueError("Q10 must be positive.")
        if self.T_ref <= 0:
            raise ValueError("T_ref must be positive (in Kelvin).")
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
        _BUILTIN_NAMES = {"Na", "K", "NaL", "KL"}
        ch_names = [ch.name for ch in self.additional_channels]
        if len(ch_names) != len(set(ch_names)):
            raise ValueError(
                f"Additional channel names must be unique, got {ch_names}."
            )
        for ch_name in ch_names:
            if ch_name in _BUILTIN_NAMES:
                raise ValueError(
                    f"Additional channel name '{ch_name}' collides with a built-in "
                    "channel name (Na, K, NaL, KL)."
                )
        logger.debug(
            "Neuron: g_Na=%.1f g_K=%.1f g_NaL=%.3f g_KL=%.3f C_m=%.2f T=%.1f K "
            "Q10=%.1f T_ref=%.1f K "
            "additional_channels=%s calcium=%s",
            self.g_Na,
            self.g_K,
            self.g_NaL,
            self.g_KL,
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
    def core_channels(self) -> tuple[IonChannel, ...]:
        """Return the four core channels built from constructor conductances.

        Returns:
            Tuple of (Na, K, NaL, KL) IonChannel objects in that order.
        """
        return (
            self.na_channel_factory(self.g_Na),
            self.k_channel_factory(self.g_K),
            self.na_leak_channel_factory(self.g_NaL),
            self.k_leak_channel_factory(self.g_KL),
        )

    @cached_property
    def all_channels(self) -> tuple[IonChannel, ...]:
        """Return all channels: core channels followed by additional channels.

        Returns:
            Tuple of all IonChannel objects in declaration order.
        """
        return self.core_channels + self.additional_channels

    @cached_property
    def all_gating_variables(self) -> tuple[GatingVariable, ...]:
        """Return a flat tuple of all gating variables across all channels.

        Returns:
            Tuple of GatingVariable objects from all channels, in the order
            the channels and their gating variables are declared.
        """
        result: list[GatingVariable] = []
        for ch in self.all_channels:
            result.extend(ch.gating_variables)
        return tuple(result)

    def calcium_current(
        self,
        V: float,
        gating_state: dict[str, float],
        e_rev_by_name: dict[str, float] | None = None,
    ) -> float:
        """Return the total current from all calcium-carrying channels.

        Sums the current from every channel (core or additional) that has
        ``carries_calcium=True``.  Used by the Ca2+ ODE to determine how much
        intracellular Ca2+ is entering the cell each time step.

        Args:
            V: Membrane voltage in mV.
            gating_state: Full gating state mapping variable name → value,
                covering both core and additional channels.
            e_rev_by_name: Optional pre-computed reversal potentials keyed by
                channel name.  When supplied, bypasses the per-call Nernst/
                Goldman computation for each Ca²⁺-carrying channel.

        Returns:
            Total calcium current in µA/cm² (positive = outward).
        """
        return sum(
            ch.compute_current(
                V,
                gating_state,
                self,
                e_rev_by_name[ch.name] if e_rev_by_name is not None else None,
            )
            for ch in self.all_channels
            if ch.carries_calcium
        )

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
