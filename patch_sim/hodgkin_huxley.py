"""This module implements the Hodgkin-Huxley model for simulating action potentials.

The model includes equations for ion channel dynamics and membrane voltage.
"""

import logging
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from .calcium import CalciumDynamics
from .channels import GatingVariable, IonChannel, IonSpecies
from .constants import (
    DEFAULT_C_M,
    DEFAULT_CA_IN,
    DEFAULT_CA_OUT,
    DEFAULT_CL_IN,
    DEFAULT_CL_OUT,
    DEFAULT_G_K,
    DEFAULT_G_L,
    DEFAULT_G_NA,
    DEFAULT_K_IN,
    DEFAULT_K_OUT,
    DEFAULT_NA_IN,
    DEFAULT_NA_OUT,
    DEFAULT_T,
    DEFAULT_V_REST,
)
from .core_channels import make_k_channel, make_leak_channel, make_na_channel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HodgkinHuxley:
    """Simulates the Hodgkin-Huxley model of action potentials.

    All parameters are fixed at construction time. The class is immutable
    (frozen) to prevent accidental mutation of parameters after the reversal
    potentials have been computed.

    Attributes:
        C_m: Membrane capacitance in uF/cm^2.
        g_Na: Maximum sodium conductance in mS/cm^2.
        g_K: Maximum potassium conductance in mS/cm^2.
        g_L: Leak conductance in mS/cm^2.
        v_rest: Resting potential in mV.
        Na_out: Extracellular sodium concentration in mM.
        Na_in: Intracellular sodium concentration in mM.
        K_out: Extracellular potassium concentration in mM.
        K_in: Intracellular potassium concentration in mM.
        Cl_out: Extracellular chloride concentration in mM.
        Cl_in: Intracellular chloride concentration in mM.
        T: Temperature in Kelvin.
        additional_channels: Tuple of additional ion channels added on top of
            the classic Na/K/leak triad.  Defaults to an empty tuple so that
            all existing code is unaffected.
        calcium_dynamics: Optional calcium dynamics model.

    Cached properties (built on first access):
        core_channels: Tuple of three IonChannel objects (Na, K, leak) built
            from the constructor conductances.
        all_channels: All channels — core_channels + additional_channels.
        all_gating_variables: Flat tuple of every gating variable across all
            channels, in channel-declaration order.
    """

    # Membrane properties
    g_Na: float = DEFAULT_G_NA
    g_K: float = DEFAULT_G_K
    g_L: float = DEFAULT_G_L
    C_m: float = DEFAULT_C_M
    v_rest: float = DEFAULT_V_REST

    # Ion concentrations (in mM)
    Na_out: float = DEFAULT_NA_OUT
    Na_in: float = DEFAULT_NA_IN
    K_out: float = DEFAULT_K_OUT
    K_in: float = DEFAULT_K_IN
    Cl_out: float = DEFAULT_CL_OUT
    Cl_in: float = DEFAULT_CL_IN
    Ca_out: float = DEFAULT_CA_OUT
    Ca_in: float = DEFAULT_CA_IN

    # Temperature in Kelvin (37°C for mammalian cells)
    T: float = DEFAULT_T

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
        if self.g_L < 0:
            raise ValueError("Leak conductance (g_L) must be non-negative.")
        if self.C_m <= 0:
            raise ValueError("Membrane capacitance (C_m) must be positive.")
        if self.T <= 0:
            raise ValueError("Temperature (T) must be positive (in Kelvin).")
        for name, value in [
            ("Na_out", self.Na_out),
            ("Na_in", self.Na_in),
            ("K_out", self.K_out),
            ("K_in", self.K_in),
            ("Cl_out", self.Cl_out),
            ("Cl_in", self.Cl_in),
            ("Ca_out", self.Ca_out),
            ("Ca_in", self.Ca_in),
        ]:
            if value <= 0:
                raise ValueError(f"Ion concentration ({name}) must be positive.")
        _BUILTIN_NAMES = {"Na", "K", "leak"}
        ch_names = [ch.name for ch in self.additional_channels]
        if len(ch_names) != len(set(ch_names)):
            raise ValueError(
                f"Additional channel names must be unique, got {ch_names}."
            )
        for ch_name in ch_names:
            if ch_name in _BUILTIN_NAMES:
                raise ValueError(
                    f"Additional channel name '{ch_name}' collides with a built-in "
                    "channel name (Na, K, leak)."
                )
        logger.debug(
            "HodgkinHuxley: g_Na=%.1f g_K=%.1f g_L=%.3f C_m=%.2f T=%.1f K "
            "additional_channels=%s calcium=%s",
            self.g_Na,
            self.g_K,
            self.g_L,
            self.C_m,
            self.T,
            ch_names if ch_names else "none",
            "enabled" if self.calcium_dynamics is not None else "disabled",
        )

    @cached_property
    def core_channels(self) -> tuple[IonChannel, ...]:
        """Return the three classic HH channels built from constructor conductances.

        Returns:
            Tuple of (Na, K, leak) IonChannel objects in that order.
        """
        return (
            make_na_channel(self.g_Na),
            make_k_channel(self.g_K),
            make_leak_channel(self.g_L),
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

    @cached_property
    def gating_index(self) -> dict[str, int]:
        """Map each gating variable name to its index in the flat state array.

        The order matches ``all_gating_variables``: gating variable at position
        ``i`` in that tuple maps to index ``i`` in the flat numpy state array.

        Returns:
            Dict mapping gating variable name to integer array index.
        """
        return {gv.name: i for i, gv in enumerate(self.all_gating_variables)}

    @cached_property
    def _channel_gate_info(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Pre-computed index/power arrays for vectorized conductance computation.

        For each channel in ``all_channels``, stores a ``(indices, powers)``
        pair of numpy arrays so that the conductance gate product can be
        evaluated as ``np.prod(state[indices] ** powers)`` without iterating
        over gating variables individually.

        Returns:
            Tuple of ``(indices, powers)`` numpy array pairs, one per channel.
        """
        result = []
        idx = self.gating_index
        for ch in self.all_channels:
            if ch.gating_variables:
                indices = np.array(
                    [idx[gv.name] for gv in ch.gating_variables], dtype=np.intp
                )
                powers = np.array(
                    [gv.power for gv in ch.gating_variables], dtype=np.float64
                )
            else:
                indices = np.empty(0, dtype=np.intp)
                powers = np.empty(0, dtype=np.float64)
            result.append((indices, powers))
        return tuple(result)

    @cached_property
    def _calcium_channel_indices(self) -> tuple[int, ...]:
        """Indices into ``all_channels`` for calcium-carrying channels.

        Used by the internal simulation loop to sum only the Ca²⁺-carrying
        channel currents when updating the intracellular Ca²⁺ ODE.

        Returns:
            Tuple of integer indices into ``all_channels``.
        """
        return tuple(i for i, ch in enumerate(self.all_channels) if ch.carries_calcium)

    @cached_property
    def _reversal_potentials(self) -> np.ndarray:
        """Pre-computed reversal potential for each channel.

        Ion concentrations are fixed model parameters, so reversal potentials
        are constant for the lifetime of the model. This avoids calling
        ``ch.reversal_potential(neuron)`` 60 000+ times per simulation.

        Returns:
            Float64 array of length ``len(all_channels)`` where entry ``i``
            is the reversal potential in mV for ``all_channels[i]``.
        """
        return np.array(
            [ch.reversal_potential(self) for ch in self.all_channels], dtype=np.float64
        )

    def calcium_current(self, V: float, gating_state: dict[str, float]) -> float:
        """Return the total current from all calcium-carrying channels.

        Sums the current from every channel (core or additional) that has
        ``carries_calcium=True``.  Used by the Ca2+ ODE to determine how much
        intracellular Ca2+ is entering the cell each time step.

        Args:
            V: Membrane voltage in mV.
            gating_state: Full gating state mapping variable name → value,
                covering both core and additional channels.

        Returns:
            Total calcium current in µA/cm² (positive = outward).
        """
        return sum(
            ch.compute_current(V, gating_state, self)
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
        if species is IonSpecies.CHLORIDE:
            return self.Cl_out, self.Cl_in
        raise ValueError(f"Unknown ion species: {species!r}")  # pragma: no cover
