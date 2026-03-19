"""This module implements the Hodgkin-Huxley model for simulating action potentials.

The model includes equations for ion channel dynamics and membrane voltage.
"""

from dataclasses import dataclass, field
from functools import cached_property

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
from .electrochemistry import nernst_potential
from .utils import safe_exp

# Threshold for detecting near-singularity in GHK-style rate equations.
# When the denominator voltage term is within this tolerance of zero, the
# L'Hôpital limit is used instead to avoid division by zero.
SINGULARITY_THRESHOLD = 1e-6


@dataclass(frozen=True)
class HodgkinHuxley:
    """Simulates the Hodgkin-Huxley model of action potentials.

    All parameters are fixed at construction time. The class is immutable
    (frozen) to prevent accidental mutation of parameters after the reversal
    potentials have been computed.

    Attributes:
        C_m (float): Membrane capacitance in uF/cm^2.
        g_Na (float): Maximum sodium conductance in mS/cm^2.
        g_K (float): Maximum potassium conductance in mS/cm^2.
        g_L (float): Leak conductance in mS/cm^2.
        v_rest (float): Resting potential in mV.
        Na_out (float): Extracellular sodium concentration in mM.
        Na_in (float): Intracellular sodium concentration in mM.
        K_out (float): Extracellular potassium concentration in mM.
        K_in (float): Intracellular potassium concentration in mM.
        Cl_out (float): Extracellular chloride concentration in mM.
        Cl_in (float): Intracellular chloride concentration in mM.
        T (float): Temperature in Kelvin.
        additional_channels: Tuple of additional ion channels added on top of
            the classic Na/K/leak triad.  Defaults to an empty tuple so that
            all existing code is unaffected.

    Cached properties (derived from ion concentrations on first access):
        E_Na (float): Sodium reversal potential in mV.
        E_K (float): Potassium reversal potential in mV.
        E_L (float): Leak reversal potential in mV.
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

    @cached_property
    def all_additional_gating_variables(self) -> tuple[GatingVariable, ...]:
        """Return a flat tuple of all gating variables across additional channels.

        Returns:
            Tuple of gating variable objects from all additional channels, in
            the order the channels and their gating variables are declared.
        """
        result: list[GatingVariable] = []
        for ch in self.additional_channels:
            result.extend(ch.gating_variables)
        return tuple(result)

    def calcium_current(self, V: float, opt_state: dict[str, float]) -> float:
        """Return the total current from all calcium-carrying additional channels.

        Sums the current from every additional channel that has
        ``carries_calcium=True``.  Used by the Ca2+ ODE to determine how much
        intracellular Ca2+ is entering the cell each time step.

        Args:
            V: Membrane voltage in mV.
            opt_state: Additional channel gating state (name → value).

        Returns:
            Total calcium current in µA/cm² (positive = outward).
        """
        return sum(
            ch.compute_current(V, opt_state, self)
            for ch in self.additional_channels
            if ch.carries_calcium
        )

    # Reversal potentials — derived from ion concentrations via the Nernst
    # equation.  Using @cached_property avoids the object.__setattr__ hack
    # that would otherwise be needed in __post_init__ on a frozen dataclass.

    def ion_concentrations(self, species: IonSpecies) -> tuple[float, float]:
        """Return the extracellular and intracellular concentrations for an ion.

        Used by :class:`~ap_sim.channels.IonChannel` to look up the
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

    @cached_property
    def E_Ca(self) -> float:
        """Calcium reversal potential in mV (z=+2)."""
        return nernst_potential(2, self.T, self.Ca_out, self.Ca_in)

    def alpha_n(self, V: float) -> float:
        """Calculate the rate constant alpha_n for potassium channel activation.

        Args:
            V: Membrane voltage in mV.

        Returns:
            The rate constant alpha_n.
        """
        if abs(V + 55) < SINGULARITY_THRESHOLD:
            # Handle near-singularity case
            # This is the limit as V approaches -55
            return 0.1
        else:
            denominator = 1 - safe_exp(-(V + 55) / 10)
            return 0.01 * (V + 55) / denominator

    def beta_n(self, V: float) -> float:
        """Calculate the rate constant beta_n for potassium channel deactivation.

        Args:
            V: Membrane voltage in mV.

        Returns:
            The rate constant beta_n.
        """
        return 0.125 * safe_exp(-(V + 65) / 80)

    def alpha_m(self, V: float) -> float:
        """Calculate the rate constant alpha_m for sodium channel activation.

        Args:
            V: Membrane voltage in mV.

        Returns:
            The rate constant alpha_m.
        """
        if abs(V + 40) < SINGULARITY_THRESHOLD:
            # Handle near-singularity case
            # This is the limit as V approaches -40
            return 1.0
        else:
            denominator = 1 - safe_exp(-(V + 40) / 10)
            return 0.1 * (V + 40) / denominator

    def beta_m(self, V: float) -> float:
        """Calculate the rate constant beta_m for sodium channel deactivation.

        Args:
            V: Membrane voltage in mV.

        Returns:
            The rate constant beta_m.
        """
        return 4.0 * safe_exp(-(V + 65) / 18)

    def alpha_h(self, V: float) -> float:
        """Calculate the rate constant alpha_h for sodium channel inactivation.

        Args:
            V: Membrane voltage in mV.

        Returns:
            The rate constant alpha_h.
        """
        return 0.07 * safe_exp(-(V + 65) / 20)

    def beta_h(self, V: float) -> float:
        """Calculate the rate constant beta_h for sodium channel reactivation.

        Args:
            V: Membrane voltage in mV.

        Returns:
            The rate constant beta_h.
        """
        return 1 / (1 + safe_exp(-(V + 35) / 10))
