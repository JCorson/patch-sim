"""Ion channel abstractions for the Hodgkin-Huxley simulator.

Provides the building blocks for defining optional ion channels that can be
added on top of the classic Na, K, and leak channels.
"""

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class GatingVariable:
    """A single gating variable with its kinetic rate functions.

    Attributes:
        name: Unique name used as a key in gating-state dicts (e.g. 'r').
        power: Exponent applied to this gate's value when computing conductance.
        alpha: Forward rate function alpha(V) in units 1/ms.
        beta: Backward rate function beta(V) in units 1/ms.
    """

    name: str
    power: int
    alpha: Callable[[float], float]
    beta: Callable[[float], float]


@runtime_checkable
class IonChannel(Protocol):
    """Structural protocol satisfied by any ion channel implementation.

    Any class with the attributes and methods listed here qualifies, without
    needing to inherit from this protocol explicitly.

    Attributes:
        name: Human-readable channel identifier (e.g. 'Ih').
        g_max: Maximum conductance in mS/cm².
        gating_variables: Tuple of GatingVariable descriptors.
    """

    name: str
    g_max: float
    gating_variables: tuple[GatingVariable, ...]

    def reversal_potential(self, neuron: Any) -> float:
        """Return the reversal potential for this channel in mV.

        Args:
            neuron: The HodgkinHuxley neuron instance (may be used to compute
                Nernst potentials for concentration-dependent channels).

        Returns:
            Reversal potential in mV.
        """
        ...

    def compute_current(self, V: float, gating_state: dict[str, float]) -> float:
        """Compute the ionic current through this channel.

        Args:
            V: Membrane voltage in mV.
            gating_state: Mapping from gating variable name to current value.

        Returns:
            Ionic current in µA/cm².
        """
        ...


@dataclass(frozen=True)
class BaseIonChannel:
    """Generic ion channel with a fixed reversal potential.

    Computes current as ``g_max * prod(gate^power) * (V - e_rev)``.

    Channels that require a concentration-dependent (Nernst) reversal potential
    can subclass this and override ``reversal_potential``.

    Attributes:
        name: Human-readable channel identifier.
        g_max: Maximum conductance in mS/cm².
        gating_variables: Tuple of GatingVariable descriptors.
        e_rev: Fixed reversal potential in mV.

    Raises:
        ValueError: If ``g_max`` is negative or if gating variable names are
            not unique within the channel.
    """

    name: str
    g_max: float
    gating_variables: tuple[GatingVariable, ...]
    e_rev: float

    def __post_init__(self) -> None:
        """Validate channel parameters on construction."""
        if self.g_max < 0:
            raise ValueError(
                f"Channel '{self.name}': g_max must be non-negative, got {self.g_max}."
            )
        names = [gv.name for gv in self.gating_variables]
        if len(names) != len(set(names)):
            raise ValueError(
                f"Channel '{self.name}': gating variable names must be unique, "
                f"got {names}."
            )

    def reversal_potential(self, neuron: Any) -> float:
        """Return the fixed reversal potential for this channel.

        Args:
            neuron: Ignored; present to satisfy the IonChannel protocol.

        Returns:
            The fixed reversal potential in mV.
        """
        return self.e_rev

    def compute_current(self, V: float, gating_state: dict[str, float]) -> float:
        """Compute the ionic current through this channel.

        Evaluates ``g_max * prod(gate^power) * (V - e_rev)``.

        Args:
            V: Membrane voltage in mV.
            gating_state: Mapping from gating variable name to current value.

        Returns:
            Ionic current in µA/cm².
        """
        g = self.g_max
        for gv in self.gating_variables:
            g *= gating_state[gv.name] ** gv.power
        return g * (V - self.e_rev)
