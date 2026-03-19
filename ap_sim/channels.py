"""Ion channel abstractions for the Hodgkin-Huxley simulator.

Provides the building blocks for defining additional ion channels that can be
added on top of the classic Na, K, and leak channels.
"""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class GatingVariable:
    """A single gating variable with its kinetic rate functions.

    Rate functions always accept ``(V, ca_i)`` — voltage in mV and
    intracellular Ca²⁺ concentration in mM.  Voltage-only gates simply ignore
    the ``ca_i`` argument.

    Attributes:
        name: Unique name used as a key in gating-state dicts (e.g. 'r').
        power: Exponent applied to this gate's value when computing conductance.
        alpha: Forward rate function ``alpha(V, ca_i)`` in units 1/ms.
        beta: Backward rate function ``beta(V, ca_i)`` in units 1/ms.
    """

    name: str
    power: int
    alpha: Callable[[float, float], float]
    beta: Callable[[float, float], float]


@dataclass(frozen=True)
class IonChannel:
    """An ion channel with a fixed reversal potential and gating mechanics.

    Computes current as ``g_max * prod(gate^power) * (V - e_rev)``.

    Attributes:
        name: Human-readable channel identifier (e.g. ``'Ih'``).
        g_max: Maximum conductance in mS/cm².
        gating_variables: Tuple of gating variable descriptors.
        e_rev: Fixed reversal potential in mV.
        carries_calcium: ``True`` for channels that carry Ca²⁺ ions (e.g.
            ICaL, ICaT, ICaN).  Used by
            :meth:`~ap_sim.hodgkin_huxley.HodgkinHuxley.calcium_current` to
            sum Ca²⁺ influx for the intracellular Ca²⁺ ODE.  Defaults to
            ``False``.

    Raises:
        ValueError: If ``g_max`` is negative or if gating variable names are
            not unique within the channel.
    """

    name: str
    g_max: float
    gating_variables: tuple["GatingVariable", ...]
    e_rev: float
    carries_calcium: bool = field(default=False)

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
            neuron: Ignored; present for interface consistency with custom
                channels that compute a Nernst potential from ion
                concentrations.

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
