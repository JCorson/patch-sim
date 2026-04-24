"""Gating-variable rate-function class hierarchy.

Every alpha/beta rate attached to a :class:`~patch_sim.channels.GatingVariable`
is an instance of :class:`Rate`.  The abstract subclasses
:class:`VoltageOnlyRate` and :class:`CalciumDependentRate` discriminate whether
a rate's output depends on ca_i.  The voltage-clamp optimizer uses this split
to tabulate V-only rates over the prescribed voltage protocol; Ca²⁺-dependent
rates are evaluated scalar-wise in the RK4 loop.

Concrete adapters :class:`VoltageOnlyFn` and :class:`CalciumDependentFn` wrap
plain ``(V, ca_i) -> float`` callables so existing module-level rate functions
can be promoted into the type hierarchy without changing their bodies.  Both
adapters are frozen dataclasses, matching the picklability contract relied on
by :func:`~patch_sim.clamp_simulations.simulate_batch`.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass


class Rate(ABC):
    """Abstract base class for gating-variable rate functions."""

    @abstractmethod
    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate the rate.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The rate in 1/ms.
        """


class VoltageOnlyRate(Rate):
    """Marker base class for rates whose output is independent of ca_i.

    Subclasses are safe to tabulate over a prescribed voltage protocol.
    """


class CalciumDependentRate(Rate):
    """Marker base class for rates whose output depends on ca_i.

    Subclasses must be evaluated scalar-wise at the current (V, ca_i).
    """


@dataclass(frozen=True)
class VoltageOnlyFn(VoltageOnlyRate):
    """Adapter wrapping a plain callable as a :class:`VoltageOnlyRate`.

    Attributes:
        fn: Plain ``(V, ca_i) -> float`` callable.  Must be independent of
            ``ca_i`` — wrap Ca²⁺-dependent functions with
            :class:`CalciumDependentFn` instead.
    """

    fn: Callable[[float, float], float]

    def __call__(self, V: float, ca_i: float) -> float:
        """Delegate to the wrapped callable.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

        Returns:
            The rate in 1/ms.
        """
        return self.fn(V, ca_i)


@dataclass(frozen=True)
class CalciumDependentFn(CalciumDependentRate):
    """Adapter wrapping a plain callable as a :class:`CalciumDependentRate`.

    Attributes:
        fn: Plain ``(V, ca_i) -> float`` callable whose output depends on
            ``ca_i``.
    """

    fn: Callable[[float, float], float]

    def __call__(self, V: float, ca_i: float) -> float:
        """Delegate to the wrapped callable.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The rate in 1/ms.
        """
        return self.fn(V, ca_i)
