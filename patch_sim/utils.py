"""Utility functions for the action potential simulator.

This module contains shared utility functions used across different modules.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

# Clipping bounds for safe_exp: chosen so that exp(100) ≈ 2.7e43, well below
# float64 overflow (~e308), while still covering all physiologically relevant
# voltage-derived exponents in the Hodgkin-Huxley rate equations.
SAFE_EXP_CLIP_MIN = -100
SAFE_EXP_CLIP_MAX = 100


def safe_exp(x: float) -> float:
    """Safely compute the exponential to avoid overflow and underflow.

    This function prevents numerical overflow/underflow by clipping the input
    values to a safe range before computing the exponential. This is particularly
    useful in neuronal modeling where exponential terms can grow very large or
    very small.

    Args:
        x: The input value.

    Returns:
        The computed exponential value, capped to prevent overflow.
    """
    if x > SAFE_EXP_CLIP_MAX:
        return math.exp(SAFE_EXP_CLIP_MAX)
    elif x < SAFE_EXP_CLIP_MIN:
        return math.exp(SAFE_EXP_CLIP_MIN)
    elif x != x:  # NaN — falls through both comparisons above
        return math.nan
    return math.exp(x)


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
