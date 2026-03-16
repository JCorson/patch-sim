"""Utility functions for the action potential simulator.

This module contains shared utility functions used across different modules.
"""

import math
from collections.abc import Callable

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


def boltzmann_cosh_rates(
    half: float,
    slope: float,
    tau_scale: float,
    tau_floor: float,
    *,
    inverted: bool = False,
    tau_cosh_scale: float | None = None,
    tau_rate: float = 1.0,
) -> tuple[Callable[[float], float], Callable[[float], float]]:
    """Return (alpha, beta) callables from Boltzmann/cosh kinetic parameters.

    Constructs a pair of rate functions for a gating variable whose steady
    state follows a Boltzmann curve and whose time constant follows a
    cosh-based expression with a floor:

        inf(V) = 1 / (1 + exp(±(V - half) / slope))
        tau(V) = tau_scale / (tau_rate * cosh((V - half) / cosh_scale))
        alpha  = inf / tau
        beta   = (1 - inf) / tau

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
        Tuple of (alpha, beta) where each is a callable ``(V: float) -> float``
        returning the corresponding rate in 1/ms.
    """
    cosh_scale: float = tau_cosh_scale if tau_cosh_scale is not None else 2.0 * slope

    def _inf(V: float) -> float:
        """Steady-state open probability at voltage V."""
        if inverted:
            return 1.0 / (1.0 + safe_exp((V - half) / slope))
        return 1.0 / (1.0 + safe_exp(-(V - half) / slope))

    def _tau(V: float) -> float:
        """Voltage-dependent time constant in ms, floored at tau_floor."""
        tau = tau_scale / (tau_rate * math.cosh((V - half) / cosh_scale))
        return max(tau, tau_floor)

    # Single-slot cache: stores (V, inf_val, tau_val) so that alpha and beta
    # called with the same V in the same RK4 sub-step share one evaluation.
    _last: list[tuple[float, float, float]] = [(float("nan"), 0.0, 0.0)]

    def _ensure(V: float) -> None:
        """Populate _last if V differs from the cached voltage."""
        if V != _last[0][0]:
            _last[0] = (V, _inf(V), _tau(V))

    def alpha(V: float) -> float:
        """Forward rate alpha = inf(V) / tau(V) in 1/ms."""
        _ensure(V)
        return _last[0][1] / _last[0][2]

    def beta(V: float) -> float:
        """Backward rate beta = (1 - inf(V)) / tau(V) in 1/ms."""
        _ensure(V)
        return (1.0 - _last[0][1]) / _last[0][2]

    return alpha, beta
