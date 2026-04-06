"""Utility functions for the action potential simulator.

This module contains shared utility functions used across different modules.
"""

import math

# Clipping bounds for safe_exp: chosen so that exp(100) ≈ 2.7e43, well below
# float64 overflow (~e308), while still covering all physiologically relevant
# voltage-derived exponents in the Hodgkin-Huxley rate equations.
SAFE_EXP_CLIP_MIN = -100
SAFE_EXP_CLIP_MAX = 100

# Clipping bound for safe_cosh: cosh is symmetric so a single bound suffices.
# cosh(100) ≈ 1.34e43, well below float64 overflow (~e308).
SAFE_COSH_CLIP = 100


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


def safe_cosh(x: float) -> float:
    """Safely compute the hyperbolic cosine to avoid overflow.

    Clips the input to [-SAFE_COSH_CLIP, SAFE_COSH_CLIP] before computing
    cosh.  Since cosh is symmetric and monotonically increasing away from
    zero, clipping at ±100 returns cosh(100) ≈ 1.34e43 for extreme inputs,
    which is finite and preserves the qualitative behaviour (very large tau
    denominator → very small time constant).

    Args:
        x: The input value.

    Returns:
        The computed hyperbolic cosine value, capped to prevent overflow.
    """
    if math.isnan(x):
        return math.nan
    if x > SAFE_COSH_CLIP:
        return math.cosh(SAFE_COSH_CLIP)
    if x < -SAFE_COSH_CLIP:
        return math.cosh(SAFE_COSH_CLIP)
    return math.cosh(x)
