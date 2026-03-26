"""Utility functions for the action potential simulator.

This module contains shared utility functions used across different modules.
"""

import math

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
