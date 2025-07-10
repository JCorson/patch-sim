"""
Utility functions for the action potential simulator.

This module contains shared utility functions used across different modules.
"""

import numpy as np


def safe_exp(x: float) -> float:
    """
    Safely compute the exponential to avoid overflow and underflow.

    This function prevents numerical overflow/underflow by clipping the input
    values to a safe range before computing the exponential. This is particularly
    useful in neuronal modeling where exponential terms can grow very large or
    very small.

    Parameters:
        x (float): The input value.

    Returns:
        float: The computed exponential value, capped to prevent overflow.
    """
    return np.exp(np.clip(x, -100, 100))
