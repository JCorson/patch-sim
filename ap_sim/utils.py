"""
Utility functions for the action potential simulator.

This module contains shared utility functions used across different modules.
"""

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Union

# Type aliases to support both scalar and array inputs
FloatOrArray = Union[float, NDArray[np.float64], pd.Series]


def safe_exp(x: FloatOrArray) -> FloatOrArray:
    """
    Safely compute the exponential to avoid overflow and underflow.

    This function prevents numerical overflow/underflow by clipping the input
    values to a safe range before computing the exponential. This is particularly
    useful in neuronal modeling where exponential terms can grow very large or
    very small.

    Parameters:
        x (FloatOrArray): The input value or array.

    Returns:
        FloatOrArray: The computed exponential value, capped to prevent overflow.
    """
    return np.exp(np.clip(x, -100, 100))
