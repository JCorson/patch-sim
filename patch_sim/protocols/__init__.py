"""Protocol generation utilities for both current and voltage clamp experiments.

This package provides functions to generate typical stimulation protocols
that can be used with current and voltage clamp simulations.
"""

from .current import (
    chirp_current,
    noise_current,
    pulse_train,
    ramp_current,
    sinusoidal_current,
    step_current,
)
from .voltage import (
    activation_sweep,
    iv_curve_protocol,
    pulse_train_voltage,
    ramp_voltage,
    step_voltage,
)

__all__ = [
    # Current clamp protocols
    "step_current",
    "ramp_current",
    "pulse_train",
    "sinusoidal_current",
    "chirp_current",
    "noise_current",
    # Voltage clamp protocols
    "step_voltage",
    "ramp_voltage",
    "pulse_train_voltage",
    "iv_curve_protocol",
    "activation_sweep",
]
