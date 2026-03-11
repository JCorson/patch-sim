"""Action Potential Simulator.

A package for simulating action potentials using the Hodgkin-Huxley model.
"""

from .hodgkin_huxley import HodgkinHuxley
from .clamp_simulations import simulate_voltage_clamp, simulate_current_clamp
from .nernst import nernst_potential
from .protocols import (
    step_current,
    ramp_current,
    pulse_train,
    sinusoidal_current,
    chirp_current,
    noise_current,
    # Voltage clamp protocols
    step_voltage,
    ramp_voltage,
    pulse_train_voltage,
    iv_curve_protocol,
    activation_protocol,
)

__all__ = [
    "HodgkinHuxley",
    "simulate_voltage_clamp",
    "simulate_current_clamp",
    "nernst_potential",
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
    "activation_protocol",
]
