"""
Action Potential Simulator

A package for simulating action potentials using the Hodgkin-Huxley model.
"""

from .hodgkin_huxley import HodgkinHuxley
from .clamp_simulations import simulate_voltage_clamp, simulate_current_clamp
from .nernst_neuron import nernst_potential
from .utils import safe_exp
from .protocols import (
    step_current,
    ramp_current,
    pulse_train,
    sinusoidal_current,
    chirp_current,
    noise_current,
)

__all__ = [
    "HodgkinHuxley",
    "simulate_voltage_clamp",
    "simulate_current_clamp",
    "nernst_potential",
    "safe_exp",
    "step_current",
    "ramp_current",
    "pulse_train",
    "sinusoidal_current",
    "chirp_current",
    "noise_current",
]
