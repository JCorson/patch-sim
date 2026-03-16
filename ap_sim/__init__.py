"""Action Potential Simulator.

A package for simulating action potentials using the Hodgkin-Huxley model.
"""

from . import constants
from .calcium import CalciumDynamics
from .channels import (
    AnyGatingVariable,
    BaseIonChannel,
    CalciumGatingVariable,
    CalciumIonChannel,
    GatingVariable,
    IonChannel,
)
from .hodgkin_huxley import HodgkinHuxley
from .additional_channels import (
    make_ih_channel,
    make_ika_channel,
    make_ikir_channel,
    make_im_channel,
    make_inap_channel,
    make_inar_channel,
)
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
    "constants",
    "AnyGatingVariable",
    "CalciumDynamics",
    "CalciumGatingVariable",
    "GatingVariable",
    "BaseIonChannel",
    "CalciumIonChannel",
    "IonChannel",
    "HodgkinHuxley",
    "make_ih_channel",
    "make_ika_channel",
    "make_ikir_channel",
    "make_im_channel",
    "make_inap_channel",
    "make_inar_channel",
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
