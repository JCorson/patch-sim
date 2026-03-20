"""Action Potential Simulator.

A package for simulating action potentials using the Hodgkin-Huxley model.
"""

from . import constants
from .calcium import CalciumDynamics
from .channels import (
    GatingVariable,
    GoldmanSpec,
    IonChannel,
    IonSpecies,
    NernstSpec,
)
from .core_channels import (
    alpha_h,
    alpha_m,
    alpha_n,
    beta_h,
    beta_m,
    beta_n,
    make_k_channel,
    make_leak_channel,
    make_na_channel,
)
from .hodgkin_huxley import HodgkinHuxley
from .additional_channels import (
    make_ican_channel,
    make_ical_channel,
    make_icat_channel,
    make_ih_channel,
    make_ika_channel,
    make_ikca_channel,
    make_ikir_channel,
    make_im_channel,
    make_inap_channel,
    make_inar_channel,
)
from .clamp_simulations import (
    simulate_voltage_clamp,
    simulate_current_clamp,
    simulate_batch,
)
from .electrochemistry import goldman_potential, nernst_potential
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
    "CalciumDynamics",
    "GatingVariable",
    "GoldmanSpec",
    "IonChannel",
    "IonSpecies",
    "NernstSpec",
    "HodgkinHuxley",
    "make_ican_channel",
    "make_ical_channel",
    "make_icat_channel",
    "make_ih_channel",
    "make_ika_channel",
    "make_ikca_channel",
    "make_ikir_channel",
    "make_im_channel",
    "make_inap_channel",
    "make_inar_channel",
    "alpha_h",
    "alpha_m",
    "alpha_n",
    "beta_h",
    "beta_m",
    "beta_n",
    "make_k_channel",
    "make_leak_channel",
    "make_na_channel",
    "simulate_voltage_clamp",
    "simulate_current_clamp",
    "simulate_batch",
    "nernst_potential",
    "goldman_potential",
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
