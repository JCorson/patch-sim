"""Patch Clamp Simulator.

A package for simulating patch clamp experiments using conductance-based neuron models.
"""

from . import constants
from .additional_channels import (
    make_ical_channel,
    make_ican_channel,
    make_icat_channel,
    make_ih_channel,
    make_ika_channel,
    make_ikca_channel,
    make_ikir_channel,
    make_im_channel,
    make_inap_channel,
    make_inar_channel,
)
from .calcium import CalciumDynamics
from .channels import (
    GatingVariable,
    GoldmanSpec,
    IonChannel,
    IonSpecies,
    NernstSpec,
)
from .clamp_simulations import (
    SimulationResult,
    simulate_batch,
    simulate_current_clamp,
    simulate_current_clamp_from_state,
    simulate_voltage_clamp,
    simulate_voltage_clamp_from_state,
)
from .constants import (
    CURRENT_CLAMP,
    CURRENT_PROTOCOLS,
    VOLTAGE_CLAMP,
    VOLTAGE_PROTOCOLS,
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
from .electrochemistry import goldman_potential, nernst_potential
from .neuron import Neuron
from .neuron_factory import (
    CHANNEL_REGISTRY,
    ChannelConfig,
    NeuronConfig,
    make_neuron,
)
from .presets import (
    NEURON_PRESET_NAMES,
    NEURON_PRESETS,
    NEURON_PROTOCOL_ADJUSTMENTS,
    PROTOCOL_PRESET_NAMES,
    PROTOCOL_PRESETS,
    build_protocol_from_preset,
)
from .protocols import (
    build_current_protocol,
    build_voltage_protocol,
    chirp_current,
    noise_current,
    pulse_train,
    pulse_train_voltage,
    ramp_current,
    ramp_voltage,
    sinusoidal_current,
    step_current,
    # Voltage clamp protocols
    step_voltage,
)

__all__ = [
    "constants",
    "CURRENT_CLAMP",
    "CURRENT_PROTOCOLS",
    "VOLTAGE_CLAMP",
    "VOLTAGE_PROTOCOLS",
    "CalciumDynamics",
    "GatingVariable",
    "GoldmanSpec",
    "IonChannel",
    "IonSpecies",
    "NernstSpec",
    "Neuron",
    "CHANNEL_REGISTRY",
    "ChannelConfig",
    "NeuronConfig",
    "make_neuron",
    "NEURON_PRESET_NAMES",
    "NEURON_PRESETS",
    "NEURON_PROTOCOL_ADJUSTMENTS",
    "PROTOCOL_PRESET_NAMES",
    "PROTOCOL_PRESETS",
    "build_protocol_from_preset",
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
    "SimulationResult",
    "simulate_voltage_clamp",
    "simulate_current_clamp",
    "simulate_voltage_clamp_from_state",
    "simulate_current_clamp_from_state",
    "simulate_batch",
    "nernst_potential",
    "goldman_potential",
    "build_current_protocol",
    "build_voltage_protocol",
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
]
