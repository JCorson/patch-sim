"""Built-in preset configurations for the patch_sim web UI."""

from typing import Any

from patch_sim.constants import DEFAULT_NEURON_PARAMS
from patch_sim_ui.constants import CURRENT_CLAMP, VOLTAGE_CLAMP

# Each preset is a dict of state variable names → values.
# Keys must match field names in AppState exactly.

PROTOCOL_PRESETS: dict[str, dict[str, Any]] = {
    "Action Potential": {
        **DEFAULT_NEURON_PARAMS,
        # Experiment
        "clamp_mode": CURRENT_CLAMP,
        # Protocol
        "protocol_type": "Step",
        "duration": 50.0,
        "current_amplitude": 10.0,
        "step_start": 10.0,
        "step_duration": 30.0,
    },
    "Subthreshold Response": {
        **DEFAULT_NEURON_PARAMS,
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "duration": 50.0,
        "current_amplitude": 4.0,
        "step_start": 10.0,
        "step_duration": 30.0,
    },
    "Repetitive Firing": {
        **DEFAULT_NEURON_PARAMS,
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "duration": 200.0,
        "current_amplitude": 15.0,
        "step_start": 10.0,
        "step_duration": 180.0,
    },
    "I-V Curve": {
        **DEFAULT_NEURON_PARAMS,
        "clamp_mode": VOLTAGE_CLAMP,
        "protocol_type": "I-V Curve",
        "duration": 20.0,
        "vc_voltage_min": -100.0,
        "vc_voltage_max": 60.0,
        "vc_voltage_step": 10.0,
        "vc_pre_pulse_duration": 5.0,
        "vc_post_pulse_duration": 5.0,
        "vc_holding_voltage": -70.0,
    },
    "Na+ Channel Activation": {
        **DEFAULT_NEURON_PARAMS,
        "g_K": 0.0,  # override: block K+ channels to isolate Na+ current
        "clamp_mode": VOLTAGE_CLAMP,
        "protocol_type": "I-V Curve",
        "duration": 20.0,
        "vc_holding_voltage": -70.0,
        "vc_voltage_min": -60.0,
        "vc_voltage_max": 60.0,
        "vc_voltage_step": 10.0,
        "vc_pre_pulse_duration": 5.0,
        "vc_post_pulse_duration": 5.0,
    },
    "Frequency Response": {
        **DEFAULT_NEURON_PARAMS,
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Chirp",
        "duration": 500.0,
        "dc_offset": 8.0,
        "amplitude": 4.0,
        "start_frequency": 1.0,
        "end_frequency": 100.0,
    },
}

NEURON_PRESETS: dict[str, dict[str, Any]] = {
    "Fast-Spiking Interneuron": {
        **DEFAULT_NEURON_PARAMS,
        # High g_Na/g_K for narrow spikes; IKa shapes inter-spike interval.
        # Long step reveals high-frequency non-adapting firing.
        "g_Na": 150.0,
        "g_K": 50.0,
        "ika_enabled": True,
        "ika_g_max": 5.0,
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "duration": 200.0,
        "current_amplitude": 20.0,
        "step_start": 10.0,
        "step_duration": 180.0,
    },
    "Pyramidal Neuron": {
        **DEFAULT_NEURON_PARAMS,
        # Ih produces voltage sag on hyperpolarization; INaP amplifies
        # subthreshold inputs; IM provides spike-frequency adaptation.
        "ih_enabled": True,
        "ih_g_max": 1.5,
        "inap_enabled": True,
        "inap_g_max": 0.5,
        "im_enabled": True,
        "im_g_max": 0.5,
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "duration": 300.0,
        "current_amplitude": 10.0,
        "step_start": 10.0,
        "step_duration": 280.0,
    },
    "Purkinje Cell": {
        **DEFAULT_NEURON_PARAMS,
        # L-type and T-type Ca²⁺ channels drive complex spiking;
        # IKCa couples Ca²⁺ influx to after-hyperpolarization.
        "ical_enabled": True,
        "ical_g_max": 1.0,
        "icat_enabled": True,
        "icat_g_max": 0.5,
        "ikca_enabled": True,
        "ikca_g_max": 2.0,
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "duration": 200.0,
        "current_amplitude": 12.0,
        "step_start": 10.0,
        "step_duration": 180.0,
    },
    "Dopaminergic Neuron": {
        **DEFAULT_NEURON_PARAMS,
        # Ih drives pacemaker sag and rebound; IM provides slow
        # oscillatory hyperpolarization. Long low-amplitude step reveals
        # slow rhythmic firing (~2–5 Hz).
        "ih_enabled": True,
        "ih_g_max": 2.0,
        "im_enabled": True,
        "im_g_max": 1.0,
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "duration": 500.0,
        "current_amplitude": 5.0,
        "step_start": 10.0,
        "step_duration": 480.0,
    },
    "Thalamic Relay": {
        **DEFAULT_NEURON_PARAMS,
        # T-type Ca²⁺ produces low-threshold spike; Ih causes
        # post-inhibitory rebound burst after hyperpolarizing step.
        "icat_enabled": True,
        "icat_g_max": 1.5,
        "ih_enabled": True,
        "ih_g_max": 1.0,
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "duration": 300.0,
        "current_amplitude": -5.0,
        "step_start": 50.0,
        "step_duration": 150.0,
    },
}

PRESETS: dict[str, dict[str, Any]] = {**PROTOCOL_PRESETS, **NEURON_PRESETS}

PROTOCOL_PRESET_NAMES: list[str] = list(PROTOCOL_PRESETS.keys())
NEURON_PRESET_NAMES: list[str] = list(NEURON_PRESETS.keys())
