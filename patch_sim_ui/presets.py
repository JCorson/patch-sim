"""Built-in preset configurations for the patch_sim web UI."""

from typing import Any

from patch_sim.constants import DEFAULT_NEURON_PARAMS
from patch_sim_ui.constants import CURRENT_CLAMP, VOLTAGE_CLAMP

# Each preset is a dict of state variable names → values.
# Keys must match field names in AppState exactly.
PRESETS: dict[str, dict[str, Any]] = {
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

PRESET_NAMES: list[str] = list(PRESETS.keys())
