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
        "pre_stimulus_duration": 10.0,
        "stimulus_duration": 30.0,
        "post_stimulus_duration": 10.0,
        "min_stimulus": 10.0,
        "max_stimulus": 10.0,
        "stimulus_step": 0.0,
    },
    "Subthreshold Response": {
        **DEFAULT_NEURON_PARAMS,
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "pre_stimulus_duration": 10.0,
        "stimulus_duration": 30.0,
        "post_stimulus_duration": 10.0,
        "min_stimulus": 4.0,
        "max_stimulus": 4.0,
        "stimulus_step": 0.0,
    },
    "Repetitive Firing": {
        **DEFAULT_NEURON_PARAMS,
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "pre_stimulus_duration": 10.0,
        "stimulus_duration": 180.0,
        "post_stimulus_duration": 10.0,
        "min_stimulus": 15.0,
        "max_stimulus": 15.0,
        "stimulus_step": 0.0,
    },
    "F-I Curve": {
        **DEFAULT_NEURON_PARAMS,
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "pre_stimulus_duration": 10.0,
        "stimulus_duration": 50.0,
        "post_stimulus_duration": 10.0,
        "min_stimulus": -10.0,
        "max_stimulus": 20.0,
        "stimulus_step": 2.5,
    },
    "I-V Curve": {
        **DEFAULT_NEURON_PARAMS,
        "clamp_mode": VOLTAGE_CLAMP,
        "protocol_type": "Step",
        "pre_stimulus_duration": 5.0,
        "stimulus_duration": 20.0,
        "post_stimulus_duration": 5.0,
        "min_stimulus": -100.0,
        "max_stimulus": 60.0,
        "stimulus_step": 10.0,
        "vc_holding_voltage": -70.0,
    },
    "Na+ Channel Activation": {
        **DEFAULT_NEURON_PARAMS,
        "g_K": 0.0,  # override: block K+ channels to isolate Na+ current
        "clamp_mode": VOLTAGE_CLAMP,
        "protocol_type": "Step",
        "pre_stimulus_duration": 5.0,
        "stimulus_duration": 20.0,
        "post_stimulus_duration": 5.0,
        "vc_holding_voltage": -70.0,
        "min_stimulus": -60.0,
        "max_stimulus": 60.0,
        "stimulus_step": 10.0,
    },
    "Frequency Response": {
        **DEFAULT_NEURON_PARAMS,
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Chirp",
        "pre_stimulus_duration": 0.0,
        "stimulus_duration": 500.0,
        "post_stimulus_duration": 0.0,
        "dc_offset": 8.0,
        "amplitude": 4.0,
        "start_frequency": 1.0,
        "end_frequency": 100.0,
    },
}

PRESET_NAMES: list[str] = list(PRESETS.keys())
