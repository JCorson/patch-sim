"""Built-in preset configurations for the patch_sim web UI."""

from typing import Any

from patch_sim.constants import (
    DEFAULT_G_ICAL,
    DEFAULT_G_ICAT,
    DEFAULT_G_ICAN,
    DEFAULT_G_IH,
    DEFAULT_G_IKA,
    DEFAULT_G_IKCA,
    DEFAULT_G_IKIR,
    DEFAULT_G_IM,
    DEFAULT_G_NAP,
    DEFAULT_G_NAR,
    DEFAULT_NEURON_PARAMS,
)
from patch_sim_ui.constants import CURRENT_CLAMP, VOLTAGE_CLAMP

# Each preset is a dict of state variable names → values.
# Keys must match field names in AppState exactly.

# Resets all auxiliary channels to disabled with default conductances.
# Spread this into every neuron preset so that loading one preset never
# leaves channels enabled that were set by a previous preset.
_DEFAULT_AUX_CHANNEL_STATE: dict[str, Any] = {
    "ih_enabled": False,
    "ih_g_max": DEFAULT_G_IH,
    "ika_enabled": False,
    "ika_g_max": DEFAULT_G_IKA,
    "inap_enabled": False,
    "inap_g_max": DEFAULT_G_NAP,
    "inar_enabled": False,
    "inar_g_max": DEFAULT_G_NAR,
    "im_enabled": False,
    "im_g_max": DEFAULT_G_IM,
    "ikir_enabled": False,
    "ikir_g_max": DEFAULT_G_IKIR,
    "ikca_enabled": False,
    "ikca_g_max": DEFAULT_G_IKCA,
    "ical_enabled": False,
    "ical_g_max": DEFAULT_G_ICAL,
    "icat_enabled": False,
    "icat_g_max": DEFAULT_G_ICAT,
    "ican_enabled": False,
    "ican_g_max": DEFAULT_G_ICAN,
}

PROTOCOL_PRESETS: dict[str, dict[str, Any]] = {
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

NEURON_PRESETS: dict[str, dict[str, Any]] = {
    "Fast-Spiking Interneuron": {
        **DEFAULT_NEURON_PARAMS,
        **_DEFAULT_AUX_CHANNEL_STATE,
        # High g_Na/g_K for narrow spikes; IKa shapes inter-spike interval.
        # Refs: Wang & Buzsáki (1996), J. Neurosci. 16:6402;
        #       Pospischil et al. (2008), Biol. Cybern. 99:427
        "g_Na": 150.0,
        "g_K": 50.0,
        "ika_enabled": True,
        "ika_g_max": 5.0,
    },
    "Pyramidal Neuron": {
        **DEFAULT_NEURON_PARAMS,
        **_DEFAULT_AUX_CHANNEL_STATE,
        # Ih produces voltage sag on hyperpolarization; INaP amplifies
        # subthreshold inputs; IM provides spike-frequency adaptation.
        # Refs: Mainen & Sejnowski (1996);
        #       Pospischil et al. (2008), Biol. Cybern. 99:427
        "ih_enabled": True,
        "ih_g_max": 1.5,
        "inap_enabled": True,
        "inap_g_max": 0.5,
        "im_enabled": True,
        "im_g_max": 0.5,
    },
    "Purkinje Cell": {
        **DEFAULT_NEURON_PARAMS,
        **_DEFAULT_AUX_CHANNEL_STATE,
        # L-type and T-type Ca²⁺ channels drive complex spiking;
        # IKCa couples Ca²⁺ influx to after-hyperpolarization.
        # Ref: De Schutter & Bower (1994), J. Neurophysiol. 71:375
        "ical_enabled": True,
        "ical_g_max": 1.0,
        "icat_enabled": True,
        "icat_g_max": 0.5,
        "ikca_enabled": True,
        "ikca_g_max": 2.0,
    },
    "Dopaminergic Neuron": {
        **DEFAULT_NEURON_PARAMS,
        **_DEFAULT_AUX_CHANNEL_STATE,
        # Ih drives pacemaker sag and rebound; IM provides slow
        # oscillatory hyperpolarization.
        # Refs: Wilson & Callaway (2000), J. Neurophysiol. 83:3084;
        #       Komendantov et al. (2004)
        "ih_enabled": True,
        "ih_g_max": 2.0,
        "im_enabled": True,
        "im_g_max": 1.0,
    },
    "Thalamic Relay": {
        **DEFAULT_NEURON_PARAMS,
        **_DEFAULT_AUX_CHANNEL_STATE,
        # T-type Ca²⁺ produces low-threshold spike; Ih causes
        # post-inhibitory rebound burst after hyperpolarizing step.
        # Ref: McCormick & Huguenard (1992), J. Neurophysiol. 68:1384
        "icat_enabled": True,
        "icat_g_max": 1.5,
        "ih_enabled": True,
        "ih_g_max": 1.0,
    },
}

# Protocol parameter overrides applied on top of a protocol preset when a
# specific neuron type is active.  Only the fields that need to differ from
# the base protocol preset are listed; everything else falls through to the
# base preset values.
#
# Structure: neuron_type → protocol_preset_name → {field: value, …}
NEURON_PROTOCOL_ADJUSTMENTS: dict[str, dict[str, dict[str, Any]]] = {
    "Fast-Spiking Interneuron": {
        # High-amplitude, standard-length step to drive rapid non-adapting firing.
        "Repetitive Firing": {
            "min_stimulus": 20.0,
            "max_stimulus": 20.0,
            "stimulus_duration": 180.0,
        },
    },
    "Pyramidal Neuron": {
        # Longer step at moderate amplitude to reveal spike-frequency adaptation.
        "Repetitive Firing": {
            "min_stimulus": 10.0,
            "max_stimulus": 10.0,
            "stimulus_duration": 280.0,
        },
    },
    "Purkinje Cell": {
        # Moderate amplitude; complex Ca²⁺-driven spiking emerges within 200 ms.
        "Repetitive Firing": {
            "min_stimulus": 12.0,
            "max_stimulus": 12.0,
            "stimulus_duration": 180.0,
        },
    },
    "Dopaminergic Neuron": {
        # Long window at low amplitude to reveal slow (~2–5 Hz) pacemaking.
        "Repetitive Firing": {
            "min_stimulus": 5.0,
            "max_stimulus": 5.0,
            "stimulus_duration": 480.0,
        },
    },
    "Thalamic Relay": {
        # Hyperpolarizing step followed by release triggers post-inhibitory
        # rebound burst via T-type Ca²⁺ channels.
        # pre=50 ms establishes baseline; post=100 ms captures the rebound burst.
        "Repetitive Firing": {
            "min_stimulus": -5.0,
            "max_stimulus": -5.0,
            "pre_stimulus_duration": 50.0,
            "stimulus_duration": 150.0,
            "post_stimulus_duration": 100.0,
        },
    },
}

PROTOCOL_PRESET_NAMES: list[str] = list(PROTOCOL_PRESETS.keys())
NEURON_PRESET_NAMES: list[str] = list(NEURON_PRESETS.keys())
