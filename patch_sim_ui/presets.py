"""Built-in preset configurations for the patch_sim web UI.

Protocol presets, neuron-protocol adjustments, and preset name lists are
re-exported from the core library.  Only the UI-specific neuron preset
format (mapping state variable names to values) is defined here.
"""

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

# Re-export from core so existing UI imports continue to work.
from patch_sim.presets import (  # noqa: F401
    NEURON_PRESET_NAMES,
    NEURON_PROTOCOL_ADJUSTMENTS,
    PROTOCOL_PRESET_NAMES,
    PROTOCOL_PRESETS,
)

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

NEURON_PRESETS: dict[str, dict[str, Any]] = {
    "Squid Giant Axon (Classic HH)": {
        **DEFAULT_NEURON_PARAMS,
        **_DEFAULT_AUX_CHANNEL_STATE,
        # Original Hodgkin-Huxley (1952) parameters — the app defaults.
        # Ref: Hodgkin & Huxley (1952), J. Physiol. 117:500
    },
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
    "Hippocampal CA1 Pyramidal": {
        **DEFAULT_NEURON_PARAMS,
        **_DEFAULT_AUX_CHANNEL_STATE,
        # Reduced g_Na/g_K vs squid axon; IKa shortens ISI; IM provides
        # spike-frequency adaptation; small Ih produces modest voltage sag;
        # Ca²⁺ channels (L, N, T) and IKCa together produce the pronounced
        # after-hyperpolarization (AHP) characteristic of CA1 cells.
        # Refs: Warman et al. (1994); Migliore et al. (1999), ModelDB #2796
        "g_Na": 35.0,
        "g_K": 10.0,
        "ika_enabled": True,
        "ika_g_max": 0.5,
        "im_enabled": True,
        "im_g_max": 0.5,
        "ih_enabled": True,
        "ih_g_max": 0.05,
        "ical_enabled": True,
        "ical_g_max": 0.5,
        "ican_enabled": True,
        "ican_g_max": 0.3,
        "icat_enabled": True,
        "icat_g_max": 0.3,
        "ikca_enabled": True,
        "ikca_g_max": 2.0,
    },
    "STN Neuron": {
        **DEFAULT_NEURON_PARAMS,
        **_DEFAULT_AUX_CHANNEL_STATE,
        # High g_Na/g_K sustain autonomous tonic firing at 5–50 Hz;
        # prominent ICaT (g_T = 5 mS/cm²) drives rebound bursts after
        # hyperpolarization; IKCa limits burst duration; Ih provides
        # pacemaker depolarization.
        # Refs: Otsuka et al. (2004); Farries & Wilson (2012), J. Neurophysiol.
        "g_Na": 49.0,
        "g_K": 57.0,
        "icat_enabled": True,
        "icat_g_max": 5.0,
        "ical_enabled": True,
        "ical_g_max": 0.5,
        "ika_enabled": True,
        "ika_g_max": 3.0,
        "ikca_enabled": True,
        "ikca_g_max": 1.0,
        "ih_enabled": True,
        "ih_g_max": 0.5,
    },
    "Thalamic Reticular Nucleus": {
        **DEFAULT_NEURON_PARAMS,
        **_DEFAULT_AUX_CHANNEL_STATE,
        # Hyperpolarised resting potential (−77 mV) and exceptionally large
        # ICaT (g_T ≈ 3.5 mS/cm²) are the hallmarks of TRN cells; these
        # combine to produce rhythmic burst firing and sleep-spindle
        # oscillations.  No auxiliary channels beyond ICaT are needed.
        # Refs: Huguenard & Prince (1992), J. Neurosci. 12:3804;
        #       Destexhe et al. (1994)
        "v_rest": -77.0,
        "icat_enabled": True,
        "icat_g_max": 3.5,
    },
    "Stomatogastric Ganglion": {
        **DEFAULT_NEURON_PARAMS,
        **_DEFAULT_AUX_CHANNEL_STATE,
        # Depolarised resting potential (−55 mV) and ~8 conductances produce
        # rhythmic bursting; large IKa and IKCa shape burst waveform; slow
        # ICaL drives plateau depolarisation; Ih contributes to inter-burst
        # pacemaker potential.  Highly parameter-sensitive — small changes
        # alter burst duty cycle substantially.
        # Refs: Prinz et al. (2003), J. Neurophysiol. 90:3998;
        #       Turrigiano et al. (1995)
        "v_rest": -55.0,
        "ika_enabled": True,
        "ika_g_max": 8.0,
        "ical_enabled": True,
        "ical_g_max": 2.0,
        "ikca_enabled": True,
        "ikca_g_max": 3.0,
        "ih_enabled": True,
        "ih_g_max": 1.5,
    },
}

# Neuron-parameter overrides applied when a protocol preset is loaded,
# regardless of which neuron type is active.  Keys must match AppState field
# names exactly.
PROTOCOL_NEURON_OVERRIDES: dict[str, dict[str, Any]] = {
    # Disable K⁺ channels so only Na⁺ current is visible.
    "Na+ Channel Activation": {"g_K": 0.0},
}
