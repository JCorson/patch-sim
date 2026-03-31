"""Built-in preset configurations for the patch_sim core library.

Contains neuron presets (as :class:`NeuronConfig` instances), protocol
presets (as plain parameter dicts), and neuron-specific protocol
adjustments — all expressed in terms of core library types so they can
be used without importing the UI package.
"""

from typing import Any

import numpy as np

from .additional_channels import (
    make_ican_channel,
    make_ical_channel,
    make_icat_channel,
    make_ih_channel,
    make_ika_channel,
    make_ikca_channel,
    make_im_channel,
    make_inap_channel,
)

from .constants import (
    CURRENT_CLAMP,
    VOLTAGE_CLAMP,
)
from .neuron_factory import ChannelConfig, NeuronConfig

# ---------------------------------------------------------------------------
# Neuron presets
# ---------------------------------------------------------------------------

NEURON_PRESETS: dict[str, NeuronConfig] = {
    "Squid Giant Axon (Classic HH)": NeuronConfig(
        # Original Hodgkin-Huxley (1952) parameters — the app defaults.
        # Ref: Hodgkin & Huxley (1952), J. Physiol. 117:500
    ),
    "Fast-Spiking Interneuron": NeuronConfig(
        # High g_Na/g_K for narrow spikes; IKa shapes inter-spike interval.
        # Refs: Wang & Buzsáki (1996), J. Neurosci. 16:6402;
        #       Pospischil et al. (2008), Biol. Cybern. 99:427
        g_Na=150.0,
        g_K=50.0,
        channels=(ChannelConfig(make_ika_channel, g_max=5.0),),
    ),
    "Pyramidal Neuron": NeuronConfig(
        # Ih produces voltage sag on hyperpolarization; INaP amplifies
        # subthreshold inputs; IM provides spike-frequency adaptation.
        # Refs: Mainen & Sejnowski (1996);
        #       Pospischil et al. (2008), Biol. Cybern. 99:427
        channels=(
            ChannelConfig(make_ih_channel, g_max=1.5),
            ChannelConfig(make_inap_channel, g_max=0.5),
            ChannelConfig(make_im_channel, g_max=0.5),
        ),
    ),
    "Purkinje Cell": NeuronConfig(
        # L-type and T-type Ca²⁺ channels drive complex spiking;
        # IKCa couples Ca²⁺ influx to after-hyperpolarization.
        # Ref: De Schutter & Bower (1994), J. Neurophysiol. 71:375
        channels=(
            ChannelConfig(make_ical_channel, g_max=1.0),
            ChannelConfig(make_icat_channel, g_max=0.5),
            ChannelConfig(make_ikca_channel, g_max=2.0),
        ),
    ),
    "Dopaminergic Neuron": NeuronConfig(
        # Ih drives pacemaker sag and rebound; IM provides slow
        # oscillatory hyperpolarization.
        # Refs: Wilson & Callaway (2000), J. Neurophysiol. 83:3084;
        #       Komendantov et al. (2004)
        channels=(
            ChannelConfig(make_ih_channel, g_max=2.0),
            ChannelConfig(make_im_channel, g_max=1.0),
        ),
    ),
    "Thalamic Relay": NeuronConfig(
        # T-type Ca²⁺ produces low-threshold spike; Ih causes
        # post-inhibitory rebound burst after hyperpolarizing step.
        # Ref: McCormick & Huguenard (1992), J. Neurophysiol. 68:1384
        channels=(
            ChannelConfig(make_icat_channel, g_max=1.5),
            ChannelConfig(make_ih_channel, g_max=1.0),
        ),
    ),
    "Hippocampal CA1 Pyramidal": NeuronConfig(
        # Reduced g_Na/g_K vs squid axon; IKa shortens ISI; IM provides
        # spike-frequency adaptation; small Ih produces modest voltage sag;
        # Ca²⁺ channels (L, N, T) and IKCa together produce the pronounced
        # after-hyperpolarization (AHP) characteristic of CA1 cells.
        # Refs: Warman et al. (1994); Migliore et al. (1999), ModelDB #2796
        g_Na=35.0,
        g_K=10.0,
        channels=(
            ChannelConfig(make_ika_channel, g_max=0.5),
            ChannelConfig(make_im_channel, g_max=0.5),
            ChannelConfig(make_ih_channel, g_max=0.05),
            ChannelConfig(make_ical_channel, g_max=0.5),
            ChannelConfig(make_ican_channel, g_max=0.3),
            ChannelConfig(make_icat_channel, g_max=0.3),
            ChannelConfig(make_ikca_channel, g_max=2.0),
        ),
    ),
    "STN Neuron": NeuronConfig(
        # High g_Na/g_K sustain autonomous tonic firing at 5–50 Hz;
        # prominent ICaT (g_T = 5 mS/cm²) drives rebound bursts after
        # hyperpolarization; IKCa limits burst duration; Ih provides
        # pacemaker depolarization.
        # Refs: Otsuka et al. (2004); Farries & Wilson (2012), J. Neurophysiol.
        g_Na=49.0,
        g_K=57.0,
        channels=(
            ChannelConfig(make_icat_channel, g_max=5.0),
            ChannelConfig(make_ical_channel, g_max=0.5),
            ChannelConfig(make_ika_channel, g_max=3.0),
            ChannelConfig(make_ikca_channel, g_max=1.0),
            ChannelConfig(make_ih_channel, g_max=0.5),
        ),
    ),
    "Thalamic Reticular Nucleus": NeuronConfig(
        # Hyperpolarised resting potential (−77 mV) and exceptionally large
        # ICaT (g_T ≈ 3.5 mS/cm²) are the hallmarks of TRN cells; these
        # combine to produce rhythmic burst firing and sleep-spindle
        # oscillations.  No auxiliary channels beyond ICaT are needed.
        # Refs: Huguenard & Prince (1992), J. Neurosci. 12:3804;
        #       Destexhe et al. (1994)
        v_rest=-77.0,
        channels=(ChannelConfig(make_icat_channel, g_max=3.5),),
    ),
    "Stomatogastric Ganglion": NeuronConfig(
        # Depolarised resting potential (−55 mV) and ~8 conductances produce
        # rhythmic bursting; large IKa and IKCa shape burst waveform; slow
        # ICaL drives plateau depolarisation; Ih contributes to inter-burst
        # pacemaker potential.  Highly parameter-sensitive — small changes
        # alter burst duty cycle substantially.
        # Refs: Prinz et al. (2003), J. Neurophysiol. 90:3998;
        #       Turrigiano et al. (1995)
        v_rest=-55.0,
        channels=(
            ChannelConfig(make_ika_channel, g_max=8.0),
            ChannelConfig(make_ical_channel, g_max=2.0),
            ChannelConfig(make_ikca_channel, g_max=3.0),
            ChannelConfig(make_ih_channel, g_max=1.5),
        ),
    ),
}

# ---------------------------------------------------------------------------
# Protocol presets
# ---------------------------------------------------------------------------

PROTOCOL_PRESETS: dict[str, dict[str, Any]] = {
    "Action Potential": {
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "pre_stimulus_duration": 10.0,
        "stimulus_duration": 30.0,
        "post_stimulus_duration": 10.0,
        "min_stimulus": 10.0,
        "max_stimulus": 10.0,
        "stimulus_step": 0.0,
    },
    "Subthreshold Response": {
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
        "g_K": 0.0,  # block K+ channels to isolate Na+ current
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

# ---------------------------------------------------------------------------
# Neuron-specific protocol adjustments
# ---------------------------------------------------------------------------

# Protocol parameter overrides applied on top of a protocol preset when a
# specific neuron type is active.  Only the fields that need to differ from
# the base protocol preset are listed.
#
# Structure: neuron_preset_name → protocol_preset_name → {field: value, …}
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
    "Hippocampal CA1 Pyramidal": {
        # Long moderate-amplitude step reveals adaptation and pronounced AHP.
        "Repetitive Firing": {
            "min_stimulus": 8.0,
            "max_stimulus": 8.0,
            "stimulus_duration": 300.0,
        },
        # IKa and IM raise the firing threshold above the default HH range.
        # Positive-only range; longer step to reveal spike-frequency adaptation.
        "F-I Curve": {
            "min_stimulus": 0.0,
            "max_stimulus": 30.0,
            "stimulus_step": 3.0,
            "stimulus_duration": 150.0,
        },
    },
    "STN Neuron": {
        # Hyperpolarizing step followed by release reveals rebound burst;
        # long post-stimulus window captures the burst dynamics.
        "Repetitive Firing": {
            "min_stimulus": -8.0,
            "max_stimulus": -8.0,
            "pre_stimulus_duration": 50.0,
            "stimulus_duration": 150.0,
            "post_stimulus_duration": 150.0,
        },
    },
    "Thalamic Reticular Nucleus": {
        # Hyperpolarizing step unlocks ICaT; long post-stimulus window
        # reveals rhythmic burst firing on release.
        "Repetitive Firing": {
            "min_stimulus": -5.0,
            "max_stimulus": -5.0,
            "pre_stimulus_duration": 50.0,
            "stimulus_duration": 200.0,
            "post_stimulus_duration": 150.0,
        },
    },
    "Stomatogastric Ganglion": {
        # Long window at low current to reveal slow (~1 Hz) rhythmic bursting.
        "Repetitive Firing": {
            "min_stimulus": 3.0,
            "max_stimulus": 3.0,
            "stimulus_duration": 800.0,
        },
        # Depolarised v_rest (−55 mV) lowers threshold; use a tighter
        # positive-only range so every sweep produces a clear burst response.
        "F-I Curve": {
            "min_stimulus": 0.0,
            "max_stimulus": 12.0,
            "stimulus_step": 1.5,
            "stimulus_duration": 300.0,
        },
    },
}

PROTOCOL_PRESET_NAMES: list[str] = list(PROTOCOL_PRESETS.keys())
NEURON_PRESET_NAMES: list[str] = list(NEURON_PRESETS.keys())

# Keys in a protocol preset that are not builder parameters.
_NON_BUILDER_KEYS: frozenset[str] = frozenset(
    {
        "clamp_mode",
        "g_Na",
        "g_K",
        "g_L",
        "C_m",
        "v_rest",
    }
)


def build_protocol_from_preset(
    preset_name: str,
    neuron_preset: str | None = None,
    sampling_frequency: float = 40_000.0,
    **overrides: Any,
) -> list[tuple[np.ndarray, str]]:
    """Build a protocol array list from a named preset.

    Looks up *preset_name* in :data:`PROTOCOL_PRESETS`, applies any
    neuron-specific adjustments from :data:`NEURON_PROTOCOL_ADJUSTMENTS` when
    *neuron_preset* is supplied, then applies any caller-supplied *overrides*
    before dispatching to :func:`build_current_protocol` or
    :func:`build_voltage_protocol`.

    Args:
        preset_name: Key in :data:`PROTOCOL_PRESETS`.
        neuron_preset: Optional key in :data:`NEURON_PRESETS`.  When given,
            neuron-specific parameter adjustments are merged on top of the base
            preset before *overrides* are applied.
        sampling_frequency: Sampling frequency in Hz.  Defaults to the
            standard simulation rate (40 kHz).
        **overrides: Additional keyword arguments that override any preset or
            neuron-adjustment values.  Use builder parameter names
            (e.g. ``min_stimulus``, ``stimulus_step``).

    Returns:
        List of (stimulus_array, sweep_label) pairs.

    Raises:
        KeyError: If *preset_name* is not in :data:`PROTOCOL_PRESETS`.
        ValueError: If the resolved parameters are invalid for the chosen
            protocol type.
    """
    # Deferred import to avoid a circular dependency at module load time.
    from .protocols.builders import build_current_protocol, build_voltage_protocol

    if preset_name not in PROTOCOL_PRESETS:
        raise KeyError(
            f"Unknown protocol preset {preset_name!r}. "
            f"Available: {list(PROTOCOL_PRESETS)}"
        )

    # Start from a copy of the base preset.
    config: dict[str, Any] = dict(PROTOCOL_PRESETS[preset_name])

    # Merge neuron-specific adjustments.
    if neuron_preset is not None:
        neuron_adjustments = NEURON_PROTOCOL_ADJUSTMENTS.get(neuron_preset, {})
        config.update(neuron_adjustments.get(preset_name, {}))

    # Apply caller overrides last.
    config.update(overrides)

    clamp_mode: str = config.pop("clamp_mode", "Current Clamp")
    protocol_type: str = config.pop("protocol_type", "Step")

    # Remove keys that are not builder parameters.
    for key in _NON_BUILDER_KEYS - {"clamp_mode"}:
        config.pop(key, None)

    if clamp_mode == "Current Clamp":
        return build_current_protocol(
            protocol_type=protocol_type,
            sampling_frequency=sampling_frequency,
            **config,
        )
    else:
        return build_voltage_protocol(
            protocol_type=protocol_type,
            sampling_frequency=sampling_frequency,
            **config,
        )
