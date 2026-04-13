"""Built-in preset configurations for the patch_sim core library.

Contains neuron presets (as :class:`NeuronConfig` instances), protocol
presets (as plain parameter dicts), and neuron-specific protocol
adjustments — all expressed in terms of core library types so they can
be used without importing the UI package.
"""

from typing import Any

import numpy as np

from .additional_channels import (
    make_ical_channel,
    make_ican_channel,
    make_icat_channel,
    make_ih_channel,
    make_ika_channel,
    make_ikca_channel,
    make_ikv31_channel,
    make_im_channel,
    make_inap_channel,
)
from .constants import (
    CA1_PYRAMIDAL,
    CORTICAL_PYRAMIDAL,
    CURRENT_CLAMP,
    DOPAMINERGIC,
    FAST_SPIKING_INTERNEURON,
    PURKINJE,
    SQUID_GIANT_AXON,
    STN,
    THALAMIC_RELAY,
    TRN,
    VOLTAGE_CLAMP,
)
from .core_channels import (
    make_pospischil_k_channel,
    make_pospischil_na_channel,
    make_stn_k_channel,
    make_stn_na_channel,
)
from .neuron_factory import ChannelConfig, NeuronConfig

# ---------------------------------------------------------------------------
# Neuron presets
# ---------------------------------------------------------------------------

NEURON_PRESETS: dict[str, NeuronConfig] = {
    SQUID_GIANT_AXON: NeuronConfig(
        # Original Hodgkin-Huxley (1952) parameters — all defaults.
        # Default ion concentrations produce HH52 reversal potentials
        # (E_Na ≈ +50, E_K ≈ −77, E_L ≈ −54 mV) so no overrides needed.
        # Ref: Hodgkin & Huxley (1952), J. Physiol. 117:500
    ),
    FAST_SPIKING_INTERNEURON: NeuronConfig(
        # High g_Na drives rapid depolarization; IKv31 (Kv3.1-type, high
        # threshold, fast deactivation) repolarizes quickly and enables
        # non-adapting high-frequency firing.  The HH delayed-rectifier is
        # retained at elevated conductance to match the overall fast spike.
        # Refs: Erisir et al. (1999), J. Neurophysiol. 82:2476;
        #       Wang & Buzsáki (1996), J. Neurosci. 16:6402
        #
        # Elevated Cl_in (19.0 mM) compensates for the large outward K⁺
        # current from g_K=50 by shifting E_L more positive, keeping the
        # zero-current equilibrium at −65 mV.
        g_Na=150.0,
        g_K=50.0,
        Cl_in=19.0,
        channels=(ChannelConfig(make_ikv31_channel, g_max=40.0),),
    ),
    CORTICAL_PYRAMIDAL: NeuronConfig(
        # Pospischil et al. (2008) Traub-Miles Na⁺/K⁺ kinetics (VT = −56.2 mV)
        # replace the default HH52 core channels to match the RS neuron model.
        # Ih produces voltage sag on hyperpolarization; INaP amplifies
        # subthreshold inputs; IM provides spike-frequency adaptation.
        # Ref: Pospischil et al. (2008), Biol. Cybern. 99:427
        #
        # K_out=3.32 produces E_K ≈ −100 mV (Pospischil target); Cl_in=7.6
        # tunes E_L so that the zero-current equilibrium is −70 mV,
        # matching the published RS model resting potential.
        #
        # g_h reduced from 1.5 → 0.3 mS/cm² and g_NaP from 0.5 → 0.1 mS/cm²
        # so that combined inward current at rest does not exceed the outward
        # leak + IM current; the original values caused spontaneous tonic firing.
        v_rest=-70.0,
        K_out=3.32,
        Cl_in=7.6,
        na_channel_factory=make_pospischil_na_channel,
        k_channel_factory=make_pospischil_k_channel,
        channels=(
            ChannelConfig(make_ih_channel, g_max=0.3),
            ChannelConfig(make_inap_channel, g_max=0.1),
            ChannelConfig(make_im_channel, g_max=0.5),
        ),
    ),
    PURKINJE: NeuronConfig(
        # L-type and T-type Ca²⁺ channels drive complex spiking;
        # IKCa couples Ca²⁺ influx to after-hyperpolarization.
        # Ref: De Schutter & Bower (1994), J. Neurophysiol. 71:375
        #
        # v_rest = −68 mV matches the published Purkinje cell resting potential.
        v_rest=-68.0,
        Cl_in=10.7,
        channels=(
            ChannelConfig(make_ical_channel, g_max=1.0),
            ChannelConfig(make_icat_channel, g_max=0.5),
            ChannelConfig(make_ikca_channel, g_max=2.0),
        ),
    ),
    DOPAMINERGIC: NeuronConfig(
        # Ih drives pacemaker sag and rebound; IM provides slow
        # oscillatory hyperpolarization.
        # Refs: Wilson & Callaway (2000), J. Neurophysiol. 83:3084;
        #       Komendantov et al. (2004)
        #
        # v_rest = −60 mV matches the published dopaminergic neuron resting
        # potential.  Cl_in = 47.0 mM shifts E_L positive to achieve this.
        v_rest=-60.0,
        Cl_in=47.0,
        channels=(
            ChannelConfig(make_ih_channel, g_max=2.0),
            ChannelConfig(make_im_channel, g_max=1.0),
        ),
    ),
    THALAMIC_RELAY: NeuronConfig(
        # T-type Ca²⁺ produces low-threshold spike; Ih causes
        # post-inhibitory rebound burst after hyperpolarizing step.
        # Ref: McCormick & Huguenard (1992), J. Neurophysiol. 68:1384
        Cl_in=10.0,
        channels=(
            ChannelConfig(make_icat_channel, g_max=1.5),
            ChannelConfig(make_ih_channel, g_max=1.0),
        ),
    ),
    CA1_PYRAMIDAL: NeuronConfig(
        # Reduced g_Na/g_K vs squid axon; IKa shortens ISI; IM provides
        # spike-frequency adaptation; small Ih produces modest voltage sag;
        # Ca²⁺ channels (L, N, T) and IKCa together produce the pronounced
        # after-hyperpolarization (AHP) characteristic of CA1 cells.
        # Refs: Warman et al. (1994); Migliore et al. (1999), ModelDB #2796
        g_Na=35.0,
        g_K=10.0,
        Cl_in=12.4,
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
    STN: NeuronConfig(
        # High-threshold Na⁺ (Otsuka 2004) and fast K⁺ DR replace the default
        # HH52 kinetics; g_Na/g_K from the original paper sustain autonomous
        # tonic firing at 5–50 Hz.  Prominent ICaT (g_T = 5 mS/cm²) drives
        # rebound bursts after hyperpolarization; IKCa limits burst duration;
        # Ih provides pacemaker depolarization.
        # Refs: Otsuka et al. (2004), J. Neurophysiol. 92:255;
        #       Farries & Wilson (2012), J. Neurophysiol.
        #
        # Mammalian Na⁺/K⁺ concentrations give E_Na ≈ +60.6, E_K ≈ −89.1 mV,
        # close to the Otsuka targets (+60, −90).  v_rest = −67 mV is the
        # stable zero-current equilibrium for this channel configuration.
        g_Na=49.0,
        g_K=57.0,
        v_rest=-67.0,
        Na_out=145.0,
        K_out=5.0,
        Cl_in=10.0,
        na_channel_factory=make_stn_na_channel,
        k_channel_factory=make_stn_k_channel,
        channels=(
            ChannelConfig(make_icat_channel, g_max=5.0),
            ChannelConfig(make_ical_channel, g_max=0.5),
            ChannelConfig(make_ika_channel, g_max=3.0),
            ChannelConfig(make_ikca_channel, g_max=1.0),
            ChannelConfig(make_ih_channel, g_max=0.5),
        ),
    ),
    TRN: NeuronConfig(
        # ICaT (g_T ≈ 3.5 mS/cm²) drives rhythmic burst firing and
        # sleep-spindle oscillations characteristic of TRN cells.
        # Refs: Huguenard & Prince (1992), J. Neurosci. 12:3804;
        #       Destexhe et al. (1994)
        #
        # Cl_in = 6.4 mM yields E_L ≈ −79 mV, pulling the zero-current
        # equilibrium to −77 mV — the biological TRN resting potential.
        # ICaT is sufficiently de-inactivated at this potential for
        # post-inhibitory rebound bursting.
        v_rest=-77.0,
        Cl_in=6.4,
        channels=(ChannelConfig(make_icat_channel, g_max=3.5),),
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
        "holding_voltage": -70.0,
    },
    "Na+ Channel Activation": {
        "clamp_mode": VOLTAGE_CLAMP,
        "protocol_type": "Step",
        "pre_stimulus_duration": 5.0,
        "stimulus_duration": 20.0,
        "post_stimulus_duration": 5.0,
        "holding_voltage": -70.0,
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
    FAST_SPIKING_INTERNEURON: {
        # Moderate amplitude to evoke a single narrow spike.
        "Action Potential": {
            "min_stimulus": 15.0,
            "max_stimulus": 15.0,
        },
        # Higher amplitude needed for non-adapting high-frequency firing with
        # the elevated Kv3.1 conductance.
        "Repetitive Firing": {
            "min_stimulus": 25.0,
            "max_stimulus": 25.0,
            "stimulus_duration": 180.0,
        },
    },
    CORTICAL_PYRAMIDAL: {
        # 800 ms at 5 µA/cm² is long enough for IM to accumulate and produce
        # clearly increasing inter-spike intervals (spike-frequency adaptation).
        "Repetitive Firing": {
            "min_stimulus": 5.0,
            "max_stimulus": 5.0,
            "stimulus_duration": 800.0,
            "pre_stimulus_duration": 50.0,
            "post_stimulus_duration": 50.0,
        },
        # Threshold is ~3–4 µA/cm²; 0 → 12 in steps of 1.5 (9 sweeps) spans
        # the subthreshold zone through fast repetitive firing.  300 ms is
        # long enough for IM-driven adaptation to be visible within each
        # suprathreshold sweep.
        "F-I Curve": {
            "min_stimulus": 0.0,
            "max_stimulus": 12.0,
            "stimulus_step": 1.5,
            "stimulus_duration": 300.0,
        },
    },
    PURKINJE: {
        # Moderate amplitude; complex Ca²⁺-driven spiking emerges within 200 ms.
        "Repetitive Firing": {
            "min_stimulus": 12.0,
            "max_stimulus": 12.0,
            "stimulus_duration": 180.0,
        },
    },
    DOPAMINERGIC: {
        # Long window at low amplitude to reveal slow (~2–5 Hz) pacemaking.
        "Repetitive Firing": {
            "min_stimulus": 5.0,
            "max_stimulus": 5.0,
            "stimulus_duration": 480.0,
        },
    },
    THALAMIC_RELAY: {
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
    CA1_PYRAMIDAL: {
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
    STN: {
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
    TRN: {
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
}

PROTOCOL_PRESET_NAMES: list[str] = list(PROTOCOL_PRESETS.keys())
NEURON_PRESET_NAMES: list[str] = list(NEURON_PRESETS.keys())


def build_protocol_from_preset(
    preset_name: str,
    neuron_preset: str | None = None,
    sampling_frequency: float = 40_000.0,
    **overrides: Any,
) -> np.ndarray:
    """Build a protocol array from a named preset.

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
        2-D array of shape ``(n_sweeps, n_samples)``.

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
