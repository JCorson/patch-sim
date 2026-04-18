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
    ACTION_POTENTIAL,
    CA1_PYRAMIDAL,
    CORTICAL_PYRAMIDAL,
    CURRENT_CLAMP,
    DOPAMINERGIC,
    FAST_SPIKING_INTERNEURON,
    FI_CURVE,
    FREQUENCY_RESPONSE,
    HYPERPOLARIZATION_STEPS,
    IV_CURVE,
    NA_CHANNEL_ACTIVATION,
    PURKINJE,
    REPETITIVE_FIRING,
    SQUID_GIANT_AXON,
    STN,
    SUBTHRESHOLD_RESPONSE,
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
        #
        # K_out=7.8 mM overrides DEFAULT_K_OUT (4.0 mM, mammalian ACSF) to
        # restore the HH52 seawater value (E_K ≈ −77 mV).
        #
        # Q10=1.0: this preset IS the room-temperature squid axon model.
        # Applying a 5.2× thermal correction to bring it to mammalian
        # temperature is not meaningful — the kinetics are already those
        # of the intact preparation.
        K_out=7.8,
        Q10=1.0,
    ),
    FAST_SPIKING_INTERNEURON: NeuronConfig(
        # High g_Na drives rapid depolarization; IKv31 (Kv3.1-type, high
        # threshold, fast deactivation) repolarizes quickly and enables
        # non-adapting high-frequency firing.  The HH delayed-rectifier is
        # retained at elevated conductance to match the overall fast spike.
        # Refs: Erisir et al. (1999), J. Neurophysiol. 82:2476;
        #       Wang & Buzsáki (1996), J. Neurosci. 16:6402
        #
        # g_NaL + g_KL = 1.5 mS/cm² gives τ_m ≈ 0.67 ms — highly leaky membrane
        # that narrows the synaptic integration window, a hallmark of FS cells.
        # Values tuned so that I_NaL + I_KL + I_channels = 0 at v_rest = −65 mV
        # with K_out=4 mM (E_K ≈ −95 mV); g_total is unchanged from the
        # previous tuning (preserving τ_m = 0.67 ms and v_rest = −65 mV).
        #
        # Q10=1.0: these channels are adapted from HH52 squid-axon kinetics
        # without a well-defined mammalian thermal reference.  Applying a 5.2×
        # Q10 factor (22→37 °C) drives Na⁺ inactivation so fast that the cell
        # enters depolarization block after the first AP.  Temperature effects
        # are already implicit in the conductance values fitted to FS cell data.
        g_Na=150.0,
        g_K=50.0,
        g_NaL=0.4065,
        g_KL=1.0935,
        Q10=1.0,
        channels=(ChannelConfig(make_ikv31_channel, g_max=40.0),),
    ),
    CORTICAL_PYRAMIDAL: NeuronConfig(
        # Pospischil et al. (2008) Traub-Miles Na⁺/K⁺ kinetics (VT = −56.2 mV)
        # replace the default HH52 core channels to match the RS neuron model.
        # Ih produces voltage sag on hyperpolarization; INaP amplifies
        # subthreshold inputs; IM provides spike-frequency adaptation.
        # Ref: Pospischil et al. (2008), Biol. Cybern. 99:427
        #
        # K_out=3.32 produces E_K ≈ −100 mV (Pospischil target).
        #
        # g_NaL + g_KL = 0.05 mS/cm² gives τ_m ≈ 20 ms and R_in ≈ 20 kΩ·cm²,
        # reflecting the high input resistance (200–400 MΩ) of RS cortical
        # pyramidal cells.  With K_out=3.32, E_K ≈ −100 mV (Pospischil target),
        # so K leak is outward at v_rest = −70 mV and absorbs most of the total
        # leak conductance.  g_NaL is very small (0.0026 mS/cm²) because the Na
        # leak inward current at −70 mV would otherwise require a large outward
        # K component to compensate.
        #
        # g_h reduced from 1.5 → 0.3 mS/cm² and g_NaP from 0.5 → 0.1 mS/cm²
        # so that combined inward current at rest does not exceed the outward
        # leak + IM current; the original values caused spontaneous tonic firing.
        #
        # T_ref = 307.15 K (34 °C): Pospischil channels were recorded and fitted
        # at 34 °C, so Q10 scaling from that reference to 37 °C (T = 310.15 K)
        # gives a factor of ~1.39× rather than the default ~5.2×.  Using the
        # HH52 reference of 22 °C causes numerical instability in this model.
        v_rest=-70.0,
        K_out=3.32,
        g_NaL=0.0026,
        g_KL=0.0474,
        T_ref=307.15,
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
        # v_rest = −70.5 mV is the zero-current equilibrium with K_out=4 mM
        # (E_K ≈ −95 mV) and g_NaL = g_total = 0.02 mS/cm² (g_KL = 0).
        # With E_K ≈ −95 mV the K⁺-leak driving force at −68 mV is too large
        # for g_total = 0.02 to balance — a pure Na⁺ leak (g_KL = 0) lets the
        # HH gated K⁺ current (g_K = 36) provide the outward balance, shifting
        # v_rest to −70.5 mV.  Published Purkinje resting potentials range from
        # −65 to −72 mV depending on preparation; −70.5 mV is within range.
        #
        # g_NaL + g_KL = 0.02 mS/cm² gives τ_m ≈ 50 ms and R_in ≈ 50 kΩ·cm²,
        # reflecting the low somatic leak conductance of Purkinje cells.
        #
        # WARNING: g_KL=0 means v_rest depends on the HH gated K⁺ current
        # (g_K=36) for outward balance.  If g_K, g_Na, or the channel list
        # changes, v_rest will shift silently.  A non-zero g_KL would be
        # biophysically cleaner but requires a higher g_total to compensate
        # the larger outward K⁺ driving force at E_K ≈ −95 mV, which would
        # shorten τ_m below the 50 ms target.  Revisit if g_K is ever retuned.
        v_rest=-70.5,
        g_NaL=0.02,
        g_KL=0.0,
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
        # potential.  g_NaL + g_KL = 0.3 mS/cm² (τ_m ≈ 3.3 ms); values tuned
        # so that I_NaL + I_KL + I_channels = 0 at v_rest = −60 mV with
        # K_out=4 mM (E_K ≈ −95 mV); g_total is unchanged (τ_m preserved).
        v_rest=-60.0,
        g_NaL=0.2646,
        g_KL=0.0354,
        channels=(
            ChannelConfig(make_ih_channel, g_max=2.0),
            ChannelConfig(make_im_channel, g_max=1.0),
        ),
    ),
    THALAMIC_RELAY: NeuronConfig(
        # T-type Ca²⁺ produces low-threshold spike; Ih causes
        # post-inhibitory rebound burst after hyperpolarizing step.
        # Ref: McCormick & Huguenard (1992), J. Neurophysiol. 68:1384
        #
        # g_NaL + g_KL = 0.1 mS/cm² gives τ_m ≈ 10 ms and R_in ≈ 10 kΩ·cm²,
        # matching moderate resting conductances in thalamic relay cells.
        # Lower total leak (< 0.1) triggers spontaneous spiking via ICaT window
        # current.  Values tuned for K_out=4 mM (E_K ≈ −95 mV) to preserve
        # v_rest = −65 mV; g_total unchanged (τ_m preserved).
        g_NaL=0.0644,
        g_KL=0.0356,
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
        #
        # g_NaL + g_KL = 0.05 mS/cm² gives τ_m ≈ 20 ms and R_in ≈ 20 kΩ·cm²,
        # matching the high input resistance measured in CA1 pyramidal cells in
        # slice recordings.  Values tuned for K_out=4 mM (E_K ≈ −95 mV) to
        # preserve v_rest = −65 mV; g_total unchanged (τ_m preserved).
        #
        # Q10=1.0: as with FSI, the HH52-derived Na⁺ kinetics lack a mammalian
        # thermal reference.  A 5.2× Q10 factor accelerates Na⁺ inactivation
        # enough to cause depolarization block after the first AP; the
        # conductance values are already calibrated for CA1 behavior.
        g_Na=35.0,
        g_K=10.0,
        g_NaL=0.0411,
        g_KL=0.0089,
        Q10=1.0,
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
        #
        # g_NaL + g_KL = 0.25 mS/cm² gives τ_m ≈ 4 ms and R_in ≈ 4 kΩ·cm².
        # Lower total leak (< 0.25) shifts the zero-current equilibrium away
        # from v_rest, breaking the resting stability of this preset.  Values
        # tuned to preserve v_rest = −67 mV.  With Na_out = 145 mM (mammalian),
        # E_Na ≈ +60.6 mV, and K_out = 5 mM gives E_K ≈ −89 mV.
        g_Na=49.0,
        g_K=57.0,
        v_rest=-67.0,
        Na_out=145.0,
        K_out=5.0,
        g_NaL=0.038,
        g_KL=0.212,
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
        # ICaT (g_T = 3.5 mS/cm²) drives rhythmic burst firing and
        # sleep-spindle oscillations characteristic of TRN cells.
        # Refs: Huguenard & Prince (1992), J. Neurosci. 12:3804;
        #       Destexhe et al. (1994)
        #
        # With K_out=4 mM (E_K ≈ −95 mV), the physiological v_rest = −77 mV
        # is now reachable: K⁺ leak has 18 mV of outward driving force at rest,
        # and a small Na⁺ leak (g_NaL = 0.0104) provides the inward current to
        # balance I_KL + I_CaT_window at −77 mV.  Previously, with E_K ≈ −77 mV
        # (K_out=7.8), the K⁺ leak had zero driving force at the target rest,
        # and the equilibrium was forced to −66 mV.
        #
        # g_NaL + g_KL = 0.08 mS/cm² preserves τ_m ≈ 12.5 ms and
        # R_in ≈ 12.5 kΩ·cm².  At v_rest = −77 mV, ICaT's inactivation gate
        # is ft_inf ≈ 0.42 — well de-inactivated for post-inhibitory rebound
        # bursting and burst character on depolarising steps.
        v_rest=-77.0,
        g_NaL=0.0104,
        g_KL=0.0696,
        channels=(ChannelConfig(make_icat_channel, g_max=3.5),),
    ),
}

# ---------------------------------------------------------------------------
# Protocol presets
# ---------------------------------------------------------------------------

PROTOCOL_PRESETS: dict[str, dict[str, Any]] = {
    ACTION_POTENTIAL: {
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "pre_stimulus_duration": 10.0,
        "stimulus_duration": 30.0,
        "post_stimulus_duration": 10.0,
        "min_stimulus": 10.0,
        "max_stimulus": 10.0,
        "stimulus_step": 0.0,
    },
    SUBTHRESHOLD_RESPONSE: {
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "pre_stimulus_duration": 10.0,
        "stimulus_duration": 30.0,
        "post_stimulus_duration": 10.0,
        "min_stimulus": 1.5,
        "max_stimulus": 1.5,
        "stimulus_step": 0.0,
    },
    REPETITIVE_FIRING: {
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "pre_stimulus_duration": 10.0,
        "stimulus_duration": 180.0,
        "post_stimulus_duration": 10.0,
        "min_stimulus": 15.0,
        "max_stimulus": 15.0,
        "stimulus_step": 0.0,
    },
    FI_CURVE: {
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "pre_stimulus_duration": 10.0,
        "stimulus_duration": 50.0,
        "post_stimulus_duration": 10.0,
        "min_stimulus": 0.0,
        "max_stimulus": 20.0,
        "stimulus_step": 2.5,
    },
    HYPERPOLARIZATION_STEPS: {
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "pre_stimulus_duration": 50.0,
        "stimulus_duration": 300.0,
        "post_stimulus_duration": 100.0,
        "min_stimulus": -10.0,
        "max_stimulus": -2.0,
        "stimulus_step": 2.0,
    },
    IV_CURVE: {
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
    NA_CHANNEL_ACTIVATION: {
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
    FREQUENCY_RESPONSE: {
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
    SQUID_GIANT_AXON: {
        # 10 µA/cm² at 30 ms (default) produces 2 spikes; the membrane
        # recovers and re-fires within the step.  10 ms is long enough to
        # reach threshold and evoke exactly one action potential.
        ACTION_POTENTIAL: {
            "stimulus_duration": 10.0,
        },
    },
    FAST_SPIKING_INTERNEURON: {
        # High total leak (g_NaL+g_KL=1.5 mS/cm²) raises the firing threshold; 20 µA/cm²
        # is safely suprathreshold.  10 ms avoids a second spike at this
        # amplitude.
        ACTION_POTENTIAL: {
            "min_stimulus": 20.0,
            "max_stimulus": 20.0,
            "stimulus_duration": 10.0,
        },
        # Higher amplitude needed for non-adapting high-frequency firing with
        # the elevated Kv3.1 conductance.
        REPETITIVE_FIRING: {
            "min_stimulus": 25.0,
            "max_stimulus": 25.0,
            "stimulus_duration": 180.0,
        },
        # Very low R_in (~0.67 kΩ·cm²) requires large currents for noticeable
        # hyperpolarization.  −20 → −5 µA/cm² gives peaks of −74 to −67 mV and
        # elicits a Kv3.1-driven rebound spike on step release at −20 µA/cm².
        HYPERPOLARIZATION_STEPS: {
            "min_stimulus": -20.0,
            "max_stimulus": -5.0,
            "stimulus_step": 5.0,
        },
    },
    CORTICAL_PYRAMIDAL: {
        # Higher R_in (g_NaL+g_KL=0.05 → R_in=20 kΩ·cm²) raises excitability; 0.5
        # µA/cm² is subthreshold where 1.5 µA/cm² (default) would spike.
        SUBTHRESHOLD_RESPONSE: {
            "min_stimulus": 0.5,
            "max_stimulus": 0.5,
        },
        # 5 µA/cm² at 15 ms evokes a single AP; 30 ms default produces 2.
        ACTION_POTENTIAL: {
            "min_stimulus": 5.0,
            "max_stimulus": 5.0,
            "stimulus_duration": 15.0,
        },
        # 800 ms at 5 µA/cm² is long enough for IM to accumulate and produce
        # clearly increasing inter-spike intervals (spike-frequency adaptation).
        REPETITIVE_FIRING: {
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
        FI_CURVE: {
            "max_stimulus": 12.0,
            "stimulus_step": 1.5,
            "stimulus_duration": 300.0,
        },
        # High R_in means small currents produce large deflections.  −5 → −1 µA/cm²
        # gives peaks of −109 to −80 mV with Ih-driven sag of 10–25 mV per step.
        HYPERPOLARIZATION_STEPS: {
            "min_stimulus": -5.0,
            "max_stimulus": -1.0,
            "stimulus_step": 1.0,
        },
    },
    PURKINJE: {
        # Low R_in at rest (active channels); 0.5 µA/cm² is subthreshold.
        SUBTHRESHOLD_RESPONSE: {
            "min_stimulus": 0.5,
            "max_stimulus": 0.5,
        },
        # 10 µA/cm² at 5 ms evokes a single AP; longer durations produce 2+.
        # Threshold rose from the old preset because v_rest shifted to −70.5 mV
        # and K⁺ driving force is stronger with K_out=4.0 mM (E_K ≈ −95 mV).
        ACTION_POTENTIAL: {
            "min_stimulus": 10.0,
            "max_stimulus": 10.0,
            "stimulus_duration": 5.0,
        },
        # Moderate amplitude; complex Ca²⁺-driven spiking emerges within 200 ms.
        REPETITIVE_FIRING: {
            "min_stimulus": 12.0,
            "max_stimulus": 12.0,
            "stimulus_duration": 180.0,
        },
        # Very high passive R_in (50 kΩ·cm²) with many active channels; currents
        # beyond −2 µA/cm² push the simulation to the numerical floor (−150 mV).
        # −2 → −0.5 µA/cm² keeps peaks in −72 to −76 mV without instability.
        HYPERPOLARIZATION_STEPS: {
            "min_stimulus": -2.0,
            "max_stimulus": -0.5,
            "stimulus_step": 0.5,
        },
    },
    DOPAMINERGIC: {
        # Firing threshold rose with K_out=4.0 mM (E_K ≈ −95 mV, stronger outward
        # drive); 15 µA/cm² at 5 ms evokes a single AP, short enough to prevent
        # a second spike from the large g_NaL inward current at rest.
        ACTION_POTENTIAL: {
            "min_stimulus": 15.0,
            "max_stimulus": 15.0,
            "stimulus_duration": 5.0,
        },
        # Long pacemaking window; 15 µA/cm² drives sustained high-frequency firing.
        REPETITIVE_FIRING: {
            "min_stimulus": 15.0,
            "max_stimulus": 15.0,
            "stimulus_duration": 480.0,
        },
        # Threshold ~1.75 µA/cm²; narrow range with finer steps to show
        # the subthreshold-to-firing transition clearly.
        FI_CURVE: {
            "max_stimulus": 12.0,
            "stimulus_step": 1.5,
            "stimulus_duration": 200.0,
        },
        # R_in ≈ 3.5 kΩ·cm²; needs larger currents for visible hyperpolarization.
        # −20 → −5 µA/cm² gives peaks of −69 to −62 mV with clear Ih-driven sag
        # (2–5 mV) and a rebound spike at step release for −15 µA/cm² and above.
        HYPERPOLARIZATION_STEPS: {
            "min_stimulus": -20.0,
            "max_stimulus": -5.0,
            "stimulus_step": 5.0,
        },
    },
    THALAMIC_RELAY: {
        # R_in increased with lower total leak (0.1 mS/cm²); 0.2 µA/cm² subthreshold.
        SUBTHRESHOLD_RESPONSE: {
            "min_stimulus": 0.2,
            "max_stimulus": 0.2,
        },
        # 20 µA/cm² at 2.5 ms evokes a single AP; threshold rose with K_out=4.0 mM
        # (E_K ≈ −95 mV), and the brief pulse prevents the ICaT rebound from
        # triggering additional oscillatory spikes.
        ACTION_POTENTIAL: {
            "min_stimulus": 20.0,
            "max_stimulus": 20.0,
            "stimulus_duration": 2.5,
        },
        # 8 µA/cm² drives sustained tonic firing via T-type Ca²⁺ and Ih
        # over 200 ms (≥52 spikes).
        REPETITIVE_FIRING: {
            "min_stimulus": 8.0,
            "max_stimulus": 8.0,
            "stimulus_duration": 200.0,
        },
        # Threshold ~0.94 µA/cm²; narrow range with 1 µA/cm² steps to
        # show the subthreshold-to-firing transition cleanly.
        FI_CURVE: {
            "max_stimulus": 10.0,
            "stimulus_step": 1.0,
            "stimulus_duration": 100.0,
        },
    },
    CA1_PYRAMIDAL: {
        # 5 µA/cm² at 15 ms evokes a single AP; 30 ms default produces 2.
        ACTION_POTENTIAL: {
            "min_stimulus": 5.0,
            "max_stimulus": 5.0,
            "stimulus_duration": 15.0,
        },
        # Long moderate-amplitude step reveals adaptation and pronounced AHP.
        # 12 µA/cm² produces 2 spikes; strong IKCa limits further firing.
        REPETITIVE_FIRING: {
            "min_stimulus": 12.0,
            "max_stimulus": 12.0,
            "stimulus_duration": 300.0,
        },
        # IKa and IM raise the firing threshold above the default HH range.
        # Positive-only range; longer step to reveal spike-frequency adaptation.
        FI_CURVE: {
            "max_stimulus": 30.0,
            "stimulus_step": 3.0,
            "stimulus_duration": 150.0,
        },
        # High R_in (≈20 kΩ·cm²) with Ih; −6 → −2 µA/cm² gives peaks of
        # −91 to −70 mV, Ih-driven sag of 2–5 mV, and rebound spikes for the
        # two most negative steps (de-inactivation of low-threshold Ca²⁺ channels).
        HYPERPOLARIZATION_STEPS: {
            "min_stimulus": -6.0,
            "max_stimulus": -2.0,
            "stimulus_step": 1.0,
        },
    },
    STN: {
        # Very low threshold (~0.27 µA/cm²); keep subthreshold well below it.
        SUBTHRESHOLD_RESPONSE: {
            "min_stimulus": 0.15,
            "max_stimulus": 0.15,
        },
        # 2 µA/cm² at 5 ms evokes a single AP; default 30 ms produces 5.
        ACTION_POTENTIAL: {
            "min_stimulus": 2.0,
            "max_stimulus": 2.0,
            "stimulus_duration": 5.0,
        },
        # Depolarizing step for sustained tonic firing; STN pacemaker kinetics
        # yield ~16 spikes at ~77 Hz at 2 µA/cm² over 200 ms.
        REPETITIVE_FIRING: {
            "min_stimulus": 2.0,
            "max_stimulus": 2.0,
            "stimulus_duration": 200.0,
        },
        # Very low threshold; fine-grained 0 → 5 µA/cm² range with 0.5 steps
        # to capture the abrupt onset of firing.
        FI_CURVE: {
            "max_stimulus": 5.0,
            "stimulus_step": 0.5,
            "stimulus_duration": 200.0,
        },
    },
    TRN: {
        # R_in increased with lower total leak (g_KL=0.08); 0.1 µA/cm² subthreshold.
        SUBTHRESHOLD_RESPONSE: {
            "min_stimulus": 0.1,
            "max_stimulus": 0.1,
        },
        # 5 µA/cm² at 5 ms evokes a single AP; the shorter window prevents
        # a second spike that Q10-scaled kinetics would otherwise allow.
        ACTION_POTENTIAL: {
            "min_stimulus": 5.0,
            "max_stimulus": 5.0,
            "stimulus_duration": 5.0,
        },
        # Depolarizing step for sustained repetitive firing via ICaT;
        # 5 µA/cm² gives ~11 spikes at ~54 Hz over 200 ms.
        REPETITIVE_FIRING: {
            "min_stimulus": 5.0,
            "max_stimulus": 5.0,
            "stimulus_duration": 200.0,
        },
        # Threshold ~1.03 µA/cm²; narrow range with 1 µA/cm² steps to
        # show the subthreshold-to-firing transition cleanly.
        FI_CURVE: {
            "max_stimulus": 10.0,
            "stimulus_step": 1.0,
            "stimulus_duration": 100.0,
        },
        # High passive R_in (≈12.5 kΩ·cm²) with Ih; −2 µA/cm² already pushes
        # v_rest (−77 mV) to the numerical floor.  −1 → −0.25 µA/cm² in 0.25
        # steps gives peaks of −79 to −89 mV — enough to de-inactivate ICaT —
        # while staying far from the −150 mV boundary.
        HYPERPOLARIZATION_STEPS: {
            "min_stimulus": -1.0,
            "max_stimulus": -0.25,
            "stimulus_step": 0.25,
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
