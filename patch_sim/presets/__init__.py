"""Built-in preset configurations for the patch_sim core library.

Contains neuron presets (as zero-arg factory functions returning
:class:`~patch_sim.Neuron`), protocol presets (as plain parameter dicts),
and neuron-specific protocol adjustments — all expressed in terms of
core library types so they can be used without importing the UI package.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from patch_sim.constants import (
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
from patch_sim.neuron import Neuron

from .ca1 import make_ca1_pyramidal
from .cortical import make_cortical_pyramidal
from .dopaminergic import make_dopaminergic
from .fast_spiking import make_fast_spiking_interneuron
from .purkinje import make_purkinje
from .squid import make_squid_giant_axon
from .stn import make_stn
from .thalamic import make_thalamic_relay
from .trn import make_trn

# ---------------------------------------------------------------------------
# Neuron presets
# ---------------------------------------------------------------------------

NEURON_PRESETS: dict[str, Callable[[], Neuron]] = {
    SQUID_GIANT_AXON: make_squid_giant_axon,
    FAST_SPIKING_INTERNEURON: make_fast_spiking_interneuron,
    CORTICAL_PYRAMIDAL: make_cortical_pyramidal,
    PURKINJE: make_purkinje,
    DOPAMINERGIC: make_dopaminergic,
    THALAMIC_RELAY: make_thalamic_relay,
    CA1_PYRAMIDAL: make_ca1_pyramidal,
    STN: make_stn,
    TRN: make_trn,
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
        # Squid inherits the base preset range (−10 → −2 µA/cm²), which
        # drives the membrane to roughly −98 mV.  At that depth the Na⁺
        # inactivation gate h fully de-inactivates (h_inf ≈ 0.996, τ_h ≈
        # 2.5 ms — reached within ~10 ms) and the K⁺ activation gate n
        # deactivates (n_inf ≈ 0.002).  On step release, m activates rapidly
        # while h is still elevated and g_K is negligible, triggering a
        # post-hyperpolarization action potential — classic HH anode-break
        # excitation (Hodgkin & Huxley, 1952).  No ICaT or Ih channels are
        # involved; the spike is an intrinsic property of the HH52 Na/K model
        # at these depths.
    },
    FAST_SPIKING_INTERNEURON: {
        # High total leak (g_NaL+g_KL=1.5 mS/cm²) raises the firing threshold;
        # 30 µA/cm² is safely suprathreshold with the retuned Pospischil
        # kinetics (g_Na=80, g_K=30).  3–6 ms all produce exactly 1 AP;
        # 5 ms sits in the middle of that stable range.  ≥8 ms triggers a
        # second AP; the previous setting (25 µA/cm² · 5 ms) fails to reach
        # threshold under the retuned conductances (issue #301).
        ACTION_POTENTIAL: {
            "min_stimulus": 30.0,
            "max_stimulus": 30.0,
            "stimulus_duration": 5.0,
        },
        # 26 µA/cm² is just above the repetitive-firing threshold for the
        # retuned cell (≈68 spikes over 180 ms).  Lower amplitudes fire only
        # 1 AP; Pospischil kinetics sustain non-adapting high-frequency firing
        # without depolarization block.
        REPETITIVE_FIRING: {
            "min_stimulus": 26.0,
            "max_stimulus": 26.0,
            "stimulus_duration": 180.0,
        },
        # Very low R_in (~0.67 kΩ·cm²) requires large currents for noticeable
        # hyperpolarization.  −20 → −5 µA/cm² gives peaks of −74 to −67 mV.
        # No post-inhibitory rebound: Pospischil Na⁺/K⁺ kinetics at 34 °C do
        # not produce the Na⁺ de-inactivation overshoot seen in HH52 kinetics.
        HYPERPOLARIZATION_STEPS: {
            "min_stimulus": -20.0,
            "max_stimulus": -5.0,
            "stimulus_step": 5.0,
        },
    },
    CORTICAL_PYRAMIDAL: {
        # 0.3 µA/cm² peaks near −65 mV (strongly subthreshold) under the
        # Pospischil Na + M-S Kv pairing introduced in #311.
        SUBTHRESHOLD_RESPONSE: {
            "min_stimulus": 0.3,
            "max_stimulus": 0.3,
        },
        # 2 µA/cm² × 15 ms evokes a single AP under the Pospischil Na +
        # M-S Kv pairing (g_Na=35, Q10=1).  At 1.0 µA/cm² the new lower
        # g_Na fails to reach AP threshold within the 15 ms step.
        ACTION_POTENTIAL: {
            "min_stimulus": 2.0,
            "max_stimulus": 2.0,
            "stimulus_duration": 15.0,
        },
        # 800 ms at 5 µA/cm² is long enough for the (now reduced) IM to
        # accumulate and produce a measurable increase in inter-spike intervals
        # (spike-frequency adaptation); under g_M=0.075 the effect is modest
        # (~10–15% ISI growth) but reliably present.
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
        # At the deepest steps the cell shows a post-hyperpolarization rebound
        # spike driven by two overlapping mechanisms: HH anode-break excitation
        # (h fully de-inactivates at −109 mV; n deactivates; m fires on release)
        # and Ih-driven overshoot (Ih activated during the step continues
        # conducting after release, transiently depolarising the membrane).  The
        # cell has no ICaT, so the rebound is not a low-threshold Ca²⁺ burst.
        HYPERPOLARIZATION_STEPS: {
            "min_stimulus": -5.0,
            "max_stimulus": -1.0,
            "stimulus_step": 1.0,
        },
    },
    PURKINJE: {
        # Purkinje is a spontaneous pacemaker (fires without external current).
        # The Subthreshold Response and Action Potential protocols are kept for
        # UI display purposes; the cell will fire spontaneously during both.
        # Subthreshold Response: small positive current to demonstrate INaP
        # amplification of subthreshold depolarization just below threshold.
        SUBTHRESHOLD_RESPONSE: {
            "min_stimulus": 0.1,
            "max_stimulus": 0.1,
        },
        # 5 µA/cm² at 5 ms superimposed on spontaneous pacemaking.
        ACTION_POTENTIAL: {
            "min_stimulus": 5.0,
            "max_stimulus": 5.0,
            "stimulus_duration": 5.0,
        },
        # 10 µA/cm² for 180 ms drives sustained tonic firing with prominent
        # Ca²⁺ contributions (ICaL/ICaT/IKCa).  This is intrinsic tonic
        # firing, not the climbing-fibre-driven "complex spike" of in-vivo
        # Purkinje cells (which is unmodelled here).
        REPETITIVE_FIRING: {
            "min_stimulus": 10.0,
            "max_stimulus": 10.0,
            "stimulus_duration": 180.0,
        },
        # R_in ≈ 22.7 kΩ·cm²; −0.5 µA/cm² hyperpolarizes by ≈ 11 mV (to −76 mV)
        # and temporarily suppresses pacemaking, revealing Ih-driven sag and
        # rebound firing on step release.  −1.5 → −0.5 µA/cm² in 0.5 µA steps.
        HYPERPOLARIZATION_STEPS: {
            "min_stimulus": -1.5,
            "max_stimulus": -0.5,
            "stimulus_step": 0.5,
        },
    },
    DOPAMINERGIC: {
        # Subthreshold: Canavier/Komendantov kinetics (VT=-67 mV) lower the
        # firing threshold to ~0.3 µA/cm² for a 30 ms step; 0.1 µA/cm² is
        # comfortably sub-threshold and produces a passive depolarisation.
        SUBTHRESHOLD_RESPONSE: {
            "min_stimulus": 0.1,
            "max_stimulus": 0.1,
        },
        # 4 µA/cm² at 5 ms evokes a single AP; lower amplitudes are subthreshold
        # and higher amplitudes (≥12 µA/cm²) fire two APs within 30 ms.
        ACTION_POTENTIAL: {
            "min_stimulus": 4.0,
            "max_stimulus": 4.0,
            "stimulus_duration": 5.0,
        },
        # SNc DA pacemaker: tonic firing throughout, accelerating modestly
        # with depolarising drive.  The somatic single-compartment model
        # does not reproduce depolarisation block at any tested amplitude
        # × duration (empirical sweep in scratch/characterize_da_block.py
        # — tonic firing up to 15 µA/cm² × 5 s and 2 µA/cm² × 10 s).
        # Real SNc DA neurons enter block above ~100 pA sustained drive
        # (Tucker et al. 2012); reproducing this requires dendritic Na
        # inactivation absent from this representation (#323).
        # The REPETITIVE_FIRING protocol uses 0.3 µA/cm² × 3000 ms,
        # producing ≥30 full APs at ~10 Hz over 3 s.  Duration must stay
        # > 180 ms (the base REPETITIVE_FIRING preset) — see
        # test_neuron_protocol_adjustments_change_stimulus_duration.
        REPETITIVE_FIRING: {
            "min_stimulus": 0.3,
            "max_stimulus": 0.3,
            "stimulus_duration": 3000.0,
        },
        # Threshold ~1 µA/cm²; 0 → 12 µA/cm² in 1.5 µA steps spans the
        # subthreshold zone through repetitive firing.  200 ms duration shows
        # the steady-state F-I relationship.
        FI_CURVE: {
            "max_stimulus": 12.0,
            "stimulus_step": 1.5,
            "stimulus_duration": 200.0,
        },
        # R_in ≈ 3.3 kΩ·cm²; −20 → −5 µA/cm² gives peaks of −106 to −71 mV
        # with clear Ih-driven sag (25–8 mV).  At step release, Ih (g=2.0 mS/cm²
        # activated during the step) drives a transient depolarisation above
        # threshold, producing a rebound spike at the most-negative step.  This is
        # an Ih-mediated rebound using Canavier/Komendantov Na⁺ kinetics (VT = −67 mV
        # gives m_inf ≈ 27% at −48 mV, enough for Ih to trigger firing).  The cell
        # has no ICaT.
        HYPERPOLARIZATION_STEPS: {
            "min_stimulus": -20.0,
            "max_stimulus": -5.0,
            "stimulus_step": 5.0,
        },
    },
    THALAMIC_RELAY: {
        # Burst-mode TC has a very low rheobase (~0.012 µA/cm²) because the
        # slow-inactivating ICaT (issue #287) and reduced g_K (=10) combine to
        # amplify any depolarisation through the LTS.  0.01 µA/cm² stays below
        # threshold while still giving a visible voltage deflection.  The
        # subthreshold margin is narrow: any stimulus ≥ ~0.05 µA/cm² already
        # crosses LTS threshold and fires an AP, so a UI user nudging this
        # value upward will hit threshold within a few clicks.
        SUBTHRESHOLD_RESPONSE: {
            "min_stimulus": 0.01,
            "max_stimulus": 0.01,
        },
        # 8 µA/cm² × 0.5 ms evokes a single AP under the retuned (g_Na=45,
        # g_K=10) preset (#307).  Excitability rose with the lower g_K, so
        # the previous 20 µA/cm² × 2.5 ms now triggers an LTS-coupled
        # rebound spike on top of the primary AP.  Shortening the pulse to
        # 0.5 ms narrows the de-inactivation window for ICaT enough that
        # only a single AP fires.
        ACTION_POTENTIAL: {
            "min_stimulus": 8.0,
            "max_stimulus": 8.0,
            "stimulus_duration": 0.5,
        },
        # 8 µA/cm² drives sustained tonic firing via T-type Ca²⁺ and Ih
        # over 200 ms (≥52 spikes).
        REPETITIVE_FIRING: {
            "min_stimulus": 8.0,
            "max_stimulus": 8.0,
            "stimulus_duration": 200.0,
        },
        # Sustained-firing threshold falls in the low-µA/cm² range; 1 µA/cm²
        # steps over [0, 10] resolve the FI relation cleanly.  Single-spike
        # rheobase is much lower (~0.02 µA/cm²) — see the SUBTHRESHOLD_RESPONSE
        # comment — but resolving that floor would need finer steps.
        FI_CURVE: {
            "max_stimulus": 10.0,
            "stimulus_step": 1.0,
            "stimulus_duration": 100.0,
        },
        # Inherits the base HYPERPOLARIZATION_STEPS range (−10 → −2 µA/cm²).
        # Sustained hyperpolarisation de-inactivates the TC-tuned ICaT
        # (g=2.5 mS/cm²); on release a textbook post-inhibitory LTS burst
        # fires (issue #287; McCormick & Huguenard 1992).  Ih (g=1.0 mS/cm²)
        # also contributes via sag and post-step overshoot.
    },
    CA1_PYRAMIDAL: {
        # 1.0 µA/cm² peaks near −50 mV (subthreshold) under the retuned preset
        # (issue #302); the default 1.5 µA/cm² is now suprathreshold because
        # INaP lowers rheobase to ~1.1 µA/cm².
        SUBTHRESHOLD_RESPONSE: {
            "min_stimulus": 1.0,
            "max_stimulus": 1.0,
        },
        # 2 µA/cm² at 15 ms evokes a single AP with the retuned preset (INaP
        # lowers rheobase from ~5 to ~1.1 µA/cm² so the previous 6 µA/cm²
        # stimulus now produces multiple spikes).
        ACTION_POTENTIAL: {
            "min_stimulus": 2.0,
            "max_stimulus": 2.0,
            "stimulus_duration": 15.0,
        },
        # Long moderate-amplitude step reveals progressive SFA driven by IM
        # accumulation and gradual IKCa activation.  12 µA/cm² × 300 ms
        # produces ~38 spikes with growing ISIs; peak [Ca²⁺]ᵢ stays well
        # within the 0.1–5 µM physiological band (issue #302 retune).
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
        # High R_in (≈20 kΩ·cm²) with Ih; −3 → −1 µA/cm² gives peaks of
        # −86 to −71 mV with Ih-driven sag of 2–10 mV.  Pospischil kinetics
        # with the retuned leak produce larger voltage deflections per µA/cm²
        # than the previous HH52 configuration; the range is reduced accordingly
        # to stay in the biologically relevant window.
        HYPERPOLARIZATION_STEPS: {
            "min_stimulus": -3.0,
            "max_stimulus": -1.0,
            "stimulus_step": 0.5,
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
        # Depolarising bias on top of the autonomous tonic train.  At
        # +2 µA/cm² the cell fires at ~36 Hz (top of the Bevan & Wilson
        # 1999 5–50 Hz autonomous range); 200 ms is long enough to
        # comfortably exceed the ≥5 spike requirement of
        # test_repetitive_firing_preset.
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
        # Inherits the base HYPERPOLARIZATION_STEPS range (−10 → −2 µA/cm²).
        # Very high ICaT conductance (g=5.0 mS/cm²) produces a prominent
        # post-inhibitory rebound burst on step release; Ih (g=1.0 mS/cm²)
        # adds a depolarising overshoot that can trigger additional spikes.
    },
    TRN: {
        # Very low threshold with HP92 kinetics; 0.01 µA/cm² is safely subthreshold.
        SUBTHRESHOLD_RESPONSE: {
            "min_stimulus": 0.01,
            "max_stimulus": 0.01,
        },
        # TRN is a tonic pacemaker post-#308 retune (~10 Hz spontaneous from
        # v_rest = −80 mV with ICaT half-deinactivated), so within the 22 ms
        # ACTION_POTENTIAL window the cell fires several APs regardless of
        # injected current — there is no "single-AP" pulse for this preset.
        # Excluded from ``test_action_potential_preset`` via
        # ``_QUIESCENT_PRESET_NAMES`` because the strict "exactly 1 AP"
        # assertion does not apply to a pacemaker.  Stimulus parameters are
        # kept at the pre-#308 values (5 µA/cm² × 2 ms) so users still see a
        # stim-evoked perturbation on top of the spontaneous tonic train.
        ACTION_POTENTIAL: {
            "min_stimulus": 5.0,
            "max_stimulus": 5.0,
            "stimulus_duration": 2.0,
        },
        # Depolarizing step for sustained repetitive firing via ICaT;
        # 3 µA/cm² gives ≥5 spikes over 200 ms with HP92 kinetics.
        REPETITIVE_FIRING: {
            "min_stimulus": 3.0,
            "max_stimulus": 3.0,
            "stimulus_duration": 200.0,
        },
        # Low threshold with HP92 kinetics; 0–5 µA/cm² in 0.5 µA steps.
        FI_CURVE: {
            "max_stimulus": 5.0,
            "stimulus_step": 0.5,
            "stimulus_duration": 100.0,
        },
        # Sweep −5 → −1 µA/cm² (in 1.0 µA steps) for 500 ms with 100 ms pre
        # and 300 ms post.  At ≤ −2 µA/cm² the cell de-inactivates ICaT and
        # activates Ih enough to fire the HP92 rebound burst on release;
        # the −1 µA step is included as a reference subthreshold sweep so
        # the burst is visible by contrast.  Step duration is 500 ms (vs
        # the 300 ms default) so Ih has time to fully activate.
        HYPERPOLARIZATION_STEPS: {
            "pre_stimulus_duration": 100.0,
            "stimulus_duration": 500.0,
            "post_stimulus_duration": 300.0,
            "min_stimulus": -5.0,
            "max_stimulus": -1.0,
            "stimulus_step": 1.0,
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
    from patch_sim.protocols.builders import (
        build_current_protocol,
        build_voltage_protocol,
    )

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
