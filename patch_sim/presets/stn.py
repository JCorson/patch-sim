"""Subthalamic nucleus pacemaker preset factory."""

from typing import Any

from patch_sim.calcium import CalciumDynamics
from patch_sim.channels import (
    make_ical_channel,
    make_icat_channel,
    make_ih_channel,
    make_ika_channel,
    make_ikca_channel,
    make_ikv31_channel,
    make_inap_channel,
    make_k_leak_channel,
    make_katp_channel,
    make_stn_na_channel,
)
from patch_sim.constants import (
    ACTION_POTENTIAL,
    FI_CURVE,
    REPETITIVE_FIRING,
    SUBTHRESHOLD_RESPONSE,
)
from patch_sim.neuron import Neuron

PROTOCOL_ADJUSTMENTS: dict[str, dict[str, Any]] = {
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
}


def make_stn() -> Neuron:
    """Subthalamic nucleus pacemaker (Otsuka 2004, Wigmore & Lacey 2000).

    Single-compartment model (~15 µm soma, area_cm2=7e-6 cm²; C ≈ 7 pF).
    The preset reproduces autonomous tonic pacemaking at ~11 Hz (Bevan &
    Wilson 1999 report 5–50 Hz in slice; ISI CV ≈ 0.05–0.15 per Hallworth,
    Wilson & Bevan 2003) and a conditional post-inhibitory rebound burst mode
    driven by ICaT.

    Pacemaking mechanism:
        INaP (g=0.05 mS/cm²) provides a persistent Na⁺ window current that
        destabilises any rest near −60 mV.  Ih (g=1.0 mS/cm²) activates
        during the AHP and drives slow depolarisation back to threshold,
        sustaining regular autonomous firing.  v_rest=−60 mV is the simulation
        starting point; this cell has no stable zero-current resting potential.
        g_NaL=0 / g_KL=0.04 mS/cm²: a balanced Na+K leak that pins I_total=0
        at v_rest prevents pacemaking; the pure-K⁺ leak lets the oscillator
        operate freely.

    IKv3.1 delayed rectifier:
        IKv3.1 (Erisir 1999, V½ = −12.4 mV, n², τ_floor=0.2 ms) is the sole
        delayed rectifier.  Wigmore & Lacey (2000) directly characterised the
        dominant somatic K current in rat STN as Kv3.1-like (V½ near −13 mV,
        threshold ≈ −38 mV) and concluded it enables high-frequency spike
        trains.  The Otsuka K factory is retained for structural symmetry but
        set to g_K=0.

    Burst mode (conditional):
        Prominent ICaT (g=5.0 mS/cm²) supports post-inhibitory rebound bursts
        when sufficient prior hyperpolarisation de-inactivates the ft gate.
        On release, ICaT and Ih together drive a high-frequency burst before
        IKCa repolarises the cell back to tonic mode.  Burst mode can also be
        triggered by NMDA-receptor activation (Beurrier et al. 1999); NMDA is
        not modelled here, so burst mode is only reachable via a
        hyperpolarising-step-and-release protocol.

    Depolarisation-block recovery:
        Three complementary mechanisms cooperate to repolarise the cell after
        sustained suprathreshold drive:

        1. INaP slow inactivation (sNaP, via ``make_inap_channel``) — removes
           >70 % of the persistent Na⁺ window current at the plateau.
        2. Fast-Na slow inactivation (sNa, via ``make_stn_na_channel``) —
           closes the residual fast-Na h-tail at the depolarised plateau
           (Otsuka 2004 h_inf ≈ 1 % at −15 mV; Do & Bean 2003 established
           slow Na inactivation in STN directly).
        3. K_ATP (Stanford & Lacey 1996; Bevan & Wilson 1999; Hahn &
           McIntyre 2010) via ``make_katp_channel(g_max=0.5)`` — a
           voltage-driven slow-activation proxy (V½ = −25 mV, τ ≈ 400 ms)
           for the metabolically gated Kir6.x channel.  Subthreshold
           availability is < 2 %, so autonomous pacemaking is unaffected.

    Known limitations:
        - ``find_zero_current_voltage_all_presets`` excludes this preset —
          autonomous oscillator with no static zero-current rest.

    References:
        - Otsuka et al. (2004), J. Neurophysiol. 92:255 (Na kinetics)
        - Bevan & Wilson (1999), J. Neurosci. 19:7617 (pacemaking, K_ATP role)
        - Hallworth, Wilson & Bevan (2003), J. Neurosci. 23:7525 (ISI regularity)
        - Beurrier et al. (1999), J. Neurosci. 19:599 (NMDA burst mode)
        - Do & Bean (2003), Neuron 39:109 (STN INaP, slow Na inactivation)
        - Magistretti & Alonso (1999), J. Gen. Physiol. 114:491 (INaP)
        - Fleidervish & Gutnick (1996), J. Physiol. 493:83 (slow Na
          inactivation, cortical pyramidal)
        - Mickus, Jung & Spruston (1999), Biophys. J. 76:846 (slow Na
          inactivation, CA1 pyramidal)
        - Stanford & Lacey (1996), J. Neurophysiol. 75:1714 (K_ATP in STN)
        - Hahn & McIntyre (2010), J. Comput. Neurosci. 28:425 (STN model
          with K_ATP)
        - Wigmore & Lacey (2000), J. Physiol. 527:493 (Kv3-like K in STN)
        - Erisir et al. (1999), J. Neurophysiol. 82:2476 (Kv3.1 kinetics)
        - Zhou & Lee (2011), Neuroscience 195:14 (Kv3 in BG output neurons)

    Returns:
        Fully-configured :class:`~patch_sim.Neuron` — ~15 µm soma
        (area_cm2=7e-6 cm²) with Otsuka Na/K core, IKv3.1, ICaT/ICaL/IKa/
        IKCa/INaP/Ih/K_ATP auxiliary channels, and tuned CalciumDynamics.
    """
    return Neuron(
        v_rest=-60.0,
        Na_out=145.0,
        K_out=5.0,
        Q10=1.0,
        T_ref=308.15,
        channels=(
            make_stn_na_channel(g_max=14.0),
            # K-leak only — no Na-leak, no HH-style core K (Kv3.1 is the
            # delayed rectifier).  See preset docstring for rationale.
            make_k_leak_channel(g_max=0.04),
            make_icat_channel(g_max=5.0),
            make_ical_channel(g_max=0.5),
            make_ika_channel(g_max=3.0),
            make_ikca_channel(g_max=1.0),
            make_ih_channel(g_max=1.0),
            make_inap_channel(g_max=0.05),
            make_ikv31_channel(g_max=1.0),
            make_katp_channel(g_max=0.5),
        ),
        # ICaT g=5.0 mS/cm² is the largest Ca conductance in any preset; low
        # alpha_ca=1.1e-5 compensates for the high Ca influx per spike, keeping
        # peak ca_i ≤ 5 µM.  ca_init is a non-equilibrium Ca²⁺ starting guess
        # (the oscillator is never truly at rest).
        calcium_dynamics=CalciumDynamics(
            alpha_ca=1.1e-5, tau_ca=20.0, ca_rest=1e-4, ca_init=7.325e-4
        ),
        area_cm2=7e-6,
    )
