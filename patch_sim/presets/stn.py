"""Subthalamic nucleus pacemaker preset factory."""

from patch_sim.calcium import CalciumDynamics
from patch_sim.channels import (
    make_ical_channel,
    make_icat_channel,
    make_ih_channel,
    make_ika_channel,
    make_ikca_channel,
    make_ikv31_channel,
    make_inap_channel,
    make_katp_channel,
    make_stn_k_channel,
    make_stn_na_channel,
)
from patch_sim.neuron import Neuron


def make_stn() -> Neuron:
    """Subthalamic nucleus pacemaker (Otsuka 2004, Wigmore & Lacey 2000).

    Returns:
        Neuron configured as an autonomous tonic pacemaker with INaP + Ih
        rhythmic drive, Kv3.1-like delayed rectifier, conditional ICaT-driven
        burst mode, and a three-mechanism depol-block recovery (sNaP, sNa,
        K_ATP).
    """
    # area_cm2 = 7e-6 cm² — ~15 µm soma typical of subthalamic projection
    # neurons.  C ≈ 7 pF in the simulation.
    #
    # Autonomous tonic pacemaker with conditional burst mode.
    #
    # PRIMARY MODE — tonic pacemaking: INaP window current destabilises
    # any rest near −60 mV; Ih activates during the AHP and drives slow
    # depolarisation back to threshold, producing autonomous firing at
    # ~11 Hz in this single-compartment model (Bevan & Wilson 1999 report
    # 5–50 Hz in slice; Hallworth, Wilson & Bevan 2003 report ISI
    # CV ≈ 0.05–0.15 — the regular single-spike pattern is exercised by
    # ``test_stn_tonic_firing_is_regular_single_spikes``).  IKv3.1
    # (Erisir 1999, V½ = −12.4 mV, n²,
    # τ_floor = 0.2 ms) replaces the Otsuka K factory as the sole
    # delayed rectifier — Wigmore & Lacey (2000) directly characterised
    # the dominant somatic K current in rat STN as Kv3.1-like (V½ near
    # −13 mV, threshold ≈ −38 mV) and concluded it "enables high-
    # frequency spike trains in SThN neurones."  Zhou & Lee (2011)
    # confirms Kv3-class channels as the rapid repolariser in basal-
    # ganglia output neurons.  The Otsuka K factory is retained for
    # structural symmetry but pinned to ``g_K=0``; this is the same
    # core-K-bypass pattern used by the cortical pyramidal (#311) and
    # FSI (#301) presets.
    #
    # CONDITIONAL MODE — burst firing: prominent ICaT (g_T = 5 mS/cm²)
    # supports post-inhibitory rebound bursts when the cell is sufficiently
    # hyperpolarised to de-inactivate the ft gate; on release, ICaT and Ih
    # together drive a high-frequency rebound burst before IKCa repolarises
    # the cell back to tonic mode.  Burst mode can also be triggered by
    # NMDA-receptor activation (Beurrier et al. 1999, J. Neurosci. 19:599);
    # NMDA is not modelled here, so burst mode is reachable in this preset
    # only via the hyperpolarising-step-and-release protocol.
    #
    # FIX — depolarization-block recovery (#324). Three complementary
    # mechanisms cooperate so the cell repolarises after a sustained
    # suprathreshold step (e.g. +5 µA/cm² × 200 ms) instead of hanging
    # on a −15 mV plateau:
    #   1. INaP slow inactivation (sNaP, Magistretti & Alonso 1999)
    #      baked into ``make_inap_channel`` — removes >70 % of the
    #      persistent Na⁺ window current at the plateau.
    #   2. Fast-Na slow inactivation (sNa, Fleidervish & Gutnick 1996;
    #      Mickus et al. 1999; Do & Bean 2003) baked into
    #      ``make_stn_na_channel`` — closes the residual ~10–20
    #      µA/cm² h-tail at the depolarised plateau that single-gate
    #      INaP slow inactivation could not reach (Otsuka 2004 h_inf
    #      ≈ 1 % at −15 mV × g_Na = 30 mS/cm²).
    #   3. K_ATP (Stanford & Lacey 1996; Bevan & Wilson 1999;
    #      Hahn & McIntyre 2010) via make_katp_channel(g_max=0.5) —
    #      provides outward K⁺ drive under sustained depolarisation,
    #      modelled here as a voltage-driven slow-activation proxy
    #      (V½ = −25 mV, τ ≈ 400 ms) for the metabolically gated
    #      Kir6.x channel.  Subthreshold availability is < 2 % so
    #      autonomous pacemaking (≤ ~+2 µA/cm²) is unaffected.
    #
    # Refs: Otsuka et al. (2004), J. Neurophysiol. 92:255 (Na kinetics);
    #       Bevan & Wilson (1999), J. Neurosci. 19:7617 (pacemaking,
    #         K_ATP role);
    #       Hallworth, Wilson & Bevan (2003), J. Neurosci. 23:7525
    #         (STN tonic ISI regularity, CV ≈ 0.05–0.15 — issue #326);
    #       Beurrier et al. (1999), J. Neurosci. 19:599 (NMDA burst mode);
    #       Do & Bean (2003), Neuron 39:109 (STN INaP, slow Na inactivation);
    #       Magistretti & Alonso (1999), J. Gen. Physiol. 114:491 (INaP);
    #       Fleidervish & Gutnick (1996), J. Physiol. 493:83 (slow Na
    #         inactivation, cortical pyramidal);
    #       Mickus, Jung & Spruston (1999), Biophys. J. 76:846 (slow Na
    #         inactivation, CA1 pyramidal);
    #       Stanford & Lacey (1996), J. Neurophysiol. 75:1714 (K_ATP in STN);
    #       Hahn & McIntyre (2010), J. Comput. Neurosci. 28:425 (STN model
    #         with K_ATP);
    #       Erecińska & Silver (1989) (ATP/ADP dynamics during firing);
    #       Wigmore & Lacey (2000), J. Physiol. 527:493 (STN Kv3-like K);
    #       Erisir et al. (1999), J. Neurophysiol. 82:2476 (Kv3.1 kinetics);
    #       Zhou & Lee (2011), Neuroscience 195:14 (Kv3 in BG output);
    #       Destexhe et al. (1993) (Ih kinetics).
    #
    # Mammalian Na⁺/K⁺ concentrations give E_Na ≈ +60.6, E_K ≈ −89.1 mV,
    # close to the Otsuka targets (+60, −90).
    #
    # v_rest = −60.0 mV is the simulation starting point (near the INaP
    # activation threshold); this cell has NO stable zero-current resting
    # potential — it is an autonomous oscillator.  Listed in the
    # ``test_find_zero_current_voltage_all_presets`` exclusion.
    #
    # Q10 = 1.0 (combined with T_ref = 308.15 K, 35 °C) holds the kinetics
    # exactly at the Otsuka 2004 reference temperature; the IKv3.1 rate
    # constants are reported by Erisir et al. (1999) at 32 °C, so the
    # combined system represents kinetics close to slice recording
    # temperature with no further runtime adjustment.
    #
    # g_NaL = 0 / g_KL = 0.04 mS/cm²: pure K⁺ background leak (Purkinje
    # pattern).  τ_m = C_m / g_KL ≈ 25 ms and R_in ≈ 25 kΩ·cm² (R_n ≈ 3.6 GΩ
    # at area = 7e-6 cm²).  A rebalanced Na+K leak that fixes I_total = 0
    # at v_rest pins the cell too stably to pacemake; the autonomous-
    # oscillator regime requires the leak NOT to bracket a zero-current root.
    #
    # Pacemaking conductances (retuned for #326 to abolish post-AHP
    # doublets and restore regular single-spike pacemaking with
    # CV(ISI) ≤ 0.15 per Hallworth, Wilson & Bevan 2003):
    #   g_NaP   = 0.05 mS/cm²: persistent Na⁺ window current that
    #     destabilises rest.  Halved from the previous 0.10 mS/cm² —
    #     the larger value rebounded the membrane within ~5 ms after
    #     each AHP, before fast-Na ``h`` had fully de-inactivated, and
    #     produced an abortive ~0 mV spikelet (the doublet artifact
    #     reported in #326).  At 0.05 mS/cm² INaP still destabilises
    #     rest (the cell remains an autonomous oscillator with no
    #     stable zero-current root) but the rebound is slow enough for
    #     ``h`` to fully de-inactivate before the next threshold
    #     crossing.
    #   g_Ih    = 1.0 mS/cm²: unchanged.  Still sizes post-AHP
    #     depolarising drive to recover the cell from AHP within the
    #     inter-spike interval at the STN rate band.
    #   g_IKv31 = 1.0 mS/cm²: sole delayed rectifier (Kv3.1, Erisir
    #     1999).  Raised from 0.2 mS/cm² to keep the AP half-width
    #     inside the 0.4–1.2 ms band after the g_Na reduction below
    #     widened it.  Wigmore & Lacey (2000) characterise Kv3.1-like
    #     as the dominant somatic K current in rat STN, so 1.0 mS/cm²
    #     is well within the channel's expressed-conductance range.
    #
    # g_Na = 14 mS/cm²: lowered from 30 mS/cm² for #326.  At g_Na = 30
    # the full-spike peak sat at ~+43 mV — well above the Bevan &
    # Wilson 1999 +5 to +25 mV range; the previous AP-peak test only
    # passed because the doublet's abortive ~0 mV spikelet pulled the
    # mean down.  At g_Na = 14 the mean peak lands near +27 mV, inside
    # the 0 to +30 mV test band and consistent with the literature
    # range from a threshold of about −51 mV (≈ 78 mV amplitude).
    #
    # ca_init = 7.325e-4 mM: the previous coupled (V, ca_i) equilibrium at
    # v_rest under the static-rest preset.  Retained as a non-zero
    # initial Ca²⁺ guess — the cell now oscillates from t = 0 so this is
    # not a true equilibrium, but it sits within the early-AP Ca²⁺ trace
    # range and avoids a spurious transient at startup.
    return Neuron(
        g_Na=14.0,
        g_K=0.0,
        v_rest=-60.0,
        Na_out=145.0,
        K_out=5.0,
        g_NaL=0.0,
        g_KL=0.04,
        Q10=1.0,
        T_ref=308.15,
        na_channel_factory=make_stn_na_channel,
        k_channel_factory=make_stn_k_channel,
        additional_channels=(
            make_icat_channel(g_max=5.0),
            make_ical_channel(g_max=0.5),
            make_ika_channel(g_max=3.0),
            make_ikca_channel(g_max=1.0),
            make_ih_channel(g_max=1.0),
            make_inap_channel(g_max=0.05),
            make_ikv31_channel(g_max=1.0),
            make_katp_channel(g_max=0.5),
        ),
        # alpha_ca/tau_ca calibrated so peak ca_i ≤ 5 µM under REPETITIVE_FIRING
        # (2 µA/cm², 200 ms).  ICaT g=5.0 mS/cm² is the largest Ca conductance in any
        # preset; low alpha_ca=1.1e-5 compensates for the high Ca influx per spike.
        calcium_dynamics=CalciumDynamics(
            alpha_ca=1.1e-5, tau_ca=20.0, ca_rest=1e-4, ca_init=7.325e-4
        ),
        area_cm2=7e-6,
    )
