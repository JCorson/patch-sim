"""Thalamic reticular neuron preset factory."""

from patch_sim.calcium import CalciumDynamics
from patch_sim.channels import (
    make_ih_channel,
    make_ikca_channel,
    make_trn_icat_channel,
    make_trn_k_channel,
    make_trn_na_channel,
)
from patch_sim.neuron import Neuron


def make_trn() -> Neuron:
    """Thalamic reticular neuron (Huguenard & Prince 1992, Bal & McCormick 1993).

    Returns:
        Neuron configured with HP92/Pospischil RE core, TRN-tuned ICaT, IKCa,
        and Ih to deliver the HP92 LTS-rebound multi-spike burst phenotype.
    """
    # area_cm2 = 7e-6 cm² — ~15 µm soma characteristic of thalamic
    # reticular neurons.  C ≈ 7 pF in the simulation.
    #
    # Channel set: ICaT + IKCa + Ih over the HP92/Pospischil RE Na⁺/K⁺ core.
    # Conductances:
    #   g_Na  = 50 mS/cm² (explicit somatic density — see below)
    #   g_K   = 24 mS/cm² (explicit somatic density — see below)
    #   g_T   = 2.85 mS/cm² (within HP92 voltage-clamp recorded range;
    #                       co-tuned with g_Na/g_K for the 5–15 spike LTS
    #                       rebound burst)
    #   g_KCa = 0.3 mS/cm² (Huguenard & Prince 1992, TRN)
    #   g_h   = 0.020 mS/cm² (slightly below the ≈0.025 reported by
    #                          Bal & McCormick 1993 for cat TRN; pre-#308
    #                          this set tonic rate to ~3 Hz alongside the
    #                          inherited HH52 Na/K defaults — at the new
    #                          (g_Na, g_K) the rate is ~10 Hz, still well
    #                          inside the test-accepted 1–15 Hz band and
    #                          the HP92 spontaneous-firing range)
    #
    # Refs: Huguenard & Prince (1992), J. Neurosci. 12:3804 (TRN
    #       low-threshold-spike biophysics; IKCa identified as the burst-
    #       terminating K⁺ current; published g_KCa);
    #       Destexhe et al. (1994), J. Neurophysiol. 72:803 (ICaT kinetics);
    #       Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (RE)
    #       (Pospischil's RE column gives g_Na, g_Kd, leak from which
    #       the kinetics and conductances here are derived);
    #       Bal & McCormick (1993), J. Physiol. 468:669 (Ih in cat TRN
    #       supports rebound burst by activating during hyperpolarisation
    #       and providing depolarising drive on release).
    #
    # IKCa: calcium-activated K⁺ current.  HP92 identify IKCa as the
    # mechanism that converts ICaT-mediated Ca²⁺ entry into outward K⁺
    # current — generating the AHP after spikes during tonic firing and
    # contributing to LTS-burst termination.
    #
    # Ih: HCN/funny current.  Activated by hyperpolarisation; its
    # depolarising contribution during the hyperpolarising step builds up
    # while ft de-inactivates, then drives V across the LTS threshold on
    # release — the canonical mechanism for triggering the post-inhibitory
    # rebound burst (Bal & McCormick 1993).  Without Ih, V's passive
    # relaxation from a hyperpolarised step does not overshoot v_rest and
    # the LTS does not fire — the LTS is unreachable in this preset
    # without Ih.  TRN g_h is small (0.020 mS/cm²) compared to TC's 1.0:
    # B&M93 report ≈ 0.025 for cat TRN.
    #
    # Huguenard & Prince (1992) / Pospischil (2008) Traub-Miles Na⁺/K⁺
    # kinetics (VT = −67 mV) replace the default HH52 core channels.
    # HH52 kinetics (fitted to room-temperature squid axon) over-accelerate
    # Na⁺ inactivation under the default Q10=3.0 scaling (22→37 °C, factor
    # ~5.2×), biologically wrong for a mammalian TRN cell.
    # HP92 channels were recorded at 36 °C, so T_ref=309.15 K limits the
    # Q10 correction to ~1.12× (36→37 °C) — a negligible adjustment that
    # preserves the published kinetics.
    #
    # g_Na = 50 mS/cm², g_K = 24 mS/cm² are explicit somatic densities
    # for the single-compartment reduction.  Without these overrides the
    # preset would silently inherit NeuronConfig's HH52 defaults
    # (g_Na = 120, g_K = 36), driving the mean tonic AP peak to ~+49 mV
    # and the AHP to ~−86 mV — both outside the Huguenard & Prince (1992)
    # TRN bands (peak +10 to +40 mV; AHP −75 to −55 mV).  Same kinetic
    # pattern and fix as the cortical pyramidal (#298), Purkinje (#299),
    # FSI (#301), DA (#304), STN (#305), TC (#307) presets; closes #308.
    # Pospischil 2008 Table 2 specifies (g_Na = 200, g_K = 20) for the
    # lumped RE single-compartment model, but at those densities the
    # MH92 Traub-Miles kinetics still overshoot peak.  (50, 24) is the
    # smallest reduction that lands every HP92 tonic AP-shape metric
    # (peak, AHP, half-width, threshold, firing rate) inside its band
    # while preserving (a) the HP92 rebound-burst phenotype (5–15
    # spikes, 200–600 Hz, single coherent burst) and (b) the v_rest =
    # −80 mV initial condition pinned by
    # ``test_trn_preset_vrest_is_physiological``.
    #
    # g_T is bumped down from 3.0 → 2.85 mS/cm² (within HP92 range) in
    # tandem to keep the LTS rebound burst from fragmenting into multiple
    # detected bursts: the lower g_K reduces post-spike repolarisation,
    # which lets the LTS plateau drive too many spikes (>15) when g_T is
    # held at 3.0; the small g_T cut shortens the plateau back into the
    # 5–15 spike band and keeps ``analyze_bursts`` finding exactly one
    # burst (required by ``test_trn_step_release_produces_hp92_rebound_burst``).
    #
    # PACEMAKER MODE.  With Ih and the elevated g_T, the cell is no longer
    # silent at zero current — it fires tonically at ~10 Hz, consistent
    # with the spontaneous firing observed in TRN slice recordings (HP92,
    # B&M93; their rates fall in the 1–15 Hz band).  v_rest = −80 mV is
    # the configured initial condition; the cell rapidly leaves this point
    # and settles into tonic firing with mean V around −70 mV.  Excluded
    # from ``test_all_presets_stable_at_rest`` for the same reason as
    # Purkinje (autonomous oscillator with no stable zero-current
    # equilibrium).  The HP92 rebound-burst phenotype is exercised by
    # ``test_trn_step_release_produces_hp92_rebound_burst``.
    #
    # g_NaL + g_KL = 0.07 mS/cm² gives τ_m ≈ 14.3 ms and
    # R_in ≈ 14.3 kΩ·cm² (physiological range 10–15 ms/kΩ·cm²).
    # The split is g_NaL = 0.0066, g_KL = 0.0634.
    # At V = −80 mV the ft gate is at its half-inactivation point
    # (ft_inf = 0.50) — well de-inactivated for post-inhibitory rebound
    # bursting.
    #
    # ICaT factory: ``make_trn_icat_channel`` (issue #295) replaces the
    # default cosh-shaped Destexhe (1994) tau with a sigmoid-shaped
    # inactivation tau — small (20 ms) at hyperpolarised V and large
    # (200 ms) at LTS-plateau V (sustains the plateau long enough for
    # the 5–15 Na⁺ spike, 200–600 Hz HP92 rebound burst).  ``ft_inf(V)``
    # is unchanged from Destexhe (1994).
    return Neuron(
        v_rest=-80.0,
        g_Na=50.0,
        g_K=24.0,
        g_NaL=0.0066,
        g_KL=0.0634,
        T_ref=309.15,
        na_channel_factory=make_trn_na_channel,
        k_channel_factory=make_trn_k_channel,
        additional_channels=(
            make_trn_icat_channel(g_max=2.85),
            make_ikca_channel(g_max=0.3),
            make_ih_channel(g_max=0.020),
        ),
        # alpha_ca/tau_ca held at the pre-#308 values: the retuned (g_Na=50,
        # g_K=24) cell genuinely fires at much higher rate than the previous
        # HH52-defaulted preset under any depolarising stimulus, so peak ca_i
        # under REPETITIVE_FIRING (3 µA/cm², 200 ms; ~70 spikes) reaches
        # ~9–10 µM — physiologically consistent with TRN somatic Ca during
        # high-frequency burst trains (cf. Cueni et al. 2008, Nat. Neurosci.
        # 11:683 on TRN dendritic [Ca²⁺]ᵢ during LTS).  Under
        # HYPERPOLARIZATION_STEPS (LTS rebound burst) peak ca_i transiently
        # rises into the 8–18 µM range, also expected for LTS-driven bursts.
        # ``test_calcium_calibration.py`` uses a TRN-specific 12 µM upper
        # bound to allow this realistic firing without flagging
        # CalciumDynamics drift.  Lower alpha_ca brings the absolute Ca
        # peak down but collapses the burst phenotype (IKCa cannot
        # terminate cleanly), so the elevated Ca is the load-bearing
        # mechanism for IKCa-driven burst termination at the literature
        # g_KCa = 0.3.
        calcium_dynamics=CalciumDynamics(alpha_ca=1.2e-5, tau_ca=20.0, ca_rest=1e-4),
        area_cm2=7e-6,
    )
