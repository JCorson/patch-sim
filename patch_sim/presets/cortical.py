"""Cortical RS pyramidal cell preset factory."""

from patch_sim.channels import (
    make_ih_channel,
    make_im_channel,
    make_inap_channel,
    make_mainen_sejnowski_kv_channel,
    make_nav12_channel,
    make_pospischil_k_channel,
)
from patch_sim.neuron import Neuron


def make_cortical_pyramidal() -> Neuron:
    """Cortical RS pyramidal cell (Pospischil 2008, Mainen-Sejnowski 1996).

    Returns:
        Neuron configured with Nav1.2 + Pospischil-K core, slow Na inactivation
        for depol-block recovery, and Mainen-Sejnowski Kv as the dominant K.
    """
    # area_cm2 = 20e-6 cm² — ~20 µm soma plus dendrites for an RS pyramidal cell.
    # With g_total ≈ 0.05 mS/cm², gives R_n ≈ 1 GΩ in the simulation; in vivo
    # values run 150–400 MΩ once dendritic filtering and additional leaks are
    # included.  Capacitance ≈ 200 pF, matching the textbook range.
    #
    # Pospischil et al. (2008) Traub-Miles Na⁺ kinetics (VT = −56.2 mV)
    # provide the fast spike upstroke and inactivation; Mainen & Sejnowski
    # (1996) Kv kinetics provide the delayed rectifier (no Pospischil K).
    # This pairing is needed for the L5 RS half-width band of 1.0–2.5 ms
    # reported by McCormick et al. (1985): Pospischil's Traub-Miles n^4
    # form has τ_n ≈ 0.3 ms near AP peak and caps half-width at ~0.5 ms,
    # whereas M-S Kv is essentially closed at rest (n_inf 50% point at
    # V ≈ +4 mV) and deactivates on a τ ~1–2 ms scale, broadening the
    # spike into the literature band (issue #311).
    # Refs: Pospischil et al. (2008), Biol. Cybern. 99:427 (Na);
    #       Mainen & Sejnowski (1996), Nature 382:363 (Kv).
    #
    # Q10 = 1.0 (no temperature scaling): Pospischil 2008 themselves
    # publish their kinetics at 34–36 °C and apply no Q10 correction; the
    # Mainen-Sejnowski Kv constants are pre-scaled from 23 → 34 °C inside
    # the rate functions (Q10=2.3 baked in, factor ~2.55).  Together this
    # represents kinetics at slice recording temperature (32–37 °C in
    # McCormick 1985) with no further runtime adjustment, giving the
    # broadest AP that still fits the peak and AHP bands.  The default
    # Q10=3.0 applied to the same kinetics produced too-fast K opening
    # and pushed half-width back below 1 ms.
    #
    # g_Na = 35 mS/cm² (vs Pospischil's published 56 for cortical RS).
    # The lower g_Na keeps the AP peak inside the +20 to +45 mV band
    # while the slower M-S Kv repolarisation widens the spike — at
    # g_Na=56 the peak hits +47 mV (just above band) and at g_Na=20 it
    # drops to +38 mV (in band but barely).  g_Na=35 places threshold
    # exactly at the -40 mV upper boundary while keeping peak ≈ +45 mV
    # at the upper boundary — the only point where all four AP-shape
    # bands (half-width, peak, threshold, AHP) are simultaneously
    # satisfied.
    #
    # k_channel_factory remains pinned to make_pospischil_k_channel for
    # symmetry with the rest of the Pospischil-Na family (FSI, CA1) but
    # is configured to ``g_K=0`` so it contributes no current.  All
    # active K conductance comes from M-S Kv (g_max=1.8 mS/cm²) wired
    # via the channels list.  Reintroducing any Pospischil K (even
    # g_K=1) drives fast early repolarisation that narrows the spike
    # below 1 ms — the slow vs fast K balance is what actually controls
    # half-width here, not the absolute K conductance.
    #
    # Trade-off: with the slow M-S Kv as the sole K channel,
    # ``find_zero_current_voltage`` with the default bracket [-100, -20]
    # cannot find a unique zero because the cell has additional zero
    # crossings between -65 and -50 mV (M-S Kv contributes very little
    # outward current at -20 mV, so the standard bracket is no longer
    # bounded by opposite-sign currents).  The cell rests stably at
    # -70 mV in simulation, but the static-gating equilibrium analysis
    # via the [-100, -20] bracket fails — see the
    # ``test_find_zero_current_voltage_all_presets`` exclusion.
    #
    # K_out=3.32 produces E_K ≈ −100 mV (Pospischil target).
    #
    # g_NaL + g_KL = 0.05 mS/cm² gives τ_m ≈ 20 ms and R_in ≈ 20 kΩ·cm²,
    # reflecting the high input resistance (200–400 MΩ) of RS cortical
    # pyramidal cells.  Split recomputed via brentq on the steady-state
    # current at -70 mV with the new active conductances: M-S Kv is
    # essentially closed at rest, so the leak split is almost pure
    # K⁺ leak (g_KL ≈ 0.0496, g_NaL ≈ 0.0004).
    #
    # Ih (g_h=0.3) and INaP (g_NaP=0.1) are project-specific additions
    # not present in Mainen-Sejnowski 1996; values tuned in PR #206 so
    # combined inward current at rest does not exceed the outward leak
    # + IM current (the original g_h=1.5, g_NaP=0.5 caused spontaneous
    # tonic firing).
    #
    # T_ref = 307.15 K (34 °C): the M-S Kv prescale and Pospischil Na
    # reference both target this temperature.
    #
    # FIX — depolarization-block recovery (#327, mirror of STN #324).
    # Two complementary slow inactivation gates cooperate so the cell
    # repolarises after a sustained suprathreshold step (e.g. +12
    # µA/cm² × 200 ms) instead of hanging on a depol-block plateau:
    #   1. INaP slow inactivation (sNaP, Magistretti & Alonso 1999)
    #      baked into ``make_inap_channel`` — removes the persistent
    #      Na⁺ window current that otherwise sustains the plateau.
    #   2. Fast-Na slow inactivation (sNa12, Fleidervish & Gutnick
    #      1996; Mickus et al. 1999) baked into
    #      ``make_nav12_channel`` — closes the residual fast-Na
    #      h-tail at the depolarised plateau that single-gate INaP
    #      slow inactivation could not reach.  Fleidervish & Gutnick
    #      1996 directly studied cortical pyramidal cells, so the
    #      same paper that motivated STN slow inactivation in #324
    #      applies a fortiori here.
    #
    # K_ATP is intentionally NOT included (unlike STN): cortical
    # pyramidal cells are not autonomous pacemakers, so the
    # metabolic-safety K_ATP rescue is not biologically motivated;
    # the two slow-inactivation gates suffice for depol-block
    # recovery.
    #
    # Refs: Fleidervish & Gutnick (1996), J. Physiol. 493:83 (slow Na
    #         inactivation directly in cortical pyramidal cells —
    #         primary source);
    #       Magistretti & Alonso (1999), J. Gen. Physiol. 114:491 (INaP
    #         slow inactivation);
    #       Mickus, Jung & Spruston (1999), Biophys. J. 76:846 (slow Na
    #         inactivation, CA1 pyramidal — same Na-channel family).
    #
    # g_Na was raised from 35 to 70 mS/cm² when slow Na inactivation
    # was opted in (#327): sNa availability at v_rest=-70 mV is ≈ 0.92
    # and drops further during sustained firing as the slow gate
    # accumulates inactivation across spikes.  This reduced effective
    # Na drive on the AP upstroke and pushed mean threshold above the
    # McCormick et al. 1985 RS band of [-55, -40] mV.  g_Na=70 (above
    # Pospischil's published 56 to compensate for the train-long sNa
    # accumulation) restores threshold into band while the slow gate
    # itself keeps peak voltage inside the +20 to +45 mV band — at
    # g_Na=90 the cell over-loads and enters depol block during the
    # sustained step.
    return Neuron(
        g_Na=70.0,
        g_K=0.0,
        v_rest=-70.0,
        K_out=3.32,
        g_NaL=0.000391,
        g_KL=0.049609,
        Q10=1.0,
        T_ref=307.15,
        na_channel_factory=make_nav12_channel,
        k_channel_factory=make_pospischil_k_channel,
        additional_channels=(
            make_ih_channel(g_max=0.3),
            make_inap_channel(g_max=0.1),
            make_im_channel(g_max=0.075),
            make_mainen_sejnowski_kv_channel(g_max=1.8),
        ),
        area_cm2=20e-6,
    )
