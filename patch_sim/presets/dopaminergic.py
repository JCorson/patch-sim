"""Substantia nigra dopaminergic neuron preset factory."""

from patch_sim.calcium import CalciumDynamics
from patch_sim.channels import (
    make_cav13_channel,
    make_dopaminergic_k_channel,
    make_dopaminergic_na_channel,
    make_ih_channel,
    make_sk_channel,
    make_snc_inap_channel,
)
from patch_sim.neuron import Neuron


def make_dopaminergic() -> Neuron:
    """Substantia nigra pars compacta dopaminergic neuron (Putzier+Drion).

    Returns:
        Neuron configured as an autonomous ~7 Hz pacemaker driven by Cav1.3 +
        INaP_SNc + SK, with Komendantov Na/K kinetics.
    """
    # SNc DA neuron — Putzier+Drion minimal pacemaker.
    #
    # Tonic autonomous pacemaker — measured ~7 Hz at zero current,
    # within the broader SNc DA in-vitro band (Wilson & Callaway 2000:
    # ~3 Hz; Liss & Roeper 2008: 1–8 Hz; Grace & Bunney 1984: 1–5 Hz).
    # Drion et al. 2011 reconciled Putzier 2009 (Cav1.3 essential) with
    # Guzman/Surmeier 2009 (INaP essential) by showing the two
    # subthreshold negative-slope conductances are interchangeable —
    # both are needed for a faithful model.  Pacemaking proceeds as:
    #   1. Cav1.3 (V½ = −31.1 mV, k = 5.35 mV; Putzier 2009) and INaP_SNc
    #      (V½ = −65 mV; Drion 2011) carry overlapping subthreshold
    #      depolarising currents; INaP starts the ramp from the AHP,
    #      Cav1.3 takes over near threshold and loads Ca²⁺ into the cell.
    #   2. AP fires; ca_i peaks at ~0.6 µM (α_ca = 5e-5 supplies enough
    #      Ca for SK to dominate the post-spike conductance landscape).
    #   3. SK (K_d = 0.3 µM, Hill n = 4; Drion 2011, scaled to 1.75 mS/cm²
    #      in this preset) opens fast, dragging V to a clean medium AHP
    #      at ~ −90 mV in <5 ms.  Without this strong SK pull the
    #      Komendantov Na window current (m_inf = 0.79 at −30 mV with
    #      VT = −67) holds V on a 20 ms plateau at −30 mV — a known
    #      pathology of single-compartment Na/K when the Cav1.3 + SK
    #      mechanism is undertuned.
    #   4. ca_i decays fast (τ_ca = 30 ms), SK closes, INaP_SNc + Cav1.3
    #      ramp resumes.  Cycle repeats at ~7 Hz.
    #
    # Channel set: only the channels that are characteristic of SNc DA.
    # No IM (cortical M-current, not SNc).  No Mainen-Sejnowski Kv
    # (cortical/Purkinje fit, not SNc) — repolarisation is carried by
    # the Komendantov Kdr alone, which avoids the competing slow K
    # timescales that produce subthreshold oscillation in the depol-block
    # window.  Ih is retained but small: Putzier showed ZD7288 has only
    # a minor effect on rate.
    #
    # Canavier (1999) / Komendantov (2004) Na/K kinetics (VT = −67 mV)
    # replace the default HH52 core channels — HH52 kinetics fitted to
    # squid axon over-accelerate inactivation in mammalian midbrain
    # cells.  Q10 = 1.0 with T_ref = 308.15 K (35 °C) holds them at the
    # published Komendantov reference temperature.
    #
    # Passive properties: area_cm2 = 50e-6 cm² gives C ≈ 50 pF, matching
    # somatic+dendritic capacitance reported for SNc DA neurons (Wolfart
    # et al. 2001: 30–80 pF).  Total leak g_total = 0.040 mS/cm² gives
    # R_in ≈ 500 MΩ and τ_m ≈ 25 ms — both in the literature band
    # (200–600 MΩ; Lacey et al. 1989; Wolfart et al. 2001) (20–30 ms).
    # The g_NaL : g_KL ratio (≈ 0.012 : 0.028) is preserved from the
    # earlier tuning so V_leak ≈ −45 mV is unchanged; only the absolute
    # leak conductance is doubled, with no additional pacemaker channel
    # retune required.
    #
    # Slow Na inactivation: ``make_dopaminergic_na_channel`` carries an
    # always-on ``sNa_da`` gate (Khaliq & Bean 2010; Tucker et al. 2012)
    # and ``make_snc_inap_channel`` carries an always-on ``sNaP_snc``
    # gate (Magistretti & Alonso 1999, V½ shifted to match the Drion
    # 2011 SNc fit).  These two slow gates make the cell biologically
    # more accurate (#330) and would, in principle, provide an escape
    # route from a depol-block plateau.
    #
    # Depol-block: real SNc DA neurons enter depolarisation block above
    # ~100 pA injected current at long sustained drive (Tucker et al.
    # 2012).  Even with the new slow-inactivation gates this single-
    # compartment somatic model does not reproduce block: empirical
    # sweep (scratch/characterize_da_block.py) confirms tonic firing at
    # every amplitude up to 15 µA/cm² and every duration up to 10 s,
    # with the 150 ms rolling-mean V never exceeding −70 mV.  Block
    # onset requires the dendritic Na inactivation pool that this
    # somatic representation lacks; the new gates are a biological
    # accuracy improvement but not a fix for #323, which tracks the
    # missing depol-block onset.
    #
    # v_rest = −55 mV is a kinematic starting point — SNc DA neurons
    # are autonomous oscillators with NO static zero-current rest.  The
    # integrator settles onto the limit cycle within one ISI.
    #
    # References:
    #   Putzier, Kullmann, Roeper (2009), J. Neurosci. 29:15414
    #     — Cav1.3 V½ drives SNc pacemaking (dynamic clamp).
    #   Drion, Massotte, Sepulchre, Seutin (2011), PLOS Comp Biol
    #     7:e1002050 — reconciliation: Cav1.3 + INaP both required.
    #   Guzman, Sanchez-Padilla, Wokosin et al. (2009), J. Neurosci.
    #     29:11011 — INaP essential, Cav1.3 dispensable in SN DA.
    #   Tucker, Huertas, Horn et al. (2012), J. Neurophysiol.
    #     108:288 — depol-block onset ~100 pA.
    #   Wolfart, Neuhoff, Franz, Roeper (2001), J. Neurosci. 21:3443
    #     — SK gates SNc tonic firing.
    #   Komendantov, Komendantova, Johnson et al. (2004),
    #     J. Neurophysiol. 91:346 — Na/K kinetics, AP-shape band.
    #   Grace & Bunney (1984), J. Neurosci. 4:2877 — 1–5 Hz in vitro.
    #   Lacey, Mercuri, North (1989), J. Physiol. 415:55 — −55 to
    #     −65 mV interspike trough.
    return Neuron(
        v_rest=-55.0,
        g_Na=10.0,
        g_K=0.5,
        g_NaL=0.012,
        g_KL=0.028,
        Q10=1.0,
        T_ref=308.15,
        na_channel_factory=make_dopaminergic_na_channel,
        k_channel_factory=make_dopaminergic_k_channel,
        additional_channels=(
            make_cav13_channel(g_max=0.04),
            make_sk_channel(g_max=1.75),
            make_snc_inap_channel(g_max=0.012),
            make_ih_channel(g_max=0.20),
        ),
        calcium_dynamics=CalciumDynamics(
            alpha_ca=5.0e-5, tau_ca=30.0, ca_rest=1e-4, ca_init=1.0e-4
        ),
        area_cm2=50e-6,
    )
