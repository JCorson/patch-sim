"""CA1 hippocampal pyramidal cell preset factory."""

from patch_sim.calcium import CalciumDynamics
from patch_sim.channels import (
    make_ical_channel,
    make_ican_channel,
    make_icat_channel,
    make_ih_channel,
    make_ika_channel,
    make_ikca_channel,
    make_im_channel,
    make_inap_channel,
    make_nav12_channel,
    make_pospischil_k_channel,
)
from patch_sim.neuron import Neuron


def make_ca1_pyramidal() -> Neuron:
    """CA1 hippocampal pyramidal cell (Pospischil, Migliore, Storm).

    Returns:
        Neuron configured with Nav1.2 + Pospischil-K core, IKa/IM/Ih, three
        Ca²⁺ channels feeding IKCa-driven SFA, and slow Na inactivation for
        depol-block recovery.
    """
    # area_cm2 = 25e-6 cm² — ~20 µm soma with the extensive 1.5 mm dendritic
    # arbor of CA1 pyramidal cells.  C ≈ 25 pF in the simulation; in vivo
    # whole-cell capacitance runs higher once dendrites are fully sampled.
    #
    # IKa shortens ISI; IM accumulates over the train to drive
    # spike-frequency adaptation (Storm 1990; Madison & Nicoll 1984); small
    # Ih produces modest voltage sag; Ca²⁺ channels (L, N, T) feed [Ca²⁺]ᵢ,
    # which slowly activates IKCa over the train and contributes to SFA;
    # INaP eliminates the ~500 ms post-AP latency that otherwise stalls
    # repetitive firing while IKCa decays.  Slow Na inactivation (sNa,
    # sNaP) added in #328 picks up part of the SFA load that previously
    # rested entirely on IM and IKCa, so g_M was trimmed 1.0→0.75 to
    # avoid double-counting adaptation; g_Na was raised 38→60 to
    # compensate for the train-long sNa accumulation that would
    # otherwise stall the train.
    # Refs: Warman et al. (1994); Migliore et al. (1999), ModelDB #2796;
    #       Storm (1990); Madison & Nicoll (1984)
    #
    # Pospischil et al. (2008) Traub-Miles Na⁺/K⁺ kinetics replace the
    # default HH52 core channels.  Q10=1.0 (matching cortical pyramidal,
    # Pospischil 2008's own no-scaling convention): the published kinetics
    # already represent slice recording temperature, and applying the
    # default 1.4× T_ref factor narrows AP half-width below the 0.7–1.5 ms
    # CA1 RS band (Spruston & Johnston 2008).  The K kinetics are slow
    # enough at Q10=1.0 to land half-width inside the band without
    # switching K families (issue #302, #311 cortical precedent).
    #
    # g_K = 3 mS/cm² (down from 10): under Q10=1.0 the slowed K kinetics
    # produce a wider AP, but at the original g_K=10 the K conductance is
    # large enough that the AP narrows back below 0.7 ms.  Lowering g_K to
    # 3 leaves enough delayed-rectifier current for orderly repolarisation
    # while keeping half-width ≈ 0.76 ms (mid-band) and AHP shallower.
    #
    # INaP (g_NaP = 0.1 mS/cm², Magistretti & Alonso 1999 kinetics): the
    # original CA1 preset (no INaP) showed a ~500 ms latency between the
    # first and second APs because IKCa stayed near saturation while
    # [Ca²⁺]ᵢ slowly decayed (issue #302).  INaP provides a small,
    # persistent inward Na⁺ window current that overcomes the post-AP
    # IKCa hyperpolarisation, restoring prompt second-AP recovery and
    # thereby allowing IM and IKCa to drive proper monotonic
    # spike-frequency adaptation rather than the inverted ramp pattern.
    #
    # IM g_max set to 0.75 mS/cm²: previously 1.0 (Storm 1990 SFA target);
    # trimmed when slow Na inactivation was opted in (#328) to avoid
    # double-counting adaptation now that the sNa and sNaP gates also
    # accumulate inactivation across the train.  IKCa stays at 2.0 —
    # only IM was reduced — keeping the Ca²⁺/IKCa-driven AHP that
    # Storm 1990 and Madison & Nicoll 1984 highlight as the deeper
    # AHP signature of CA1.
    #
    # alpha_ca = 5e-6, tau_ca = 100 ms: original (alpha_ca=2.1e-5,
    # tau_ca=20 ms) saturated IKCa via the Hill function (K_d = 1 µM)
    # within the first AP and pinned the cell hyperpolarised for ~500 ms
    # while [Ca²⁺]ᵢ decayed.  Reducing alpha_ca by ~4× per-AP rise plus
    # extending tau_ca to 100 ms keeps [Ca²⁺]ᵢ well below K_d initially
    # and lets it accumulate gradually over the train, so IKCa progresses
    # smoothly from low to moderate activation as a SFA driver rather than
    # saturating at AP 1.  Note this changes the calibration target: peak
    # [Ca²⁺]ᵢ under sustained 7 µA/cm² firing is now ~1 µM rather than the
    # earlier 5 µM band.
    #
    # g_NaL + g_KL = 0.05 mS/cm² gives τ_m ≈ 20 ms and R_in ≈ 20 kΩ·cm²,
    # matching the high input resistance measured in CA1 pyramidal cells
    # in slice recordings.  Values tuned for K_out=4 mM (E_K ≈ −95 mV)
    # via brentq on the steady-state current at v_rest = −65 mV with the
    # new active conductances (INaP shifts the inward/outward balance).
    #
    # FIX — depolarization-block recovery (#328, mirror of #324/#327).
    # Two complementary slow inactivation gates cooperate so the cell
    # repolarises after a sustained suprathreshold step (e.g. +30
    # µA/cm² × 200 ms — CA1's deeper IKCa AHP makes it less excitable
    # than cortical pyramidal, so the +12 µA/cm² level used in #327 is
    # too weak to engage the gates here) instead of hanging on a
    # depol-block plateau:
    #   1. INaP slow inactivation (sNaP, Magistretti & Alonso 1999)
    #      baked into ``make_inap_channel`` — removes the persistent
    #      Na⁺ window current that otherwise sustains the plateau.
    #   2. Fast-Na slow inactivation (sNa12, Mickus, Jung & Spruston
    #      1999) baked into ``make_nav12_channel`` — closes the
    #      residual fast-Na h-tail at the depolarised plateau that
    #      single-gate INaP slow inactivation could not reach.
    #      Mickus et al. 1999 directly studied CA1 pyramidal cells,
    #      so the slow Na inactivation kinetics motivated for STN in
    #      #324 (which cited the same paper) apply a fortiori here.
    #
    # K_ATP is intentionally NOT included (unlike STN): CA1 pyramidal
    # cells are not autonomous pacemakers, so the metabolic-safety
    # K_ATP rescue is not biologically motivated; the two slow-
    # inactivation gates suffice for depol-block recovery.
    #
    # Refs: Mickus, Jung & Spruston (1999), Biophys. J. 76:846 (slow
    #         Na inactivation directly in CA1 pyramidal cells —
    #         primary source);
    #       Magistretti & Alonso (1999), J. Gen. Physiol. 114:491 (INaP
    #         slow inactivation).
    return Neuron(
        g_Na=60.0,
        g_K=3.0,
        g_NaL=0.020854,
        g_KL=0.029146,
        Q10=1.0,
        T_ref=307.15,
        na_channel_factory=make_nav12_channel,
        k_channel_factory=make_pospischil_k_channel,
        additional_channels=(
            make_ika_channel(g_max=0.5),
            make_im_channel(g_max=0.75),
            make_ih_channel(g_max=0.05),
            make_ical_channel(g_max=0.5),
            make_ican_channel(g_max=0.3),
            make_icat_channel(g_max=0.3),
            make_ikca_channel(g_max=2.0),
            make_inap_channel(g_max=0.1),
        ),
        # alpha_ca/tau_ca rebalanced so [Ca²⁺]ᵢ accumulates gradually over the
        # 500 ms test train rather than saturating IKCa within the first AP
        # (issue #302).  Three Ca channel types (ICaL + ICaN + ICaT) feed the
        # pool; with the smaller alpha_ca the per-AP rise is below K_d=1 µM,
        # and IKCa progressively activates as the cell continues to fire.
        # ca_init is the coupled (V, ca_i) equilibrium at v_rest=−65 mV: ICaT
        # window current at rest keeps ca_i elevated above ca_rest, so the
        # simulation needs to start at the actual coupled rest rather than
        # ca_rest.  Use find_coupled_equilibrium to recompute if any channel
        # parameters change.
        calcium_dynamics=CalciumDynamics(
            alpha_ca=5e-6, tau_ca=100.0, ca_rest=1e-4, ca_init=2.5845e-4
        ),
        area_cm2=25e-6,
    )
