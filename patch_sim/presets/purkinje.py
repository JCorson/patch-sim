"""Cerebellar Purkinje cell preset factory."""

from patch_sim.calcium import CalciumDynamics
from patch_sim.channels import (
    make_ical_channel,
    make_icat_channel,
    make_ih_channel,
    make_ikca_channel,
    make_inap_channel,
    make_inar_channel,
    make_purkinje_k_channel,
    make_purkinje_na_channel,
)
from patch_sim.neuron import Neuron


def make_purkinje() -> Neuron:
    """Cerebellar Purkinje cell (De Schutter & Bower 1994, Carter & Bean 2009).

    Returns:
        Neuron configured as an autonomous tonic pacemaker with INaP/Ih
        rhythmic drive and Carter-Bean slow Na inactivation for depol-block
        recovery.
    """
    # area_cm2 = 250e-6 cm² — ~25 µm soma with the massive dendritic tree
    # characteristic of cerebellar Purkinje cells.  C ≈ 250 pF aligns with
    # somatic recordings (Roth & Häusser 2001).
    #
    # Tonic pacemaker (autonomous, intrinsic): INaP window current destabilises
    # the rest, Ih drives recovery from AHP back to threshold → autonomous
    # oscillation producing regular tonic simple spikes.  Complex-spike
    # "bursts" observed in vivo are climbing-fibre synaptic events (Raman &
    # Bean 1997 — INaR-supported repriming under climbing-fibre EPSPs); they
    # are NOT intrinsic and cannot be produced by this single-compartment,
    # current-clamp preset, which has no climbing-fibre input.
    # Refs: De Schutter & Bower (1994), J. Neurophysiol. 71:375;
    #       Raman & Bean (1997), Neuron 19:881 (NaR);
    #       Raman & Bean (1999), J. Neurosci. 19:4663 (NaP/NaR pacemaking);
    #       Häusser & Clark (1997), J. Neurosci. 17:2358 (−55 to −65 mV range)
    #
    # De Schutter & Bower (1994) Traub-Miles Na⁺/K⁺ kinetics (VT = −58 mV)
    # replace the default HH52 core channels.  HH52 kinetics (fitted to
    # room-temperature squid axon) over-accelerate Na⁺ inactivation under
    # the default Q10=3.0 scaling (22→37 °C, factor ~5.2×), biologically
    # wrong for a mammalian cerebellar Purkinje cell.  DSB94 channels were
    # recorded at 32 °C, so T_ref=305.15 K limits the Q10 correction to
    # ~1.73× (32→37 °C), preserving the published kinetics.
    #
    # Pacemaking mechanism:
    #   1. INaP (Magistretti & Alonso 1999, half = −52.6 mV) provides a
    #      persistent inward Na⁺ window current that destabilises any rest
    #      near −65 mV — the cell fires spontaneously without external input.
    #   2. After each AP the cell undergoes a deep AHP near E_K (≈ −95 mV).
    #   3. Ih (Destexhe et al. 1993, half ≈ −82 mV, g = 1.0 mS/cm²) activates
    #      during the AHP and drives a slow depolarisation back to threshold,
    #      producing rhythmic spontaneous firing at ~10–20 Hz in this
    #      single-compartment model (in vivo rate is 50–100 Hz with dendritic
    #      inputs and network drive).
    #
    # v_rest = −65.0 mV is the simulation starting point (near the INaP
    # activation threshold); this cell has NO stable zero-current resting
    # potential — it is an autonomous oscillator.
    #
    # g_NaL = 0 / g_KL = 0.044 mS/cm²: pure K⁺ background leak.
    # τ_m = C_m / g_KL ≈ 22.7 ms and R_in ≈ 22.7 kΩ·cm², consistent with
    # somatic recordings in cerebellar slice (Roth & Häusser 2001).
    #
    # g_Na = 30 mS/cm² and g_K = 10 mS/cm² are explicit somatic densities
    # for the single-compartment reduction.  g_Na sits at the lower end of
    # the somatic NaF range reported by Khaliq et al. (2003) (~30–50
    # mS/cm²); g_K bundles delayed-rectifier and other K⁺ currents that
    # exist as separate compartments in the original DSB94 multi-
    # compartment model (somatic g_Kdr = 4.5 alone).  These values place
    # the AP peak in the +10 to +40 mV band of Häusser & Clark (1997) and
    # the AHP near −68 mV, inside the −55 to −72 mV band of Raman & Bean
    # (1999).
    #
    # FIX — depolarization-block recovery (#329, mirror of STN #324 /
    # cortical pyramidal #327 / CA1 pyramidal #328).  Two complementary
    # slow inactivation gates cooperate so the cell repolarises after a
    # sustained suprathreshold step (e.g. +10 µA/cm² × 200 ms) instead
    # of hanging on a depol-block plateau:
    #   1. INaP slow inactivation (sNaP, Magistretti & Alonso 1999)
    #      baked into ``make_inap_channel`` — removes the persistent
    #      Na⁺ window current that otherwise sustains the plateau.
    #   2. Fast-Na slow inactivation (sNa, Carter & Bean 2009) baked
    #      into ``make_purkinje_na_channel`` — closes the residual
    #      fast-Na h-tail at the depolarised plateau that single-gate
    #      INaP slow inactivation could not reach.  Carter & Bean 2009
    #      directly demonstrated cumulative slow inactivation in
    #      cerebellar Purkinje somatic Na⁺ channels, so this is the
    #      primary cellular reference for Purkinje (companion to
    #      Do & Bean 2003 for STN).
    #
    # K_ATP is intentionally NOT included (unlike STN): although
    # Purkinje cells are intrinsic pacemakers, Carter & Bean's own
    # demonstration of depol-block recovery uses the two slow-
    # inactivation gates alone, so the metabolic-safety K_ATP rescue
    # added for STN is not biologically motivated here.
    #
    # No conductance retuning was needed when the two slow gates were
    # opted in: at g_Na=30, g_K=10, g_NaP=0.1 (the values calibrated
    # in #299 / #314) the spontaneous AP shape and 10–50 Hz pacemaker
    # rate stay in band (rate ≈ 30 Hz, peak ≈ +30 mV, half-width ≈
    # 0.39 ms, threshold ≈ −42 mV, AHP ≈ −68 mV — all inside the
    # Häusser & Clark / Raman & Bean tolerances).  The Purkinje preset
    # relies primarily on Ih for AHP recovery and on the moderate
    # g_Na density to set AP threshold, so the reduction in Na⁺
    # availability from the always-on sNa gate (s_inf ≈ 0.87 at
    # v_rest=-65 mV) is offset by sNaP curtailing the persistent Na⁺
    # ramp; the two gates together stabilise tonic firing.
    #
    # Refs: Carter & Bean (2009), Neuron 64:898 (slow Na inactivation
    #         directly in cerebellar Purkinje cells — primary source);
    #       Magistretti & Alonso (1999), J. Gen. Physiol. 114:491 (INaP
    #         slow inactivation);
    #       Do & Bean (2003), Neuron 39:109 (Na pacemaker channels in
    #         STN — companion paper).
    return Neuron(
        v_rest=-65.0,
        g_Na=30.0,
        g_K=10.0,
        g_NaL=0.0,
        g_KL=0.044,
        T_ref=305.15,
        na_channel_factory=make_purkinje_na_channel,
        k_channel_factory=make_purkinje_k_channel,
        additional_channels=(
            make_ical_channel(g_max=1.0),
            make_icat_channel(g_max=0.5),
            make_ikca_channel(g_max=2.0),
            make_inap_channel(g_max=0.1),
            make_inar_channel(g_max=0.1),
            make_ih_channel(g_max=1.0),
        ),
        # alpha_ca/tau_ca calibrated so peak ca_i ≤ 5 µM under REPETITIVE_FIRING.
        # tau_ca=20 ms (vs default 200 ms) prevents inter-spike accumulation at 10 Hz
        # pacemaking; alpha_ca=1.5e-5 gives a ~2.6 µM peak at 10 µA/cm² for 180 ms.
        calcium_dynamics=CalciumDynamics(alpha_ca=1.5e-5, tau_ca=20.0, ca_rest=1e-4),
        area_cm2=250e-6,
    )
