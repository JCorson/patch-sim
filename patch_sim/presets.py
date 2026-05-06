"""Built-in preset configurations for the patch_sim core library.

Contains neuron presets (as :class:`NeuronConfig` instances), protocol
presets (as plain parameter dicts), and neuron-specific protocol
adjustments — all expressed in terms of core library types so they can
be used without importing the UI package.
"""

from typing import Any

import numpy as np

from .additional_channels import (
    make_cav13_channel,
    make_ical_channel,
    make_ican_channel,
    make_icat_channel,
    make_ih_channel,
    make_ika_channel,
    make_ikca_channel,
    make_ikv31_channel,
    make_im_channel,
    make_inap_channel,
    make_inar_channel,
    make_katp_channel,
    make_sk_channel,
    make_snc_inap_channel,
    make_thalamic_relay_icat_channel,
    make_trn_icat_channel,
)
from .calcium import CalciumDynamics
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
    make_dopaminergic_k_channel,
    make_dopaminergic_na_channel,
    make_mainen_sejnowski_kv_channel,
    make_nav11_channel,
    make_nav12_channel,
    make_pospischil_k_channel,
    make_purkinje_k_channel,
    make_purkinje_na_channel,
    make_stn_k_channel,
    make_stn_na_channel,
    make_thalamic_relay_k_channel,
    make_thalamic_relay_na_channel,
    make_trn_k_channel,
    make_trn_na_channel,
)
from .neuron_factory import ChannelConfig, NeuronConfig

# ---------------------------------------------------------------------------
# Neuron presets
# ---------------------------------------------------------------------------

NEURON_PRESETS: dict[str, NeuronConfig] = {
    SQUID_GIANT_AXON: NeuronConfig(
        # area_cm2 left as None: HH52 was characterised on a giant axon segment
        # rather than a whole somatic cell, so absolute MΩ/pF values are not
        # the conventional way to report squid passive properties.  Per-area
        # density units (kΩ·cm², µF/cm²) remain the meaningful display.
        #
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
        # area_cm2 = 3e-6 cm² — compact ~10 µm soma, minimal dendrites.
        # With g_total ≈ 1.5 mS/cm², gives R_n ≈ 220 MΩ and C ≈ 3 pF
        # (matching reported FS interneuron values; Erisir et al. 1999).
        #
        # Active conductances (g_Na=80, g_K=30, g_IKv3.1=20) deliver the
        # fast-spiking phenotype while keeping the AP peak and AHP depth inside
        # the bands reported by Erisir et al. (1999) and Kawaguchi (1995):
        #   AP peak  ~+39 mV  (band +10 to +40 mV)
        #   AHP depth ~−74 mV (band −78 to −60 mV)
        # The earlier g_Na=150, g_K=50, g_IKv3.1=40 setting overshot to peak
        # ~+44 mV and AHP ~−81 mV — both Na and K drive were larger than
        # needed, pulling V toward E_Na on the upstroke and E_K on the
        # undershoot.  Halving each conductance preserves the high firing rate
        # (~237 Hz at 30 µA/cm², well inside the 100–500 Hz FS band) and keeps
        # half-width ~0.30 ms (band 0.25–0.7 ms), while bringing peak and AHP
        # into band (issue #301).
        #
        # IKv3.1 (Kv3.1-type, V₁/₂=−12.4 mV, fast deactivation) drives the
        # late-AP repolarization phase and supports non-adapting firing; the
        # Pospischil delayed rectifier handles the bulk of the early
        # repolarization.  Refs: Erisir et al. (1999), J. Neurophysiol. 82:2476;
        # Kawaguchi (1995), J. Neurosci. 15:2638; Wang & Buzsáki (1996),
        # J. Neurosci. 16:6402.
        #
        # Pospischil et al. (2008) Traub-Miles Na⁺/K⁺ kinetics replace the
        # default HH52 core channels.  HH52 kinetics (fitted to room-temperature
        # squid axon) over-accelerate Na⁺ inactivation under the default
        # Q10=3.0 scaling (22→37 °C, factor ~5.2×), causing depolarization
        # block after the first AP — biologically wrong for a fast-spiking cell.
        # Pospischil kinetics were characterized at mammalian temperature (34 °C),
        # so T_ref=307.15 K limits the Q10 factor to ~1.4× and the cell sustains
        # non-adapting high-frequency firing as expected.
        #
        # Fast Na⁺ uses ``make_nav11_channel`` (Pospischil base + a weak
        # Nav1.1-flavoured slow inactivation gate, V½ = −45 mV, τ_floor =
        # 5000 ms) so the model captures the biological fact that Nav1.1
        # has slow inactivation (Patel et al. 2015) while the kinetics
        # remain too slow to dominate at FSI firing rates.  At v_rest =
        # −65 mV the gate sits at sNa11_inf ≈ 0.92, so g_Na is bumped
        # from the previous 80 → 88 mS/cm² to compensate for the ~8 %
        # rest-availability reduction; AP peak (~+38 mV), AHP (~−74 mV),
        # half-width (~0.30 ms), and ~235 Hz firing at 30 µA/cm² stay
        # within the Erisir / Kawaguchi / Wang-Buzsáki bands.  A higher-
        # fidelity isoform-fitted overhaul (true Nav1.1 activation/fast-
        # inactivation kinetics from Hu & Jonas 2014) is tracked as a
        # follow-up.
        #
        # g_NaL + g_KL = 1.5 mS/cm² gives τ_m ≈ 0.67 ms — highly leaky membrane
        # that narrows the synaptic integration window, a hallmark of FS cells.
        # Values originally tuned so that I_NaL + I_KL + I_channels = 0 at
        # v_rest = −65 mV with K_out=4 mM (E_K ≈ −95 mV) and the previous
        # higher-conductance setting.  At rest the Pospischil m³h ≈ 1.1e-8 and
        # n⁴ ≈ 4.4e-9, and IKv3.1 nk² ≈ 1.3e-4 — small enough that halving
        # each active conductance does not perturb v_rest measurably; the leak
        # split is therefore kept identical, and `test_fs_no_spontaneous_firing`
        # confirms the cell still rests stably below threshold.
        g_Na=88.0,
        g_K=30.0,
        g_NaL=0.3115,
        g_KL=1.1885,
        T_ref=307.15,
        na_channel_factory=make_nav11_channel,
        k_channel_factory=make_pospischil_k_channel,
        channels=(ChannelConfig(make_ikv31_channel, g_max=20.0),),
        area_cm2=3e-6,
    ),
    CORTICAL_PYRAMIDAL: NeuronConfig(
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
        channels=(
            ChannelConfig(make_ih_channel, g_max=0.3),
            ChannelConfig(make_inap_channel, g_max=0.1),
            ChannelConfig(make_im_channel, g_max=0.075),
            ChannelConfig(make_mainen_sejnowski_kv_channel, g_max=1.8),
        ),
        area_cm2=20e-6,
    ),
    PURKINJE: NeuronConfig(
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
        v_rest=-65.0,
        g_Na=30.0,
        g_K=10.0,
        g_NaL=0.0,
        g_KL=0.044,
        T_ref=305.15,
        na_channel_factory=make_purkinje_na_channel,
        k_channel_factory=make_purkinje_k_channel,
        channels=(
            ChannelConfig(make_ical_channel, g_max=1.0),
            ChannelConfig(make_icat_channel, g_max=0.5),
            ChannelConfig(make_ikca_channel, g_max=2.0),
            ChannelConfig(make_inap_channel, g_max=0.1),
            ChannelConfig(make_inar_channel, g_max=0.1),
            ChannelConfig(make_ih_channel, g_max=1.0),
        ),
        # alpha_ca/tau_ca calibrated so peak ca_i ≤ 5 µM under REPETITIVE_FIRING.
        # tau_ca=20 ms (vs default 200 ms) prevents inter-spike accumulation at 10 Hz
        # pacemaking; alpha_ca=1.5e-5 gives a ~2.6 µM peak at 10 µA/cm² for 180 ms.
        calcium_dynamics=CalciumDynamics(alpha_ca=1.5e-5, tau_ca=20.0, ca_rest=1e-4),
        area_cm2=250e-6,
    ),
    DOPAMINERGIC: NeuronConfig(
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
        v_rest=-55.0,
        g_Na=10.0,
        g_K=0.5,
        g_NaL=0.012,
        g_KL=0.028,
        Q10=1.0,
        T_ref=308.15,
        na_channel_factory=make_dopaminergic_na_channel,
        k_channel_factory=make_dopaminergic_k_channel,
        channels=(
            ChannelConfig(make_cav13_channel, g_max=0.04),
            ChannelConfig(make_sk_channel, g_max=1.75),
            ChannelConfig(make_snc_inap_channel, g_max=0.012),
            ChannelConfig(make_ih_channel, g_max=0.20),
        ),
        calcium_dynamics=CalciumDynamics(
            alpha_ca=5.0e-5, tau_ca=30.0, ca_rest=1e-4, ca_init=1.0e-4
        ),
        area_cm2=50e-6,
    ),
    THALAMIC_RELAY: NeuronConfig(
        # area_cm2 = 12e-6 cm² — ~20 µm soma with modest dendrites.
        # C ≈ 12 pF and R_n ≈ 80 MΩ within the simulation, in line with
        # thalamic relay cell measurements.
        #
        # T-type Ca²⁺ produces a low-threshold spike (LTS) plateau; Ih
        # provides the depolarising post-inhibitory rebound that activates
        # the LTS.  Together they generate the multi-spike burst (3–7 Na⁺
        # spikes at 200–500 Hz on the LTS plateau) that defines TC burst
        # mode (Sherman & Guillery 1996; Llinás & Jahnsen 1982; issue #287).
        # Refs: McCormick & Huguenard (1992), J. Neurophysiol. 68:1384;
        #       Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (TC)
        #
        # McCormick-Huguenard (1992) / Pospischil (2008) Traub-Miles Na⁺/K⁺
        # kinetics (VT = −52 mV) replace the default HH52 core channels.
        # HH52 kinetics (fitted to room-temperature squid axon) over-accelerate
        # Na⁺ inactivation under the default Q10=3.0 scaling (22→37 °C, factor
        # ~5.2×), biologically wrong for a mammalian thalamic relay cell.
        # MH92 channels were recorded at 36 °C, so T_ref=309.15 K limits the
        # Q10 correction to ~1.12× (36→37 °C) — a negligible adjustment that
        # preserves the published kinetics.
        #
        # g_K = 18 mS/cm² (vs the HH52 default of 36): the lower
        # delayed-rectifier conductance reduces the deep K⁺-driven AHP that
        # otherwise collapses the LTS plateau after the first Na⁺ spike.
        # Pospischil 2008 Table 2 specifies g_K = 7 mS/cm² for TC, but at
        # that value the cell has multiple zero-current crossings in the
        # default [-100, -20] mV bracket used by find_coupled_equilibrium
        # (the test_find_zero_current_voltage_all_presets coupled-equilibrium
        # test breaks).  g_K = 18 is the smallest value that both supports
        # the multi-spike LTS burst AND keeps the equilibrium bracket valid.
        #
        # g_T = 2.5 mS/cm²: Pospischil 2008 specifies 2.0; bumped slightly
        # to land squarely within the 200–500 Hz / 3–7 spike acceptance band
        # (issue #287).
        #
        # ICaT factory: ``make_thalamic_relay_icat_channel`` — TC-tuned
        # variant with slower inactivation (ft tau_scale=100 ms vs the
        # global Destexhe-1994 default of 20 ms).  M&H 1992 report
        # tau_h_T ≈ 25–40 ms in the depolarised range; the slower
        # inactivation is required to sustain the LTS plateau for the
        # full multi-spike burst.
        #
        # g_NaL = 0, g_KL ≈ 0.18 mS/cm²: purely K⁺ background leak,
        # τ_m ≈ 5.6 ms and R_in ≈ 5.6 kΩ·cm² — both within the physiological
        # bounds in test_preset_passive_properties_in_physiological_range.
        # The MH92 K⁺ kinetics produce negligible tonic window current at rest
        # (n_inf ≈ 0.004 vs HH52 n_inf ≈ 0.32), so the leak must be pure-K⁺
        # to balance the ICaT and Ih window inward currents at rest.
        #
        # WARNING: v_rest depends on ICaT (g=2.5) and Ih (g=1.0) window
        # currents at rest, not purely on the leak ratio.  With dynamic
        # E_Ca, the ICaT window current at rest elevates ca_i above ca_rest,
        # shifting E_Ca and moving the coupled equilibrium.  If g_CaT, g_Ih,
        # or the ICaT factory are ever retuned, re-run
        # find_coupled_equilibrium to recompute v_rest, g_KL, and ca_init.
        g_K=18.0,
        v_rest=-68.8121,
        g_NaL=0.0,
        g_KL=0.18,
        T_ref=309.15,
        na_channel_factory=make_thalamic_relay_na_channel,
        k_channel_factory=make_thalamic_relay_k_channel,
        channels=(
            ChannelConfig(make_thalamic_relay_icat_channel, g_max=2.5),
            ChannelConfig(make_ih_channel, g_max=1.0),
        ),
        # alpha_ca/tau_ca calibrated so peak ca_i stays in the 0.1–5 µM
        # physiological band under REPETITIVE_FIRING (8 µA/cm², 200 ms).
        # tau_ca=20 ms allows inter-burst clearance.  ca_init is the coupled
        # (V, ca_i) equilibrium at v_rest: ICaT window current at rest keeps
        # ca_i elevated above ca_rest.  Use find_coupled_equilibrium to
        # recompute if CalciumDynamics or channel parameters change.
        calcium_dynamics=CalciumDynamics(
            alpha_ca=2.6e-5, tau_ca=20.0, ca_rest=1e-4, ca_init=7.4123e-4
        ),
        area_cm2=12e-6,
    ),
    CA1_PYRAMIDAL: NeuronConfig(
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
        g_Na=60.0,
        g_K=3.0,
        g_NaL=0.020854,
        g_KL=0.029146,
        Q10=1.0,
        T_ref=307.15,
        na_channel_factory=make_nav12_channel,
        k_channel_factory=make_pospischil_k_channel,
        channels=(
            ChannelConfig(make_ika_channel, g_max=0.5),
            ChannelConfig(make_im_channel, g_max=0.75),
            ChannelConfig(make_ih_channel, g_max=0.05),
            ChannelConfig(make_ical_channel, g_max=0.5),
            ChannelConfig(make_ican_channel, g_max=0.3),
            ChannelConfig(make_icat_channel, g_max=0.3),
            ChannelConfig(make_ikca_channel, g_max=2.0),
            ChannelConfig(make_inap_channel, g_max=0.1),
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
    ),
    STN: NeuronConfig(
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
        channels=(
            ChannelConfig(make_icat_channel, g_max=5.0),
            ChannelConfig(make_ical_channel, g_max=0.5),
            ChannelConfig(make_ika_channel, g_max=3.0),
            ChannelConfig(make_ikca_channel, g_max=1.0),
            ChannelConfig(make_ih_channel, g_max=1.0),
            ChannelConfig(make_inap_channel, g_max=0.05),
            ChannelConfig(make_ikv31_channel, g_max=1.0),
            ChannelConfig(make_katp_channel, g_max=0.5),
        ),
        # alpha_ca/tau_ca calibrated so peak ca_i ≤ 5 µM under REPETITIVE_FIRING
        # (2 µA/cm², 200 ms).  ICaT g=5.0 mS/cm² is the largest Ca conductance in any
        # preset; low alpha_ca=1.1e-5 compensates for the high Ca influx per spike.
        calcium_dynamics=CalciumDynamics(
            alpha_ca=1.1e-5, tau_ca=20.0, ca_rest=1e-4, ca_init=7.325e-4
        ),
        area_cm2=7e-6,
    ),
    TRN: NeuronConfig(
        # area_cm2 = 7e-6 cm² — ~15 µm soma characteristic of thalamic
        # reticular neurons.  C ≈ 7 pF in the simulation.
        #
        # Channel set: ICaT + IKCa + Ih over the HP92/Pospischil RE Na⁺/K⁺ core.
        # Conductances:
        #   g_T   = 3.0 mS/cm² (within HP92 voltage-clamp recorded range;
        #                       tuned for the 5–15 spike LTS rebound burst)
        #   g_KCa = 0.3 mS/cm² (Huguenard & Prince 1992, TRN)
        #   g_h   = 0.020 mS/cm² (slightly below the ≈0.025 reported by
        #                          Bal & McCormick 1993 for cat TRN; tuned to
        #                          set tonic firing rate to ~3 Hz)
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
        # PACEMAKER MODE.  With Ih and the elevated g_T, the cell is no longer
        # silent at zero current — it fires tonically at ~3 Hz, consistent
        # with the spontaneous firing observed in TRN slice recordings (HP92,
        # B&M93).  v_rest = −80 mV is the configured initial condition; the
        # cell rapidly leaves this point and settles into tonic firing with
        # mean V around −77 mV.  Excluded from
        # ``test_all_presets_stable_at_rest`` for the same reason as Purkinje
        # (autonomous oscillator with no stable zero-current equilibrium).
        # The HP92 rebound-burst phenotype is exercised by
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
        v_rest=-80.0,
        g_NaL=0.0066,
        g_KL=0.0634,
        T_ref=309.15,
        na_channel_factory=make_trn_na_channel,
        k_channel_factory=make_trn_k_channel,
        channels=(
            ChannelConfig(make_trn_icat_channel, g_max=3.0),
            ChannelConfig(make_ikca_channel, g_max=0.3),
            ChannelConfig(make_ih_channel, g_max=0.020),
        ),
        # alpha_ca/tau_ca calibrated so peak ca_i stays in the 0.1–5 µM
        # physiological band under REPETITIVE_FIRING (3 µA/cm², 200 ms;
        # peak ≈ 3.7 µM at g_T = 3.0).  Under HYPERPOLARIZATION_STEPS
        # (LTS rebound burst) peak ca_i transiently rises into the
        # 8–18 µM range — this is biologically expected for LTS-driven
        # bursts in TRN soma (cf. Cueni et al. 2008, Nature Neurosci.
        # 11:683 — TRN dendritic [Ca²⁺]ᵢ during LTS) and is required for
        # IKCa-driven burst termination at the literature g_KCa = 0.3.
        # Lower alpha_ca brings the rebound Ca into [0.1, 5] µM but
        # collapses the burst phenotype (IKCa cannot terminate cleanly),
        # so we accept the LTS-specific transient as a feature.
        calcium_dynamics=CalciumDynamics(alpha_ca=1.2e-5, tau_ca=20.0, ca_rest=1e-4),
        area_cm2=7e-6,
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
        # slow-inactivating ICaT (issue #287) and reduced g_K (=18) combine to
        # amplify any depolarisation through the LTS.  0.01 µA/cm² stays below
        # threshold while still giving a visible voltage deflection.  The
        # subthreshold margin is narrow: any stimulus ≥ ~0.05 µA/cm² already
        # crosses LTS threshold and fires an AP, so a UI user nudging this
        # value upward will hit threshold within a few clicks.
        SUBTHRESHOLD_RESPONSE: {
            "min_stimulus": 0.01,
            "max_stimulus": 0.01,
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
        # 5 µA/cm² at 2 ms evokes a single AP with HP92 kinetics.
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
