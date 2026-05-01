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
    make_inar_channel,
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
    make_pospischil_k_channel,
    make_pospischil_na_channel,
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
        # High g_Na drives rapid depolarization; IKv31 (Kv3.1-type, high
        # threshold, fast deactivation) repolarizes quickly and enables
        # non-adapting high-frequency firing.  The HH delayed-rectifier is
        # retained at elevated conductance to match the overall fast spike.
        # Refs: Erisir et al. (1999), J. Neurophysiol. 82:2476;
        #       Wang & Buzsáki (1996), J. Neurosci. 16:6402
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
        # g_NaL + g_KL = 1.5 mS/cm² gives τ_m ≈ 0.67 ms — highly leaky membrane
        # that narrows the synaptic integration window, a hallmark of FS cells.
        # Values tuned so that I_NaL + I_KL + I_channels = 0 at v_rest = −65 mV
        # with K_out=4 mM (E_K ≈ −95 mV) and Pospischil channel steady-state
        # currents; g_total is unchanged (preserving τ_m = 0.67 ms).
        g_Na=150.0,
        g_K=50.0,
        g_NaL=0.3115,
        g_KL=1.1885,
        T_ref=307.15,
        na_channel_factory=make_pospischil_na_channel,
        k_channel_factory=make_pospischil_k_channel,
        channels=(ChannelConfig(make_ikv31_channel, g_max=40.0),),
        area_cm2=3e-6,
    ),
    CORTICAL_PYRAMIDAL: NeuronConfig(
        # area_cm2 = 20e-6 cm² — ~20 µm soma plus dendrites for an RS pyramidal cell.
        # With g_total ≈ 0.05 mS/cm², gives R_n ≈ 1 GΩ in the simulation; in vivo
        # values run 150–400 MΩ once dendritic filtering and additional leaks are
        # included.  Capacitance ≈ 200 pF, matching the textbook range.
        #
        # Pospischil et al. (2008) Traub-Miles Na⁺/K⁺ kinetics (VT = −56.2 mV)
        # replace the default HH52 core channels to match the RS neuron model.
        # Conductances follow Pospischil 2008 Table 2 (cortical RS column):
        # g_Na=56, g_Kd=6, g_M=0.075 mS/cm² — the previous defaults (g_Na=120,
        # g_K=36) carried over from HH52 and produced AP peaks (~+47 mV) and
        # AHPs (~−85 mV) outside the L5 RS literature ranges (McCormick et al.
        # 1985; Connors & Gutnick 1990) because the K⁺ side was ~6× over-spec
        # relative to Pospischil's published values.  Ih produces voltage sag
        # on hyperpolarization; INaP amplifies subthreshold inputs; IM
        # provides spike-frequency adaptation.
        # Ref: Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2.
        #
        # Known limitation: AP half-width remains ~0.5 ms vs the 1.0–2.5 ms
        # literature band.  The Traub-Miles n-gate has a tau ~0.3 ms near
        # spike peak at mammalian temperature, intrinsically capping the
        # achievable half-width.  Resolving this requires a slower
        # delayed-rectifier kinetic family (e.g. Mainen-Sejnowski Kv) and is
        # out of scope for the conductance-rebalance fix tracked in #298.
        #
        # K_out=3.32 produces E_K ≈ −100 mV (Pospischil target).
        #
        # g_NaL + g_KL = 0.05 mS/cm² gives τ_m ≈ 20 ms and R_in ≈ 20 kΩ·cm²,
        # reflecting the high input resistance (200–400 MΩ) of RS cortical
        # pyramidal cells.  Split recomputed via find_zero_current_voltage for
        # v_rest = −70 mV with the new g_K=6: the much weaker delayed-rectifier
        # outward current at rest pushes nearly all leak onto the K⁺ side
        # (g_KL ≈ 0.0497) so g_NaL collapses to ~0.0003 mS/cm².
        #
        # Ih (g_h=0.3) and INaP (g_NaP=0.1) are project-specific additions not
        # present in Pospischil 2008; values tuned in PR #206 so combined
        # inward current at rest does not exceed the outward leak + IM current
        # (the original g_h=1.5, g_NaP=0.5 caused spontaneous tonic firing).
        #
        # T_ref = 307.15 K (34 °C): Pospischil channels were recorded and fitted
        # at 34 °C, so Q10 scaling from that reference to 37 °C (T = 310.15 K)
        # gives a factor of ~1.39× rather than the default ~5.2×.  Using the
        # HH52 reference of 22 °C causes numerical instability in this model.
        g_Na=56.0,
        g_K=6.0,
        v_rest=-70.0,
        K_out=3.32,
        g_NaL=0.000298,
        g_KL=0.049702,
        T_ref=307.15,
        na_channel_factory=make_pospischil_na_channel,
        k_channel_factory=make_pospischil_k_channel,
        channels=(
            ChannelConfig(make_ih_channel, g_max=0.3),
            ChannelConfig(make_inap_channel, g_max=0.1),
            ChannelConfig(make_im_channel, g_max=0.075),
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
        v_rest=-65.0,
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
        # area_cm2 = 7e-6 cm² — ~15 µm soma typical of midbrain SNc DA neurons.
        # Yields C ≈ 7 pF and R_n ≈ 25–50 MΩ in the simulation.
        #
        # Ih drives pacemaker sag and rebound; IM provides slow
        # oscillatory hyperpolarization.
        # Refs: Wilson & Callaway (2000), J. Neurophysiol. 83:3084;
        #       Canavier (1999), J. Comput. Neurosci. 6:49;
        #       Komendantov et al. (2004), J. Neurophysiol. 91:346
        #
        # Canavier (1999) / Komendantov (2004) Traub-Miles Na⁺/K⁺ kinetics
        # (VT = −67 mV) replace the default HH52 core channels.  HH52 kinetics
        # (fitted to room-temperature squid axon) over-accelerate Na⁺ inactivation
        # under the default Q10=3.0 scaling (22→37 °C, factor ~5.2×), biologically
        # wrong for a mammalian midbrain DA cell.  The Canavier/Komendantov kinetics
        # at ~35 °C give VT = −67 mV (equivalent to α_m = 0.32*(V+54)/...), which
        # produces m_inf ≈ 5.6% at rest — the Na⁺ window current that supports
        # Ih-driven post-hyperpolarization rebound spiking in SNc neurons (Wilson &
        # Callaway 2000).  T_ref = 308.15 K (35 °C) limits the Q10 correction to
        # ~1.26× (35→37 °C), preserving the published kinetics.
        #
        # v_rest = −62.5 mV: with VT = −67 mV and g_Ih = 2.0 mS/cm², the HCN
        # channel provides ~6% activation at −60 mV, yielding ~4 µA/cm² inward
        # current that shifts the zero-current equilibrium to −62.5 mV.  This is
        # within the published resting-potential range for SNc DA neurons
        # (Grace & Bunney 1983; Lacey et al. 1989: −60 to −65 mV).
        #
        # g_NaL + g_KL = 0.3 mS/cm² (τ_m ≈ 3.3 ms); split tuned so that
        # I_NaL + I_KL + I_channels = 0 at v_rest = −62.5 mV with K_out=4 mM
        # (E_K ≈ −95 mV) and the Canavier/Komendantov steady-state currents.
        v_rest=-62.5,
        g_NaL=0.0615,
        g_KL=0.2385,
        T_ref=308.15,
        na_channel_factory=make_dopaminergic_na_channel,
        k_channel_factory=make_dopaminergic_k_channel,
        channels=(
            ChannelConfig(make_ih_channel, g_max=2.0),
            ChannelConfig(make_im_channel, g_max=1.0),
        ),
        area_cm2=7e-6,
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
        # Reduced g_Na/g_K vs squid axon; IKa shortens ISI; IM provides
        # spike-frequency adaptation; small Ih produces modest voltage sag;
        # Ca²⁺ channels (L, N, T) and IKCa together produce the pronounced
        # after-hyperpolarization (AHP) characteristic of CA1 cells.
        # Refs: Warman et al. (1994); Migliore et al. (1999), ModelDB #2796
        #
        # Pospischil et al. (2008) Traub-Miles Na⁺/K⁺ kinetics replace the
        # default HH52 core channels.  HH52 kinetics over-accelerate Na⁺
        # inactivation under default Q10=3.0 scaling (factor ~5.2×), causing
        # depolarization block after the first AP.  Pospischil kinetics were
        # characterized at 34 °C, so T_ref=307.15 K reduces the Q10 factor
        # to ~1.4× and the cell sustains repetitive firing as expected.
        #
        # g_NaL + g_KL = 0.05 mS/cm² gives τ_m ≈ 20 ms and R_in ≈ 20 kΩ·cm²,
        # matching the high input resistance measured in CA1 pyramidal cells in
        # slice recordings.  Values tuned for K_out=4 mM (E_K ≈ −95 mV) with
        # Pospischil channel steady-state currents to preserve v_rest = −65 mV;
        # g_total unchanged (τ_m preserved).
        g_Na=35.0,
        g_K=10.0,
        g_NaL=0.0232,
        g_KL=0.0268,
        T_ref=307.15,
        na_channel_factory=make_pospischil_na_channel,
        k_channel_factory=make_pospischil_k_channel,
        channels=(
            ChannelConfig(make_ika_channel, g_max=0.5),
            ChannelConfig(make_im_channel, g_max=0.5),
            ChannelConfig(make_ih_channel, g_max=0.05),
            ChannelConfig(make_ical_channel, g_max=0.5),
            ChannelConfig(make_ican_channel, g_max=0.3),
            ChannelConfig(make_icat_channel, g_max=0.3),
            ChannelConfig(make_ikca_channel, g_max=2.0),
        ),
        # alpha_ca/tau_ca calibrated so peak ca_i ≤ 5 µM under REPETITIVE_FIRING
        # (12 µA/cm², 300 ms).  Three Ca channel types (ICaL + ICaN + ICaT) combined
        # with IKCa require lower alpha_ca than single-channel presets to stay in band.
        calcium_dynamics=CalciumDynamics(alpha_ca=2.1e-5, tau_ca=20.0, ca_rest=1e-4),
        area_cm2=25e-6,
    ),
    STN: NeuronConfig(
        # area_cm2 = 7e-6 cm² — ~15 µm soma typical of subthalamic projection
        # neurons.  C ≈ 7 pF in the simulation.
        #
        # Autonomous tonic pacemaker with conditional burst mode.
        #
        # PRIMARY MODE — tonic pacemaking: high-threshold Na⁺ (Otsuka 2004) and
        # fast K⁺ DR replace the default HH52 kinetics; g_Na/g_K from the
        # original paper sustain autonomous tonic firing at 5–50 Hz in vivo
        # (~80 Hz here under a 2 µA/cm² depolarising bias).
        #
        # CONDITIONAL MODE — burst firing: prominent ICaT (g_T = 5 mS/cm²)
        # supports post-inhibitory rebound bursts when the cell is sufficiently
        # hyperpolarised to de-inactivate the ft gate; on release, ICaT and Ih
        # together drive a high-frequency rebound burst before IKCa repolarises
        # the cell back to tonic mode.  Burst mode can also be triggered by
        # NMDA-receptor activation (Beurrier et al. 1999, J. Neurosci. 19:599);
        # NMDA is not modelled here, so burst mode is reachable in this preset
        # only via the hyperpolarising-step-and-release protocol.
        # Refs: Otsuka et al. (2004), J. Neurophysiol. 92:255;
        #       Bevan & Wilson (1999), J. Neurosci. 19:7617;
        #       Beurrier et al. (1999), J. Neurosci. 19:599 (NMDA burst mode);
        #       Farries & Wilson (2012), J. Neurophysiol.
        #
        # Mammalian Na⁺/K⁺ concentrations give E_Na ≈ +60.6, E_K ≈ −89.1 mV,
        # close to the Otsuka targets (+60, −90).  v_rest = −67 mV is the
        # stable zero-current equilibrium for this channel configuration.
        #
        # g_NaL + g_KL = 0.25 mS/cm² gives τ_m ≈ 4 ms and R_in ≈ 4 kΩ·cm².
        # Lower total leak (< 0.25) shifts the zero-current equilibrium away
        # from v_rest.  With Na_out = 145 mM (mammalian), E_Na ≈ +60.6 mV,
        # and K_out = 5 mM gives E_K ≈ −89 mV.
        # v_rest = −68.02 mV: with dynamic E_Ca, ICaT and ICaL window currents
        # at rest elevate ca_i above ca_rest, shifting E_Ca and moving the
        # coupled equilibrium to −68.02 mV (vs −67 mV static).
        # Re-run find_coupled_equilibrium if channel parameters change.
        g_Na=49.0,
        g_K=57.0,
        v_rest=-68.0152,
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
        # alpha_ca/tau_ca calibrated so peak ca_i ≤ 5 µM under REPETITIVE_FIRING
        # (2 µA/cm², 200 ms).  ICaT g=5.0 mS/cm² is the largest Ca conductance in any
        # preset; low alpha_ca=1.1e-5 compensates for the high Ca influx per spike.
        # ca_init is the coupled (V, ca_i) equilibrium at v_rest: ICaT/ICaL window
        # currents keep ca_i elevated above ca_rest at rest.
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
        # High total leak (g_NaL+g_KL=1.5 mS/cm²) raises the firing threshold; 25 µA/cm²
        # is safely suprathreshold with Pospischil kinetics.  5 ms is the minimum
        # duration that reaches threshold at this amplitude; 5–10 ms gives exactly 1 AP.
        ACTION_POTENTIAL: {
            "min_stimulus": 25.0,
            "max_stimulus": 25.0,
            "stimulus_duration": 5.0,
        },
        # 26 µA/cm² is just above the repetitive-firing threshold; lower amplitudes
        # fire only 1 AP.  Pospischil kinetics sustain non-adapting high-frequency
        # firing without depolarization block.
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
        # With Pospischil 2008 RS conductances (g_K=6) the cell is more
        # excitable than under the previous HH52 defaults; 0.3 µA/cm² peaks
        # near −63 mV (strongly subthreshold) while 0.5 already triggers an AP.
        SUBTHRESHOLD_RESPONSE: {
            "min_stimulus": 0.3,
            "max_stimulus": 0.3,
        },
        # 1 µA/cm² at 15 ms evokes a single AP under the Pospischil RS
        # conductances; the previous 5 µA/cm² stimulus now produces 4 APs.
        ACTION_POTENTIAL: {
            "min_stimulus": 1.0,
            "max_stimulus": 1.0,
            "stimulus_duration": 15.0,
        },
        # 800 ms at 5 µA/cm² is long enough for the (now reduced) IM to
        # accumulate and produce a measurable increase in inter-spike intervals
        # (spike-frequency adaptation); under Pospischil g_M=0.075 the effect
        # is modest (~10% ISI growth) but reliably present.
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
        # Long pacemaking window; 2 µA/cm² drives sustained supra-threshold
        # firing (≥5 APs) over 480 ms with Canavier/Komendantov kinetics.
        # Duration must stay at 480 ms — this override is used as a regression
        # target in test_neuron_protocol_adjustments_change_stimulus_duration.
        REPETITIVE_FIRING: {
            "min_stimulus": 2.0,
            "max_stimulus": 2.0,
            "stimulus_duration": 480.0,
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
        # 6 µA/cm² at 15 ms evokes a single AP with Pospischil kinetics;
        # 5 µA/cm² falls below threshold with the retuned leak conductances.
        ACTION_POTENTIAL: {
            "min_stimulus": 6.0,
            "max_stimulus": 6.0,
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
        # Inherits the base HYPERPOLARIZATION_STEPS range (−10 → −2 µA/cm²).
        # Very high ICaT conductance (g=5.0 mS/cm²) produces a prominent
        # post-inhibitory rebound burst on step release; Ih (g=0.5 mS/cm²)
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
