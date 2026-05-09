"""CA1 hippocampal pyramidal cell preset factory."""

from typing import Any

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
    make_k_leak_channel,
    make_na_leak_channel,
    make_nav12_channel,
    make_pospischil_k_channel,
)
from patch_sim.constants import (
    ACTION_POTENTIAL,
    FI_CURVE,
    HYPERPOLARIZATION_STEPS,
    REPETITIVE_FIRING,
    SUBTHRESHOLD_RESPONSE,
)
from patch_sim.neuron import Neuron

PROTOCOL_ADJUSTMENTS: dict[str, dict[str, Any]] = {
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
}


def make_ca1_pyramidal() -> Neuron:
    """CA1 hippocampal pyramidal cell (Pospischil, Migliore, Storm).

    Single-compartment CA1 pyramidal cell (~20 µm soma with 1.5 mm dendritic
    arbor, area_cm2=25e-6 cm²; C ≈ 25 pF).  The preset reproduces
    spike-frequency adaptation (SFA) driven cooperatively by IM, IKCa, and
    slow Na inactivation, with a half-width of ~0.76 ms (band 0.7–1.5 ms;
    Spruston & Johnston 2008).

    Na/K kinetics and temperature:
        Pospischil et al. (2008) Traub-Miles Na⁺/K⁺ kinetics (via
        ``make_nav12_channel`` and ``make_pospischil_k_channel``) replace the
        HH52 defaults.  Q10=1.0 with T_ref=307.15 K: the published kinetics
        already represent slice recording temperature; no further runtime
        scaling is applied.  g_K=3 mS/cm² (low relative to HH52) keeps the
        delayed-rectifier current small enough that the slowed Pospischil K
        kinetics at Q10=1.0 broaden the AP into the 0.7–1.5 ms CA1 half-width
        band without switching K families.

    Spike-frequency adaptation mechanism:
        Three overlapping adaptation mechanisms cooperate:

        1. IM (g=0.75 mS/cm²; Storm 1990; Madison & Nicoll 1984) accumulates
           M-current over the spike train to progressively reduce firing rate.
        2. ICaL + ICaN + ICaT (g=0.5, 0.3, 0.3 mS/cm²) load [Ca²⁺]ᵢ per
           spike; IKCa (g=2.0 mS/cm²) activates gradually over the train
           (IKCa K_d=1 µM, Hill n=2), contributing additional AHP and SFA.
           alpha_ca=5e-6 and tau_ca=100 ms keep peak [Ca²⁺]ᵢ below K_d
           initially so IKCa builds progressively rather than saturating at
           the first AP.
        3. Slow Na inactivation (sNa12 and sNaP; see below) accumulates
           across spikes and reduces effective Na⁺ drive through the train.

        INaP (g=0.1 mS/cm²; Magistretti & Alonso 1999) provides a persistent
        Na⁺ window current that overcomes post-AP IKCa hyperpolarisation,
        ensuring prompt second-AP recovery and maintaining a monotonic SFA
        ramp rather than a long post-burst pause.

    Slow Na inactivation and depolarisation-block recovery:
        Two complementary slow gates ensure the cell repolarises after
        sustained suprathreshold drive rather than hanging on a plateau:

        1. Fast-Na slow inactivation (sNa12, via ``make_nav12_channel``) —
           closes the residual fast-Na h-tail during prolonged depolarisation.
           Mickus, Jung & Spruston (1999) characterised this gate directly
           in CA1 pyramidal cells.  g_Na=60 mS/cm² compensates for the
           train-long sNa12 accumulation.
        2. INaP slow inactivation (sNaP, via ``make_inap_channel``) — removes
           the persistent Na⁺ window current that otherwise sustains the
           plateau (Magistretti & Alonso 1999).

        K_ATP is intentionally excluded: CA1 pyramidal cells are not
        autonomous pacemakers; the two slow gates suffice for depol-block
        recovery.

    Passive properties:
        g_NaL + g_KL = 0.05 mS/cm² gives τ_m ≈ 20 ms and R_in ≈ 20 kΩ·cm²,
        matching the high input resistance of CA1 pyramidal cells in slice
        recordings.  ca_init is the coupled (V, ca_i) equilibrium at
        v_rest=−65 mV (ICaT window current elevates ca_i above ca_rest); use
        ``find_coupled_equilibrium`` to recompute if channel parameters change.

    References:
        - Pospischil et al. (2008), Biol. Cybern. 99:427 (Traub-Miles Na/K)
        - Migliore et al. (1999), J. Neurophysiol. 81:1379 (ModelDB #2796)
        - Storm (1990), J. Physiol. 422:327 (IM-driven SFA in CA1)
        - Madison & Nicoll (1984), J. Physiol. 354:319 (SFA mechanism)
        - Mickus, Jung & Spruston (1999), Biophys. J. 76:846 (slow Na
          inactivation directly in CA1 pyramidal cells — primary source)
        - Magistretti & Alonso (1999), J. Gen. Physiol. 114:491 (INaP slow
          inactivation)
        - Spruston & Johnston (2008), in Dendrites (AP-shape bands)

    Returns:
        Fully-configured :class:`~patch_sim.Neuron` — ~20 µm soma
        (area_cm2=25e-6 cm²) with Nav1.2/Pospischil-K core, slow Na
        inactivation, IKa/IM/Ih/ICaL/ICaN/ICaT/IKCa/INaP auxiliary channels,
        and tuned CalciumDynamics.
    """
    return Neuron(
        Q10=1.0,
        T_ref=307.15,
        channels=(
            make_nav12_channel(g_max=60.0),
            make_pospischil_k_channel(g_max=3.0),
            make_na_leak_channel(g_max=0.020854),
            make_k_leak_channel(g_max=0.029146),
            make_ika_channel(g_max=0.5),
            make_im_channel(g_max=0.75),
            make_ih_channel(g_max=0.05),
            make_ical_channel(g_max=0.5),
            make_ican_channel(g_max=0.3),
            make_icat_channel(g_max=0.3),
            make_ikca_channel(g_max=2.0),
            make_inap_channel(g_max=0.1),
        ),
        # alpha_ca=5e-6 / tau_ca=100 ms: three Ca sources (ICaL + ICaN + ICaT)
        # feed the pool; the small alpha_ca keeps the per-AP rise below IKCa
        # K_d=1 µM so IKCa activates gradually over the train as a SFA driver.
        # ca_init is the coupled (V, ca_i) equilibrium at v_rest=−65 mV.
        # Use find_coupled_equilibrium to recompute if channel parameters change.
        calcium_dynamics=CalciumDynamics(
            alpha_ca=5e-6, tau_ca=100.0, ca_rest=1e-4, ca_init=2.5845e-4
        ),
        area_cm2=25e-6,
    )
