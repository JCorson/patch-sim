"""Thalamocortical relay neuron preset factory."""

from typing import Any

from patch_sim.calcium import CalciumDynamics
from patch_sim.channels import (
    make_ih_channel,
    make_k_leak_channel,
    make_thalamic_relay_icat_channel,
    make_thalamic_relay_k_channel,
    make_thalamic_relay_na_channel,
)
from patch_sim.constants import (
    ACTION_POTENTIAL,
    FI_CURVE,
    REPETITIVE_FIRING,
    SUBTHRESHOLD_RESPONSE,
)
from patch_sim.neuron import Neuron

PROTOCOL_ADJUSTMENTS: dict[str, dict[str, Any]] = {
    # Burst-mode TC has a very low rheobase (~0.012 µA/cm²) because the
    # slow-inactivating ICaT (issue #287) and reduced g_K (=10) combine to
    # amplify any depolarisation through the LTS.  0.01 µA/cm² stays below
    # threshold while still giving a visible voltage deflection.  The
    # subthreshold margin is narrow: any stimulus ≥ ~0.05 µA/cm² already
    # crosses LTS threshold and fires an AP, so a UI user nudging this
    # value upward will hit threshold within a few clicks.
    SUBTHRESHOLD_RESPONSE: {
        "min_stimulus": 0.01,
        "max_stimulus": 0.01,
    },
    # 8 µA/cm² × 0.5 ms evokes a single AP under the retuned (g_Na=45,
    # g_K=10) preset (#307).  Excitability rose with the lower g_K, so
    # the previous 20 µA/cm² × 2.5 ms now triggers an LTS-coupled
    # rebound spike on top of the primary AP.  Shortening the pulse to
    # 0.5 ms narrows the de-inactivation window for ICaT enough that
    # only a single AP fires.
    ACTION_POTENTIAL: {
        "min_stimulus": 8.0,
        "max_stimulus": 8.0,
        "stimulus_duration": 0.5,
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
}


def make_thalamic_relay() -> Neuron:
    """Thalamocortical relay neuron (McCormick & Huguenard 1992).

    Single-compartment model (~20 µm soma with modest dendrites,
    area_cm2=12e-6 cm²; C ≈ 12 pF, R_n ≈ 80 MΩ).  The preset reproduces
    the post-inhibitory rebound LTS burst (3–7 Na⁺ spikes at 200–500 Hz on
    the LTS plateau) that defines TC burst mode (Sherman & Guillery 1996;
    Llinás & Jahnsen 1982).

    LTS burst mechanism:
        ICaT (``make_thalamic_relay_icat_channel``, g=2.5 mS/cm²) produces a
        low-threshold Ca²⁺ spike plateau when the ft gate is de-inactivated by
        prior hyperpolarisation.  Ih (g=1.0 mS/cm²) activates during
        hyperpolarisation and provides depolarising drive on release, ensuring
        the membrane crosses the LTS threshold after a hyperpolarising step.
        Without Ih, passive relaxation does not overshoot the LTS threshold.

    ICaT factory and slow inactivation:
        ``make_thalamic_relay_icat_channel`` uses a sigmoid-shaped ft
        inactivation time constant (tau_scale=100 ms) rather than the
        cosh-shaped Destexhe (1994) default.  McCormick & Huguenard (1992)
        report tau_h_T ≈ 25–40 ms in the depolarised range; the slower
        inactivation sustains the LTS plateau long enough to support the full
        multi-spike burst.

    Na/K kinetics and temperature:
        McCormick-Huguenard (1992) / Pospischil (2008) Traub-Miles Na⁺/K⁺
        kinetics (VT = −52 mV) replace the HH52 defaults.  MH92 channels were
        recorded at 36 °C; T_ref=309.15 K limits the runtime Q10 correction to
        ~1.12× (36→37 °C), preserving the published kinetics.  g_Na=45 mS/cm²
        and g_K=10 mS/cm² are explicit somatic densities for the
        single-compartment reduction; these land every M&H92 AP-shape metric
        (peak, AHP, half-width, threshold, firing rate) inside its acceptance
        band while preserving the LTS burst phenotype.

    Passive properties:
        g_NaL=0 / g_KL=0.19 mS/cm²: pure K⁺ background leak; τ_m ≈ 5.26 ms
        and R_in ≈ 5.26 kΩ·cm².  The MH92 K⁺ kinetics produce negligible
        window current at rest (n_inf ≈ 0.004), so the leak must be pure-K⁺
        to balance the ICaT and Ih window inward currents.

    Coupled equilibrium:
        v_rest and ca_init are the coupled (V, ca_i) equilibrium at rest:
        the ICaT window current elevates ca_i above ca_rest even at v_rest.
        If g_CaT, g_Ih, or the ICaT factory are ever retuned, re-run
        ``find_coupled_equilibrium`` to recompute v_rest, g_KL, and ca_init.

    References:
        - McCormick & Huguenard (1992), J. Neurophysiol. 68:1384 (TC Na/K/ICaT)
        - Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (TC params)
        - Sherman & Guillery (1996), J. Neurophysiol. 76:1367 (TC burst mode)
        - Llinás & Jahnsen (1982), Nature 297:406 (LTS in TC neurons)

    Returns:
        Fully-configured :class:`~patch_sim.Neuron` — ~20 µm soma
        (area_cm2=12e-6 cm²) with MH92/Pospischil Na/K core, TC-tuned ICaT,
        Ih, and tuned CalciumDynamics.
    """
    return Neuron(
        v_rest=-69.2550,
        T_ref=309.15,
        channels=(
            make_thalamic_relay_na_channel(g_max=45.0),
            make_thalamic_relay_k_channel(g_max=10.0),
            # K-leak only — pure-K⁺ background.
            make_k_leak_channel(g_max=0.19),
            make_thalamic_relay_icat_channel(g_max=2.5),
            make_ih_channel(g_max=1.0),
        ),
        # ca_init is the coupled (V, ca_i) equilibrium at v_rest — ICaT window
        # current at rest elevates ca_i above ca_rest.  Use find_coupled_equilibrium
        # to recompute if CalciumDynamics or channel parameters change.
        calcium_dynamics=CalciumDynamics(
            alpha_ca=2.6e-5, tau_ca=20.0, ca_rest=1e-4, ca_init=6.9107e-4
        ),
        area_cm2=12e-6,
    )
