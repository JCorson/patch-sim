"""Thalamocortical relay neuron preset factory."""

from patch_sim.calcium import CalciumDynamics
from patch_sim.channels import (
    make_ih_channel,
    make_thalamic_relay_icat_channel,
    make_thalamic_relay_k_channel,
    make_thalamic_relay_na_channel,
)
from patch_sim.neuron import Neuron


def make_thalamic_relay() -> Neuron:
    """Thalamocortical relay neuron (McCormick & Huguenard 1992).

    Returns:
        Neuron configured with M&H92 / Pospischil Na/K core, slow-inactivating
        TC-tuned ICaT, and Ih to deliver the LTS-rebound burst phenotype.
    """
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
    # g_Na = 45 mS/cm², g_K = 10 mS/cm² are explicit somatic densities
    # for the single-compartment reduction.  Without these overrides the
    # preset would silently inherit NeuronConfig's HH52 defaults
    # (g_Na = 120, g_K = 36), driving the mean tonic AP peak to ~+49 mV
    # and the AHP to ~−77 mV — both outside the McCormick & Huguenard
    # (1992) thalamic-relay bands (peak +10 to +40 mV; AHP −75 to −55
    # mV).  Pospischil 2008 Table 2 specifies (g_Na = 90, g_K = 7) for
    # the lumped TC single-compartment model, but at those densities the
    # MH92 Traub-Miles kinetics still overshoot peak (~+45 mV) — Pospischil
    # presumably absorbs the residual peak into different leak/area
    # choices.  (45, 10) is the smallest reduction that lands every M&H92
    # AP-shape metric (peak, AHP, half-width, threshold, firing rate)
    # comfortably inside its band while preserving the LTS rebound burst
    # (3–7 spikes, 200–500 Hz).  g_KL is bumped from 0.18 → 0.19 in
    # tandem so that total resting outward conductance compensates for
    # the lower g_K, keeping the cell genuinely quiescent at rest under
    # zero injected current (without the bump, the lowered delayed-
    # rectifier damping lets numerical noise grow into a slow LTS-coupled
    # mini-burst over the 1-second test window).  Same kinetic pattern
    # and fix as the cortical pyramidal (#298), Purkinje (#299), FSI
    # (#301), DA (#304), and STN (#305) presets; closes #307.
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
    # g_NaL = 0, g_KL = 0.19 mS/cm²: purely K⁺ background leak,
    # τ_m ≈ 5.26 ms and R_in ≈ 5.26 kΩ·cm² — both within the physiological
    # bounds in test_preset_passive_properties_in_physiological_range
    # (band [5.0, 9.0]).  Cannot raise g_KL further to 0.20 without
    # putting τ_m on the bound boundary.  The MH92 K⁺ kinetics produce
    # negligible tonic window current at rest (n_inf ≈ 0.004 vs HH52
    # n_inf ≈ 0.32), so the leak must be pure-K⁺ to balance the ICaT
    # and Ih window inward currents at rest.
    #
    # WARNING: v_rest depends on ICaT (g=2.5) and Ih (g=1.0) window
    # currents at rest, not purely on the leak ratio.  With dynamic
    # E_Ca, the ICaT window current at rest elevates ca_i above ca_rest,
    # shifting E_Ca and moving the coupled equilibrium.  If g_CaT, g_Ih,
    # or the ICaT factory are ever retuned, re-run
    # find_coupled_equilibrium to recompute v_rest, g_KL, and ca_init.
    return Neuron(
        g_Na=45.0,
        g_K=10.0,
        v_rest=-69.2550,
        g_NaL=0.0,
        g_KL=0.19,
        T_ref=309.15,
        na_channel_factory=make_thalamic_relay_na_channel,
        k_channel_factory=make_thalamic_relay_k_channel,
        additional_channels=(
            make_thalamic_relay_icat_channel(g_max=2.5),
            make_ih_channel(g_max=1.0),
        ),
        # alpha_ca/tau_ca calibrated so peak ca_i stays in the 0.1–5 µM
        # physiological band under REPETITIVE_FIRING (8 µA/cm², 200 ms).
        # tau_ca=20 ms allows inter-burst clearance.  ca_init is the coupled
        # (V, ca_i) equilibrium at v_rest: ICaT window current at rest keeps
        # ca_i elevated above ca_rest.  Use find_coupled_equilibrium to
        # recompute if CalciumDynamics or channel parameters change.
        calcium_dynamics=CalciumDynamics(
            alpha_ca=2.6e-5, tau_ca=20.0, ca_rest=1e-4, ca_init=6.9107e-4
        ),
        area_cm2=12e-6,
    )
