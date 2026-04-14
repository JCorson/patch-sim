"""Dedicated membrane test for passive membrane property characterisation.

Provides :func:`run_membrane_test` which applies a fixed small hyperpolarising
current step to a **passive-only** copy of the neuron and extracts its input
resistance, membrane time constant, and membrane capacitance via
:mod:`~patch_sim.analysis.passive_properties`.

In patch-clamp physiology a membrane test (or "seal test") is run at the start
of every recording to verify seal quality and obtain baseline passive properties.
The standard technique blocks voltage-gated channels (TTX for Na⁺, TEA for K⁺)
so the recorded response reflects only the passive RC membrane rather than
being contaminated by slow gating relaxations.

This module simulates the same procedure: active conductances are zeroed
(g_Na = g_K = 0, no auxiliary channels) and the resting potential is set to
the leak equilibrium (E_L), so the step response is a clean single exponential.
The resulting R_in, τ_m, and C_m match the set membrane parameters rather than
the apparent (gating-contaminated) values from the full model.

Constants:
    MEMBRANE_TEST_CURRENT: Fixed step amplitude in µA/cm² (hyperpolarising).
    MEMBRANE_TEST_PRE_MS: Pre-stimulus duration in ms.
    MEMBRANE_TEST_STEP_MS: Step duration in ms.
    MEMBRANE_TEST_POST_MS: Post-stimulus duration in ms.
"""

from __future__ import annotations

from patch_sim.analysis.passive_properties import (
    PassiveProperties,
    analyze_passive_from_result,
)
from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ, simulate_current_clamp
from patch_sim.electrochemistry import nernst_potential
from patch_sim.neuron import Neuron
from patch_sim.protocols.current import step_current

#: Injected current amplitude in µA/cm².  Negative (hyperpolarising) to avoid
#: activating any residual conductances.  Small enough to stay well below the
#: rheobase of any standard preset (minimum rheobase ≈ 0.27 µA/cm²).
MEMBRANE_TEST_CURRENT: float = -0.5

#: Duration of the pre-stimulus baseline period in ms.
MEMBRANE_TEST_PRE_MS: float = 10.0

#: Duration of the current step in ms.  Long enough to reach steady state for
#: any physiologically plausible τ_m (< 25 ms) while keeping run time short.
MEMBRANE_TEST_STEP_MS: float = 50.0

#: Duration of the post-stimulus recovery period in ms.
MEMBRANE_TEST_POST_MS: float = 10.0


def run_membrane_test(neuron: Neuron) -> PassiveProperties | None:
    """Run a dedicated membrane test on the given neuron and extract passive properties.

    Constructs a passive-only copy of the neuron (g_Na = g_K = 0, no auxiliary
    channels, v_rest = E_L) to simulate the equivalent of pharmacological channel
    block (TTX + TEA).  This ensures the step response is a clean single
    exponential driven by the passive RC circuit, giving an accurate R_in, τ_m,
    and C_m rather than values contaminated by K⁺ channel deactivation sag.

    The passive copy preserves g_L, C_m, ion concentrations, and temperature
    from the original neuron so that R_in = 1/g_L and C_m match the configured
    membrane parameters exactly.

    Args:
        neuron: A fully configured :class:`~patch_sim.neuron.Neuron` instance.
            Its g_L, C_m, ion concentrations, and temperature are used;
            active conductances (g_Na, g_K, auxiliary channels) are blocked.

    Returns:
        A :class:`~patch_sim.analysis.passive_properties.PassiveProperties`
        instance when extraction succeeds, or ``None`` when the analysis cannot
        converge (should not occur for a pure RC circuit).
    """
    # Compute the leak equilibrium potential so the passive neuron starts at
    # rest with no net current, preventing drift during the baseline period.
    e_l = float(
        nernst_potential(
            z=-1,
            T=neuron.T,
            ion_concentration_out=neuron.Cl_out,
            ion_concentration_in=neuron.Cl_in,
        )
    )

    # Passive-only neuron: g_Na = g_K = 0, no auxiliary channels.
    # Equivalent to pharmacological channel block in a real experiment.
    passive_neuron = Neuron(
        g_Na=0.0,
        g_K=0.0,
        g_L=neuron.g_L,
        C_m=neuron.C_m,
        v_rest=e_l,
        Na_out=neuron.Na_out,
        Na_in=neuron.Na_in,
        K_out=neuron.K_out,
        K_in=neuron.K_in,
        Cl_out=neuron.Cl_out,
        Cl_in=neuron.Cl_in,
        Ca_out=neuron.Ca_out,
        Ca_in=neuron.Ca_in,
        T=neuron.T,
    )

    total_ms = MEMBRANE_TEST_PRE_MS + MEMBRANE_TEST_STEP_MS + MEMBRANE_TEST_POST_MS
    stimulus = step_current(
        duration=total_ms,
        current_amplitude=MEMBRANE_TEST_CURRENT,
        step_start=MEMBRANE_TEST_PRE_MS,
        step_duration=MEMBRANE_TEST_STEP_MS,
        sampling_frequency=SIM_SAMPLING_FREQ,
    )
    result = simulate_current_clamp(passive_neuron, stimulus)
    return analyze_passive_from_result(
        result,
        current_amplitude=MEMBRANE_TEST_CURRENT,
        stim_start_ms=MEMBRANE_TEST_PRE_MS,
        stim_end_ms=MEMBRANE_TEST_PRE_MS + MEMBRANE_TEST_STEP_MS,
    )
