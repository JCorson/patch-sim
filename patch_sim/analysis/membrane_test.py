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

This module simulates the same procedure: only ungated (passive) channels are
retained on a copy of the neuron, and the resting potential is set to the leak
equilibrium (E_L) so the step response is a clean single exponential. The
resulting R_in, τ_m, and C_m match the set membrane parameters rather than the
apparent (gating-contaminated) values from the full model.

Constants:
    MEMBRANE_TEST_CURRENT: Fixed step amplitude in µA/cm² (hyperpolarising).
    MEMBRANE_TEST_PRE_MS: Pre-stimulus duration in ms.
    MEMBRANE_TEST_STEP_MS: Step duration in ms.
    MEMBRANE_TEST_POST_MS: Post-stimulus duration in ms.
"""

from __future__ import annotations

import dataclasses

from patch_sim.analysis.passive_properties import (
    PassiveProperties,
    analyze_passive_properties,
)
from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ, simulate_current_clamp
from patch_sim.neuron import Neuron
from patch_sim.protocols.current import step_current

#: Injected current amplitude in µA/cm².  Negative (hyperpolarising) to avoid
#: activating any residual conductances.  Small enough to stay well below the
#: rheobase of any standard preset (minimum rheobase ≈ 0.27 µA/cm²).
MEMBRANE_TEST_CURRENT: float = -0.5

#: Duration of the pre-stimulus baseline period in ms.
MEMBRANE_TEST_PRE_MS: float = 10.0

#: Duration of the current step in ms.  Long enough to reach steady state for
#: any physiologically plausible τ_m (< 55 ms) while keeping run time short.
#: Must span ≥5×τ_m for the slowest preset (Purkinje, τ_m ≈ 50 ms).
MEMBRANE_TEST_STEP_MS: float = 250.0

#: Duration of the post-stimulus recovery period in ms.
MEMBRANE_TEST_POST_MS: float = 10.0


def run_membrane_test(neuron: Neuron) -> PassiveProperties | None:
    """Run a dedicated membrane test on the given neuron and extract passive properties.

    Constructs a passive-only copy of the neuron — only ungated channels are
    retained, and ``v_rest`` is set to the leak equilibrium (E_L) — to simulate
    the equivalent of pharmacological channel block (TTX + TEA).  This ensures
    the step response is a clean single exponential driven by the passive RC
    circuit, giving an accurate R_in, τ_m, and C_m rather than values
    contaminated by K⁺ channel deactivation sag.

    Passive channels are identified by ``ch.gating_variables == ()``; their
    ``g_max`` (along with the neuron's ``C_m``, ion concentrations, and
    temperature) is preserved so R_in = 1 / Σ g_max and C_m match the
    configured membrane parameters.

    When ``neuron.area_cm2`` is set, the returned :class:`PassiveProperties`
    carries absolute MΩ / pF counterparts alongside the per-area density
    values.  When it is ``None``, only density values are populated.

    Args:
        neuron: A fully configured :class:`~patch_sim.neuron.Neuron` instance.
            Only its ungated channels survive the passive copy; gated channels
            (Na, K, auxiliary) are blocked.  ``neuron.area_cm2``, if set, is
            forwarded to :func:`analyze_passive_properties` to populate
            absolute units.

    Returns:
        A :class:`~patch_sim.analysis.passive_properties.PassiveProperties`
        instance when extraction succeeds, or ``None`` when the analysis cannot
        converge (should not occur for a pure RC circuit).
    """
    # Effective leak equilibrium: weighted average of each passive channel's
    # reversal potential.  This is the zero-current voltage of the passive
    # circuit: Σ g_max_i * (V - E_i) = 0  →  V = Σ g_max_i * E_i / Σ g_max_i.
    passive_channels = tuple(ch for ch in neuron.channels if not ch.gating_variables)
    g_total = sum(ch.g_max for ch in passive_channels)
    if g_total > 0:
        e_l = (
            sum(ch.g_max * ch.reversal_potential(neuron) for ch in passive_channels)
            / g_total
        )
    else:
        e_l = neuron.v_rest

    # Passive-only neuron: only ungated channels survive.  Equivalent to
    # pharmacological channel block in a real experiment — guarantees a pure RC
    # circuit regardless of which gated channels the original neuron carried.
    passive_neuron = dataclasses.replace(neuron, channels=passive_channels, v_rest=e_l)

    total_ms = MEMBRANE_TEST_PRE_MS + MEMBRANE_TEST_STEP_MS + MEMBRANE_TEST_POST_MS
    stimulus = step_current(
        duration=total_ms,
        current_amplitude=MEMBRANE_TEST_CURRENT,
        step_start=MEMBRANE_TEST_PRE_MS,
        step_duration=MEMBRANE_TEST_STEP_MS,
        sampling_frequency=SIM_SAMPLING_FREQ,
    )
    result = simulate_current_clamp(passive_neuron, stimulus)
    # Use an extended fit window (150 ms) because the passive-only simulation
    # has no gating-variable relaxations to distort the fit, and Purkinje
    # cells have τ_m ≈ 50 ms requiring ≥3×τ_m in the fit window.
    return analyze_passive_properties(
        result["time"],
        result["voltage"],
        current_amplitude=MEMBRANE_TEST_CURRENT,
        stim_start_ms=MEMBRANE_TEST_PRE_MS,
        stim_end_ms=MEMBRANE_TEST_PRE_MS + MEMBRANE_TEST_STEP_MS,
        max_fit_window_ms=150.0,
        area_cm2=neuron.area_cm2,
    )
