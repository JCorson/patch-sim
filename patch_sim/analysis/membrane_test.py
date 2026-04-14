"""Dedicated membrane test for passive membrane property characterisation.

Provides :func:`run_membrane_test` which applies a fixed small hyperpolarising
current step to a neuron and extracts its input resistance, membrane time
constant, and membrane capacitance via :mod:`~patch_sim.analysis.passive_properties`.

In patch-clamp physiology a membrane test (or "seal test") is run at the start
of every recording to verify seal quality and obtain baseline passive properties.
This module provides the simulation equivalent: a canonical, repeatable protocol
independent of whatever experiment the user is running.

Constants:
    MEMBRANE_TEST_CURRENT: Fixed step amplitude in µA/cm² (hyperpolarising).
    MEMBRANE_TEST_PRE_MS: Pre-stimulus duration in ms.
    MEMBRANE_TEST_STEP_MS: Step duration in ms.
    MEMBRANE_TEST_POST_MS: Post-stimulus duration in ms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from patch_sim.analysis.passive_properties import (
    PassiveProperties,
    analyze_passive_from_result,
)
from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ, simulate_current_clamp
from patch_sim.protocols.current import step_current

if TYPE_CHECKING:
    from patch_sim.neuron import Neuron

#: Injected current amplitude in µA/cm².  Negative (hyperpolarising) to avoid
#: activating voltage-gated Na⁺ channels.  Small enough to stay well below the
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

    Generates a small hyperpolarising current step, simulates it with
    :func:`~patch_sim.clamp_simulations.simulate_current_clamp`, and extracts
    R_in, τ_m, and C_m via
    :func:`~patch_sim.analysis.passive_properties.analyze_passive_from_result`.

    The protocol parameters (:data:`MEMBRANE_TEST_CURRENT`,
    :data:`MEMBRANE_TEST_PRE_MS`, :data:`MEMBRANE_TEST_STEP_MS`,
    :data:`MEMBRANE_TEST_POST_MS`) are fixed internal constants — this is not a
    user-configurable protocol.

    Args:
        neuron: A fully configured :class:`~patch_sim.neuron.Neuron` instance.

    Returns:
        A :class:`~patch_sim.analysis.passive_properties.PassiveProperties`
        instance when extraction succeeds, or ``None`` when the sweep is
        unexpectedly suprathreshold or the analysis cannot converge.
    """
    total_ms = MEMBRANE_TEST_PRE_MS + MEMBRANE_TEST_STEP_MS + MEMBRANE_TEST_POST_MS
    stimulus = step_current(
        duration=total_ms,
        current_amplitude=MEMBRANE_TEST_CURRENT,
        step_start=MEMBRANE_TEST_PRE_MS,
        step_duration=MEMBRANE_TEST_STEP_MS,
        sampling_frequency=SIM_SAMPLING_FREQ,
    )
    result = simulate_current_clamp(neuron, stimulus)
    return analyze_passive_from_result(
        result,
        current_amplitude=MEMBRANE_TEST_CURRENT,
        stim_start_ms=MEMBRANE_TEST_PRE_MS,
        stim_end_ms=MEMBRANE_TEST_PRE_MS + MEMBRANE_TEST_STEP_MS,
    )
