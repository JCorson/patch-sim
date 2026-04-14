"""Tests for the dedicated membrane test protocol.

Covers the run_membrane_test() function in patch_sim.analysis.membrane_test
and verifies the module-level protocol constants.
"""

from __future__ import annotations

import patch_sim
from patch_sim.analysis.membrane_test import (
    MEMBRANE_TEST_CURRENT,
    MEMBRANE_TEST_POST_MS,
    MEMBRANE_TEST_PRE_MS,
    MEMBRANE_TEST_STEP_MS,
    run_membrane_test,
)
from patch_sim.neuron import Neuron


def test_run_membrane_test_default_neuron(hh_model: Neuron) -> None:
    """Membrane test on the default HH neuron returns passive properties.

    The HH model has active conductances that complicate τ extraction, so this
    test only verifies that the function returns a result and that R_in is
    positive and within a plausible range.  The time constant and derived
    capacitance are checked to be positive without tight bounds because the
    HH step response is not a pure single exponential.
    """
    props = run_membrane_test(hh_model)
    assert props is not None, "Expected PassiveProperties, got None"
    assert props.input_resistance > 0, "R_in must be positive"
    assert props.input_resistance < 5.0, "R_in unrealistically large (> 5 kΩ·cm²)"
    assert props.time_constant > 0, "τ_m must be positive"
    if props.membrane_capacitance is not None:
        assert props.membrane_capacitance > 0, "C_m must be positive"


def test_run_membrane_test_always_subthreshold() -> None:
    """Membrane test stays subthreshold for every standard neuron preset.

    The fixed -0.5 µA/cm² step must not elicit action potentials in any preset
    when the neuron is built with the correct channel kinetics for that preset,
    guaranteeing that the protocol always returns valid passive properties.
    """
    for name, preset_config in patch_sim.NEURON_PRESETS.items():
        neuron = patch_sim.make_neuron(preset_config)
        props = run_membrane_test(neuron)
        assert props is not None, (
            f"Preset '{name}': expected PassiveProperties but got None "
            "(suggests suprathreshold response to the membrane test step)"
        )


def test_membrane_test_sensitive_to_g_l() -> None:
    """Changing g_L changes the measured R_in.

    With g_Na and g_K set to zero the model is a pure RC circuit, so R_in
    equals 1/g_L exactly.  The neurons must be initialised at the leak
    equilibrium potential (E_L) so that there is no drift during the
    pre-stimulus baseline period.  Doubling g_L should halve R_in, confirming
    that the membrane test propagates biophysical changes to the output.
    """
    default_neuron = Neuron()
    # Leak reversal potential (Cl⁻, z=−1) for the default ion concentrations
    e_l = float(
        patch_sim.nernst_potential(
            -1, default_neuron.T, default_neuron.Cl_out, default_neuron.Cl_in
        )
    )
    neuron_low_gl = Neuron(g_Na=0.0, g_K=0.0, g_L=0.15, v_rest=e_l)
    neuron_high_gl = Neuron(g_Na=0.0, g_K=0.0, g_L=0.6, v_rest=e_l)

    props_low = run_membrane_test(neuron_low_gl)
    props_high = run_membrane_test(neuron_high_gl)

    assert props_low is not None
    assert props_high is not None
    # With only leak conductance: R_in = 1/g_L (kΩ·cm²)
    # g_L=0.15 → R_in ≈ 6.67,  g_L=0.6 → R_in ≈ 1.67
    assert props_low.input_resistance > props_high.input_resistance, (
        "Higher g_L should produce lower R_in"
    )


def test_membrane_test_protocol_constants() -> None:
    """Protocol constants are positive and define a sensible stimulus window.

    The step must be long enough (> 2 ms) for the exponential fit to converge,
    and all durations must be strictly positive.
    """
    assert MEMBRANE_TEST_CURRENT < 0, "Step must be hyperpolarising (< 0)"
    assert MEMBRANE_TEST_PRE_MS > 0, "Pre-stimulus duration must be positive"
    assert MEMBRANE_TEST_STEP_MS > 2.0, "Step must be > 2 ms for reliable fitting"
    assert MEMBRANE_TEST_POST_MS > 0, "Post-stimulus duration must be positive"
