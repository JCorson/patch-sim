"""Tests for the dedicated membrane test protocol.

Covers the run_membrane_test() function in patch_sim.analysis.membrane_test
and verifies the module-level protocol constants.
"""

from __future__ import annotations

import dataclasses

import pytest

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
    """Membrane test on the default HH neuron recovers the set passive parameters.

    The membrane test simulates with g_Na = g_K = 0 (channel block) and
    v_rest = E_L_eff, giving a clean single-exponential response.  The measured
    R_in, τ_m, and C_m should closely match the analytically expected values
    for the default HH parameters (g_NaL+g_KL = 0.3 mS/cm², C_m = 1.0 µF/cm²):
      R_in = 1/(g_NaL+g_KL) ≈ 3.33 kΩ·cm²
      τ_m  = C_m/(g_NaL+g_KL) ≈ 3.33 ms
      C_m  ≈ 1.0 µF/cm²
    """
    props = run_membrane_test(hh_model)
    assert props is not None, "Expected PassiveProperties, got None"
    assert props.input_resistance == pytest.approx(1.0 / 0.3, abs=0.1)
    assert props.time_constant == pytest.approx(1.0 / 0.3, abs=0.1)
    assert props.membrane_capacitance is not None
    assert props.membrane_capacitance == pytest.approx(1.0, abs=0.1)


def test_run_membrane_test_always_subthreshold() -> None:
    """Membrane test stays subthreshold for every standard neuron preset.

    The fixed -0.5 µA/cm² step must not elicit action potentials in any preset
    when the neuron is built with the correct channel kinetics for that preset,
    guaranteeing that the protocol always returns valid passive properties.
    """
    for name, factory in patch_sim.NEURON_PRESETS.items():
        neuron = factory()
        props = run_membrane_test(neuron)
        assert props is not None, (
            f"Preset '{name}': expected PassiveProperties but got None "
            "(suggests suprathreshold response to the membrane test step)"
        )


def test_membrane_test_sensitive_to_g_leak() -> None:
    """Changing total leak conductance produces the expected change in R_in.

    The membrane test runs in passive mode (channels blocked internally), so
    R_in = 1/(g_NaL+g_KL) exactly.  This test verifies the relationship holds
    and that R_in matches the analytical expectation.
    """
    from patch_sim.channels import make_k_leak_channel, make_na_leak_channel

    neuron_low_gl = Neuron(
        channels=(
            make_na_leak_channel(g_max=0.027),
            make_k_leak_channel(g_max=0.123),
        )
    )  # total = 0.15
    neuron_high_gl = Neuron(
        channels=(
            make_na_leak_channel(g_max=0.108),
            make_k_leak_channel(g_max=0.492),
        )
    )  # total = 0.6

    props_low = run_membrane_test(neuron_low_gl)
    props_high = run_membrane_test(neuron_high_gl)

    assert props_low is not None
    assert props_high is not None
    # R_in = 1/g_total: 0.15 → 6.67 kΩ·cm², 0.6 → 1.67 kΩ·cm²
    assert props_low.input_resistance == pytest.approx(1.0 / 0.15, abs=0.1)
    assert props_high.input_resistance == pytest.approx(1.0 / 0.6, abs=0.1)
    assert props_low.input_resistance > props_high.input_resistance


def test_membrane_test_protocol_constants() -> None:
    """Protocol constants are positive and define a sensible stimulus window.

    The step must be long enough (> 2 ms) for the exponential fit to converge,
    and all durations must be strictly positive.
    """
    assert MEMBRANE_TEST_CURRENT < 0, "Step must be hyperpolarising (< 0)"
    assert MEMBRANE_TEST_PRE_MS > 0, "Pre-stimulus duration must be positive"
    assert MEMBRANE_TEST_STEP_MS > 2.0, "Step must be > 2 ms for reliable fitting"
    assert MEMBRANE_TEST_POST_MS > 0, "Post-stimulus duration must be positive"


def test_run_membrane_test_default_no_area_returns_density_only(
    hh_model: Neuron,
) -> None:
    """Default call (no area_cm2) leaves absolute MΩ / pF fields ``None``."""
    props = run_membrane_test(hh_model)
    assert props is not None
    assert props.input_resistance_mohm is None
    assert props.membrane_capacitance_pf is None
    assert props.area_cm2 is None


def test_run_membrane_test_with_area_returns_absolute(hh_model: Neuron) -> None:
    """A neuron carrying ``area_cm2`` produces absolute MΩ / pF outputs.

    For the default HH model (g_NaL+g_KL = 0.3 mS/cm², C_m = 1.0 µF/cm²) and
    area = 10×10⁻⁶ cm²: R_n ≈ 1/0.3 / 10e-6 / 1000 ≈ 333 MΩ and
    C ≈ 1.0 × 10e-6 × 1e6 ≈ 10 pF.  Surface area is read off the Neuron
    itself rather than passed as a separate argument.
    """
    area = 10e-6
    neuron_with_area = dataclasses.replace(hh_model, area_cm2=area)
    props = run_membrane_test(neuron_with_area)
    assert props is not None
    assert props.area_cm2 == pytest.approx(area)
    assert props.input_resistance_mohm == pytest.approx(
        1.0 / 0.3 / area / 1000.0, abs=10.0
    )
    assert props.membrane_capacitance_pf is not None
    assert props.membrane_capacitance_pf == pytest.approx(1.0 * area * 1e6, abs=0.1)
