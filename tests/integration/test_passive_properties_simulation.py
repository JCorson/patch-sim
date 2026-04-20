"""Integration tests for patch_sim.analysis.passive_properties with real HH simulations.

Verifies that passive property analysis (R_in, τ_m, C_m) returns physiologically
plausible values and correct guard behaviour when run on simulate_current_clamp output.
Unit tests with synthetic traces live in tests/unit/test_passive_properties.py.
"""

import pytest

import patch_sim
from patch_sim.analysis.passive_properties import (
    PassiveProperties,
    analyze_passive_properties,
    is_subthreshold,
)

_PRE_MS = 50.0
_STIM_MS = 200.0
_POST_MS = 50.0


def _run_subthreshold_sim(
    hh_model: patch_sim.Neuron,
    current_amplitude: float = -1.0,
) -> tuple[patch_sim.SimulationResult, float, float]:
    """Run a short subthreshold current clamp simulation.

    Args:
        hh_model: Neuron instance to simulate.
        current_amplitude: Injected current in µA/cm² (should be subthreshold).

    Returns:
        Tuple of (result, stim_start_ms, stim_end_ms).
    """
    stim_start = _PRE_MS
    stim_end = _PRE_MS + _STIM_MS
    stimulus = patch_sim.step_current(
        duration=_PRE_MS + _STIM_MS + _POST_MS,
        current_amplitude=current_amplitude,
        step_start=stim_start,
        step_duration=_STIM_MS,
    )
    result = patch_sim.simulate_current_clamp(hh_model, stimulus)
    return result, stim_start, stim_end


@pytest.fixture(scope="module")
def _subthreshold_passive() -> tuple[patch_sim.SimulationResult, PassiveProperties]:
    """Run one subthreshold sim and analyze passive properties once per module.

    Returns:
        Tuple of (simulation_result, passive_properties).
    """
    neuron = patch_sim.Neuron(K_out=7.8)
    result, stim_start, stim_end = _run_subthreshold_sim(neuron, -1.0)
    props = analyze_passive_properties(
        result["time"],
        result["voltage"],
        current_amplitude=-1.0,
        stim_start_ms=stim_start,
        stim_end_ms=stim_end,
    )
    assert props is not None, "Subthreshold sim should yield analyzable passive props"
    return result, props


@pytest.fixture(scope="module")
def _suprathreshold_result() -> patch_sim.SimulationResult:
    """Run one suprathreshold sim once per module.

    Returns:
        Simulation result for a spiking trace.
    """
    neuron = patch_sim.Neuron(K_out=7.8)
    stimulus = patch_sim.step_current(
        duration=_PRE_MS + _STIM_MS + _POST_MS,
        current_amplitude=20.0,
        step_start=_PRE_MS,
        step_duration=_STIM_MS,
    )
    return patch_sim.simulate_current_clamp(neuron, stimulus)


# ---------------------------------------------------------------------------
# is_subthreshold with real simulations
# ---------------------------------------------------------------------------


def test_is_subthreshold_rejects_spiking(
    _suprathreshold_result: patch_sim.SimulationResult,
) -> None:
    """Suprathreshold stimulus produces spikes; is_subthreshold returns False."""
    result = _suprathreshold_result
    assert is_subthreshold(result["time"], result["voltage"]) is False


# ---------------------------------------------------------------------------
# analyze_passive_properties — guard conditions with HH simulation
# ---------------------------------------------------------------------------


def test_returns_none_for_suprathreshold(
    _suprathreshold_result: patch_sim.SimulationResult,
) -> None:
    """analyze_passive_properties returns None for a spiking trace."""
    result = _suprathreshold_result
    props = analyze_passive_properties(
        result["time"],
        result["voltage"],
        current_amplitude=20.0,
        stim_start_ms=_PRE_MS,
        stim_end_ms=_PRE_MS + _STIM_MS,
    )
    assert props is None


# ---------------------------------------------------------------------------
# analyze_passive_properties — HH model integration
# ---------------------------------------------------------------------------


def test_input_resistance_hh_model(
    _subthreshold_passive: tuple[patch_sim.SimulationResult, PassiveProperties],
) -> None:
    """R_in from a real HH simulation is positive and in a plausible range.

    The HH model at rest has g_NaL+g_KL = 0.3 mS/cm² plus active conductances.
    Total conductance is typically 0.5–2.0 mS/cm², giving R_in in the range
    0.5–2.0 kΩ·cm².  This test verifies the sign and order of magnitude only,
    since the exact value depends on gating variables at the new steady state.
    """
    _, props = _subthreshold_passive
    assert props.input_resistance > 0.0
    assert props.input_resistance < 5.0  # below 1/(g_NaL+g_KL)=3.33 with tolerance


def test_time_constant_hh_model(
    _subthreshold_passive: tuple[patch_sim.SimulationResult, PassiveProperties],
) -> None:
    """τₘ from a real HH simulation is positive and within a plausible range.

    Standard HH: C_m = 1.0 µF/cm², total g ≈ 0.5–1.5 mS/cm², so τₘ is
    expected to be between 0.5 and 20 ms.  The active conductances at rest
    mean the effective τₘ is considerably shorter than the passive C_m / (g_NaL+g_KL)
    ≈ 3.33 ms estimate, so this test verifies the order of magnitude only.
    """
    _, props = _subthreshold_passive
    assert props.time_constant > 0.0
    assert props.time_constant < 20.0


def test_membrane_capacitance_hh_model(
    _subthreshold_passive: tuple[patch_sim.SimulationResult, PassiveProperties],
) -> None:
    """Derived Cₘ from the HH model is positive and in a plausible range.

    Cₘ = τₘ / R_in.  With τₘ < 20 ms and R_in ≈ 0.5–5 kΩ·cm², the derived
    Cₘ should be in the range 0.05–10 µF/cm².  This test verifies sign and
    rough order of magnitude only.
    """
    _, props = _subthreshold_passive
    assert props.membrane_capacitance is not None
    assert props.membrane_capacitance > 0.0
    assert props.membrane_capacitance < 10.0


def test_fit_converged_flag_hh_model(
    _subthreshold_passive: tuple[patch_sim.SimulationResult, PassiveProperties],
) -> None:
    """Exponential fit converges for a clean subthreshold HH sweep."""
    _, props = _subthreshold_passive
    assert props.fit_converged is True
