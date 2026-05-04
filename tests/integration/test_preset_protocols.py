"""Integration tests for each neuron preset × stimulation protocol combination.

Verifies that every neuron preset produces biologically reasonable responses
under each of the built-in stimulation protocols.  Failures here indicate
either a miscalibrated neuron preset or a broken protocol adjustment.

These tests complement the per-neuron behavioral tests (e.g.
``test_cortical_pyramidal.py``) with broad coverage across all presets.
"""

from __future__ import annotations

import numpy as np
import pytest

from patch_sim.clamp_simulations import (
    simulate_current_clamp,
    simulate_voltage_clamp,
)
from patch_sim.constants import (
    ACTION_POTENTIAL,
    PURKINJE,
    REPETITIVE_FIRING,
    STN,
    TRN,
)
from patch_sim.neuron_factory import make_neuron
from patch_sim.presets import (
    NEURON_PRESET_NAMES,
    NEURON_PRESETS,
    build_protocol_from_preset,
)

# Purkinje is a spontaneous pacemaker — it fires without external current.
# The "zero-current rest" assumption underlying these three tests does not hold:
# - test_action_potential_preset: cell fires spontaneously before the stimulus
# - test_subthreshold_response_preset: no stimulus is subthreshold for a pacemaker
# - test_fi_curve_preset: the zero-current (first) sweep produces spontaneous APs
# Dedicated Purkinje tests live in tests/integration/test_purkinje.py.
#
# TRN is excluded for the same reason after issue #295: with Ih added, the cell
# fires tonically at ~3 Hz at zero current (consistent with HP92/B&M93 slice
# recordings).  The HP92 rebound-burst phenotype is exercised by
# ``test_trn_step_release_produces_hp92_rebound_burst``.
#
# STN is excluded post-#305: the autonomous-pacemaker preset fires
# spontaneously at ~20 Hz from v_rest = −60 mV (Bevan & Wilson 1999).  The
# tonic-pacemaker phenotype and conditional rebound-burst phenotype are
# exercised by tests/integration/test_stn.py and
# ``test_stn_conditional_burst_mode_under_hyperpolarising_step_release``.
_QUIESCENT_PRESET_NAMES = [
    p for p in NEURON_PRESET_NAMES if p not in (PURKINJE, TRN, STN)
]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CURRENT_CLAMP_REQUIRED = {"time", "voltage", "INa", "IK", "INaL", "IKL", "Itotal"}
_VOLTAGE_CLAMP_REQUIRED = {"time", "voltage", "Itotal", "INa", "IK", "INaL", "IKL"}


def _count_action_potentials(voltage: np.ndarray, threshold: float = 0.0) -> int:
    """Count upward threshold crossings in a voltage trace.

    Args:
        voltage: 1-D array of membrane voltages in mV.
        threshold: Voltage threshold for AP detection in mV.

    Returns:
        Number of action potentials detected.
    """
    count = 0
    above = False
    for v in voltage:
        if v > threshold and not above:
            count += 1
            above = True
        elif v < threshold:
            above = False
    return count


def _assert_current_clamp_valid(
    result: np.ndarray,
    v_min: float = -120.0,
    v_max: float = 80.0,
) -> None:
    """Assert structural and numerical validity of a current-clamp result.

    Args:
        result: Structured numpy array returned by ``simulate_current_clamp``.
        v_min: Lower bound for physiologically acceptable voltage in mV.
        v_max: Upper bound for physiologically acceptable voltage in mV.

    Raises:
        AssertionError: If any structural or numerical check fails.
    """
    assert isinstance(result, np.ndarray)
    assert result.dtype.names is not None
    col_names = set(result.dtype.names)
    for required in _CURRENT_CLAMP_REQUIRED:
        assert required in col_names, f"Missing column '{required}'"
    assert np.isfinite(result["voltage"]).all(), "voltage contains non-finite values"
    assert float(result["voltage"].min()) >= v_min, (
        f"voltage {result['voltage'].min():.1f} mV below {v_min} mV"
    )
    assert float(result["voltage"].max()) <= v_max, (
        f"voltage {result['voltage'].max():.1f} mV above {v_max} mV"
    )


def _assert_voltage_clamp_valid(result: np.ndarray) -> None:
    """Assert structural and numerical validity of a voltage-clamp result.

    Args:
        result: Structured numpy array returned by ``simulate_voltage_clamp``.

    Raises:
        AssertionError: If any structural or numerical check fails.
    """
    assert isinstance(result, np.ndarray)
    assert result.dtype.names is not None
    col_names = set(result.dtype.names)
    for required in _VOLTAGE_CLAMP_REQUIRED:
        assert required in col_names, f"Missing column '{required}'"
    assert np.isfinite(result["Itotal"]).all(), "Itotal contains non-finite values"


# ---------------------------------------------------------------------------
# Current clamp: Action Potential protocol
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", _QUIESCENT_PRESET_NAMES)
def test_action_potential_preset(preset_name: str) -> None:
    """Action Potential protocol elicits exactly one action potential.

    The adjusted single-step stimulus is tuned per neuron to produce a
    single suprathreshold spike.  The test asserts exactly 1 AP so that
    miscalibrated stimuli (producing 0 or ≥2 spikes) are caught as failures.
    Purkinje is excluded because it fires spontaneously (see test_purkinje.py).

    Args:
        preset_name: Name of the neuron preset under test.
    """
    neuron = make_neuron(NEURON_PRESETS[preset_name])
    protocol = build_protocol_from_preset(ACTION_POTENTIAL, neuron_preset=preset_name)
    result = simulate_current_clamp(neuron, current_external=protocol[0])

    _assert_current_clamp_valid(result)

    n_aps = _count_action_potentials(result["voltage"])
    assert n_aps == 1, (
        f"{preset_name}: expected exactly 1 AP under Action Potential "
        f"protocol, got {n_aps}"
    )
    assert float(result["voltage"].max()) > 0.0, (
        f"{preset_name}: AP peak {result['voltage'].max():.1f} mV is below 0 mV"
    )


# ---------------------------------------------------------------------------
# Current clamp: Subthreshold Response protocol
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", _QUIESCENT_PRESET_NAMES)
def test_subthreshold_response_preset(preset_name: str) -> None:
    """Subthreshold Response protocol produces no action potentials.

    The weak stimulus is calibrated per neuron to stay below AP threshold.
    The test asserts no threshold crossings and that peak voltage remains
    below 0 mV.  Purkinje is excluded because it fires spontaneously and has
    no subthreshold regime (see test_purkinje.py).

    Args:
        preset_name: Name of the neuron preset under test.
    """
    neuron = make_neuron(NEURON_PRESETS[preset_name])
    protocol = build_protocol_from_preset(
        "Subthreshold Response", neuron_preset=preset_name
    )
    result = simulate_current_clamp(neuron, current_external=protocol[0])

    _assert_current_clamp_valid(result)

    n_aps = _count_action_potentials(result["voltage"])
    assert n_aps == 0, (
        f"{preset_name}: expected 0 APs under Subthreshold Response protocol, "
        f"got {n_aps}"
    )
    assert float(result["voltage"].max()) < 0.0, (
        f"{preset_name}: peak voltage {result['voltage'].max():.1f} mV reached "
        "or exceeded 0 mV despite subthreshold stimulus"
    )


# ---------------------------------------------------------------------------
# Current clamp: Repetitive Firing protocol
# ---------------------------------------------------------------------------

_DEFAULT_MIN_REPETITIVE_APS = 5


@pytest.mark.parametrize("preset_name", NEURON_PRESET_NAMES)
def test_repetitive_firing_preset(preset_name: str) -> None:
    """Repetitive Firing protocol elicits multiple action potentials.

    The long suprathreshold step is designed to produce sustained repetitive
    discharge.  All presets must produce at least ``_DEFAULT_MIN_REPETITIVE_APS``
    (5) APs.

    Args:
        preset_name: Name of the neuron preset under test.
    """
    neuron = make_neuron(NEURON_PRESETS[preset_name])
    protocol = build_protocol_from_preset(REPETITIVE_FIRING, neuron_preset=preset_name)
    result = simulate_current_clamp(neuron, current_external=protocol[0])

    _assert_current_clamp_valid(result)

    n_aps = _count_action_potentials(result["voltage"])
    assert n_aps >= _DEFAULT_MIN_REPETITIVE_APS, (
        f"{preset_name}: expected ≥{_DEFAULT_MIN_REPETITIVE_APS} APs under Repetitive "
        f"Firing protocol, got {n_aps}"
    )


# ---------------------------------------------------------------------------
# Current clamp: F-I Curve protocol
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", _QUIESCENT_PRESET_NAMES)
def test_fi_curve_preset(preset_name: str) -> None:
    """F-I Curve protocol shows increasing AP count with increasing current.

    Checks that:

    - The zero-current (first) sweep produces no APs.
    - The maximum-current (last) sweep produces at least as many APs as the
      first sweep (monotonicity).
    - Each sweep result is structurally valid with all values finite.

    Purkinje is excluded because its zero-current sweep produces spontaneous
    APs (see test_purkinje.py).

    Args:
        preset_name: Name of the neuron preset under test.
    """
    neuron = make_neuron(NEURON_PRESETS[preset_name])
    protocol = build_protocol_from_preset("F-I Curve", neuron_preset=preset_name)

    assert len(protocol) >= 2, f"{preset_name}: F-I Curve must have ≥2 sweeps"

    first_result = simulate_current_clamp(neuron, current_external=protocol[0])
    last_result = simulate_current_clamp(neuron, current_external=protocol[-1])

    _assert_current_clamp_valid(first_result)
    _assert_current_clamp_valid(last_result)

    n_first = _count_action_potentials(first_result["voltage"])
    n_last = _count_action_potentials(last_result["voltage"])

    assert n_first == 0, (
        f"{preset_name}: zero-current sweep should produce 0 APs, got {n_first}"
    )
    assert n_last >= n_first, (
        f"{preset_name}: last F-I sweep ({n_last} APs) should have ≥ first sweep "
        f"({n_first} APs)"
    )


# ---------------------------------------------------------------------------
# Current clamp: Frequency Response protocol
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", NEURON_PRESET_NAMES)
def test_frequency_response_preset(preset_name: str) -> None:
    """Frequency Response protocol produces a measurable voltage response.

    The chirp stimulus sweeps from 1 to 100 Hz and elicits a mixture of
    subthreshold and suprathreshold responses.  The test asserts that the
    voltage trace is not flat: voltage range must exceed 2 mV and standard
    deviation must exceed 0.5 mV.

    Args:
        preset_name: Name of the neuron preset under test.
    """
    neuron = make_neuron(NEURON_PRESETS[preset_name])
    protocol = build_protocol_from_preset(
        "Frequency Response", neuron_preset=preset_name
    )
    result = simulate_current_clamp(neuron, current_external=protocol[0])

    _assert_current_clamp_valid(result)

    v_std = float(np.std(result["voltage"]))
    v_range = float(result["voltage"].max() - result["voltage"].min())

    assert v_std > 0.5, (
        f"{preset_name}: Frequency Response voltage std {v_std:.3f} mV ≤ 0.5 mV "
        "(neuron may not be responding to the chirp stimulus)"
    )
    assert v_range > 2.0, (
        f"{preset_name}: Frequency Response voltage range {v_range:.3f} mV ≤ 2.0 mV "
        "(neuron may not be responding to the chirp stimulus)"
    )


# ---------------------------------------------------------------------------
# Voltage clamp: I-V Curve protocol
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", NEURON_PRESET_NAMES)
def test_iv_curve_preset(preset_name: str) -> None:
    """I-V Curve protocol completes without errors across all voltage steps.

    Iterates every sweep and asserts that the voltage-clamp result is
    structurally valid with all values finite.

    Args:
        preset_name: Name of the neuron preset under test.
    """
    neuron = make_neuron(NEURON_PRESETS[preset_name])
    protocol = build_protocol_from_preset("I-V Curve", neuron_preset=preset_name)

    assert len(protocol) >= 1, f"{preset_name}: I-V Curve must have ≥1 sweep"

    for i, sweep in enumerate(protocol):
        result = simulate_voltage_clamp(neuron, voltage_protocol=sweep)
        _assert_voltage_clamp_valid(result)
        assert np.isfinite(result["INa"]).all(), (
            f"{preset_name} sweep {i}: INa contains non-finite values"
        )
        assert np.isfinite(result["IK"]).all(), (
            f"{preset_name} sweep {i}: IK contains non-finite values"
        )


# ---------------------------------------------------------------------------
# Voltage clamp: Na+ Channel Activation protocol
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", NEURON_PRESET_NAMES)
def test_na_channel_activation_preset(preset_name: str) -> None:
    """Na⁺ Channel Activation protocol reveals inward Na⁺ current at depolarization.

    Each sweep must produce a structurally valid, finite result.  At the most
    depolarized command voltage (+60 mV) the Na⁺ current must show an inward
    (negative) peak of at least 1 µA/cm², confirming Na⁺ channel activation.

    Args:
        preset_name: Name of the neuron preset under test.
    """
    neuron = make_neuron(NEURON_PRESETS[preset_name])
    protocol = build_protocol_from_preset(
        "Na+ Channel Activation", neuron_preset=preset_name
    )

    assert len(protocol) >= 1, f"{preset_name}: Na+ Channel Activation needs ≥1 sweep"

    last_result = None
    for i, sweep in enumerate(protocol):
        result = simulate_voltage_clamp(neuron, voltage_protocol=sweep)
        _assert_voltage_clamp_valid(result)
        assert np.isfinite(result["INa"]).all(), (
            f"{preset_name} sweep {i}: INa contains non-finite values"
        )
        last_result = result

    # At the most depolarized step, Na⁺ current must be clearly inward.
    assert last_result is not None
    ina_min = float(last_result["INa"].min())
    assert ina_min < -1.0, (
        f"{preset_name}: INa minimum at max depolarization = {ina_min:.3f} µA/cm²; "
        "expected < -1.0 µA/cm² (inward Na⁺ current not detected)"
    )
