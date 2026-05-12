"""Tests for patch_sim.protocols.builders and build_protocol_from_preset."""

import numpy as np
import pytest

import patch_sim
from patch_sim.constants import ACTION_POTENTIAL, REPETITIVE_FIRING
from patch_sim.presets import NEURON_PRESET_NAMES, PROTOCOL_PRESETS
from patch_sim.protocols.builders import (
    build_current_protocol,
    build_voltage_protocol,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLING_FREQUENCY = 10_000.0  # Hz


def _is_valid_protocol_array(result: np.ndarray) -> bool:
    """Return True if result is a non-empty 2-D finite ndarray."""
    return (
        isinstance(result, np.ndarray)
        and result.ndim == 2
        and result.size > 0
        and bool(np.all(np.isfinite(result)))
    )


# ---------------------------------------------------------------------------
# build_current_protocol — core import
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "protocol_type",
    ["Step", "Ramp", "Pulse Train", "Sinusoidal", "Chirp", "Noise"],
)
def test_current_protocol_types(protocol_type: str) -> None:
    """Each current clamp protocol type returns a valid single-sweep list."""
    result = build_current_protocol(
        protocol_type=protocol_type,
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    assert _is_valid_protocol_array(result)
    assert result.shape[0] == 1


@pytest.mark.parametrize(
    "protocol_type",
    ["Step", "Ramp", "Pulse Train"],
)
def test_voltage_protocol_types(protocol_type: str) -> None:
    """Each voltage clamp protocol type returns a valid single-sweep list."""
    result = build_voltage_protocol(
        protocol_type=protocol_type,
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    assert _is_valid_protocol_array(result)
    assert result.shape[0] == 1


# ---------------------------------------------------------------------------
# build_voltage_protocol — "Inactivation" two-pulse protocol
# ---------------------------------------------------------------------------


def _inactivation_protocol() -> tuple[np.ndarray, float, float, float, list[float]]:
    """Build a small Inactivation protocol and return it with its parameters.

    Returns:
        A 5-tuple ``(arrays, pre_ms, stim_ms, post_ms, prepulses)`` where
        ``arrays`` has shape ``(5, n_samples)``, the durations are in ms and
        ``prepulses`` lists the five conditioning prepulse voltages in mV.
    """
    pre_ms, stim_ms, post_ms = 50.0, 10.0, 5.0
    min_v, max_v, step_v = -100.0, -20.0, 20.0
    arrays = build_voltage_protocol(
        protocol_type="Inactivation",
        sampling_frequency=SAMPLING_FREQUENCY,
        pre_stimulus_duration=pre_ms,
        stimulus_duration=stim_ms,
        post_stimulus_duration=post_ms,
        min_stimulus=min_v,
        max_stimulus=max_v,
        stimulus_step=step_v,
        test_pulse_voltage=0.0,
    )
    n_steps = round((max_v - min_v) / step_v) + 1
    return arrays, pre_ms, stim_ms, post_ms, list(np.linspace(min_v, max_v, n_steps))


def test_build_voltage_protocol_inactivation_shape() -> None:
    """The Inactivation protocol returns one sweep per conditioning prepulse."""
    arrays, *_ = _inactivation_protocol()
    assert _is_valid_protocol_array(arrays)
    assert arrays.shape[0] == 5


def test_build_voltage_protocol_inactivation_holding_varies_step_fixed() -> None:
    """Each sweep holds at its prepulse and steps to the fixed test pulse."""
    arrays, pre_ms, stim_ms, _post_ms, prepulses = _inactivation_protocol()
    # Sample indices comfortably inside the prepulse, test-pulse and tail.
    i_prepulse = int(0.5 * pre_ms * 1e-3 * SAMPLING_FREQUENCY)
    i_test = int((pre_ms + 0.5 * stim_ms) * 1e-3 * SAMPLING_FREQUENCY)
    i_tail = int((pre_ms + stim_ms + 1.0) * 1e-3 * SAMPLING_FREQUENCY)
    for row, prepulse in zip(arrays, prepulses):
        assert row[i_prepulse] == pytest.approx(prepulse)
        assert row[i_test] == pytest.approx(0.0)
        assert row[i_tail] == pytest.approx(prepulse)


def test_build_voltage_protocol_inactivation_requires_prepulse_range() -> None:
    """A missing prepulse range (min == max) is rejected."""
    with pytest.raises(ValueError, match="prepulse range"):
        build_voltage_protocol(
            protocol_type="Inactivation",
            sampling_frequency=SAMPLING_FREQUENCY,
            min_stimulus=-80.0,
            max_stimulus=-80.0,
            stimulus_step=10.0,
        )


def test_build_voltage_protocol_inactivation_requires_positive_step() -> None:
    """A zero step over a prepulse range is rejected."""
    with pytest.raises(ValueError, match="stimulus_step"):
        build_voltage_protocol(
            protocol_type="Inactivation",
            sampling_frequency=SAMPLING_FREQUENCY,
            min_stimulus=-100.0,
            max_stimulus=-20.0,
            stimulus_step=0.0,
        )


def test_build_voltage_protocol_inactivation_rejects_min_gt_max() -> None:
    """A prepulse range with min > max is rejected."""
    with pytest.raises(ValueError, match="min_stimulus"):
        build_voltage_protocol(
            protocol_type="Inactivation",
            sampling_frequency=SAMPLING_FREQUENCY,
            min_stimulus=-20.0,
            max_stimulus=-100.0,
            stimulus_step=10.0,
        )


def test_build_protocol_from_preset_steady_state_inactivation() -> None:
    """The Steady-State Inactivation preset builds a valid multi-sweep protocol."""
    result = patch_sim.build_protocol_from_preset(
        patch_sim.constants.STEADY_STATE_INACTIVATION,
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    assert _is_valid_protocol_array(result)
    assert result.shape[0] > 1


# ---------------------------------------------------------------------------
# build_protocol_from_preset — exported from patch_sim top-level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", list(PROTOCOL_PRESETS.keys()))
def test_build_protocol_from_preset_base(preset_name: str) -> None:
    """Each preset produces a valid protocol list when called with no neuron preset."""
    result = patch_sim.build_protocol_from_preset(
        preset_name,
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    assert _is_valid_protocol_array(result), (
        f"Preset '{preset_name}' produced an invalid protocol list"
    )


@pytest.mark.parametrize("neuron_name", NEURON_PRESET_NAMES)
def test_build_protocol_from_preset_repetitive_firing_with_neuron(
    neuron_name: str,
) -> None:
    """Repetitive Firing preset with every neuron preset returns a valid protocol."""
    result = patch_sim.build_protocol_from_preset(
        REPETITIVE_FIRING,
        neuron_preset=neuron_name,
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    assert _is_valid_protocol_array(result)


def test_build_protocol_from_preset_overrides_applied() -> None:
    """Caller overrides take precedence over the preset values."""
    result = patch_sim.build_protocol_from_preset(
        ACTION_POTENTIAL,
        sampling_frequency=SAMPLING_FREQUENCY,
        min_stimulus=5.0,
        max_stimulus=5.0,
    )
    assert _is_valid_protocol_array(result)
    assert result.shape[0] == 1


def test_build_protocol_from_preset_multi_sweep() -> None:
    """F-I Curve preset produces multiple sweeps."""
    result = patch_sim.build_protocol_from_preset(
        "F-I Curve",
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    assert _is_valid_protocol_array(result)
    assert result.shape[0] > 1


def test_build_protocol_from_preset_voltage_clamp() -> None:
    """I-V Curve preset (voltage clamp) produces multiple sweeps."""
    result = patch_sim.build_protocol_from_preset(
        "I-V Curve",
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    assert _is_valid_protocol_array(result)
    assert result.shape[0] > 1


def test_build_protocol_from_preset_unknown_raises() -> None:
    """Passing an unknown preset name raises KeyError."""
    with pytest.raises(KeyError, match="Unknown protocol preset"):
        patch_sim.build_protocol_from_preset("NonExistentPreset")
