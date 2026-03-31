"""Tests for patch_sim.protocols.builders and build_protocol_from_preset."""

import numpy as np
import pytest

import patch_sim
from patch_sim.protocols.builders import (
    build_current_protocol,
    build_voltage_protocol,
)
from patch_sim.presets import PROTOCOL_PRESETS, NEURON_PRESET_NAMES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLING_FREQUENCY = 10_000.0  # Hz


def _is_valid_protocol_list(result: list[tuple[np.ndarray, str]]) -> bool:
    """Return True if result is a non-empty list of (finite ndarray, str) pairs."""
    if not isinstance(result, list) or len(result) == 0:
        return False
    return all(
        isinstance(arr, np.ndarray)
        and arr.size > 0
        and bool(np.all(np.isfinite(arr)))
        and isinstance(label, str)
        for arr, label in result
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
    assert _is_valid_protocol_list(result)
    assert len(result) == 1
    assert result[0][1] == ""


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
    assert _is_valid_protocol_list(result)
    assert len(result) == 1
    assert result[0][1] == ""


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
    assert _is_valid_protocol_list(result), (
        f"Preset '{preset_name}' produced an invalid protocol list"
    )


@pytest.mark.parametrize("neuron_name", NEURON_PRESET_NAMES)
def test_build_protocol_from_preset_repetitive_firing_with_neuron(
    neuron_name: str,
) -> None:
    """Repetitive Firing preset with every neuron preset returns a valid protocol."""
    result = patch_sim.build_protocol_from_preset(
        "Repetitive Firing",
        neuron_preset=neuron_name,
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    assert _is_valid_protocol_list(result)


def test_build_protocol_from_preset_overrides_applied() -> None:
    """Caller overrides take precedence over the preset values."""
    result = patch_sim.build_protocol_from_preset(
        "Action Potential",
        sampling_frequency=SAMPLING_FREQUENCY,
        min_stimulus=5.0,
        max_stimulus=5.0,
    )
    assert _is_valid_protocol_list(result)
    assert len(result) == 1


def test_build_protocol_from_preset_multi_sweep() -> None:
    """F-I Curve preset produces multiple sweeps."""
    result = patch_sim.build_protocol_from_preset(
        "F-I Curve",
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    assert _is_valid_protocol_list(result)
    assert len(result) > 1


def test_build_protocol_from_preset_voltage_clamp() -> None:
    """I-V Curve preset (voltage clamp) produces multiple sweeps with mV labels."""
    result = patch_sim.build_protocol_from_preset(
        "I-V Curve",
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    assert _is_valid_protocol_list(result)
    assert len(result) > 1
    assert all("mV" in label for _, label in result)


def test_build_protocol_from_preset_unknown_raises() -> None:
    """Passing an unknown preset name raises KeyError."""
    with pytest.raises(KeyError, match="Unknown protocol preset"):
        patch_sim.build_protocol_from_preset("NonExistentPreset")
