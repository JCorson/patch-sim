"""Tests for patch_sim.presets — preset data structures and build_protocol_from_preset.

Focuses on the preset catalogue itself and verifies that neuron-specific
protocol adjustments are actually applied (not just that the function runs
without error).
"""

import numpy as np
import pytest

import patch_sim
from patch_sim.presets import (
    NEURON_PRESET_NAMES,
    NEURON_PRESETS,
    NEURON_PROTOCOL_ADJUSTMENTS,
    PROTOCOL_PRESET_NAMES,
    PROTOCOL_PRESETS,
    build_protocol_from_preset,
)

SAMPLING_FREQUENCY = 10_000.0  # Hz — fast enough for correct shapes


# ---------------------------------------------------------------------------
# Preset catalogue integrity
# ---------------------------------------------------------------------------


def test_neuron_preset_names_matches_neuron_presets_keys() -> None:
    """NEURON_PRESET_NAMES is exactly the keys of NEURON_PRESETS in order."""
    assert NEURON_PRESET_NAMES == list(NEURON_PRESETS.keys())


def test_protocol_preset_names_matches_protocol_presets_keys() -> None:
    """PROTOCOL_PRESET_NAMES is exactly the keys of PROTOCOL_PRESETS in order."""
    assert PROTOCOL_PRESET_NAMES == list(PROTOCOL_PRESETS.keys())


def test_neuron_protocol_adjustments_keys_are_known_neuron_presets() -> None:
    """Every key in NEURON_PROTOCOL_ADJUSTMENTS names a known neuron preset."""
    for neuron_name in NEURON_PROTOCOL_ADJUSTMENTS:
        assert neuron_name in NEURON_PRESETS, (
            f"Adjustment key {neuron_name!r} is not a known neuron preset"
        )


def test_neuron_protocol_adjustment_protocol_keys_are_known() -> None:
    """Every protocol name in NEURON_PROTOCOL_ADJUSTMENTS is a known preset."""
    for neuron_name, proto_map in NEURON_PROTOCOL_ADJUSTMENTS.items():
        for proto_name in proto_map:
            assert proto_name in PROTOCOL_PRESETS, (
                f"Adjustment for neuron {neuron_name!r} references unknown "
                f"protocol preset {proto_name!r}"
            )


# ---------------------------------------------------------------------------
# build_protocol_from_preset — each preset produces valid output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", PROTOCOL_PRESET_NAMES)
def test_build_protocol_from_preset_produces_valid_output(preset_name: str) -> None:
    """Each protocol preset returns a non-empty list of (finite ndarray, str) pairs."""
    result = build_protocol_from_preset(
        preset_name, sampling_frequency=SAMPLING_FREQUENCY
    )
    assert isinstance(result, list) and len(result) > 0
    for arr, label in result:
        assert isinstance(arr, np.ndarray) and arr.size > 0
        assert bool(np.all(np.isfinite(arr))), (
            f"Non-finite values in preset '{preset_name}'"
        )
        assert isinstance(label, str)


def test_build_protocol_from_preset_unknown_raises_key_error() -> None:
    """A non-existent preset name raises KeyError."""
    with pytest.raises(KeyError, match="Unknown protocol preset"):
        build_protocol_from_preset("NoSuchPreset")


# ---------------------------------------------------------------------------
# Neuron-specific protocol adjustments are applied
# ---------------------------------------------------------------------------


def _protocol_total_samples(preset_name: str, neuron_name: str | None) -> int:
    """Return the total number of samples in the first sweep of a preset."""
    result = build_protocol_from_preset(
        preset_name,
        neuron_preset=neuron_name,
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    return result[0][0].size


def test_neuron_protocol_adjustments_change_stimulus_duration() -> None:
    """Neuron adjustments that change stimulus_duration produce a different length.

    The Dopaminergic Neuron adjusts Repetitive Firing from 180 ms to 480 ms —
    the longer duration must result in a longer stimulus array.
    """
    base_samples = _protocol_total_samples("Repetitive Firing", None)
    adjusted_samples = _protocol_total_samples(
        "Repetitive Firing", "Dopaminergic Neuron"
    )
    assert adjusted_samples > base_samples, (
        "Dopaminergic Neuron adjustment should produce a longer array"
    )


def test_neuron_protocol_adjustments_not_applied_for_unknown_neuron() -> None:
    """Passing a neuron preset with no adjustments returns the base-preset output."""
    # "Squid Giant Axon (Classic HH)" has no adjustments defined.
    base_samples = _protocol_total_samples("Repetitive Firing", None)
    squid_samples = _protocol_total_samples(
        "Repetitive Firing", "Squid Giant Axon (Classic HH)"
    )
    assert squid_samples == base_samples


@pytest.mark.parametrize(
    "neuron_name",
    [
        n
        for n in NEURON_PRESET_NAMES
        if n in NEURON_PROTOCOL_ADJUSTMENTS
        and "Repetitive Firing" in NEURON_PROTOCOL_ADJUSTMENTS[n]
    ],
)
def test_repetitive_firing_adjusted_for_all_configured_neurons(
    neuron_name: str,
) -> None:
    """Every neuron with a Repetitive Firing adjustment produces a valid protocol."""
    result = build_protocol_from_preset(
        "Repetitive Firing",
        neuron_preset=neuron_name,
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    assert isinstance(result, list) and len(result) == 1
    arr, _label = result[0]
    assert arr.size > 0 and bool(np.all(np.isfinite(arr)))


def test_caller_overrides_take_precedence_over_neuron_adjustments() -> None:
    """Caller-supplied overrides win over both the base preset and neuron adjustments.

    Dopaminergic Neuron sets stimulus_duration=480; override to 50 ms.
    The resulting array should match a plain 50 ms stimulus.
    """
    base_50 = build_protocol_from_preset(
        "Repetitive Firing",
        sampling_frequency=SAMPLING_FREQUENCY,
        pre_stimulus_duration=10.0,
        stimulus_duration=50.0,
        post_stimulus_duration=10.0,
    )
    overridden = build_protocol_from_preset(
        "Repetitive Firing",
        neuron_preset="Dopaminergic Neuron",
        sampling_frequency=SAMPLING_FREQUENCY,
        pre_stimulus_duration=10.0,
        stimulus_duration=50.0,
        post_stimulus_duration=10.0,
    )
    assert base_50[0][0].size == overridden[0][0].size


# ---------------------------------------------------------------------------
# Top-level patch_sim re-export
# ---------------------------------------------------------------------------


def test_build_protocol_from_preset_exported_from_patch_sim() -> None:
    """build_protocol_from_preset is accessible directly from the patch_sim package."""
    assert callable(patch_sim.build_protocol_from_preset)
    result = patch_sim.build_protocol_from_preset(
        "Action Potential", sampling_frequency=SAMPLING_FREQUENCY
    )
    assert isinstance(result, list) and len(result) == 1
