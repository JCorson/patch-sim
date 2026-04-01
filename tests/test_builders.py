"""Tests for patch_sim.protocols.builders.

These tests call build_current_protocol and build_voltage_protocol directly,
requiring no Reflex runtime.
"""

import numpy as np
import pytest

from patch_sim.protocols.builders import (
    build_current_protocol,
    build_voltage_protocol,
)
from patch_sim.presets import PROTOCOL_PRESETS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLING_FREQUENCY = 10000.0  # Hz — matches UI default


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
# Parametrised: one test per CC protocol type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "protocol_type",
    ["Step", "Ramp", "Pulse Train", "Sinusoidal", "Chirp", "Noise"],
)
def test_current_protocol_returns_valid_list(protocol_type: str) -> None:
    """Each current clamp protocol returns a single-element list with a valid array."""
    result = build_current_protocol(
        protocol_type=protocol_type,
        sampling_frequency=SAMPLING_FREQUENCY,
        pre_stimulus_duration=5.0,
        stimulus_duration=40.0,
        post_stimulus_duration=5.0,
    )
    assert _is_valid_protocol_list(result), (
        f"Protocol '{protocol_type}' returned an invalid protocol list"
    )
    assert len(result) == 1
    assert result[0][1] == ""


# ---------------------------------------------------------------------------
# Parametrised: one test per VC protocol type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "protocol_type",
    ["Step", "Ramp", "Pulse Train"],
)
def test_single_sweep_voltage_protocol_returns_valid_list(protocol_type: str) -> None:
    """Single-sweep voltage protocols return a one-element list with a valid array."""
    result = build_voltage_protocol(
        protocol_type=protocol_type,
        sampling_frequency=SAMPLING_FREQUENCY,
        pre_stimulus_duration=5.0,
        stimulus_duration=10.0,
        post_stimulus_duration=5.0,
    )
    assert _is_valid_protocol_list(result), (
        f"Protocol '{protocol_type}' returned an invalid protocol list"
    )
    assert len(result) == 1
    assert result[0][1] == ""


def test_step_multi_sweep_returns_list_per_current_level() -> None:
    """Step with a range returns one (array, label) pair per current level."""
    result = build_current_protocol(
        protocol_type="Step",
        sampling_frequency=SAMPLING_FREQUENCY,
        pre_stimulus_duration=5.0,
        stimulus_duration=40.0,
        post_stimulus_duration=5.0,
        min_stimulus=0.0,
        max_stimulus=10.0,
        stimulus_step=5.0,
    )
    # 0 to 10 in steps of 5 → [0, 5, 10] = 3 sweeps
    assert _is_valid_protocol_list(result)
    assert len(result) == 3
    for arr, label in result:
        assert label != "", "Each multi-sweep Step should have a non-empty label"
        assert "µA/cm²" in label


def test_step_multi_sweep_voltage_returns_list_per_voltage_level() -> None:
    """Step with a voltage range returns one (array, label) pair per voltage step."""
    result = build_voltage_protocol(
        protocol_type="Step",
        sampling_frequency=SAMPLING_FREQUENCY,
        pre_stimulus_duration=5.0,
        stimulus_duration=20.0,
        post_stimulus_duration=5.0,
        min_stimulus=-40.0,
        max_stimulus=40.0,
        stimulus_step=20.0,
    )
    # -40 to +40 in steps of 20 → [-40, -20, 0, +20, +40] = 5 sweeps
    assert _is_valid_protocol_list(result)
    assert len(result) == 5
    for arr, label in result:
        assert label != "", "Each multi-sweep Step should have a non-empty label"
        assert "mV" in label


def test_step_single_via_equal_range_returns_single_sweep() -> None:
    """Step with min == max returns a single-element list regardless of step value."""
    result = build_current_protocol(
        protocol_type="Step",
        sampling_frequency=SAMPLING_FREQUENCY,
        min_stimulus=10.0,
        max_stimulus=10.0,
        stimulus_step=5.0,
    )
    assert len(result) == 1
    assert result[0][1] == ""


def test_voltage_step_single_via_equal_range_returns_single_sweep() -> None:
    """Voltage Step with min == max returns a single-element list regardless of step."""
    result = build_voltage_protocol(
        protocol_type="Step",
        sampling_frequency=SAMPLING_FREQUENCY,
        min_stimulus=-40.0,
        max_stimulus=-40.0,
        stimulus_step=10.0,
    )
    assert len(result) == 1
    assert result[0][1] == ""


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


def test_unknown_current_protocol_raises() -> None:
    """Unrecognised CC protocol type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown current protocol"):
        build_current_protocol(
            protocol_type="BadType",
            sampling_frequency=SAMPLING_FREQUENCY,
        )


def test_unknown_voltage_protocol_raises() -> None:
    """Unrecognised VC protocol type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown voltage protocol"):
        build_voltage_protocol(
            protocol_type="BadType",
            sampling_frequency=SAMPLING_FREQUENCY,
        )


def test_current_pulse_width_ge_interval_raises() -> None:
    """pulse_width >= pulse_interval raises ValueError for Pulse Train."""
    with pytest.raises(ValueError, match="pulse_width"):
        build_current_protocol(
            protocol_type="Pulse Train",
            sampling_frequency=SAMPLING_FREQUENCY,
            pulse_width=5.0,
            pulse_interval=5.0,
        )


def test_voltage_pulse_width_ge_interval_raises() -> None:
    """pulse_width >= pulse_interval raises ValueError for Voltage Pulse Train."""
    with pytest.raises(ValueError, match="pulse_width"):
        build_voltage_protocol(
            protocol_type="Pulse Train",
            sampling_frequency=SAMPLING_FREQUENCY,
            pulse_width=5.0,
            pulse_interval=5.0,
        )


def test_current_step_zero_with_range_raises() -> None:
    """Step=0.0 with min != max raises ValueError for current clamp Step."""
    with pytest.raises(ValueError, match="stimulus_step must be > 0"):
        build_current_protocol(
            protocol_type="Step",
            sampling_frequency=SAMPLING_FREQUENCY,
            min_stimulus=0.0,
            max_stimulus=10.0,
            stimulus_step=0.0,
        )


def test_current_min_gt_max_raises() -> None:
    """min_stimulus > max_stimulus raises ValueError for current clamp Step."""
    with pytest.raises(ValueError, match="min_stimulus"):
        build_current_protocol(
            protocol_type="Step",
            sampling_frequency=SAMPLING_FREQUENCY,
            min_stimulus=20.0,
            max_stimulus=10.0,
            stimulus_step=5.0,
        )


def test_current_negative_step_raises() -> None:
    """Negative stimulus_step raises ValueError for current clamp Step."""
    with pytest.raises(ValueError, match="stimulus_step must be >= 0"):
        build_current_protocol(
            protocol_type="Step",
            sampling_frequency=SAMPLING_FREQUENCY,
            min_stimulus=0.0,
            max_stimulus=10.0,
            stimulus_step=-1.0,
        )


def test_voltage_step_zero_with_range_raises() -> None:
    """Step=0.0 with min != max raises ValueError for voltage clamp Step."""
    with pytest.raises(ValueError, match="stimulus_step must be > 0"):
        build_voltage_protocol(
            protocol_type="Step",
            sampling_frequency=SAMPLING_FREQUENCY,
            min_stimulus=-40.0,
            max_stimulus=40.0,
            stimulus_step=0.0,
        )


def test_voltage_min_gt_max_raises() -> None:
    """min_stimulus > max_stimulus raises ValueError for voltage clamp Step."""
    with pytest.raises(ValueError, match="min_stimulus"):
        build_voltage_protocol(
            protocol_type="Step",
            sampling_frequency=SAMPLING_FREQUENCY,
            min_stimulus=40.0,
            max_stimulus=-40.0,
            stimulus_step=10.0,
        )


def test_voltage_negative_step_raises() -> None:
    """Negative stimulus_step raises ValueError for voltage clamp Step."""
    with pytest.raises(ValueError, match="stimulus_step must be >= 0"):
        build_voltage_protocol(
            protocol_type="Step",
            sampling_frequency=SAMPLING_FREQUENCY,
            min_stimulus=-40.0,
            max_stimulus=40.0,
            stimulus_step=-5.0,
        )


# ---------------------------------------------------------------------------
# Preset integration: each preset produces valid output from its builder
# ---------------------------------------------------------------------------


_NON_BUILDER_KEYS = {"clamp_mode", "protocol_type", "sampling_frequency"}


@pytest.mark.parametrize("preset_name", list(PROTOCOL_PRESETS.keys()))
def test_preset_produces_valid_protocol(preset_name: str) -> None:
    """For each preset, building the corresponding protocol returns a valid list."""
    config = PROTOCOL_PRESETS[preset_name]
    mode = config.get("clamp_mode", "Current Clamp")
    protocol_type = config.get("protocol_type", "Step")
    sampling_frequency = float(config.get("sampling_frequency", SAMPLING_FREQUENCY))
    kwargs = {
        k: float(v)
        for k, v in config.items()
        if k not in _NON_BUILDER_KEYS and isinstance(v, (int, float))
    }

    if mode == "Current Clamp":
        result = build_current_protocol(
            protocol_type=protocol_type,
            sampling_frequency=sampling_frequency,
            **kwargs,
        )
    else:
        result = build_voltage_protocol(
            protocol_type=protocol_type,
            sampling_frequency=sampling_frequency,
            **kwargs,
        )

    assert _is_valid_protocol_list(result), (
        f"Preset '{preset_name}' produced an invalid protocol list"
    )
