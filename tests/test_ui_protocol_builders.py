"""Regression tests for patch_sim_ui.protocol_builders.

These tests call build_current_protocol and build_voltage_protocol directly,
requiring no Reflex runtime.
"""

import numpy as np
import pytest

from patch_sim_ui.protocol_builders import (
    build_current_protocol,
    build_voltage_protocol,
)
from patch_sim_ui.presets import PRESETS

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
        duration=50.0,
        sampling_frequency=SAMPLING_FREQUENCY,
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
        duration=20.0,
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    assert _is_valid_protocol_list(result), (
        f"Protocol '{protocol_type}' returned an invalid protocol list"
    )
    assert len(result) == 1
    assert result[0][1] == ""


def test_iv_curve_returns_multi_sweep_list() -> None:
    """I-V Curve returns one (array, label) pair per voltage step."""
    result = build_voltage_protocol(
        protocol_type="I-V Curve",
        duration=20.0,
        sampling_frequency=SAMPLING_FREQUENCY,
        vc_voltage_min=-40.0,
        vc_voltage_max=40.0,
        vc_voltage_step=20.0,
    )
    # -40 to +40 in steps of 20 → [-40, -20, 0, +20, +40] = 5 sweeps
    assert _is_valid_protocol_list(result)
    assert len(result) == 5
    for arr, label in result:
        assert label != "", "Each I-V Curve sweep should have a non-empty label"
        assert "mV" in label


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


def test_unknown_current_protocol_raises() -> None:
    """Unrecognised CC protocol type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown current protocol"):
        build_current_protocol(
            protocol_type="BadType",
            duration=50.0,
            sampling_frequency=SAMPLING_FREQUENCY,
        )


def test_unknown_voltage_protocol_raises() -> None:
    """Unrecognised VC protocol type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown voltage protocol"):
        build_voltage_protocol(
            protocol_type="BadType",
            duration=20.0,
            sampling_frequency=SAMPLING_FREQUENCY,
        )


def test_current_pulse_width_ge_interval_raises() -> None:
    """pulse_width >= pulse_interval raises ValueError for Pulse Train."""
    with pytest.raises(ValueError, match="pulse_width"):
        build_current_protocol(
            protocol_type="Pulse Train",
            duration=50.0,
            sampling_frequency=SAMPLING_FREQUENCY,
            pulse_width=5.0,
            pulse_interval=5.0,
        )


def test_voltage_pulse_width_ge_interval_raises() -> None:
    """vc_pulse_width >= vc_pulse_interval raises ValueError for Pulse Train."""
    with pytest.raises(ValueError, match="pulse_width"):
        build_voltage_protocol(
            protocol_type="Pulse Train",
            duration=50.0,
            sampling_frequency=SAMPLING_FREQUENCY,
            vc_pulse_width=5.0,
            vc_pulse_interval=5.0,
        )


# ---------------------------------------------------------------------------
# Preset integration: each preset produces valid output from its builder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
def test_preset_produces_valid_protocol(preset_name: str) -> None:
    """For each preset, building the corresponding protocol returns a valid list."""
    config = PRESETS[preset_name]
    mode = config.get("clamp_mode", "Current Clamp")
    protocol_type = config.get("protocol_type", "Step")
    duration = float(config.get("duration", 50.0))
    sampling_frequency = float(config.get("sampling_frequency", SAMPLING_FREQUENCY))

    if mode == "Current Clamp":
        kwargs = {
            k: float(v)
            for k, v in config.items()
            if k
            not in {
                "clamp_mode",
                "protocol_type",
                "duration",
                "sampling_frequency",
                "g_Na",
                "g_K",
                "g_L",
                "C_m",
                "v_rest",
                "Na_out",
                "Na_in",
                "K_out",
                "K_in",
                "Cl_out",
                "Cl_in",
                "Ca_out",
                "Ca_in",
                "T",
            }
            and isinstance(v, (int, float))
        }
        result = build_current_protocol(
            protocol_type=protocol_type,
            duration=duration,
            sampling_frequency=sampling_frequency,
            **kwargs,
        )
    else:
        kwargs = {
            k: float(v)
            for k, v in config.items()
            if k
            not in {
                "clamp_mode",
                "protocol_type",
                "duration",
                "sampling_frequency",
                "g_Na",
                "g_K",
                "g_L",
                "C_m",
                "v_rest",
                "Na_out",
                "Na_in",
                "K_out",
                "K_in",
                "Cl_out",
                "Cl_in",
                "Ca_out",
                "Ca_in",
                "T",
            }
            and isinstance(v, (int, float))
        }
        result = build_voltage_protocol(
            protocol_type=protocol_type,
            duration=duration,
            sampling_frequency=sampling_frequency,
            **kwargs,
        )

    assert _is_valid_protocol_list(result), (
        f"Preset '{preset_name}' produced an invalid protocol list"
    )
