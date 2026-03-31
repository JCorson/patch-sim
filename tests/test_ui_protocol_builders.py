"""Verify that patch_sim_ui.presets still re-exports core symbols via the shim.

These tests guard against accidental breakage of the UI shim layer: all
symbols that were once defined directly in patch_sim_ui.presets must
continue to be importable from there (they are now re-exported from the
core library).
"""

from patch_sim_ui.presets import (
    NEURON_PRESET_NAMES,
    NEURON_PRESETS,
    NEURON_PROTOCOL_ADJUSTMENTS,
    PROTOCOL_PRESET_NAMES,
    PROTOCOL_PRESETS,
)
from patch_sim.presets import (
    NEURON_PRESET_NAMES as _CORE_NEURON_PRESET_NAMES,
    PROTOCOL_PRESETS as _CORE_PROTOCOL_PRESETS,
)


def test_protocol_presets_shim_equals_core() -> None:
    """PROTOCOL_PRESETS imported from the UI shim is the same object as the core one."""
    assert PROTOCOL_PRESETS is _CORE_PROTOCOL_PRESETS


def test_neuron_preset_names_shim_equals_core() -> None:
    """NEURON_PRESET_NAMES imported from the UI shim equals the core list."""
    assert NEURON_PRESET_NAMES == _CORE_NEURON_PRESET_NAMES


def test_protocol_preset_names_shim_non_empty() -> None:
    """PROTOCOL_PRESET_NAMES from the shim is a non-empty list of strings."""
    assert isinstance(PROTOCOL_PRESET_NAMES, list)
    assert len(PROTOCOL_PRESET_NAMES) > 0
    assert all(isinstance(n, str) for n in PROTOCOL_PRESET_NAMES)


def test_neuron_presets_shim_non_empty() -> None:
    """NEURON_PRESETS from the shim is a non-empty dict."""
    assert isinstance(NEURON_PRESETS, dict)
    assert len(NEURON_PRESETS) > 0


def test_neuron_protocol_adjustments_shim_non_empty() -> None:
    """NEURON_PROTOCOL_ADJUSTMENTS from the shim is a non-empty dict."""
    assert isinstance(NEURON_PROTOCOL_ADJUSTMENTS, dict)
    assert len(NEURON_PROTOCOL_ADJUSTMENTS) > 0


def test_shim_neuron_presets_are_ui_state_dicts() -> None:
    """NEURON_PRESETS from the shim contains flat UI-state dicts.

    The UI shim converts NeuronConfig presets to flat AppState dicts via
    neuron_config_to_ui_state, so each value must be a plain dict.
    """
    for name, state in NEURON_PRESETS.items():
        assert isinstance(state, dict), (
            f"UI preset '{name}' should be a dict, got {type(state)}"
        )
        assert "g_Na" in state, f"UI preset '{name}' missing 'g_Na'"
        assert "g_K" in state, f"UI preset '{name}' missing 'g_K'"
