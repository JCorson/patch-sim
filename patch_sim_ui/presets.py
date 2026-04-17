"""Built-in preset configurations for the patch_sim web UI.

Only the UI-specific neuron preset format (mapping state variable names to
values) is defined here.  All protocol presets, adjustments, and name lists
live in the core library and should be imported from ``patch_sim.presets``
directly.
"""

from dataclasses import Field
from dataclasses import fields as dc_fields
from typing import Any

from patch_sim.constants import (
    ACTION_POTENTIAL,
    DEFAULT_G_ICAL,
    DEFAULT_G_ICAN,
    DEFAULT_G_ICAT,
    DEFAULT_G_IH,
    DEFAULT_G_IKA,
    DEFAULT_G_IKCA,
    DEFAULT_G_IKIR,
    DEFAULT_G_IKV31,
    DEFAULT_G_IM,
    DEFAULT_G_NAP,
    DEFAULT_G_NAR,
    NA_CHANNEL_ACTIVATION,
    SQUID_GIANT_AXON,
)
from patch_sim.neuron_factory import CHANNEL_REGISTRY, NeuronConfig
from patch_sim.presets import NEURON_PRESETS

# Enumerate NeuronConfig scalar fields once; derived constants reuse this tuple.
_NEURON_CONFIG_SCALAR_META: tuple[Field[Any], ...] = tuple(
    f for f in dc_fields(NeuronConfig) if f.type == "float"
)

#: Ordered tuple of NeuronConfig scalar field names — the single source of
#: truth that drives NeuronState field declarations, neuron_config_to_ui_state,
#: and _build_neuron kwargs.
NEURON_CONFIG_SCALAR_FIELDS: tuple[str, ...] = tuple(
    f.name for f in _NEURON_CONFIG_SCALAR_META
)

#: Default float values for each scalar field, keyed by field name.
NEURON_CONFIG_SCALAR_DEFAULTS: dict[str, float] = {
    f.name: f.default
    for f in _NEURON_CONFIG_SCALAR_META
    if isinstance(f.default, float)
}

# Default conductances for each auxiliary channel, keyed by channel name.
_DEFAULT_G_MAX: dict[str, float] = {
    "ih": DEFAULT_G_IH,
    "ika": DEFAULT_G_IKA,
    "ikv31": DEFAULT_G_IKV31,
    "inap": DEFAULT_G_NAP,
    "inar": DEFAULT_G_NAR,
    "im": DEFAULT_G_IM,
    "ikir": DEFAULT_G_IKIR,
    "ikca": DEFAULT_G_IKCA,
    "ical": DEFAULT_G_ICAL,
    "icat": DEFAULT_G_ICAT,
    "ican": DEFAULT_G_ICAN,
}

# Reverse map: factory function → channel name.
_FACTORY_TO_NAME: dict[Any, str] = {v: k for k, v in CHANNEL_REGISTRY.items()}


def neuron_config_to_ui_state(config: NeuronConfig) -> dict[str, Any]:
    """Convert a :class:`~patch_sim.NeuronConfig` to a flat NeuronState dict.

    Produces a mapping whose keys exactly match ``NeuronState`` field names so
    that it can be unpacked with ``setattr`` in ``load_neuron_preset``.

    All auxiliary channels that are absent from *config.channels* are set to
    disabled with their default maximum conductances, so that loading a preset
    never leaves channels enabled that were set by a previous preset.

    Args:
        config: Core neuron configuration to convert.

    Returns:
        Flat dict of ``{field_name: value}`` pairs for ``NeuronState``.
    """
    state: dict[str, Any] = {
        name: getattr(config, name) for name in NEURON_CONFIG_SCALAR_FIELDS
    }

    # Disable all auxiliary channels with default conductances.
    for name in CHANNEL_REGISTRY:
        state[f"{name}_enabled"] = False
        state[f"{name}_g_max"] = _DEFAULT_G_MAX[name]

    # Enable channels present in config.
    for cc in config.channels:
        name = _FACTORY_TO_NAME[cc.factory]
        state[f"{name}_enabled"] = True
        state[f"{name}_g_max"] = cc.g_max

    return state


# Flat UI-state dicts derived from core NeuronConfig presets.
NEURON_UI_PRESETS: dict[str, dict[str, Any]] = {
    name: neuron_config_to_ui_state(cfg) for name, cfg in NEURON_PRESETS.items()
}

#: Default neuron preset applied on app startup and reset.
DEFAULT_NEURON_PRESET: str = SQUID_GIANT_AXON

#: Default protocol preset applied on app startup and reset.
DEFAULT_PROTOCOL_PRESET: str = ACTION_POTENTIAL

# Neuron-parameter overrides applied when a protocol preset is loaded,
# regardless of which neuron type is active.  Keys must match NeuronState field
# names exactly.
PROTOCOL_NEURON_OVERRIDES: dict[str, dict[str, Any]] = {
    # Disable K⁺ channels so only Na⁺ current is visible.
    NA_CHANNEL_ACTIVATION: {"g_K": 0.0},
}
