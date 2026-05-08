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
    DEFAULT_G_CAV13,
    DEFAULT_G_ICAL,
    DEFAULT_G_ICAN,
    DEFAULT_G_ICAT,
    DEFAULT_G_IH,
    DEFAULT_G_IKA,
    DEFAULT_G_IKCA,
    DEFAULT_G_IKIR,
    DEFAULT_G_IKV31,
    DEFAULT_G_IM,
    DEFAULT_G_KATP,
    DEFAULT_G_MSKV,
    DEFAULT_G_NAP,
    DEFAULT_G_NAR,
    DEFAULT_G_SK,
    NA_CHANNEL_ACTIVATION,
    SQUID_GIANT_AXON,
)
from patch_sim.neuron import Neuron
from patch_sim.presets import NEURON_PRESETS
from patch_sim_ui.channels import ADDITIONAL_CHANNELS

# Enumerate Neuron scalar fields once; derived constants reuse this tuple.
# Neuron does not use ``from __future__ import annotations``, so f.type is
# the actual float class rather than the string "float".
_NEURON_CONFIG_SCALAR_META: tuple[Field[Any], ...] = tuple(
    f for f in dc_fields(Neuron) if f.type is float
)

#: Ordered tuple of Neuron scalar field names — the single source of
#: truth that drives NeuronState field declarations, neuron_to_ui_state,
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
    "mskv": DEFAULT_G_MSKV,
    "inap": DEFAULT_G_NAP,
    "inar": DEFAULT_G_NAR,
    "im": DEFAULT_G_IM,
    "katp": DEFAULT_G_KATP,
    "ikir": DEFAULT_G_IKIR,
    "ikca": DEFAULT_G_IKCA,
    "ical": DEFAULT_G_ICAL,
    "cav13": DEFAULT_G_CAV13,
    "icat": DEFAULT_G_ICAT,
    "ican": DEFAULT_G_ICAN,
    "sk": DEFAULT_G_SK,
}

# Map from IonChannel.name to ChannelMeta.id for all channels used by presets.
#
# Channels whose current_key starts with "I" (the convention for *current*
# names like "Ih", "IKv31", "ICaT") have an IonChannel.name equal to the key
# with the leading "I" stripped (e.g. name="h" for current_key="Ih").
# Channels whose current_key does not start with "I" (e.g. "Kv", "Cav1.3",
# "SK") have an IonChannel.name equal to the key directly.
# Variant channels with a divergent name (e.g. SNc INaP -> "NaP_SNc") are
# mapped explicitly so they collapse onto the canonical UI toggle.
CHANNEL_NAME_TO_ID: dict[str, str] = {}
for _ch_meta in ADDITIONAL_CHANNELS:
    if _ch_meta.current_key.startswith("I"):
        CHANNEL_NAME_TO_ID[_ch_meta.current_key[1:]] = _ch_meta.id
    else:
        CHANNEL_NAME_TO_ID[_ch_meta.current_key] = _ch_meta.id
CHANNEL_NAME_TO_ID["NaP_SNc"] = "inap"


def neuron_to_ui_state(neuron: Neuron) -> dict[str, Any]:
    """Convert a :class:`~patch_sim.Neuron` to a flat NeuronState dict.

    Produces a mapping whose keys exactly match ``NeuronState`` field names so
    that it can be unpacked with ``setattr`` in ``load_neuron_preset``.

    All auxiliary channels that are absent from *neuron.additional_channels*
    are set to disabled with their default maximum conductances, so that
    loading a preset never leaves channels enabled that were set by a previous
    preset.

    Args:
        neuron: Core neuron instance to convert.

    Returns:
        Flat dict of ``{field_name: value}`` pairs for ``NeuronState``.
    """
    state: dict[str, Any] = {
        name: getattr(neuron, name) for name in NEURON_CONFIG_SCALAR_FIELDS
    }

    # Disable all auxiliary channels with default conductances.
    for ch_meta in ADDITIONAL_CHANNELS:
        state[ch_meta.enabled_field] = False
        state[ch_meta.g_max_field] = _DEFAULT_G_MAX[ch_meta.id]

    # Enable channels present on the Neuron instance.  Variant factories
    # (e.g. make_snc_inap_channel) produce channels with names that are
    # resolved via CHANNEL_NAME_TO_ID so the lookup collapses variants onto
    # the same UI toggle automatically.
    for ch in neuron.additional_channels:
        ui_id = CHANNEL_NAME_TO_ID.get(ch.name)
        if ui_id is not None:
            state[f"{ui_id}_enabled"] = True
            state[f"{ui_id}_g_max"] = ch.g_max

    return state


# Flat UI-state dicts derived from core Neuron presets.
NEURON_UI_PRESETS: dict[str, dict[str, Any]] = {
    name: neuron_to_ui_state(factory()) for name, factory in NEURON_PRESETS.items()
}

#: Maps preset name -> set of UI channel ids the preset's factory produces.
#: Single source of truth for which auxiliary-channel rows render in the
#: neuron panel and which trace-visibility checkboxes appear in the sweep
#: manager.  Computed once at import time.  Unknown preset names default to
#: an empty frozenset so the UI can render zero rows safely.
PRESET_CHANNEL_IDS: dict[str, frozenset[str]] = {
    name: frozenset(
        ui_id
        for ch in factory().additional_channels
        if (ui_id := CHANNEL_NAME_TO_ID.get(ch.name)) is not None
    )
    for name, factory in NEURON_PRESETS.items()
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
