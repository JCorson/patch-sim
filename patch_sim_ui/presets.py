"""Built-in preset configurations for the patch_sim web UI.

Only the UI-specific neuron preset format (mapping state variable names to
values) is defined here.  All protocol presets, adjustments, and name lists
live in the core library and should be imported from ``patch_sim.presets``
directly.
"""

from collections.abc import Callable
from dataclasses import Field
from dataclasses import fields as dc_fields
from typing import Any

from patch_sim.channels import IonChannel, IonSpecies, NernstSpec
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
from patch_sim_ui.channels import CHANNELS

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

# Map from IonChannel.name to ChannelMeta.id for every channel in the registry.
#
# Channels whose current_key starts with "I" (the convention for *current*
# names like "Ih", "IKv31", "ICaT") have an IonChannel.name equal to the key
# with the leading "I" stripped (e.g. name="h" for current_key="Ih").
# Channels whose current_key does not start with "I" (e.g. "Na", "K", "NaL",
# "KL", "Kv", "Cav1.3", "SK") have an IonChannel.name equal to the key
# directly.  Variant channels with a divergent name (e.g. SNc INaP -> "NaP_SNc")
# are mapped explicitly so they collapse onto the canonical UI toggle.
CHANNEL_NAME_TO_ID: dict[str, str] = {}
for _ch_meta in CHANNELS:
    if _ch_meta.current_key.startswith("I") and _ch_meta.current_key != "I":
        CHANNEL_NAME_TO_ID[_ch_meta.current_key[1:]] = _ch_meta.id
    else:
        CHANNEL_NAME_TO_ID[_ch_meta.current_key] = _ch_meta.id
CHANNEL_NAME_TO_ID["NaP_SNc"] = "inap"

#: Default conductance for each channel's UI slider, keyed by channel UI id.
#:
#: HH-classic core defaults are pulled from the canonical Squid Giant Axon
#: ``Neuron`` object; auxiliary channel defaults come from the core library's
#: per-channel ``DEFAULT_G_*`` constants (their authoritative home).  Used
#: both to seed ``NeuronState`` g_max vars at module load and to initialise
#: every slider in :func:`neuron_to_ui_state` before per-preset overrides
#: are applied.
_SQUID_G_MAX: dict[str, float] = {
    CHANNEL_NAME_TO_ID[ch.name]: ch.g_max
    for ch in NEURON_PRESETS[SQUID_GIANT_AXON]().channels
}
DEFAULT_G_MAX: dict[str, float] = _SQUID_G_MAX | {
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


def neuron_to_ui_state(neuron: Neuron) -> dict[str, Any]:
    """Convert a :class:`~patch_sim.Neuron` to a flat NeuronState dict.

    Produces a mapping whose keys exactly match ``NeuronState`` field names so
    that it can be unpacked with ``setattr`` in ``load_neuron_preset``.

    For each registered channel, the corresponding ``{id}_g_max`` key is set
    to the channel's preset-tuned ``g_max`` if present on *neuron*, otherwise
    the registry default in :data:`DEFAULT_G_MAX`.  Visibility (which slider
    appears in the panel) is driven separately by :data:`PRESET_CHANNEL_IDS`,
    so the default fallback only matters when the user later switches to a
    preset that does include the channel.

    Args:
        neuron: Core neuron instance to convert.

    Returns:
        Flat dict of ``{field_name: value}`` pairs for ``NeuronState``.
    """
    state: dict[str, Any] = {
        name: getattr(neuron, name) for name in NEURON_CONFIG_SCALAR_FIELDS
    }

    # Initialise every channel slider to its registry default.
    for ch_meta in CHANNELS:
        state[ch_meta.g_max_field] = DEFAULT_G_MAX[ch_meta.id]

    # Override with preset-tuned values for channels present on the Neuron.
    # Variant factories (e.g. make_snc_inap_channel) produce channels with
    # names that resolve via CHANNEL_NAME_TO_ID, so the lookup collapses
    # variants onto the same UI slider automatically.
    for ch in neuron.channels:
        ui_id = CHANNEL_NAME_TO_ID.get(ch.name)
        if ui_id is not None:
            state[f"{ui_id}_g_max"] = ch.g_max

    return state


# Flat UI-state dicts derived from core Neuron presets.
NEURON_UI_PRESETS: dict[str, dict[str, Any]] = {
    name: neuron_to_ui_state(factory()) for name, factory in NEURON_PRESETS.items()
}

#: Maps preset name -> set of UI channel ids the preset's factory produces.
#: Single source of truth for which channel rows render in the neuron panel
#: and which trace-visibility checkboxes appear in the sweep manager.
#: Computed once at import time.  Unknown preset names default to an empty
#: frozenset so the UI can render zero rows safely.
PRESET_CHANNEL_IDS: dict[str, frozenset[str]] = {
    name: frozenset(
        ui_id
        for ch in factory().channels
        if (ui_id := CHANNEL_NAME_TO_ID.get(ch.name)) is not None
    )
    for name, factory in NEURON_PRESETS.items()
}

#: Default neuron preset applied on app startup and reset.
DEFAULT_NEURON_PRESET: str = SQUID_GIANT_AXON

#: Default protocol preset applied on app startup and reset.
DEFAULT_PROTOCOL_PRESET: str = ACTION_POTENTIAL


def _is_sodium_carrying(channel: IonChannel) -> bool:
    """Return True if ``channel`` carries Na⁺ ions through a Nernst reversal.

    Used by the Na+ Channel Activation protocol override to decide which
    sliders to leave alone (Na, NaL, INaP, INaR, INaP_SNc) versus zero
    (every K-, Ca-, or Goldman-mixed-cation channel).

    Args:
        channel: An ion channel from a Neuron preset.

    Returns:
        ``True`` iff ``channel.reversal_spec`` is a ``NernstSpec`` with
        species ``IonSpecies.SODIUM``.  ``GoldmanSpec`` channels (Ih)
        return ``False`` even though they pass some Na⁺.
    """
    spec = channel.reversal_spec
    return isinstance(spec, NernstSpec) and spec.species is IonSpecies.SODIUM


def _na_channel_activation_override(neuron_preset_name: str) -> dict[str, Any]:
    """Build the Na+ Channel Activation slider override for ``neuron_preset_name``.

    Walks the active preset's channels and emits ``{"<ui_id>_g_max": 0.0}``
    for every channel that does NOT carry sodium.  Channels not in
    :data:`CHANNEL_NAME_TO_ID` (i.e. with no UI slider) are skipped.

    Args:
        neuron_preset_name: Key into :data:`patch_sim.presets.NEURON_PRESETS`.
            Unknown names produce an empty dict (no overrides applied).

    Returns:
        Mapping of NeuronState slider field name to ``0.0``.  Only
        non-Na-carrying channels are included; Na/NaL/NaP/NaR sliders
        keep their current values.
    """
    factory = NEURON_PRESETS.get(neuron_preset_name)
    if factory is None:
        return {}
    overrides: dict[str, Any] = {}
    for ch in factory().channels:
        if _is_sodium_carrying(ch):
            continue
        ui_id = CHANNEL_NAME_TO_ID.get(ch.name)
        if ui_id is not None:
            overrides[f"{ui_id}_g_max"] = 0.0
    return overrides


# Neuron-parameter overrides applied when a protocol preset is loaded,
# regardless of which neuron type is active.  Each value is either:
#   - a static ``dict[str, Any]`` of NeuronState field name → value, or
#   - a callable ``(neuron_preset_name: str) -> dict[str, Any]`` that
#     computes the override based on which channels the active preset has.
# The callable form lets the Na+ Channel Activation override adapt to
# whichever delayed rectifier each preset uses (HH-style "K", Mainen-Sejnowski
# "Kv", IKv3.1, etc.) instead of hard-coding a single slider name.
PROTOCOL_NEURON_OVERRIDES: dict[
    str, dict[str, Any] | Callable[[str], dict[str, Any]]
] = {
    # Zero every non-Na-carrying channel so only the Na⁺ current contributes
    # to the voltage-clamp trace — the textbook TEA/4-AP/Cd²⁺/Cs⁺ cocktail.
    NA_CHANNEL_ACTIVATION: _na_channel_activation_override,
}
