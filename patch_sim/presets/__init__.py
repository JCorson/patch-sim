"""Built-in preset configurations for the patch_sim core library.

Aggregates per-cell-type preset modules.  Each submodule owns a zero-arg
factory function that returns a fully-configured :class:`~patch_sim.Neuron`
plus a ``PROTOCOL_ADJUSTMENTS`` dict listing protocol-parameter overrides
specific to that cell type.  Protocol presets and the dispatcher live in
:mod:`patch_sim.presets.protocols`.
"""

from collections.abc import Callable
from typing import Any

from patch_sim.constants import (
    CA1_PYRAMIDAL,
    CORTICAL_PYRAMIDAL,
    DOPAMINERGIC,
    FAST_SPIKING_INTERNEURON,
    PURKINJE,
    SQUID_GIANT_AXON,
    STN,
    THALAMIC_RELAY,
    TRN,
)
from patch_sim.neuron import Neuron

from . import (
    ca1,
    cortical,
    dopaminergic,
    fast_spiking,
    purkinje,
    squid,
    stn,
    thalamic,
    trn,
)
from .ca1 import make_ca1_pyramidal
from .cortical import make_cortical_pyramidal
from .dopaminergic import make_dopaminergic
from .fast_spiking import make_fast_spiking_interneuron
from .protocols import (
    PROTOCOL_PRESET_NAMES,
    PROTOCOL_PRESETS,
    build_protocol_from_preset,
)
from .purkinje import make_purkinje
from .squid import make_squid_giant_axon
from .stn import make_stn
from .thalamic import make_thalamic_relay
from .trn import make_trn

NEURON_PRESETS: dict[str, Callable[[], Neuron]] = {
    SQUID_GIANT_AXON: make_squid_giant_axon,
    FAST_SPIKING_INTERNEURON: make_fast_spiking_interneuron,
    CORTICAL_PYRAMIDAL: make_cortical_pyramidal,
    PURKINJE: make_purkinje,
    DOPAMINERGIC: make_dopaminergic,
    THALAMIC_RELAY: make_thalamic_relay,
    CA1_PYRAMIDAL: make_ca1_pyramidal,
    STN: make_stn,
    TRN: make_trn,
}

NEURON_PROTOCOL_ADJUSTMENTS: dict[str, dict[str, dict[str, Any]]] = {
    SQUID_GIANT_AXON: squid.PROTOCOL_ADJUSTMENTS,
    FAST_SPIKING_INTERNEURON: fast_spiking.PROTOCOL_ADJUSTMENTS,
    CORTICAL_PYRAMIDAL: cortical.PROTOCOL_ADJUSTMENTS,
    PURKINJE: purkinje.PROTOCOL_ADJUSTMENTS,
    DOPAMINERGIC: dopaminergic.PROTOCOL_ADJUSTMENTS,
    THALAMIC_RELAY: thalamic.PROTOCOL_ADJUSTMENTS,
    CA1_PYRAMIDAL: ca1.PROTOCOL_ADJUSTMENTS,
    STN: stn.PROTOCOL_ADJUSTMENTS,
    TRN: trn.PROTOCOL_ADJUSTMENTS,
}

NEURON_PRESET_NAMES: list[str] = list(NEURON_PRESETS.keys())

__all__ = [
    "NEURON_PRESETS",
    "NEURON_PRESET_NAMES",
    "NEURON_PROTOCOL_ADJUSTMENTS",
    "PROTOCOL_PRESETS",
    "PROTOCOL_PRESET_NAMES",
    "build_protocol_from_preset",
    "make_ca1_pyramidal",
    "make_cortical_pyramidal",
    "make_dopaminergic",
    "make_fast_spiking_interneuron",
    "make_purkinje",
    "make_squid_giant_axon",
    "make_stn",
    "make_thalamic_relay",
    "make_trn",
]
