"""Ion channel subpackage for the conductance-based neuron simulator.

Re-exports the base channel types and all factory functions so external callers
can import from ``patch_sim.channels`` (or ``patch_sim``) without knowing the
internal submodule layout.
"""

from .base import (
    _CA_NERNST_FLOOR,
    GatingVariable,
    GoldmanSpec,
    IonChannel,
    IonSpecies,
    NernstSpec,
    ReversalSpec,
)

__all__ = [
    "GatingVariable",
    "GoldmanSpec",
    "IonChannel",
    "IonSpecies",
    "NernstSpec",
    "ReversalSpec",
    "_CA_NERNST_FLOOR",
]
