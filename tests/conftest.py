"""Shared pytest fixtures for the ap-sim test suite."""

import pytest

from patch_sim.neuron import Neuron
from patch_sim.presets import make_squid_giant_axon


@pytest.fixture
def hh_model() -> Neuron:
    """HH52 squid giant axon neuron instance for all tests.

    Returns the canonical squid preset so tests written against the classic
    axon (K_out=7.8 mM, Q10=1.0, full HH52 channel set) remain correct even
    though ``Neuron()`` itself has empty channels by default.
    """
    return make_squid_giant_axon()
