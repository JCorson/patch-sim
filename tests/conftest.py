"""Shared pytest fixtures for the ap-sim test suite."""

import pytest

from patch_sim.neuron import Neuron


@pytest.fixture
def hh_model() -> Neuron:
    """Default Hodgkin-Huxley neuron instance for all tests."""
    return Neuron()
