"""Shared pytest fixtures for the ap-sim test suite."""

import pytest
from patch_sim.hodgkin_huxley import HodgkinHuxley


@pytest.fixture
def hh_model() -> HodgkinHuxley:
    """Default Hodgkin-Huxley neuron instance for all tests."""
    return HodgkinHuxley()
