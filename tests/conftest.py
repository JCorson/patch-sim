"""Shared pytest fixtures for the ap-sim test suite."""

import pytest

from patch_sim.neuron import Neuron


@pytest.fixture
def hh_model() -> Neuron:
    """HH52 squid giant axon neuron instance for all tests.

    Uses K_out=7.8 mM (HH52 seawater value) explicitly so that tests written
    against the classic squid axon remain correct even though DEFAULT_K_OUT
    is 4.0 mM (physiological mammalian ACSF).
    """
    return Neuron(K_out=7.8)
