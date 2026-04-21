"""Tests for the reversal-potential cache helper in clamp_simulations."""

import pytest

from patch_sim.additional_channels import make_ical_channel, make_ikca_channel
from patch_sim.calcium import CalciumDynamics
from patch_sim.clamp_simulations import _compute_reversal_potentials
from patch_sim.neuron import Neuron


def test_compute_reversal_potentials_covers_all_channels() -> None:
    """_compute_reversal_potentials keys match all channels; values match live calls."""
    neuron = Neuron(
        additional_channels=(make_ical_channel(), make_ikca_channel()),
        calcium_dynamics=CalciumDynamics(),
    )
    cache = _compute_reversal_potentials(neuron)

    assert set(cache.keys()) == {ch.name for ch in neuron.all_channels}
    for ch in neuron.all_channels:
        assert cache[ch.name] == pytest.approx(ch.reversal_potential(neuron))


def test_compute_reversal_potentials_hh_only() -> None:
    """_compute_reversal_potentials works for a plain HH neuron with no extra channels.

    Verifies that the helper handles the minimal case (core channels only) and
    that every returned value matches the per-channel live computation.
    """
    neuron = Neuron()
    cache = _compute_reversal_potentials(neuron)

    assert set(cache.keys()) == {ch.name for ch in neuron.all_channels}
    for ch in neuron.all_channels:
        assert cache[ch.name] == pytest.approx(ch.reversal_potential(neuron))
