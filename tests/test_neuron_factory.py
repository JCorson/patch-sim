"""Tests for patch_sim.neuron_factory — NeuronConfig, make_neuron, and CalciumDynamics.

Covers automatic calcium-dynamics detection and custom configuration.
"""

import pytest

from patch_sim.additional_channels import make_ical_channel, make_icat_channel
from patch_sim.calcium import CalciumDynamics
from patch_sim.constants import (
    CA1_PYRAMIDAL,
    PURKINJE,
    STN,
    STOMATOGASTRIC_GANGLION,
    THALAMIC_RELAY,
    TRN,
)
from patch_sim.hodgkin_huxley import HodgkinHuxley
from patch_sim.neuron_factory import ChannelConfig, NeuronConfig, make_neuron
from patch_sim.presets import NEURON_PRESET_NAMES, NEURON_PRESETS

# ---------------------------------------------------------------------------
# Presets that include at least one calcium-carrying channel.
# ICaL, ICaT, and ICaN all use IonSpecies.CALCIUM.
# ---------------------------------------------------------------------------

_CALCIUM_PRESETS = {
    PURKINJE,
    THALAMIC_RELAY,
    CA1_PYRAMIDAL,
    STN,
    TRN,
    STOMATOGASTRIC_GANGLION,
}

_NON_CALCIUM_PRESETS = set(NEURON_PRESET_NAMES) - _CALCIUM_PRESETS


# ---------------------------------------------------------------------------
# make_neuron — each preset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", NEURON_PRESET_NAMES)
def test_make_neuron_each_preset(preset_name: str) -> None:
    """make_neuron returns a HodgkinHuxley instance for every built-in preset."""
    config = NEURON_PRESETS[preset_name]
    model = make_neuron(config)
    assert isinstance(model, HodgkinHuxley)


# ---------------------------------------------------------------------------
# Calcium auto-detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", sorted(_CALCIUM_PRESETS))
def test_make_neuron_calcium_preset_gets_calcium_dynamics(preset_name: str) -> None:
    """Presets with calcium channels receive a CalciumDynamics instance."""
    config = NEURON_PRESETS[preset_name]
    model = make_neuron(config)
    assert isinstance(model.calcium_dynamics, CalciumDynamics), (
        f"Preset '{preset_name}' should have CalciumDynamics but got None"
    )


@pytest.mark.parametrize("preset_name", sorted(_NON_CALCIUM_PRESETS))
def test_make_neuron_non_calcium_preset_no_calcium_dynamics(preset_name: str) -> None:
    """Presets without calcium channels have calcium_dynamics=None."""
    config = NEURON_PRESETS[preset_name]
    model = make_neuron(config)
    assert model.calcium_dynamics is None, (
        f"Preset '{preset_name}' should have no CalciumDynamics but got one"
    )


def test_make_neuron_single_calcium_channel_triggers_dynamics() -> None:
    """A NeuronConfig with just one calcium channel gets CalciumDynamics."""
    config = NeuronConfig(channels=(ChannelConfig(make_ical_channel, g_max=1.0),))
    model = make_neuron(config)
    assert isinstance(model.calcium_dynamics, CalciumDynamics)


def test_make_neuron_no_channels_no_calcium_dynamics() -> None:
    """A NeuronConfig with no additional channels has no CalciumDynamics."""
    config = NeuronConfig()
    model = make_neuron(config)
    assert model.calcium_dynamics is None


# ---------------------------------------------------------------------------
# Custom NeuronConfig
# ---------------------------------------------------------------------------


def test_make_neuron_custom_conductances() -> None:
    """Custom conductances from NeuronConfig are reflected on the HH model."""
    config = NeuronConfig(g_Na=150.0, g_K=50.0, g_L=0.2)
    model = make_neuron(config)
    assert isinstance(model, HodgkinHuxley)
    assert model.g_Na == pytest.approx(150.0)
    assert model.g_K == pytest.approx(50.0)
    assert model.g_L == pytest.approx(0.2)


def test_make_neuron_custom_concentrations() -> None:
    """Custom ion concentrations from NeuronConfig are reflected on the HH model."""
    config = NeuronConfig(Na_out=100.0, K_in=180.0)
    model = make_neuron(config)
    assert model.Na_out == pytest.approx(100.0)
    assert model.K_in == pytest.approx(180.0)


def test_make_neuron_additional_channels_attached() -> None:
    """Additional channels specified in NeuronConfig are attached to the model."""
    config = NeuronConfig(channels=(ChannelConfig(make_icat_channel, g_max=2.0),))
    model = make_neuron(config)
    assert len(model.additional_channels) == 1
