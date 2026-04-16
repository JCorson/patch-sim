"""Tests for patch_sim.neuron_factory — NeuronConfig, make_neuron, and CalciumDynamics.

Covers automatic calcium-dynamics detection and custom configuration.
"""

import pytest

from patch_sim.additional_channels import make_ical_channel, make_icat_channel
from patch_sim.calcium import CalciumDynamics
from patch_sim.channels import IonChannel
from patch_sim.constants import (
    CA1_PYRAMIDAL,
    PURKINJE,
    STN,
    THALAMIC_RELAY,
    TRN,
)
from patch_sim.neuron import Neuron
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
}

_NON_CALCIUM_PRESETS = set(NEURON_PRESET_NAMES) - _CALCIUM_PRESETS


# ---------------------------------------------------------------------------
# make_neuron — each preset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", NEURON_PRESET_NAMES)
def test_make_neuron_each_preset(preset_name: str) -> None:
    """make_neuron returns a Neuron instance for every built-in preset."""
    config = NEURON_PRESETS[preset_name]
    model = make_neuron(config)
    assert isinstance(model, Neuron)


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
    assert isinstance(model, Neuron)
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


# ---------------------------------------------------------------------------
# Core channel factory pass-through
# ---------------------------------------------------------------------------


def test_make_neuron_passes_na_factory() -> None:
    """make_neuron forwards na_channel_factory from NeuronConfig to Neuron."""
    from patch_sim.core_channels import make_na_channel

    calls: list[float] = []

    def recording_factory(g_max: float) -> IonChannel:
        """Na factory that records invocations."""
        calls.append(g_max)
        return make_na_channel(g_max)

    config = NeuronConfig(g_Na=88.0, na_channel_factory=recording_factory)
    model = make_neuron(config)
    assert model.na_channel_factory is recording_factory
    _ = model.core_channels
    assert calls == [88.0]


def test_make_neuron_passes_k_factory() -> None:
    """make_neuron forwards k_channel_factory from NeuronConfig to Neuron."""
    from patch_sim.core_channels import make_k_channel

    def alt_k(g_max: float) -> IonChannel:
        """Alternate K factory."""
        return make_k_channel(g_max)

    config = NeuronConfig(k_channel_factory=alt_k)
    model = make_neuron(config)
    assert model.k_channel_factory is alt_k


def test_make_neuron_passes_leak_factory() -> None:
    """make_neuron forwards leak_channel_factory from NeuronConfig to Neuron."""
    from patch_sim.core_channels import make_leak_channel

    def alt_leak(g_max: float) -> IonChannel:
        """Alternate leak factory."""
        return make_leak_channel(g_max)

    config = NeuronConfig(leak_channel_factory=alt_leak)
    model = make_neuron(config)
    assert model.leak_channel_factory is alt_leak


def test_neuron_config_default_factories_are_hh52() -> None:
    """NeuronConfig default factories are the HH52 squid axon functions."""
    from patch_sim.core_channels import (
        make_k_channel,
        make_leak_channel,
        make_na_channel,
    )

    config = NeuronConfig()
    assert config.na_channel_factory is make_na_channel
    assert config.k_channel_factory is make_k_channel
    assert config.leak_channel_factory is make_leak_channel


# ---------------------------------------------------------------------------
# Q10 pass-through
# ---------------------------------------------------------------------------


def test_make_neuron_passes_q10_and_t_ref() -> None:
    """make_neuron forwards Q10 and T_ref from NeuronConfig to Neuron."""
    config = NeuronConfig(Q10=2.0, T_ref=300.0)
    model = make_neuron(config)
    assert model.Q10 == pytest.approx(2.0)
    assert model.T_ref == pytest.approx(300.0)


@pytest.mark.parametrize("Q10", [0.0, -1.0])
def test_neuron_config_non_positive_q10_raises(Q10: float) -> None:
    """NeuronConfig rejects non-positive Q10 at construction time."""
    with pytest.raises(ValueError, match="Q10"):
        NeuronConfig(Q10=Q10)


@pytest.mark.parametrize("T_ref", [0.0, -1.0])
def test_neuron_config_non_positive_t_ref_raises(T_ref: float) -> None:
    """NeuronConfig rejects non-positive T_ref at construction time."""
    with pytest.raises(ValueError, match="T_ref"):
        NeuronConfig(T_ref=T_ref)
