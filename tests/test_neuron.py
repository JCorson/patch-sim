"""Tests for the core Neuron model functionality."""

import dataclasses

import pytest

from patch_sim.channels import IonChannel, IonSpecies
from patch_sim.core_channels import (
    make_k_channel,
    make_leak_channel,
    make_na_channel,
)
from patch_sim.neuron import Neuron


def test_initialization(hh_model: Neuron) -> None:
    """Test that the model is initialized with correct parameters."""
    assert hh_model.C_m == pytest.approx(1.0)
    assert hh_model.g_Na == pytest.approx(120.0)
    assert hh_model.g_K == pytest.approx(36.0)
    assert hh_model.g_L == pytest.approx(0.3)


def test_core_channels_structure(hh_model: Neuron) -> None:
    """core_channels contains Na, K, and leak IonChannels."""
    chs = hh_model.core_channels
    assert len(chs) == 3
    assert {ch.name for ch in chs} == {"Na", "K", "leak"}
    assert all(isinstance(ch, IonChannel) for ch in chs)


def test_core_channels_conductances(hh_model: Neuron) -> None:
    """core_channels channels carry the constructor g_max values."""
    chs = {ch.name: ch for ch in hh_model.core_channels}
    assert chs["Na"].g_max == pytest.approx(hh_model.g_Na)
    assert chs["K"].g_max == pytest.approx(hh_model.g_K)
    assert chs["leak"].g_max == pytest.approx(hh_model.g_L)


def test_all_channels_no_additional(hh_model: Neuron) -> None:
    """all_channels equals core_channels when there are no additional channels."""
    assert hh_model.all_channels == hh_model.core_channels


def test_all_channels_with_additional() -> None:
    """all_channels appends additional channels after the core three."""
    extra = make_na_channel(g_max=5.0)
    extra_named = dataclasses.replace(extra, name="NaExtra")
    neuron = Neuron(additional_channels=(extra_named,))
    assert len(neuron.all_channels) == 4
    assert neuron.all_channels[3].name == "NaExtra"


def test_all_gating_variables_no_additional(hh_model: Neuron) -> None:
    """all_gating_variables has exactly 3 variables for the default HH model."""
    gvs = hh_model.all_gating_variables
    names = [gv.name for gv in gvs]
    assert "m" in names
    assert "h" in names
    assert "n" in names
    assert len(gvs) == 3


def test_all_gating_variables_with_additional() -> None:
    """all_gating_variables includes gating vars from additional channels."""
    # Rename the gating variable to avoid name collision with the core K channel
    from patch_sim.channels import GatingVariable, NernstSpec
    from patch_sim.core_channels import alpha_n, beta_n

    gv_new = GatingVariable(
        name="kextra_activation", power=4, alpha=alpha_n, beta=beta_n
    )
    extra_ch2 = IonChannel(
        name="Kextra",
        g_max=1.0,
        gating_variables=(gv_new,),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )
    neuron = Neuron(additional_channels=(extra_ch2,))
    gvs = neuron.all_gating_variables
    names = [gv.name for gv in gvs]
    assert "kextra_activation" in names
    assert len(gvs) == 4  # 3 core + 1 extra


def test_custom_initialization() -> None:
    """Test that the model can be initialized with custom parameters."""
    custom_g_Na = 100.0
    custom_model = Neuron(g_Na=custom_g_Na)

    assert custom_model.g_Na == pytest.approx(custom_g_Na)

    # Other parameters should still have default values
    assert custom_model.C_m == pytest.approx(1.0)
    assert custom_model.g_K == pytest.approx(36.0)
    assert custom_model.g_L == pytest.approx(0.3)


def test_frozen_immutability(hh_model: Neuron) -> None:
    """Assigning to a frozen dataclass field must raise FrozenInstanceError."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        hh_model.g_Na = 999.0  # ty: ignore[invalid-assignment]


def test_reversal_potentials_from_core_channels(hh_model: Neuron) -> None:
    """Core channel reversal potentials match direct Nernst calculation."""
    from patch_sim.electrochemistry import nernst_potential

    na_ch, k_ch, leak_ch = hh_model.core_channels
    expected_E_Na = nernst_potential(1, hh_model.T, hh_model.Na_out, hh_model.Na_in)
    expected_E_K = nernst_potential(1, hh_model.T, hh_model.K_out, hh_model.K_in)
    expected_E_L = nernst_potential(-1, hh_model.T, hh_model.Cl_out, hh_model.Cl_in)

    assert na_ch.reversal_potential(hh_model) == pytest.approx(expected_E_Na)
    assert k_ch.reversal_potential(hh_model) == pytest.approx(expected_E_K)
    assert leak_ch.reversal_potential(hh_model) == pytest.approx(expected_E_L)


def test_reversal_potentials_in_physiological_range(hh_model: Neuron) -> None:
    """Core channel reversal potentials are in expected physiological ranges."""
    na_ch, k_ch, leak_ch = hh_model.core_channels
    assert 45.0 < na_ch.reversal_potential(hh_model) < 55.0
    assert -80.0 < k_ch.reversal_potential(hh_model) < -70.0
    assert -60.0 < leak_ch.reversal_potential(hh_model) < -50.0


def test_custom_ion_concentrations_shift_reversal_potentials() -> None:
    """Changing ion concentrations must produce shifted reversal potentials."""
    from patch_sim.electrochemistry import nernst_potential

    custom_model = Neuron(Na_out=200.0, K_in=100.0, T=293.15)
    default_model = Neuron()

    na_custom, k_custom, _ = custom_model.core_channels
    na_default, k_default, _ = default_model.core_channels

    # Higher extracellular Na+ → more positive E_Na
    assert na_custom.reversal_potential(custom_model) > na_default.reversal_potential(
        default_model
    )

    # Lower intracellular K+ (100 vs 140 mM) → K_out/K_in ratio is larger → more +ve E_K
    assert k_custom.reversal_potential(custom_model) > k_default.reversal_potential(
        default_model
    )

    # Values must agree with direct Nernst calculation
    assert na_custom.reversal_potential(custom_model) == pytest.approx(
        nernst_potential(1, custom_model.T, custom_model.Na_out, custom_model.Na_in)
    )
    assert k_custom.reversal_potential(custom_model) == pytest.approx(
        nernst_potential(1, custom_model.T, custom_model.K_out, custom_model.K_in)
    )


def test_calcium_reversal_potential() -> None:
    """Calcium channel reversal potential matches Nernst with z=2."""
    from patch_sim.channels import IonChannel, NernstSpec
    from patch_sim.electrochemistry import nernst_potential

    model = Neuron()
    ca_ch = IonChannel(
        name="CaTest",
        g_max=1.0,
        gating_variables=(),
        reversal_spec=NernstSpec(IonSpecies.CALCIUM),
    )
    expected = nernst_potential(2, model.T, model.Ca_out, model.Ca_in)
    assert ca_ch.reversal_potential(model) == pytest.approx(expected)
    assert ca_ch.reversal_potential(model) > 100.0


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("g_Na", [-1.0, -0.001])
def test_negative_g_Na_raises(g_Na: float) -> None:
    """Negative sodium conductance must raise ValueError."""
    with pytest.raises(ValueError, match="g_Na"):
        Neuron(g_Na=g_Na)


@pytest.mark.parametrize("g_K", [-1.0, -0.001])
def test_negative_g_K_raises(g_K: float) -> None:
    """Negative potassium conductance must raise ValueError."""
    with pytest.raises(ValueError, match="g_K"):
        Neuron(g_K=g_K)


@pytest.mark.parametrize("g_L", [-1.0, -0.001])
def test_negative_g_L_raises(g_L: float) -> None:
    """Negative leak conductance must raise ValueError."""
    with pytest.raises(ValueError, match="g_L"):
        Neuron(g_L=g_L)


@pytest.mark.parametrize("C_m", [0.0, -1.0])
def test_non_positive_capacitance_raises(C_m: float) -> None:
    """Non-positive membrane capacitance must raise ValueError."""
    with pytest.raises(ValueError, match="C_m"):
        Neuron(C_m=C_m)


@pytest.mark.parametrize("T", [0.0, -1.0])
def test_non_positive_temperature_raises(T: float) -> None:
    """Non-positive temperature must raise ValueError."""
    with pytest.raises(ValueError, match="Temperature"):
        Neuron(T=T)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"Na_out": 0.0},
        {"Na_out": -1.0},
        {"Na_in": 0.0},
        {"Na_in": -1.0},
        {"K_out": 0.0},
        {"K_out": -1.0},
        {"K_in": 0.0},
        {"K_in": -1.0},
        {"Cl_out": 0.0},
        {"Cl_out": -1.0},
        {"Cl_in": 0.0},
        {"Cl_in": -1.0},
        {"Ca_out": 0.0},
        {"Ca_out": -1.0},
        {"Ca_in": 0.0},
        {"Ca_in": -1.0},
    ],
)
def test_non_positive_ion_concentration_raises(kwargs: dict) -> None:
    """Non-positive ion concentration must raise ValueError."""
    with pytest.raises(ValueError, match="concentration"):
        Neuron(**kwargs)


# ---------------------------------------------------------------------------
# ion_concentrations tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "species, expected_out_attr, expected_in_attr",
    [
        (IonSpecies.SODIUM, "Na_out", "Na_in"),
        (IonSpecies.POTASSIUM, "K_out", "K_in"),
        (IonSpecies.CALCIUM, "Ca_out", "Ca_in"),
        (IonSpecies.CHLORIDE, "Cl_out", "Cl_in"),
    ],
)
def test_ion_concentrations_returns_correct_pair(
    hh_model: Neuron,
    species: IonSpecies,
    expected_out_attr: str,
    expected_in_attr: str,
) -> None:
    """ion_concentrations returns the correct (C_out, C_in) pair for each species."""
    c_out, c_in = hh_model.ion_concentrations(species)
    assert c_out == pytest.approx(getattr(hh_model, expected_out_attr))
    assert c_in == pytest.approx(getattr(hh_model, expected_in_attr))


def test_ion_concentrations_reflects_custom_values() -> None:
    """ion_concentrations must return user-supplied concentration values."""
    model = Neuron(Na_out=200.0, K_in=100.0, Ca_out=5.0, Cl_in=20.0)
    assert model.ion_concentrations(IonSpecies.SODIUM) == pytest.approx((200.0, 15.0))
    assert model.ion_concentrations(IonSpecies.POTASSIUM) == pytest.approx((7.8, 100.0))
    assert model.ion_concentrations(IonSpecies.CALCIUM) == pytest.approx((5.0, 0.0001))
    assert model.ion_concentrations(IonSpecies.CHLORIDE) == pytest.approx((120.0, 20.0))


# ---------------------------------------------------------------------------
# Core channel factory fields
# ---------------------------------------------------------------------------


def test_default_factories_are_hh52() -> None:
    """Default core channel factories are the HH52 squid axon functions."""
    model = Neuron()
    assert model.na_channel_factory is make_na_channel
    assert model.k_channel_factory is make_k_channel
    assert model.leak_channel_factory is make_leak_channel


def test_custom_na_factory_is_used() -> None:
    """A custom na_channel_factory is called when building core_channels."""
    calls: list[float] = []

    def recording_na_factory(g_max: float) -> IonChannel:
        """Na factory that records g_max and delegates to the default."""
        calls.append(g_max)
        return make_na_channel(g_max)

    model = Neuron(g_Na=99.0, na_channel_factory=recording_na_factory)
    _ = model.core_channels
    assert calls == [99.0]
    assert model.core_channels[0].name == "Na"
    assert model.core_channels[0].g_max == pytest.approx(99.0)


def test_custom_k_factory_is_used() -> None:
    """A custom k_channel_factory is called when building core_channels."""
    calls: list[float] = []

    def recording_k_factory(g_max: float) -> IonChannel:
        """K factory that records g_max and delegates to the default."""
        calls.append(g_max)
        return make_k_channel(g_max)

    model = Neuron(g_K=77.0, k_channel_factory=recording_k_factory)
    _ = model.core_channels
    assert calls == [77.0]
    assert model.core_channels[1].name == "K"
    assert model.core_channels[1].g_max == pytest.approx(77.0)


def test_custom_leak_factory_is_used() -> None:
    """A custom leak_channel_factory is called when building core_channels."""
    calls: list[float] = []

    def recording_leak_factory(g_max: float) -> IonChannel:
        """Leak factory that records g_max and delegates to the default."""
        calls.append(g_max)
        return make_leak_channel(g_max)

    model = Neuron(g_L=0.5, leak_channel_factory=recording_leak_factory)
    _ = model.core_channels
    assert calls == [0.5]
    assert model.core_channels[2].name == "leak"
    assert model.core_channels[2].g_max == pytest.approx(0.5)


def test_custom_factory_replaces_all_three_core_channels() -> None:
    """Replacing all three factories produces the expected channel names."""

    def alt_na(g_max: float) -> IonChannel:
        """Alternate Na factory returning HH52 channel."""
        return make_na_channel(g_max)

    def alt_k(g_max: float) -> IonChannel:
        """Alternate K factory returning HH52 channel."""
        return make_k_channel(g_max)

    def alt_leak(g_max: float) -> IonChannel:
        """Alternate leak factory returning HH52 channel."""
        return make_leak_channel(g_max)

    model = Neuron(
        na_channel_factory=alt_na,
        k_channel_factory=alt_k,
        leak_channel_factory=alt_leak,
    )
    assert [ch.name for ch in model.core_channels] == ["Na", "K", "leak"]


# ---------------------------------------------------------------------------
# Q10 temperature scaling
# ---------------------------------------------------------------------------


def test_q10_defaults() -> None:
    """Default Neuron has Q10=3.0 and T_ref=295.15 K."""
    model = Neuron()
    assert model.Q10 == pytest.approx(3.0)
    assert model.T_ref == pytest.approx(295.15)


def test_q10_factor_at_reference_temperature() -> None:
    """q10_factor is exactly 1.0 when T equals T_ref."""
    model = Neuron(T=295.15, T_ref=295.15)
    assert model.q10_factor == pytest.approx(1.0)


def test_q10_factor_ten_degrees_above_reference() -> None:
    """q10_factor equals Q10 when T is exactly 10 K above T_ref."""
    model = Neuron(T=305.15, T_ref=295.15, Q10=3.0)
    assert model.q10_factor == pytest.approx(3.0)


def test_q10_factor_ten_degrees_below_reference() -> None:
    """q10_factor is the reciprocal of Q10 when T is 10 K below T_ref."""
    model = Neuron(T=285.15, T_ref=295.15, Q10=3.0)
    assert model.q10_factor == pytest.approx(1.0 / 3.0)


def test_q10_of_one_gives_factor_one_regardless_of_temperature() -> None:
    """When Q10=1.0 the scaling factor is always 1.0."""
    model = Neuron(T=320.0, T_ref=295.15, Q10=1.0)
    assert model.q10_factor == pytest.approx(1.0)


@pytest.mark.parametrize("Q10", [0.0, -1.0])
def test_non_positive_q10_raises(Q10: float) -> None:
    """Non-positive Q10 must raise ValueError."""
    with pytest.raises(ValueError, match="Q10"):
        Neuron(Q10=Q10)


@pytest.mark.parametrize("T_ref", [0.0, -1.0])
def test_non_positive_t_ref_raises(T_ref: float) -> None:
    """Non-positive T_ref must raise ValueError."""
    with pytest.raises(ValueError, match="T_ref"):
        Neuron(T_ref=T_ref)
