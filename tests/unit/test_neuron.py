"""Tests for the core Neuron model functionality."""

import dataclasses

import pytest

from patch_sim.channels import (
    IonChannel,
    IonSpecies,
    NernstSpec,
    make_k_channel,
    make_na_channel,
)
from patch_sim.neuron import Neuron
from patch_sim.presets import make_squid_giant_axon


def test_initialization(hh_model: Neuron) -> None:
    """The HH52 fixture initializes with the canonical Na/K/leak conductances."""
    assert hh_model.C_m == pytest.approx(1.0)
    by_name = {ch.name: ch for ch in hh_model.channels}
    assert by_name["Na"].g_max == pytest.approx(120.0)
    assert by_name["K"].g_max == pytest.approx(36.0)
    assert by_name["NaL"].g_max == pytest.approx(0.054)
    assert by_name["KL"].g_max == pytest.approx(0.246)


def test_default_neuron_has_empty_channels() -> None:
    """A bare ``Neuron()`` carries no channels — presets supply the list."""
    assert Neuron().channels == ()


def test_channels_structure(hh_model: Neuron) -> None:
    """The squid preset's channels tuple holds 4 IonChannels named Na/K/NaL/KL."""
    chs = hh_model.channels
    assert len(chs) == 4
    assert {ch.name for ch in chs} == {"Na", "K", "NaL", "KL"}
    assert all(isinstance(ch, IonChannel) for ch in chs)


def test_channels_with_extra() -> None:
    """Channels list may carry any number of channels in declaration order."""
    extra = make_na_channel(g_max=5.0)
    extra_named = dataclasses.replace(extra, name="NaExtra")
    neuron = Neuron(
        channels=(
            make_na_channel(g_max=120.0),
            make_k_channel(g_max=36.0),
            extra_named,
        )
    )
    assert len(neuron.channels) == 3
    assert neuron.channels[2].name == "NaExtra"


def test_all_gating_variables_squid(hh_model: Neuron) -> None:
    """Squid HH52 has gating variables m, h, n (NaL/KL are leak)."""
    gvs = hh_model.all_gating_variables
    names = [gv.name for gv in gvs]
    assert "m" in names
    assert "h" in names
    assert "n" in names
    assert len(gvs) == 3


def test_all_gating_variables_extra() -> None:
    """all_gating_variables flattens gates from every channel in order."""
    from patch_sim.channels import GatingVariable, alpha_n, beta_n

    gv_new = GatingVariable(
        name="kextra_activation", power=4, alpha=alpha_n, beta=beta_n
    )
    extra_ch = IonChannel(
        name="Kextra",
        g_max=1.0,
        gating_variables=(gv_new,),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )
    neuron = Neuron(channels=(make_na_channel(g_max=120.0), extra_ch))
    gvs = neuron.all_gating_variables
    names = [gv.name for gv in gvs]
    assert "kextra_activation" in names
    assert names == ["m", "h", "kextra_activation"]


def test_duplicate_channel_names_raise() -> None:
    """Channels with duplicate names must raise ValueError."""
    ch1 = make_na_channel(g_max=120.0)
    ch2 = make_na_channel(g_max=80.0)
    with pytest.raises(ValueError, match="unique"):
        Neuron(channels=(ch1, ch2))


def test_frozen_immutability(hh_model: Neuron) -> None:
    """Assigning to a frozen dataclass field must raise FrozenInstanceError."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        hh_model.C_m = 999.0  # ty: ignore[invalid-assignment]


def test_reversal_potentials_from_channels(hh_model: Neuron) -> None:
    """Channel reversal potentials match direct Nernst calculation."""
    from patch_sim.electrochemistry import nernst_potential

    by_name = {ch.name: ch for ch in hh_model.channels}
    expected_E_Na = nernst_potential(1, hh_model.T, hh_model.Na_out, hh_model.Na_in)
    expected_E_K = nernst_potential(1, hh_model.T, hh_model.K_out, hh_model.K_in)

    assert by_name["Na"].reversal_potential(hh_model) == pytest.approx(expected_E_Na)
    assert by_name["K"].reversal_potential(hh_model) == pytest.approx(expected_E_K)
    assert by_name["NaL"].reversal_potential(hh_model) == pytest.approx(expected_E_Na)
    assert by_name["KL"].reversal_potential(hh_model) == pytest.approx(expected_E_K)


def test_reversal_potentials_in_physiological_range(hh_model: Neuron) -> None:
    """Channel reversal potentials are in expected physiological ranges."""
    by_name = {ch.name: ch for ch in hh_model.channels}
    assert 45.0 < by_name["Na"].reversal_potential(hh_model) < 55.0
    assert -80.0 < by_name["K"].reversal_potential(hh_model) < -70.0
    assert 45.0 < by_name["NaL"].reversal_potential(hh_model) < 55.0
    assert -80.0 < by_name["KL"].reversal_potential(hh_model) < -70.0


def test_custom_ion_concentrations_shift_reversal_potentials() -> None:
    """Changing ion concentrations must produce shifted reversal potentials."""
    from patch_sim.electrochemistry import nernst_potential

    custom_model = dataclasses.replace(
        make_squid_giant_axon(), Na_out=200.0, K_in=100.0, T=293.15
    )
    default_model = make_squid_giant_axon()

    by_custom = {ch.name: ch for ch in custom_model.channels}
    by_default = {ch.name: ch for ch in default_model.channels}

    assert by_custom["Na"].reversal_potential(custom_model) > by_default[
        "Na"
    ].reversal_potential(default_model)

    assert by_custom["K"].reversal_potential(custom_model) > by_default[
        "K"
    ].reversal_potential(default_model)

    assert by_custom["Na"].reversal_potential(custom_model) == pytest.approx(
        nernst_potential(1, custom_model.T, custom_model.Na_out, custom_model.Na_in)
    )
    assert by_custom["K"].reversal_potential(custom_model) == pytest.approx(
        nernst_potential(1, custom_model.T, custom_model.K_out, custom_model.K_in)
    )


def test_calcium_reversal_potential() -> None:
    """Calcium channel reversal potential matches Nernst with z=2."""
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
    model = Neuron(Na_out=200.0, K_in=100.0, Ca_out=5.0)
    assert model.ion_concentrations(IonSpecies.SODIUM) == pytest.approx((200.0, 15.0))
    assert model.ion_concentrations(IonSpecies.POTASSIUM) == pytest.approx((4.0, 100.0))
    assert model.ion_concentrations(IonSpecies.CALCIUM) == pytest.approx((5.0, 0.0001))


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


def test_default_area_cm2_is_none() -> None:
    """A default-constructed Neuron has ``area_cm2 is None``.

    Surface area is optional analysis-layer metadata; the ODE solver does
    not depend on it.
    """
    assert Neuron().area_cm2 is None


def test_custom_area_cm2_is_preserved() -> None:
    """A positive area_cm2 supplied at construction is preserved on the Neuron."""
    assert Neuron(area_cm2=20e-6).area_cm2 == pytest.approx(20e-6)


@pytest.mark.parametrize("area", [0.0, -1e-6])
def test_non_positive_area_cm2_raises(area: float) -> None:
    """Non-positive area_cm2 must raise ValueError when explicitly supplied."""
    with pytest.raises(ValueError, match="area_cm2"):
        Neuron(area_cm2=area)
