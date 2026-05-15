"""Tests for the ion channel framework.

Covers IonChannel math, GatingVariable steady states, Ih kinetics,
backward compatibility with no additional channels, and validation errors.
The trailing "Core (HH-style) channels" section also covers HH52 rate
functions and the per-preset Na/K factory variants (Pospischil, Nav1.1,
Nav1.2, Mainen-Sejnowski, STN, Purkinje, Dopaminergic).
"""

import dataclasses
import math
import pickle
from typing import Any

import numpy as np
import pytest

from patch_sim.calcium import CalciumDynamics
from patch_sim.channels import (
    DOPAMINERGIC_VT,
    MAINEN_SEJNOWSKI_KV_PRESCALE,
    MAINEN_SEJNOWSKI_KV_VHALF,
    POSPISCHIL_VT,
    PURKINJE_VT,
    GatingVariable,
    GoldmanSpec,
    IonChannel,
    IonSpecies,
    NernstSpec,
    alpha_h,
    alpha_m,
    alpha_n,
    beta_h,
    beta_m,
    beta_n,
    dopaminergic_alpha_h,
    dopaminergic_alpha_m,
    dopaminergic_alpha_n,
    dopaminergic_beta_h,
    dopaminergic_beta_m,
    dopaminergic_beta_n,
    mainen_sejnowski_alpha_h,
    mainen_sejnowski_alpha_m,
    mainen_sejnowski_alpha_n,
    mainen_sejnowski_beta_h,
    mainen_sejnowski_beta_m,
    mainen_sejnowski_beta_n,
    make_dopaminergic_k_channel,
    make_dopaminergic_na_channel,
    make_ical_channel,
    make_ican_channel,
    make_icat_channel,
    make_ih_channel,
    make_ika_channel,
    make_ikca_channel,
    make_ikir_channel,
    make_im_channel,
    make_inap_channel,
    make_inar_channel,
    make_k_channel,
    make_k_leak_channel,
    make_katp_channel,
    make_mainen_sejnowski_kv_channel,
    make_mainen_sejnowski_na_channel,
    make_na_channel,
    make_na_leak_channel,
    make_nav11_channel,
    make_nav12_channel,
    make_pospischil_k_channel,
    make_purkinje_k_channel,
    make_purkinje_na_channel,
    make_snc_inap_channel,
    make_stn_na_channel,
    make_trn_icat_channel,
    make_trn_na_channel,
    pospischil_alpha_h,
    pospischil_alpha_m,
    pospischil_alpha_n,
    pospischil_beta_h,
    pospischil_beta_m,
    pospischil_beta_n,
    purkinje_alpha_h,
    purkinje_alpha_m,
    purkinje_alpha_n,
    purkinje_beta_h,
    purkinje_beta_m,
    purkinje_beta_n,
    trn_alpha_h,
    trn_alpha_m,
    trn_beta_h,
)
from patch_sim.channels.auxiliary import (
    _alpha_a,
    _alpha_b,
    _alpha_d,
    _alpha_dn,
    _alpha_dt,
    _alpha_f,
    _alpha_fn,
    _alpha_ft,
    _alpha_hr,
    _alpha_kATP,
    _alpha_kir,
    _alpha_p,
    _alpha_q,
    _alpha_r,
    _alpha_s,
    _alpha_sNaP,
    _alpha_w,
    _beta_a,
    _beta_b,
    _beta_d,
    _beta_dn,
    _beta_dt,
    _beta_f,
    _beta_fn,
    _beta_ft,
    _beta_hr,
    _beta_kATP,
    _beta_kir,
    _beta_p,
    _beta_q,
    _beta_r,
    _beta_s,
    _beta_sNaP,
    _beta_w,
)
from patch_sim.channels.pospischil import (
    _nav11_alpha_sNa,
    _nav11_beta_sNa,
    _nav12_alpha_sNa,
    _nav12_beta_sNa,
)
from patch_sim.channels.purkinje import _purkinje_alpha_sNa, _purkinje_beta_sNa
from patch_sim.channels.snc import (
    _alpha_sNaP_snc,
    _beta_sNaP_snc,
    _dopaminergic_alpha_sNa,
    _dopaminergic_beta_sNa,
)
from patch_sim.channels.stn import (
    _stn_alpha_h,
    _stn_alpha_m,
    _stn_alpha_sNa,
    _stn_beta_h,
    _stn_beta_m,
    _stn_beta_sNa,
)
from patch_sim.clamp_simulations import simulate_current_clamp, simulate_voltage_clamp
from patch_sim.electrochemistry import nernst_potential
from patch_sim.neuron import Neuron
from patch_sim.presets import make_squid_giant_axon
from patch_sim.protocols import step_current, step_voltage
from patch_sim.rates import CalciumDependentFn, Rate, VoltageOnlyFn


def _hh_with(*extras: IonChannel, **overrides: Any) -> Neuron:
    """Build an HH52 squid neuron with the given extra channels appended.

    Used by tests that previously relied on ``Neuron(additional_channels=...)``
    auto-supplying the HH52 core.  Extra ``overrides`` are applied via
    :func:`dataclasses.replace` after the channels are assembled.
    """
    base = make_squid_giant_axon()
    return dataclasses.replace(base, channels=base.channels + extras, **overrides)


# ---------------------------------------------------------------------------
# GatingVariable
# ---------------------------------------------------------------------------


def test_gating_variable_steady_state_in_bounds():
    """Ih gating variable steady state is in [0, 1] for all physiological voltages."""
    for V in np.linspace(-120.0, 0.0, 50):
        a = _alpha_r(V, 0.0)
        b = _beta_r(V, 0.0)
        assert a >= 0, f"alpha_r negative at V={V}"
        assert b >= 0, f"beta_r negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"steady state {ss} out of [0,1] at V={V}"


def test_ih_kinetics_alpha_increases_with_hyperpolarization():
    """alpha_r should increase as voltage becomes more negative (Ih is HCN-type)."""
    alpha_at_minus100 = _alpha_r(-100.0, 0.0)
    alpha_at_minus65 = _alpha_r(-65.0, 0.0)
    alpha_at_minus40 = _alpha_r(-40.0, 0.0)
    assert alpha_at_minus100 > alpha_at_minus65 > alpha_at_minus40


def test_ih_kinetics_steady_state_higher_at_hyperpolarized():
    """Ih r steady state is higher at -100 mV than at -65 mV."""
    a65, b65 = _alpha_r(-65.0, 0.0), _beta_r(-65.0, 0.0)
    a100, b100 = _alpha_r(-100.0, 0.0), _beta_r(-100.0, 0.0)
    ss65 = a65 / (a65 + b65)
    ss100 = a100 / (a100 + b100)
    assert ss100 > ss65


# ---------------------------------------------------------------------------
# IonChannel
# ---------------------------------------------------------------------------


def _make_simple_channel(g_max: float = 1.0) -> IonChannel:
    """Helper: create a K⁺ channel with a single linear gating variable (power=1)."""
    gv = GatingVariable(
        name="x",
        power=1,
        alpha=VoltageOnlyFn(lambda V, ca_i: 0.1),
        beta=VoltageOnlyFn(lambda V, ca_i: 0.1),
    )
    return IonChannel(
        name="test",
        g_max=g_max,
        gating_variables=(gv,),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )


def test_base_ion_channel_compute_current_math():
    """compute_current returns g_max * gate^power * (V - E_rev)."""
    neuron = Neuron()
    ch = _make_simple_channel(g_max=2.0)
    e_rev = ch.reversal_potential(neuron)
    # gate value = 0.5, power = 1 → g = 2.0 * 0.5^1 = 1.0
    result = ch.compute_current(V=0.0, gating_state={"x": 0.5}, neuron=neuron)
    assert result == pytest.approx(2.0 * 0.5 * (0.0 - e_rev))


def test_base_ion_channel_power_two():
    """compute_current correctly raises the gate to its power."""
    neuron = Neuron()
    gv = GatingVariable(
        name="y",
        power=2,
        alpha=VoltageOnlyFn(lambda V, ca_i: 0.1),
        beta=VoltageOnlyFn(lambda V, ca_i: 0.1),
    )
    ch = IonChannel(
        name="pow2",
        g_max=1.0,
        gating_variables=(gv,),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )
    e_rev = ch.reversal_potential(neuron)
    # gate=0.5, power=2 → g = 1.0 * 0.5^2 = 0.25
    result = ch.compute_current(V=10.0, gating_state={"y": 0.5}, neuron=neuron)
    assert result == pytest.approx(1.0 * (0.5**2) * (10.0 - e_rev))


def test_base_ion_channel_reversal_potential_uses_nernst():
    """reversal_potential() computes K⁺ Nernst potential from neuron concentrations."""
    from patch_sim.electrochemistry import nernst_potential

    neuron = Neuron()
    ch = _make_simple_channel()
    expected = nernst_potential(1, neuron.T, neuron.K_out, neuron.K_in)
    assert ch.reversal_potential(neuron) == pytest.approx(expected)


def test_base_ion_channel_zero_current_at_reversal():
    """Current is zero when V equals E_rev (the Nernst potential)."""
    neuron = Neuron()
    ch = _make_simple_channel(g_max=1.0)
    e_rev = ch.reversal_potential(neuron)
    result = ch.compute_current(V=e_rev, gating_state={"x": 0.8}, neuron=neuron)
    assert result == pytest.approx(0.0)


def test_base_ion_channel_satisfies_protocol():
    """IonChannel is an instance of the IonChannel dataclass."""
    ch = _make_simple_channel()
    assert isinstance(ch, IonChannel)


# ---------------------------------------------------------------------------
# Dynamic E_Ca via compute_current(ca_i=...)
# ---------------------------------------------------------------------------


def _make_ca_channel(g_max: float = 1.0) -> IonChannel:
    """Helper: create a minimal ICaL-type Ca²⁺ channel (carries_calcium=True)."""
    gv = GatingVariable(
        name="d",
        power=1,
        alpha=VoltageOnlyFn(lambda V, ca_i: 0.1),
        beta=VoltageOnlyFn(lambda V, ca_i: 0.1),
    )
    return IonChannel(
        name="CaTest",
        g_max=g_max,
        gating_variables=(gv,),
        reversal_spec=NernstSpec(IonSpecies.CALCIUM),
        carries_calcium=True,
    )


def test_compute_current_ca_uses_live_ca_i() -> None:
    """Ca²⁺ channel computes E_Ca from live ca_i instead of the cache.

    The current should equal g_max * gate^power * (V - nernst(2, T, Ca_out, ca_i)).
    """
    neuron = Neuron()
    ch = _make_ca_channel(g_max=2.0)
    ca_i = 1e-3  # 1 µM
    gate = 0.5
    expected_e_ca = nernst_potential(2, neuron.T, neuron.Ca_out, ca_i)
    expected_current = 2.0 * gate * (0.0 - expected_e_ca)
    result = ch.compute_current(
        V=0.0, gating_state={"d": gate}, neuron=neuron, ca_i=ca_i
    )
    assert result == pytest.approx(expected_current)


def test_compute_current_ca_falls_back_to_ca_in_when_ca_i_none() -> None:
    """Ca²⁺ channel falls back to neuron.Ca_in when ca_i is not provided."""
    neuron = Neuron()
    ch = _make_ca_channel(g_max=1.0)
    expected_e_ca = nernst_potential(2, neuron.T, neuron.Ca_out, neuron.Ca_in)
    expected_current = 1.0 * 0.5 * (0.0 - expected_e_ca)
    result = ch.compute_current(V=0.0, gating_state={"d": 0.5}, neuron=neuron)
    assert result == pytest.approx(expected_current)


def test_compute_current_ca_floors_zero_ca_i() -> None:
    """Ca²⁺ channel does not raise when ca_i=0.0; the Nernst floor is applied."""
    neuron = Neuron()
    ch = _make_ca_channel(g_max=1.0)
    # Should not raise despite ca_i=0 (which would cause log(0) without the floor).
    result = ch.compute_current(V=0.0, gating_state={"d": 0.5}, neuron=neuron, ca_i=0.0)
    assert isinstance(result, float)
    assert not math.isnan(result)


def test_reversal_potentials_excludes_ca_channels() -> None:
    """neuron.reversal_potentials does not include Ca²⁺-carrying channels.

    Only channels where carries_calcium=False should be in the cache so that
    Ca channels always recompute E_Ca from live ca_i.
    """
    ca_ch = _make_ca_channel(g_max=1.0)
    k_ch = _make_simple_channel(g_max=1.0)
    neuron = _hh_with(ca_ch, k_ch, calcium_dynamics=CalciumDynamics())
    cache = neuron.reversal_potentials
    # K⁺ channel must be in the cache.
    assert "test" in cache
    # Ca²⁺ channel must NOT be in the cache — it uses live-ca_i Nernst.
    assert "CaTest" not in cache


def test_ikca_channel_in_reversal_potentials_cache() -> None:
    """IKCa (K⁺ channel whose gating depends on ca_i) stays in the cache.

    IKCa is a K⁺ channel (carries_calcium=False).  Only its gating rate
    functions use ca_i; its reversal potential is a fixed K⁺ Nernst value.
    """
    ikca = make_ikca_channel(g_max=1.0)
    neuron = _hh_with(ikca, calcium_dynamics=CalciumDynamics())
    assert "KCa" in neuron.reversal_potentials


# ---------------------------------------------------------------------------
# IonSpecies, NernstSpec, GoldmanSpec
# ---------------------------------------------------------------------------


def test_ion_species_valences():
    """Each IonSpecies must carry the correct valence."""
    assert IonSpecies.SODIUM.valence == 1
    assert IonSpecies.POTASSIUM.valence == 1
    assert IonSpecies.CALCIUM.valence == 2


def test_ion_species_symbols():
    """Each IonSpecies must carry the correct chemical symbol."""
    assert IonSpecies.SODIUM.symbol == "Na"
    assert IonSpecies.POTASSIUM.symbol == "K"
    assert IonSpecies.CALCIUM.symbol == "Ca"


def test_nernst_spec_stores_species():
    """NernstSpec stores the ion species it was created with."""
    spec = NernstSpec(species=IonSpecies.POTASSIUM)
    assert spec.species is IonSpecies.POTASSIUM


def test_goldman_spec_stores_permeabilities():
    """GoldmanSpec stores the permeability tuple it was created with."""
    spec = GoldmanSpec(
        permeabilities=((IonSpecies.SODIUM, 0.289), (IonSpecies.POTASSIUM, 1.0))
    )
    assert spec.permeabilities[0] == (IonSpecies.SODIUM, 0.289)
    assert spec.permeabilities[1] == (IonSpecies.POTASSIUM, 1.0)


def test_goldman_spec_divalent_ion_raises():
    """GoldmanSpec must reject divalent ions (use NernstSpec for Ca2+)."""
    with pytest.raises(ValueError, match="monovalent"):
        GoldmanSpec(permeabilities=((IonSpecies.CALCIUM, 1.0),))


def test_goldman_spec_negative_permeability_raises():
    """GoldmanSpec must reject negative permeabilities."""
    with pytest.raises(ValueError, match="non-negative"):
        GoldmanSpec(
            permeabilities=((IonSpecies.SODIUM, -0.1), (IonSpecies.POTASSIUM, 1.0))
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_base_ion_channel_negative_gmax_raises():
    """Negative g_max raises ValueError."""
    with pytest.raises(ValueError, match="g_max must be non-negative"):
        _make_simple_channel(g_max=-1.0)


def test_base_ion_channel_duplicate_gating_names_raises():
    """Duplicate gating variable names within a channel raise ValueError."""
    gv1 = GatingVariable(
        name="x",
        power=1,
        alpha=VoltageOnlyFn(lambda V, ca_i: 0.1),
        beta=VoltageOnlyFn(lambda V, ca_i: 0.1),
    )
    gv2 = GatingVariable(
        name="x",
        power=2,
        alpha=VoltageOnlyFn(lambda V, ca_i: 0.2),
        beta=VoltageOnlyFn(lambda V, ca_i: 0.2),
    )
    with pytest.raises(ValueError, match="names must be unique"):
        IonChannel(
            name="dup",
            g_max=1.0,
            gating_variables=(gv1, gv2),
            reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
        )


def test_hh_duplicate_additional_channel_names_raises():
    """Duplicate additional channel names on Neuron raise ValueError."""
    ch = make_ih_channel()
    with pytest.raises(ValueError, match="names must be unique"):
        _hh_with(ch, ch)


def test_hh_duplicate_with_existing_na_raises():
    """Adding a second channel named 'Na' to a neuron that already has one raises."""
    gv = GatingVariable(
        name="r",
        power=1,
        alpha=VoltageOnlyFn(lambda V, ca_i: 0.1),
        beta=VoltageOnlyFn(lambda V, ca_i: 0.1),
    )
    ch = IonChannel(
        name="Na",
        g_max=0.1,
        gating_variables=(gv,),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )
    with pytest.raises(ValueError, match="names must be unique"):
        _hh_with(ch)


# ---------------------------------------------------------------------------
# make_ih_channel
# ---------------------------------------------------------------------------


def test_make_ih_channel_defaults():
    """make_ih_channel() produces a channel with the expected defaults."""
    from patch_sim.constants import DEFAULT_G_IH, DEFAULT_IH_P_NA

    ch = make_ih_channel()
    assert ch.name == "h"
    assert ch.g_max == pytest.approx(DEFAULT_G_IH)
    assert isinstance(ch.reversal_spec, GoldmanSpec)
    assert ch.reversal_spec.permeabilities[0] == (IonSpecies.SODIUM, DEFAULT_IH_P_NA)
    assert ch.reversal_spec.permeabilities[1] == (IonSpecies.POTASSIUM, 1.0)
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "r"
    assert ch.gating_variables[0].power == 1


def test_make_ih_channel_custom_params():
    """make_ih_channel accepts custom g_max and p_na."""
    ch = make_ih_channel(g_max=0.5, p_na=0.33)
    assert ch.g_max == pytest.approx(0.5)
    assert isinstance(ch.reversal_spec, GoldmanSpec)
    assert ch.reversal_spec.permeabilities[0][1] == pytest.approx(0.33)


# ---------------------------------------------------------------------------
# Backward compatibility — no optional channels
# ---------------------------------------------------------------------------


def test_current_clamp_no_additional_channels_identical_columns(hh_model):
    """simulate_current_clamp with no additional channels has exact classic columns."""
    stim = step_current(
        duration=10.0,
        current_amplitude=10.0,
        step_start=2.0,
        step_duration=5.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(hh_model, stim)
    expected = {
        "time",
        "voltage",
        "INa",
        "IK",
        "INaL",
        "IKL",
        "Itotal",
        "n",
        "m",
        "h",
    }
    assert result.dtype.names is not None
    assert set(result.dtype.names) == expected


def test_voltage_clamp_no_additional_channels_identical_columns(hh_model):
    """simulate_voltage_clamp with no additional channels has exact classic columns."""
    prot = step_voltage(
        duration=10.0,
        voltage_amplitude=0.0,
        step_start=2.0,
        step_duration=5.0,
        holding_voltage=-70.0,
        sampling_frequency=40000.0,
    )
    result = simulate_voltage_clamp(hh_model, prot)
    expected = {
        "time",
        "voltage",
        "Itotal",
        "INa",
        "IK",
        "INaL",
        "IKL",
        "n",
        "m",
        "h",
    }
    assert result.dtype.names is not None
    assert set(result.dtype.names) == expected


def test_current_clamp_empty_neuron_holds_v_rest():
    """A Neuron with no channels has no membrane currents; voltage stays at v_rest."""
    stim = step_current(
        duration=5.0,
        current_amplitude=0.0,
        step_start=1.0,
        step_duration=1.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(Neuron(), stim)
    expected = np.full_like(result["voltage"], Neuron().v_rest)
    np.testing.assert_allclose(result["voltage"], expected, atol=1e-12)


# ---------------------------------------------------------------------------
# h channel integration tests
# ---------------------------------------------------------------------------


def test_current_clamp_with_ih_extra_columns():
    """Current clamp with h channel adds Ih and r columns."""
    neuron = _hh_with(make_ih_channel())
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result.dtype.names is not None
    assert "Ih" in result.dtype.names
    assert "r" in result.dtype.names


def test_current_clamp_ih_gating_variable_in_bounds():
    """Ih gating variable r stays in [0, 1] during current clamp."""
    neuron = _hh_with(make_ih_channel())
    stim = step_current(
        duration=30.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=20.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result["r"].min() >= 0.0
    assert result["r"].max() <= 1.0


def test_voltage_clamp_with_ih_extra_columns():
    """Voltage clamp with h channel adds Ih and r columns."""
    neuron = _hh_with(make_ih_channel())
    prot = step_voltage(
        duration=20.0,
        voltage_amplitude=-40.0,
        step_start=5.0,
        step_duration=10.0,
        holding_voltage=-70.0,
        sampling_frequency=40000.0,
    )
    result = simulate_voltage_clamp(neuron, prot)
    assert result.dtype.names is not None
    assert "Ih" in result.dtype.names
    assert "r" in result.dtype.names


def test_voltage_clamp_total_current_includes_ih():
    """total_current includes Ih contribution: I_total == I_Na + I_K + I_L + I_Ih."""
    neuron = _hh_with(make_ih_channel())
    prot = step_voltage(
        duration=10.0,
        voltage_amplitude=-40.0,
        step_start=2.0,
        step_duration=5.0,
        holding_voltage=-70.0,
        sampling_frequency=40000.0,
    )
    result = simulate_voltage_clamp(neuron, prot)
    expected = (
        result["INa"] + result["IK"] + result["INaL"] + result["IKL"] + result["Ih"]
    )
    np.testing.assert_allclose(result["Itotal"], expected, rtol=1e-10)


def test_multiple_optional_channels_coexist():
    """Two distinct optional channels can coexist and each contributes columns."""
    ch1 = make_ih_channel(g_max=0.1)
    gv2 = GatingVariable(
        name="q",
        power=1,
        alpha=VoltageOnlyFn(lambda V, ca_i: 0.05),
        beta=VoltageOnlyFn(lambda V, ca_i: 0.05),
    )
    ch2 = IonChannel(
        name="q",
        g_max=0.05,
        gating_variables=(gv2,),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )
    neuron = _hh_with(ch1, ch2)
    stim = step_current(
        duration=10.0,
        current_amplitude=5.0,
        step_start=2.0,
        step_duration=5.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result.dtype.names is not None
    assert "Ih" in result.dtype.names
    assert "Iq" in result.dtype.names
    assert "r" in result.dtype.names
    assert "q" in result.dtype.names


# ---------------------------------------------------------------------------
# IKa rate functions
# ---------------------------------------------------------------------------


def test_ika_gating_variable_steady_state_in_bounds():
    """IKa gating variables a_inf and b_inf are in [0, 1] for physiological voltages."""
    for V in np.linspace(-120.0, 0.0, 50):
        for alpha_fn, beta_fn in ((_alpha_a, _beta_a), (_alpha_b, _beta_b)):
            a = alpha_fn(V, 0.0)
            b = beta_fn(V, 0.0)
            assert a >= 0, f"alpha negative at V={V}"
            assert b >= 0, f"beta negative at V={V}"
            ss = a / (a + b)
            assert 0.0 <= ss <= 1.0, f"steady state {ss} out of [0,1] at V={V}"


def test_ika_kinetics_activation_increases_with_depolarization():
    """IKa a_inf (activation) is higher at depolarized voltages."""

    def a_inf(V: float) -> float:
        """Activation steady-state at voltage V."""
        a, b = _alpha_a(V, 0.0), _beta_a(V, 0.0)
        return a / (a + b)

    assert a_inf(-20.0) > a_inf(-65.0) > a_inf(-100.0)


def test_ika_kinetics_inactivation_decreases_with_depolarization():
    """IKa b_inf (inactivation) is lower at depolarized voltages."""

    def b_inf(V: float) -> float:
        """Inactivation steady-state at voltage V."""
        a, b = _alpha_b(V, 0.0), _beta_b(V, 0.0)
        return a / (a + b)

    assert b_inf(-100.0) > b_inf(-65.0) > b_inf(-20.0)


# ---------------------------------------------------------------------------
# make_ika_channel
# ---------------------------------------------------------------------------


def test_make_ika_channel_defaults():
    """make_ika_channel() produces a channel with the expected defaults."""
    from patch_sim.constants import DEFAULT_G_IKA

    ch = make_ika_channel()
    assert ch.name == "Ka"
    assert ch.g_max == pytest.approx(DEFAULT_G_IKA)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM
    assert len(ch.gating_variables) == 2
    assert ch.gating_variables[0].name == "a"
    assert ch.gating_variables[0].power == 1
    assert ch.gating_variables[1].name == "b"
    assert ch.gating_variables[1].power == 1


def test_make_ika_channel_custom_params():
    """make_ika_channel accepts custom g_max."""
    ch = make_ika_channel(g_max=10.0)
    assert ch.g_max == pytest.approx(10.0)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM


# ---------------------------------------------------------------------------
# IKa integration tests
# ---------------------------------------------------------------------------


def test_current_clamp_with_ika_extra_columns():
    """Current clamp with Ka channel adds IKa, a, and b columns."""
    neuron = _hh_with(make_ika_channel())
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result.dtype.names is not None
    assert "IKa" in result.dtype.names
    assert "a" in result.dtype.names
    assert "b" in result.dtype.names


def test_current_clamp_ika_gating_in_bounds():
    """IKa gating variables a and b stay in [0, 1] during current clamp."""
    neuron = _hh_with(make_ika_channel())
    stim = step_current(
        duration=30.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=20.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result["a"].min() >= 0.0
    assert result["a"].max() <= 1.0
    assert result["b"].min() >= 0.0
    assert result["b"].max() <= 1.0


def test_voltage_clamp_with_ika_extra_columns():
    """Voltage clamp with Ka channel adds IKa, a, and b columns."""
    neuron = _hh_with(make_ika_channel())
    prot = step_voltage(
        duration=20.0,
        voltage_amplitude=0.0,
        step_start=5.0,
        step_duration=10.0,
        holding_voltage=-70.0,
        sampling_frequency=40000.0,
    )
    result = simulate_voltage_clamp(neuron, prot)
    assert result.dtype.names is not None
    assert "IKa" in result.dtype.names
    assert "a" in result.dtype.names
    assert "b" in result.dtype.names


def test_ika_and_ih_coexist():
    """Ka and h channels can coexist and each contributes its columns."""
    neuron = _hh_with(make_ika_channel(), make_ih_channel())
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result.dtype.names is not None
    assert "IKa" in result.dtype.names
    assert "Ih" in result.dtype.names
    assert "a" in result.dtype.names
    assert "b" in result.dtype.names
    assert "r" in result.dtype.names


# ---------------------------------------------------------------------------
# IKv31 (Kv3.1-type K+)
# ---------------------------------------------------------------------------


def test_ikv31_gating_steady_state_in_bounds():
    """IKv31 gating variable nk_inf is in [0, 1] for physiological voltages."""
    from patch_sim.channels.auxiliary import _ikv31_alpha_nk, _ikv31_beta_nk

    for V in range(-100, 61):
        alpha = _ikv31_alpha_nk(float(V), 0.0)
        beta = _ikv31_beta_nk(float(V), 0.0)
        nk_inf = alpha / (alpha + beta)
        assert 0.0 <= nk_inf <= 1.0, f"nk_inf={nk_inf} out of bounds at V={V}"


def test_ikv31_near_zero_activation_at_rest():
    """IKv31 nk_inf is near zero at resting potential (-65 mV).

    The high activation threshold of Kv3.1 means virtually no outward current
    at rest — this is the key property that fixes issue #155.
    """
    from patch_sim.channels.auxiliary import _ikv31_alpha_nk, _ikv31_beta_nk

    alpha = _ikv31_alpha_nk(-65.0, 0.0)
    beta = _ikv31_beta_nk(-65.0, 0.0)
    nk_inf = alpha / (alpha + beta)
    assert nk_inf < 0.02, f"nk_inf={nk_inf} should be near zero at -65 mV"


def test_ikv31_strong_activation_depolarized():
    """IKv31 nk_inf is well above 0.5 at 0 mV (depolarized)."""
    from patch_sim.channels.auxiliary import _ikv31_alpha_nk, _ikv31_beta_nk

    alpha = _ikv31_alpha_nk(0.0, 0.0)
    beta = _ikv31_beta_nk(0.0, 0.0)
    nk_inf = alpha / (alpha + beta)
    assert nk_inf > 0.5, f"nk_inf={nk_inf} should be above 0.5 at 0 mV"


def test_make_ikv31_channel_defaults():
    """make_ikv31_channel() produces a channel with the expected defaults."""
    from patch_sim.channels import make_ikv31_channel
    from patch_sim.constants import DEFAULT_G_IKV31

    ch = make_ikv31_channel()
    assert ch.name == "Kv31"
    assert ch.g_max == pytest.approx(DEFAULT_G_IKV31)
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "nk"
    assert ch.gating_variables[0].power == 2


def test_current_clamp_with_ikv31():
    """Current clamp with IKv31 channel adds Kv31 and nk columns."""
    from patch_sim.channels import make_ikv31_channel

    neuron = _hh_with(make_ikv31_channel())
    stimulus = np.zeros(int(40_000 * 0.05))
    result = simulate_current_clamp(neuron=neuron, current_external=stimulus)
    assert "IKv31" in result.dtype.names
    assert "nk" in result.dtype.names


# ---------------------------------------------------------------------------
# INaP rate functions
# ---------------------------------------------------------------------------


def test_inap_gating_variable_steady_state_in_bounds():
    """INaP gating variable p_inf is in [0, 1] for physiological voltages."""
    for V in np.linspace(-120.0, 0.0, 50):
        a = _alpha_p(V, 0.0)
        b = _beta_p(V, 0.0)
        assert a >= 0, f"alpha_p negative at V={V}"
        assert b >= 0, f"beta_p negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"steady state {ss} out of [0,1] at V={V}"


def test_inap_activation_increases_with_depolarization():
    """INaP p_inf (activation) is higher at depolarized voltages."""

    def p_inf(V: float) -> float:
        """Activation steady-state at voltage V."""
        a, b = _alpha_p(V, 0.0), _beta_p(V, 0.0)
        return a / (a + b)

    assert p_inf(-20.0) > p_inf(-53.0) > p_inf(-100.0)


def test_inap_subthreshold_activation():
    """INaP p_inf is substantially activated below spike threshold (-52.6 mV half)."""

    def p_inf(V: float) -> float:
        """Activation steady-state at voltage V."""
        a, b = _alpha_p(V, 0.0), _beta_p(V, 0.0)
        return a / (a + b)

    # At half-activation voltage p_inf should be ~0.5
    assert p_inf(-52.6) == pytest.approx(0.5, abs=0.01)
    # Subthreshold range (-55 to -40 mV) should have partial activation
    assert 0.0 < p_inf(-55.0) < 0.5
    assert 0.5 < p_inf(-40.0) < 1.0


# ---------------------------------------------------------------------------
# INaP slow-inactivation rate functions (Magistretti & Alonso 1999)
# ---------------------------------------------------------------------------


def _sNaP_inf(V: float) -> float:
    """INaP slow-inactivation steady-state availability at voltage V."""
    a, b = _alpha_sNaP(V, 0.0), _beta_sNaP(V, 0.0)
    return a / (a + b)


def test_inap_slow_inactivation_steady_state_in_bounds():
    """INaP slow-inactivation s_inf is in [0, 1] for physiological voltages."""
    for V in np.linspace(-120.0, 0.0, 50):
        a = _alpha_sNaP(V, 0.0)
        b = _beta_sNaP(V, 0.0)
        assert a >= 0, f"alpha_sNaP negative at V={V}"
        assert b >= 0, f"beta_sNaP negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"sNaP steady state {ss} out of [0,1] at V={V}"


def test_inap_slow_inactivation_decreases_with_depolarization():
    """Availability is highest at hyperpolarized voltages (inactivation)."""
    assert _sNaP_inf(-80.0) > _sNaP_inf(-45.0) > _sNaP_inf(-10.0)


def test_inap_slow_inactivation_half_voltage():
    """V½ for sNaP sits at -45 mV (within Magistretti & Alonso 1999's spread)."""
    assert _sNaP_inf(-45.0) == pytest.approx(0.5, abs=0.01)


def test_inap_slow_inactivation_resting_availability():
    """Near-rest availability is high enough to preserve pacemaking."""
    # At -65 mV the gate must be near 1 so opting in does not silence cells
    # whose firing depends on the resting INaP window current.
    assert _sNaP_inf(-65.0) > 0.9
    assert _sNaP_inf(-75.0) > 0.95


def test_inap_slow_inactivation_blocks_depol_plateau():
    """At depolarized plateau voltages sNaP closes, providing block escape."""
    # The depol-block plateau in #324 hung at ≈ −15 mV; sNaP must close
    # firmly there so g_INaP_eff = g_max * p * sNaP collapses.
    assert _sNaP_inf(-15.0) < 0.05


def test_inap_slow_inactivation_tau_is_slow():
    """τ_sNaP at V½ stays distinctly slow vs the fast p activation (~6 ms)."""
    a, b = _alpha_sNaP(-45.0, 0.0), _beta_sNaP(-45.0, 0.0)
    tau = 1.0 / (a + b)
    assert tau > 100.0, f"sNaP tau at V½ is {tau:.1f} ms, expected > 100 ms"


# ---------------------------------------------------------------------------
# make_inap_channel
# ---------------------------------------------------------------------------


def test_make_inap_channel_defaults():
    """make_inap_channel() returns the (p, sNaP) two-gate INaP topology."""
    from patch_sim.constants import DEFAULT_G_NAP

    ch = make_inap_channel()
    assert ch.name == "NaP"
    assert ch.g_max == pytest.approx(DEFAULT_G_NAP)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.SODIUM
    assert len(ch.gating_variables) == 2
    assert ch.gating_variables[0].name == "p"
    assert ch.gating_variables[0].power == 1
    assert ch.gating_variables[1].name == "sNaP"
    assert ch.gating_variables[1].power == 1


def test_make_inap_channel_custom_params():
    """make_inap_channel accepts custom g_max."""
    ch = make_inap_channel(g_max=1.0)
    assert ch.g_max == pytest.approx(1.0)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.SODIUM


# ---------------------------------------------------------------------------
# SNc INaP slow inactivation rate functions (sNaP_snc gate; Magistretti &
# Alonso 1999 V½ shifted to match the Drion 2011 SNc fit).  Always-on gate
# added in #330.
# ---------------------------------------------------------------------------


def _sNaP_snc_inf(V: float) -> float:
    """Steady-state availability of the SNc INaP slow inactivation gate.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Steady-state availability of the sNaP_snc gate at voltage V, in [0, 1].
    """
    a, b = _alpha_sNaP_snc(V, 0.0), _beta_sNaP_snc(V, 0.0)
    return a / (a + b)


def test_snc_inap_slow_inactivation_steady_state_in_bounds():
    """The sNaP_snc rates are positive and steady state in [0, 1] across V."""
    for V in (-120.0, -100.0, -75.0, -55.0, -45.0, -30.0, -15.0, 0.0, 30.0):
        a = _alpha_sNaP_snc(V, 0.0)
        b = _beta_sNaP_snc(V, 0.0)
        assert a >= 0, f"alpha_sNaP_snc negative at V={V}"
        assert b >= 0, f"beta_sNaP_snc negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"sNaP_snc steady state {ss} out of [0,1] at V={V}"


def test_snc_inap_slow_inactivation_decreases_with_depolarization():
    """The sNaP_snc availability decreases monotonically with depolarization."""
    assert _sNaP_snc_inf(-80.0) > _sNaP_snc_inf(-55.0) > _sNaP_snc_inf(-15.0)


def test_snc_inap_slow_inactivation_half_voltage():
    """V½ for sNaP_snc sits at -55 mV (Drion 2011 shift from M&A 1999)."""
    assert _sNaP_snc_inf(-55.0) == pytest.approx(0.5, abs=0.01)


def test_snc_inap_slow_inactivation_resting_availability():
    """SNc DA cycles through −90 to −55 mV; sNaP_snc must stay open in the trough.

    The looser lower bound at v_rest = −55 mV (vs entorhinal sNaP at 0.94
    at −65 mV) is expected: SNc rests more depolarized, and the SNc-shifted
    V½ tracks.  The cycle hyperpolarized end (≈ −75 mV) is what matters
    for recovery between spikes, and there sNaP_snc > 0.94.
    """
    assert _sNaP_snc_inf(-75.0) > 0.94
    assert _sNaP_snc_inf(-55.0) == pytest.approx(0.5, abs=0.01)


def test_snc_inap_slow_inactivation_blocks_depol_plateau():
    """At depolarized plateau voltages sNaP_snc closes, providing block escape."""
    # At ≈ −15 mV the slow gate must close firmly so the residual SNc
    # persistent Na current (g_NaP_SNc * pSNc * sNaP_snc) collapses.
    assert _sNaP_snc_inf(-15.0) < 0.05


def test_snc_inap_slow_inactivation_tau_is_slow():
    """τ_sNaP_snc at V½ stays distinctly slow vs the fast pSNc activation (~5 ms)."""
    a, b = _alpha_sNaP_snc(-55.0, 0.0), _beta_sNaP_snc(-55.0, 0.0)
    tau = 1.0 / (a + b)
    assert tau > 100.0, f"sNaP_snc tau at V½ is {tau:.1f} ms, expected > 100 ms"


# ---------------------------------------------------------------------------
# make_snc_inap_channel
# ---------------------------------------------------------------------------


def test_make_snc_inap_channel_defaults():
    """make_snc_inap_channel() exposes pSNc and sNaP_snc gates by default."""
    from patch_sim.constants import DEFAULT_G_NAP_SNC

    ch = make_snc_inap_channel()
    assert ch.name == "NaP_SNc"
    assert ch.g_max == pytest.approx(DEFAULT_G_NAP_SNC)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.SODIUM
    assert len(ch.gating_variables) == 2
    assert ch.gating_variables[0].name == "pSNc"
    assert ch.gating_variables[0].power == 1
    assert ch.gating_variables[1].name == "sNaP_snc"
    assert ch.gating_variables[1].power == 1


def test_make_snc_inap_channel_custom_params():
    """make_snc_inap_channel accepts custom g_max."""
    ch = make_snc_inap_channel(g_max=0.05)
    assert ch.g_max == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# INaP integration tests
# ---------------------------------------------------------------------------


def test_current_clamp_with_inap_extra_columns():
    """Current clamp with INaP channel exposes both p and sNaP gating columns."""
    neuron = _hh_with(make_inap_channel())
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result.dtype.names is not None
    assert "INaP" in result.dtype.names
    assert "p" in result.dtype.names
    assert "sNaP" in result.dtype.names


def test_voltage_clamp_with_inap_extra_columns():
    """Voltage clamp with NaP channel adds INaP and p columns."""
    neuron = _hh_with(make_inap_channel())
    prot = step_voltage(
        duration=20.0,
        voltage_amplitude=0.0,
        step_start=5.0,
        step_duration=10.0,
        holding_voltage=-70.0,
        sampling_frequency=40000.0,
    )
    result = simulate_voltage_clamp(neuron, prot)
    assert result.dtype.names is not None
    assert "INaP" in result.dtype.names
    assert "p" in result.dtype.names


def test_current_clamp_inap_gating_in_bounds():
    """INaP gating variable p stays in [0, 1] during current clamp."""
    neuron = _hh_with(make_inap_channel())
    stim = step_current(
        duration=30.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=20.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result["p"].min() >= 0.0
    assert result["p"].max() <= 1.0


def test_current_clamp_inap_slow_inactivation_gating_in_bounds():
    """Both p and sNaP gates stay in [0, 1] during current clamp."""
    neuron = _hh_with(make_inap_channel())
    stim = step_current(
        duration=30.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=20.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result["p"].min() >= 0.0
    assert result["p"].max() <= 1.0
    assert result["sNaP"].min() >= 0.0
    assert result["sNaP"].max() <= 1.0


# ---------------------------------------------------------------------------
# INaR rate functions
# ---------------------------------------------------------------------------


def test_inar_s_gating_steady_state_in_bounds():
    """INaR activation variable s_inf is in [0, 1] for physiological voltages."""
    for V in np.linspace(-120.0, 0.0, 50):
        a = _alpha_s(V, 0.0)
        b = _beta_s(V, 0.0)
        assert a >= 0, f"alpha_s negative at V={V}"
        assert b >= 0, f"beta_s negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"s steady state {ss} out of [0,1] at V={V}"


def test_inar_hr_gating_steady_state_in_bounds():
    """INaR unblocking variable hr_inf is in [0, 1] for physiological voltages."""
    for V in np.linspace(-120.0, 0.0, 50):
        a = _alpha_hr(V, 0.0)
        b = _beta_hr(V, 0.0)
        assert a >= 0, f"alpha_hr negative at V={V}"
        assert b >= 0, f"beta_hr negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"hr steady state {ss} out of [0,1] at V={V}"


def test_inar_activation_increases_with_depolarization():
    """INaR s_inf (activation) is higher at depolarized voltages."""

    def s_inf(V: float) -> float:
        """Activation steady-state at voltage V."""
        a, b = _alpha_s(V, 0.0), _beta_s(V, 0.0)
        return a / (a + b)

    assert s_inf(-20.0) > s_inf(-42.0) > s_inf(-100.0)


def test_inar_unblocking_decreases_with_depolarization():
    """INaR hr_inf (unblocking) is lower at depolarized voltages (more blocked)."""

    def hr_inf(V: float) -> float:
        """Unblocking steady-state at voltage V."""
        a, b = _alpha_hr(V, 0.0), _beta_hr(V, 0.0)
        return a / (a + b)

    assert hr_inf(-100.0) > hr_inf(-55.0) > hr_inf(-20.0)


def test_inar_rates_non_negative():
    """All four INaR rate functions are non-negative across physiological voltages."""
    for V in np.linspace(-120.0, 60.0, 100):
        assert _alpha_s(V, 0.0) >= 0, f"alpha_s negative at V={V}"
        assert _beta_s(V, 0.0) >= 0, f"beta_s negative at V={V}"
        assert _alpha_hr(V, 0.0) >= 0, f"alpha_hr negative at V={V}"
        assert _beta_hr(V, 0.0) >= 0, f"beta_hr negative at V={V}"


# ---------------------------------------------------------------------------
# make_inar_channel
# ---------------------------------------------------------------------------


def test_make_inar_channel_defaults():
    """make_inar_channel() produces a channel with the expected defaults."""
    from patch_sim.constants import DEFAULT_G_NAR

    ch = make_inar_channel()
    assert ch.name == "NaR"
    assert ch.g_max == pytest.approx(DEFAULT_G_NAR)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.SODIUM
    assert len(ch.gating_variables) == 2
    assert ch.gating_variables[0].name == "s"
    assert ch.gating_variables[0].power == 1
    assert ch.gating_variables[1].name == "hr"
    assert ch.gating_variables[1].power == 1


def test_make_inar_channel_custom_params():
    """make_inar_channel accepts custom g_max."""
    ch = make_inar_channel(g_max=0.5)
    assert ch.g_max == pytest.approx(0.5)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.SODIUM


# ---------------------------------------------------------------------------
# INaR integration tests
# ---------------------------------------------------------------------------


def test_current_clamp_with_inar_extra_columns():
    """Current clamp with NaR channel adds INaR, s, and hr columns."""
    neuron = _hh_with(make_inar_channel())
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result.dtype.names is not None
    assert "INaR" in result.dtype.names
    assert "s" in result.dtype.names
    assert "hr" in result.dtype.names


def test_voltage_clamp_with_inar_extra_columns():
    """Voltage clamp with NaR channel adds INaR, s, and hr columns."""
    neuron = _hh_with(make_inar_channel())
    prot = step_voltage(
        duration=20.0,
        voltage_amplitude=0.0,
        step_start=5.0,
        step_duration=10.0,
        holding_voltage=-70.0,
        sampling_frequency=40000.0,
    )
    result = simulate_voltage_clamp(neuron, prot)
    assert result.dtype.names is not None
    assert "INaR" in result.dtype.names
    assert "s" in result.dtype.names
    assert "hr" in result.dtype.names


def test_current_clamp_inar_gating_in_bounds():
    """INaR gating variables s and hr stay in [0, 1] during current clamp."""
    neuron = _hh_with(make_inar_channel())
    stim = step_current(
        duration=30.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=20.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result["s"].min() >= 0.0
    assert result["s"].max() <= 1.0
    assert result["hr"].min() >= 0.0
    assert result["hr"].max() <= 1.0


def test_inap_and_inar_coexist():
    """NaP and NaR channels can coexist and each contributes columns."""
    neuron = _hh_with(make_inap_channel(), make_inar_channel())
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result.dtype.names is not None
    assert "INaP" in result.dtype.names
    assert "INaR" in result.dtype.names
    assert "p" in result.dtype.names
    assert "s" in result.dtype.names
    assert "hr" in result.dtype.names


def test_inap_slow_inactivation_and_inar_no_gate_collision():
    """INaP slow inactivation and INaR coexist with distinct gate columns.

    Regression test for the gate-naming hazard noted in the channel
    docstring: the gating-state dictionary is keyed by gate name only, so
    INaP's slow inactivation gate must be named ``sNaP`` (not ``s``) to
    avoid aliasing with :func:`make_inar_channel`'s activation gate.
    """
    neuron = _hh_with(make_inap_channel(), make_inar_channel())
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result.dtype.names is not None
    # Both gates present and distinct
    assert "sNaP" in result.dtype.names
    assert "s" in result.dtype.names


def test_all_additional_channels_coexist():
    """All seven additional channels (Ih, IKa, INaP, INaR, IM, IKir, IKCa) coexist."""
    from patch_sim.calcium import CalciumDynamics

    neuron = _hh_with(
        make_ih_channel(),
        make_ika_channel(),
        make_inap_channel(),
        make_inar_channel(),
        make_im_channel(),
        make_ikir_channel(),
        make_ikca_channel(),
        calcium_dynamics=CalciumDynamics(),
    )
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result.dtype.names is not None
    for col in (
        "Ih",
        "IKa",
        "INaP",
        "INaR",
        "IM",
        "IKir",
        "IKCa",
    ):
        assert col in result.dtype.names
    for gate in ("r", "a", "b", "p", "s", "hr", "w", "kir", "q"):
        assert gate in result.dtype.names


# ---------------------------------------------------------------------------
# I_M rate functions
# ---------------------------------------------------------------------------


def test_im_gating_variable_steady_state_in_bounds():
    """IM gating variable w_inf is in [0, 1] for physiological voltages."""
    for V in np.linspace(-120.0, 0.0, 50):
        a = _alpha_w(V, 0.0)
        b = _beta_w(V, 0.0)
        assert a >= 0, f"alpha_w negative at V={V}"
        assert b >= 0, f"beta_w negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"steady state {ss} out of [0,1] at V={V}"


def test_im_activation_increases_with_depolarization():
    """IM w_inf (activation) is higher at depolarized voltages."""

    def w_inf(V: float) -> float:
        """Activation steady-state at voltage V."""
        a, b = _alpha_w(V, 0.0), _beta_w(V, 0.0)
        return a / (a + b)

    assert w_inf(-20.0) > w_inf(-35.0) > w_inf(-100.0)


def test_im_slow_kinetics():
    """IM tau_w is greater than 50 ms near the half-activation voltage (-35 mV)."""
    tau = 1.0 / (_alpha_w(-35.0, 0.0) + _beta_w(-35.0, 0.0))
    assert tau > 50.0


# ---------------------------------------------------------------------------
# make_im_channel
# ---------------------------------------------------------------------------


def test_make_im_channel_defaults():
    """make_im_channel() produces a channel with the expected defaults."""
    from patch_sim.constants import DEFAULT_G_IM

    ch = make_im_channel()
    assert ch.name == "M"
    assert ch.g_max == pytest.approx(DEFAULT_G_IM)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "w"
    assert ch.gating_variables[0].power == 1


def test_make_im_channel_custom_params():
    """make_im_channel accepts custom g_max."""
    ch = make_im_channel(g_max=1.0)
    assert ch.g_max == pytest.approx(1.0)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM


# ---------------------------------------------------------------------------
# I_M integration tests
# ---------------------------------------------------------------------------


def test_current_clamp_with_im_extra_columns():
    """Current clamp with M channel adds IM and w columns."""
    neuron = _hh_with(make_im_channel())
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result.dtype.names is not None
    assert "IM" in result.dtype.names
    assert "w" in result.dtype.names


def test_voltage_clamp_with_im_extra_columns():
    """Voltage clamp with M channel adds IM and w columns."""
    neuron = _hh_with(make_im_channel())
    prot = step_voltage(
        duration=20.0,
        voltage_amplitude=0.0,
        step_start=5.0,
        step_duration=10.0,
        holding_voltage=-70.0,
        sampling_frequency=40000.0,
    )
    result = simulate_voltage_clamp(neuron, prot)
    assert result.dtype.names is not None
    assert "IM" in result.dtype.names
    assert "w" in result.dtype.names


def test_current_clamp_im_gating_in_bounds():
    """IM gating variable w stays in [0, 1] during current clamp."""
    neuron = _hh_with(make_im_channel())
    stim = step_current(
        duration=30.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=20.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result["w"].min() >= 0.0
    assert result["w"].max() <= 1.0


# ---------------------------------------------------------------------------
# I_K_ATP rate functions (kATP gate; voltage-driven proxy for the
# metabolically gated Kir6.x channel; #324)
# ---------------------------------------------------------------------------


def _kATP_inf(V: float) -> float:
    """Steady-state K_ATP activation at voltage V."""
    a, b = _alpha_kATP(V, 0.0), _beta_kATP(V, 0.0)
    return a / (a + b)


def test_katp_steady_state_in_bounds():
    """The kATP rates are positive and steady state in [0, 1] across V."""
    for V in np.linspace(-120.0, 30.0, 50):
        a = _alpha_kATP(V, 0.0)
        b = _beta_kATP(V, 0.0)
        assert a >= 0, f"alpha_kATP negative at V={V}"
        assert b >= 0, f"beta_kATP negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"kATP steady state {ss} out of [0,1] at V={V}"


def test_katp_increases_with_depolarization():
    """The kATP activation rises monotonically with depolarization."""
    assert _kATP_inf(-65.0) < _kATP_inf(-25.0) < _kATP_inf(0.0)


def test_katp_half_voltage():
    """V½ for kATP sits at -25 mV (Hahn & McIntyre 2010 fit)."""
    assert _kATP_inf(-25.0) == pytest.approx(0.5, abs=0.01)


def test_katp_subthreshold_closed():
    """Subthreshold kATP availability is near zero so rest is uncorrupted."""
    # Autonomous tonic firing cycles around -60 mV; kATP must stay closed
    # at and below the autonomous threshold so background firing is not
    # silenced by an unwanted outward K+ leak.
    assert _kATP_inf(-65.0) < 0.02


def test_katp_plateau_open():
    """At the depol-block plateau kATP opens strongly to provide block escape."""
    # The plateau sits ≈ −15 mV; kATP must open enough there to dominate
    # the residual fast-Na inward drive.
    assert _kATP_inf(-15.0) > 0.7


def test_katp_tau_is_slow():
    """The τ_kATP at V½ stays distinctly slow vs spike kinetics."""
    a, b = _alpha_kATP(-25.0, 0.0), _beta_kATP(-25.0, 0.0)
    tau = 1.0 / (a + b)
    assert tau > 200.0, f"kATP tau at V½ is {tau:.1f} ms, expected > 200 ms"


# ---------------------------------------------------------------------------
# make_katp_channel
# ---------------------------------------------------------------------------


def test_make_katp_channel_defaults():
    """make_katp_channel() produces a channel with the expected defaults."""
    from patch_sim.constants import DEFAULT_G_KATP

    ch = make_katp_channel()
    assert ch.name == "KATP"
    assert ch.g_max == pytest.approx(DEFAULT_G_KATP)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "kATP"
    assert ch.gating_variables[0].power == 1


def test_make_katp_channel_custom_params():
    """make_katp_channel accepts custom g_max."""
    ch = make_katp_channel(g_max=1.0)
    assert ch.g_max == pytest.approx(1.0)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM


# ---------------------------------------------------------------------------
# I_Kir rate functions
# ---------------------------------------------------------------------------


def test_ikir_gating_variable_steady_state_in_bounds():
    """IKir gating variable kir_inf is in [0, 1] for physiological voltages."""
    for V in np.linspace(-120.0, 0.0, 50):
        a = _alpha_kir(V, 0.0)
        b = _beta_kir(V, 0.0)
        assert a >= 0, f"alpha_kir negative at V={V}"
        assert b >= 0, f"beta_kir negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"steady state {ss} out of [0,1] at V={V}"


def test_ikir_activation_increases_with_hyperpolarization():
    """IKir kir_inf is higher at hyperpolarized voltages (inverted rectifier)."""

    def kir_inf(V: float) -> float:
        """Activation steady-state at voltage V."""
        a, b = _alpha_kir(V, 0.0), _beta_kir(V, 0.0)
        return a / (a + b)

    assert kir_inf(-100.0) > kir_inf(-80.0) > kir_inf(-40.0)


def test_ikir_fast_kinetics():
    """IKir tau_kir is at most 10 ms across physiological voltages."""
    for V in np.linspace(-120.0, 0.0, 50):
        tau = 1.0 / (_alpha_kir(V, 0.0) + _beta_kir(V, 0.0))
        assert tau <= 10.0, f"tau_kir too slow at V={V}"


# ---------------------------------------------------------------------------
# make_ikir_channel
# ---------------------------------------------------------------------------


def test_make_ikir_channel_defaults():
    """make_ikir_channel() produces a channel with the expected defaults."""
    from patch_sim.constants import DEFAULT_G_IKIR

    ch = make_ikir_channel()
    assert ch.name == "Kir"
    assert ch.g_max == pytest.approx(DEFAULT_G_IKIR)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "kir"
    assert ch.gating_variables[0].power == 1


def test_make_ikir_channel_custom_params():
    """make_ikir_channel accepts custom g_max."""
    ch = make_ikir_channel(g_max=0.5)
    assert ch.g_max == pytest.approx(0.5)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM


# ---------------------------------------------------------------------------
# I_Kir integration tests
# ---------------------------------------------------------------------------


def test_current_clamp_with_ikir_extra_columns():
    """Current clamp with Kir channel adds IKir and kir columns."""
    neuron = _hh_with(make_ikir_channel())
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result.dtype.names is not None
    assert "IKir" in result.dtype.names
    assert "kir" in result.dtype.names


def test_voltage_clamp_with_ikir_extra_columns():
    """Voltage clamp with Kir channel adds IKir and kir columns."""
    neuron = _hh_with(make_ikir_channel())
    prot = step_voltage(
        duration=20.0,
        voltage_amplitude=0.0,
        step_start=5.0,
        step_duration=10.0,
        holding_voltage=-70.0,
        sampling_frequency=40000.0,
    )
    result = simulate_voltage_clamp(neuron, prot)
    assert result.dtype.names is not None
    assert "IKir" in result.dtype.names
    assert "kir" in result.dtype.names


def test_current_clamp_ikir_gating_in_bounds():
    """IKir gating variable kir stays in [0, 1] during current clamp."""
    neuron = _hh_with(make_ikir_channel())
    stim = step_current(
        duration=30.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=20.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result["kir"].min() >= 0.0
    assert result["kir"].max() <= 1.0


# ---------------------------------------------------------------------------
# Calcium-sensitive gating variable infrastructure
# ---------------------------------------------------------------------------


def test_calcium_gating_variable_in_integrator():
    """A channel with a Ca²⁺-sensitive GatingVariable runs without error."""
    cg = GatingVariable(
        name="q_test",
        power=1,
        alpha=CalciumDependentFn(lambda V, ca_i: 0.1 * ca_i if ca_i > 0 else 0.0),
        beta=VoltageOnlyFn(lambda V, ca_i: 0.1),
    )
    ch = IonChannel(
        name="Test",
        g_max=0.5,
        gating_variables=(cg,),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )
    from patch_sim.calcium import CalciumDynamics

    neuron = _hh_with(
        ch,
        calcium_dynamics=CalciumDynamics(),
    )
    stim = step_current(
        duration=10.0,
        current_amplitude=0.0,
        step_start=2.0,
        step_duration=5.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result.dtype.names is not None
    assert "ITest" in result.dtype.names
    assert "q_test" in result.dtype.names


def test_calcium_gating_variable_steady_state_depends_on_ca():
    """GatingVariable steady state differs for different ca_i when Ca2+-sensitive."""
    cg = GatingVariable(
        name="q_test2",
        power=1,
        alpha=CalciumDependentFn(lambda V, ca_i: ca_i / (ca_i + 0.001)),
        beta=CalciumDependentFn(lambda V, ca_i: 1.0 - ca_i / (ca_i + 0.001)),
    )
    V = -65.0
    ca_low = 1e-4
    ca_high = 1e-2
    a_low, b_low = cg.alpha(V, ca_low), cg.beta(V, ca_low)
    a_high, b_high = cg.alpha(V, ca_high), cg.beta(V, ca_high)
    ss_low = a_low / (a_low + b_low)
    ss_high = a_high / (a_high + b_high)
    assert ss_high > ss_low


def test_existing_channels_unaffected_by_calcium_gating_infra():
    """Voltage-only channels still work alongside Ca²⁺-sensitive gate infrastructure."""
    neuron = _hh_with(make_ih_channel(), make_ika_channel())
    stim = step_current(
        duration=10.0,
        current_amplitude=10.0,
        step_start=2.0,
        step_duration=5.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result.dtype.names is not None
    assert "Ih" in result.dtype.names
    assert "IKa" in result.dtype.names
    assert result["r"].min() >= 0.0
    assert result["r"].max() <= 1.0


# ---------------------------------------------------------------------------
# I_KCa rate functions
# ---------------------------------------------------------------------------


def test_ikca_gating_steady_state_in_bounds():
    """IKCa gating variable q_inf is in [0, 1] for physiological V and ca_i."""
    for V in np.linspace(-120.0, 60.0, 20):
        for ca in (1e-4, 1e-3, 1e-2):
            a = _alpha_q(V, ca)
            b = _beta_q(V, ca)
            assert a >= 0, f"alpha_q negative at V={V}, ca={ca}"
            assert b >= 0, f"beta_q negative at V={V}, ca={ca}"
            ss = a / (a + b)
            assert 0.0 <= ss <= 1.0, f"steady state {ss} out of [0,1] at V={V}, ca={ca}"


def test_ikca_activation_increases_with_calcium():
    """IKCa q_inf is higher at higher [Ca²⁺]ᵢ at a fixed voltage."""
    from patch_sim.channels.auxiliary import _ikca_q_inf

    V = -20.0
    assert _ikca_q_inf(V, 1e-2) > _ikca_q_inf(V, 1e-3) > _ikca_q_inf(V, 1e-4)


def test_ikca_activation_increases_with_depolarization():
    """IKCa q_inf is higher at depolarized voltages at fixed [Ca²⁺]ᵢ."""
    from patch_sim.channels.auxiliary import _ikca_q_inf

    ca = 1e-3
    assert _ikca_q_inf(20.0, ca) > _ikca_q_inf(-20.0, ca) > _ikca_q_inf(-80.0, ca)


def test_ikca_zero_calcium_gives_zero_activation():
    """IKCa q_inf is zero when [Ca²⁺]ᵢ is zero, regardless of voltage."""
    from patch_sim.channels.auxiliary import _ikca_q_inf

    for V in np.linspace(-120.0, 60.0, 10):
        assert _ikca_q_inf(V, 0.0) == 0.0, f"q_inf non-zero at V={V} with ca=0"


def test_ikca_rates_non_negative():
    """IKCa alpha_q and beta_q are non-negative across physiological range."""
    for V in np.linspace(-120.0, 60.0, 30):
        for ca in (0.0, 1e-4, 1e-3):
            assert _alpha_q(V, ca) >= 0, f"alpha_q negative at V={V}, ca={ca}"
            assert _beta_q(V, ca) >= 0, f"beta_q negative at V={V}, ca={ca}"


# ---------------------------------------------------------------------------
# make_ikca_channel
# ---------------------------------------------------------------------------


def test_make_ikca_channel_defaults():
    """make_ikca_channel() produces a channel with the expected defaults."""
    from patch_sim.constants import DEFAULT_G_IKCA

    ch = make_ikca_channel()
    assert ch.name == "KCa"
    assert ch.g_max == pytest.approx(DEFAULT_G_IKCA)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "q"
    assert ch.gating_variables[0].power == 1


def test_make_ikca_channel_custom_params():
    """make_ikca_channel accepts a custom g_max."""
    ch = make_ikca_channel(g_max=2.0)
    assert ch.g_max == pytest.approx(2.0)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM


def test_ikca_is_not_calcium_ion_channel():
    """IKCa does not carry Ca²⁺ — carries_calcium is False."""
    ch = make_ikca_channel()
    assert not ch.carries_calcium


# ---------------------------------------------------------------------------
# I_KCa integration tests
# ---------------------------------------------------------------------------


def test_current_clamp_with_ikca():
    """Current clamp with KCa channel adds IKCa and q columns."""
    from patch_sim.calcium import CalciumDynamics

    neuron = _hh_with(
        make_ikca_channel(),
        calcium_dynamics=CalciumDynamics(),
    )
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result.dtype.names is not None
    assert "IKCa" in result.dtype.names
    assert "q" in result.dtype.names


def test_current_clamp_ikca_gating_in_bounds():
    """IKCa gating variable q stays in [0, 1] during current clamp."""
    from patch_sim.calcium import CalciumDynamics

    neuron = _hh_with(
        make_ikca_channel(),
        calcium_dynamics=CalciumDynamics(),
    )
    stim = step_current(
        duration=30.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=20.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result["q"].min() >= 0.0
    assert result["q"].max() <= 1.0


# ---------------------------------------------------------------------------
# ICaL — L-type Ca²⁺ channel
# ---------------------------------------------------------------------------


def test_ical_gating_steady_state_in_bounds():
    """ICaL gating variable steady states are in [0, 1] for physiological voltages."""
    for V in np.linspace(-120.0, 60.0, 60):
        for alpha_fn, beta_fn in ((_alpha_d, _beta_d), (_alpha_f, _beta_f)):
            a = alpha_fn(V, 0.0)
            b = beta_fn(V, 0.0)
            assert a >= 0, f"alpha negative at V={V}"
            assert b >= 0, f"beta negative at V={V}"
            ss = a / (a + b)
            assert 0.0 <= ss <= 1.0, f"steady state {ss} out of [0,1] at V={V}"


def test_ical_activation_increases_with_depolarization():
    """ICaL d_inf (activation) is higher at depolarized voltages."""

    def d_inf(V: float) -> float:
        """Activation steady-state at voltage V."""
        a, b = _alpha_d(V, 0.0), _beta_d(V, 0.0)
        return a / (a + b)

    assert d_inf(20.0) > d_inf(-30.0) > d_inf(-80.0)


def test_make_ical_channel_defaults():
    """make_ical_channel() produces a channel with the expected defaults."""
    from patch_sim.constants import DEFAULT_G_ICAL

    ch = make_ical_channel()
    assert ch.name == "CaL"
    assert ch.g_max == pytest.approx(DEFAULT_G_ICAL)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.CALCIUM
    assert len(ch.gating_variables) == 2
    assert ch.gating_variables[0].name == "d"
    assert ch.gating_variables[0].power == 2
    assert ch.gating_variables[1].name == "f"
    assert ch.gating_variables[1].power == 1
    assert ch.carries_calcium


def test_current_clamp_with_ical_extra_columns():
    """Current clamp with CaL channel adds ICaL, d, and f columns."""
    from patch_sim.calcium import CalciumDynamics

    neuron = _hh_with(
        make_ical_channel(),
        calcium_dynamics=CalciumDynamics(),
    )
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result.dtype.names is not None
    assert "ICaL" in result.dtype.names
    assert "d" in result.dtype.names
    assert "f" in result.dtype.names


def test_current_clamp_ical_gating_in_bounds():
    """ICaL gating variables d and f stay in [0, 1] during current clamp."""
    from patch_sim.calcium import CalciumDynamics

    neuron = _hh_with(
        make_ical_channel(),
        calcium_dynamics=CalciumDynamics(),
    )
    stim = step_current(
        duration=30.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=20.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result["d"].min() >= 0.0
    assert result["d"].max() <= 1.0
    assert result["f"].min() >= 0.0
    assert result["f"].max() <= 1.0


def test_voltage_clamp_with_ical_extra_columns():
    """Voltage clamp with CaL channel adds ICaL, d, and f columns."""
    from patch_sim.calcium import CalciumDynamics

    neuron = _hh_with(
        make_ical_channel(),
        calcium_dynamics=CalciumDynamics(),
    )
    prot = step_voltage(
        duration=20.0,
        voltage_amplitude=0.0,
        step_start=5.0,
        step_duration=10.0,
        holding_voltage=-70.0,
        sampling_frequency=40000.0,
    )
    result = simulate_voltage_clamp(neuron, prot)
    assert result.dtype.names is not None
    assert "ICaL" in result.dtype.names
    assert "d" in result.dtype.names
    assert "f" in result.dtype.names


# ---------------------------------------------------------------------------
# ICaT — T-type Ca²⁺ channel
# ---------------------------------------------------------------------------


def test_icat_gating_steady_state_in_bounds():
    """ICaT gating variable steady states are in [0, 1] for physiological voltages."""
    for V in np.linspace(-120.0, 60.0, 60):
        for alpha_fn, beta_fn in ((_alpha_dt, _beta_dt), (_alpha_ft, _beta_ft)):
            a = alpha_fn(V, 0.0)
            b = beta_fn(V, 0.0)
            assert a >= 0, f"alpha negative at V={V}"
            assert b >= 0, f"beta negative at V={V}"
            ss = a / (a + b)
            assert 0.0 <= ss <= 1.0, f"steady state {ss} out of [0,1] at V={V}"


def test_icat_activation_increases_with_depolarization():
    """ICaT dt_inf (activation) is higher at less-negative voltages."""

    def dt_inf(V: float) -> float:
        """Activation steady-state at voltage V."""
        a, b = _alpha_dt(V, 0.0), _beta_dt(V, 0.0)
        return a / (a + b)

    assert dt_inf(-20.0) > dt_inf(-60.0) > dt_inf(-100.0)


def test_make_icat_channel_defaults():
    """make_icat_channel() produces a channel with the expected defaults."""
    from patch_sim.constants import DEFAULT_G_ICAT

    ch = make_icat_channel()
    assert ch.name == "CaT"
    assert ch.g_max == pytest.approx(DEFAULT_G_ICAT)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.CALCIUM
    assert len(ch.gating_variables) == 2
    assert ch.gating_variables[0].name == "dt"
    assert ch.gating_variables[0].power == 2
    assert ch.gating_variables[1].name == "ft"
    assert ch.gating_variables[1].power == 1
    assert ch.carries_calcium


def test_current_clamp_with_icat_extra_columns():
    """Current clamp with CaT channel adds ICaT, dt, and ft columns."""
    from patch_sim.calcium import CalciumDynamics

    neuron = _hh_with(
        make_icat_channel(),
        calcium_dynamics=CalciumDynamics(),
    )
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result.dtype.names is not None
    assert "ICaT" in result.dtype.names
    assert "dt" in result.dtype.names
    assert "ft" in result.dtype.names


def test_current_clamp_icat_gating_in_bounds():
    """ICaT gating variables dt and ft stay in [0, 1] during current clamp."""
    from patch_sim.calcium import CalciumDynamics

    neuron = _hh_with(
        make_icat_channel(),
        calcium_dynamics=CalciumDynamics(),
    )
    stim = step_current(
        duration=30.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=20.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result["dt"].min() >= 0.0
    assert result["dt"].max() <= 1.0
    assert result["ft"].min() >= 0.0
    assert result["ft"].max() <= 1.0


def test_voltage_clamp_with_icat_extra_columns():
    """Voltage clamp with CaT channel adds ICaT, dt, and ft columns."""
    from patch_sim.calcium import CalciumDynamics

    neuron = _hh_with(
        make_icat_channel(),
        calcium_dynamics=CalciumDynamics(),
    )
    prot = step_voltage(
        duration=20.0,
        voltage_amplitude=0.0,
        step_start=5.0,
        step_duration=10.0,
        holding_voltage=-70.0,
        sampling_frequency=40000.0,
    )
    result = simulate_voltage_clamp(neuron, prot)
    assert result.dtype.names is not None
    assert "ICaT" in result.dtype.names
    assert "dt" in result.dtype.names
    assert "ft" in result.dtype.names


def test_make_trn_icat_channel_defaults():
    """make_trn_icat_channel() produces a channel with the expected defaults."""
    from patch_sim.constants import DEFAULT_G_ICAT

    ch = make_trn_icat_channel()
    assert ch.name == "CaT"
    assert ch.g_max == pytest.approx(DEFAULT_G_ICAT)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.CALCIUM
    assert len(ch.gating_variables) == 2
    assert ch.gating_variables[0].name == "dt"
    assert ch.gating_variables[0].power == 2
    assert ch.gating_variables[1].name == "ft"
    assert ch.gating_variables[1].power == 1
    assert ch.carries_calcium


def _trn_icat_ft_inf_at(V: float) -> float:
    """Compute the TRN ICaT ft_inf at voltage V from the channel rate functions.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Steady-state inactivation probability ft_inf = alpha / (alpha + beta).
    """
    ch = make_trn_icat_channel()
    ft_var = next(gv for gv in ch.gating_variables if gv.name == "ft")
    alpha = ft_var.alpha(V, 0.0)
    beta = ft_var.beta(V, 0.0)
    return alpha / (alpha + beta)


def _trn_icat_tau_ft_at(V: float) -> float:
    """Compute the TRN ICaT tau_ft at voltage V from the channel rate functions.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Time constant tau_ft = 1 / (alpha + beta) in ms.
    """
    ch = make_trn_icat_channel()
    ft_var = next(gv for gv in ch.gating_variables if gv.name == "ft")
    alpha = ft_var.alpha(V, 0.0)
    beta = ft_var.beta(V, 0.0)
    return 1.0 / (alpha + beta)


def test_trn_icat_ft_inf_matches_destexhe_at_key_voltages():
    """TRN ICaT ft_inf preserves the Destexhe (1994) shape (half=-80, slope=-9).

    The TRN factory changes only tau_ft; ft_inf is bit-identical to the
    global ICaT default to keep ft_inf-at-rest invariants for the TRN preset
    (issue #295).
    """
    assert _trn_icat_ft_inf_at(-80.0) == pytest.approx(0.50, abs=0.02)
    assert _trn_icat_ft_inf_at(-90.0) == pytest.approx(0.75, abs=0.02)
    assert _trn_icat_ft_inf_at(-60.0) == pytest.approx(0.10, abs=0.02)


def test_trn_icat_tau_ft_is_sigmoid_in_voltage():
    """TRN ICaT tau_ft increases monotonically from ~20 ms to ~200 ms.

    Sigmoid-shaped tau is the core invariant that distinguishes the TRN
    factory from the cosh-shaped Destexhe (1994) default: small at
    hyperpolarized V (rest stability) and large at LTS-plateau V (sustained
    plateau for 5–15 Na⁺ spikes per Huguenard & Prince 1992).
    """
    tau_at_minus_90 = _trn_icat_tau_ft_at(-90.0)
    tau_at_0 = _trn_icat_tau_ft_at(0.0)
    assert tau_at_minus_90 == pytest.approx(20.0, abs=2.0)
    assert tau_at_0 == pytest.approx(200.0, abs=2.0)

    voltages = np.linspace(-90.0, 0.0, 19)
    taus = np.array([_trn_icat_tau_ft_at(float(v)) for v in voltages])
    assert np.all(np.diff(taus) > 0.0), (
        f"tau_ft must be strictly increasing in V across [-90, 0] mV; "
        f"got non-monotonic samples taus={taus}"
    )


# ---------------------------------------------------------------------------
# TRN Na⁺ — optional depolarized h-gate V½ shift
# ---------------------------------------------------------------------------


def _trn_na_h_gate(channel: IonChannel) -> GatingVariable:
    """Return the h gating variable of a TRN Na⁺ channel.

    Args:
        channel: An IonChannel produced by make_trn_na_channel.

    Returns:
        The ``h`` GatingVariable.
    """
    return next(gv for gv in channel.gating_variables if gv.name == "h")


def test_trn_na_unshifted_h_gate_reuses_cached_rate_objects():
    """h_v_half_shift=0.0 keeps the module-level trn_alpha_h/beta_h objects.

    Object identity matters: the unshifted path must not allocate a wrapper
    so callers relying on rate-object identity (e.g. the voltage-clamp
    tabulation cache) keep hitting the shared instances.
    """
    h = _trn_na_h_gate(make_trn_na_channel(g_max=1.0))
    assert h.alpha is trn_alpha_h
    assert h.beta is trn_beta_h


def test_trn_na_h_v_half_shift_evaluates_at_shifted_voltage():
    """The shifted h gate evaluates Traub-Miles α_h/β_h at V − h_v_half_shift.

    A +5 mV depolarized V½ shift means the gate at membrane voltage V
    behaves as the unshifted gate at V − 5 mV — larger α_h (faster recovery
    from inactivation) at the LTS-plateau voltages.  m kinetics stay on the
    shared TRN_VT and must be untouched by the shift.
    """
    shift = 5.0
    ch = make_trn_na_channel(g_max=1.0, h_v_half_shift=shift)
    h = _trn_na_h_gate(ch)
    m = next(gv for gv in ch.gating_variables if gv.name == "m")
    for V in (-80.0, -65.0, -30.0, 0.0, 30.0):
        assert h.alpha(V, 0.0) == pytest.approx(trn_alpha_h(V - shift, 0.0))
        assert h.beta(V, 0.0) == pytest.approx(trn_beta_h(V - shift, 0.0))
        assert m.alpha(V, 0.0) == pytest.approx(trn_alpha_m(V, 0.0))


def test_trn_na_shifted_h_rate_survives_pickle_round_trip():
    """Shifted h rates pickle cleanly (simulate_batch ships them to workers).

    The shift is a frozen dataclass rather than a closure specifically so
    simulate_batch can pickle the channel when handing it to a worker
    process; a closure would raise PicklingError here.
    """
    shift = 5.0
    h = _trn_na_h_gate(make_trn_na_channel(g_max=1.0, h_v_half_shift=shift))
    alpha = pickle.loads(pickle.dumps(h.alpha))
    beta = pickle.loads(pickle.dumps(h.beta))
    for V in (-80.0, -30.0, 0.0, 30.0):
        assert alpha(V, 0.0) == pytest.approx(trn_alpha_h(V - shift, 0.0))
        assert beta(V, 0.0) == pytest.approx(trn_beta_h(V - shift, 0.0))


# ---------------------------------------------------------------------------
# ICaN — N-type Ca²⁺ channel
# ---------------------------------------------------------------------------


def test_ican_gating_steady_state_in_bounds():
    """ICaN gating variable steady states are in [0, 1] for physiological voltages."""
    for V in np.linspace(-120.0, 60.0, 60):
        for alpha_fn, beta_fn in ((_alpha_dn, _beta_dn), (_alpha_fn, _beta_fn)):
            a = alpha_fn(V, 0.0)
            b = beta_fn(V, 0.0)
            assert a >= 0, f"alpha negative at V={V}"
            assert b >= 0, f"beta negative at V={V}"
            ss = a / (a + b)
            assert 0.0 <= ss <= 1.0, f"steady state {ss} out of [0,1] at V={V}"


def test_ican_activation_increases_with_depolarization():
    """ICaN dn_inf (activation) is higher at depolarized voltages."""

    def dn_inf(V: float) -> float:
        """Activation steady-state at voltage V."""
        a, b = _alpha_dn(V, 0.0), _beta_dn(V, 0.0)
        return a / (a + b)

    assert dn_inf(20.0) > dn_inf(-30.0) > dn_inf(-80.0)


def test_make_ican_channel_defaults():
    """make_ican_channel() produces a channel with the expected defaults."""
    from patch_sim.constants import DEFAULT_G_ICAN

    ch = make_ican_channel()
    assert ch.name == "CaN"
    assert ch.g_max == pytest.approx(DEFAULT_G_ICAN)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.CALCIUM
    assert len(ch.gating_variables) == 2
    assert ch.gating_variables[0].name == "dn"
    assert ch.gating_variables[0].power == 2
    assert ch.gating_variables[1].name == "fn"
    assert ch.gating_variables[1].power == 1
    assert ch.carries_calcium


def test_current_clamp_with_ican_extra_columns():
    """Current clamp with CaN channel adds ICaN, dn, and fn columns."""
    from patch_sim.calcium import CalciumDynamics

    neuron = _hh_with(
        make_ican_channel(),
        calcium_dynamics=CalciumDynamics(),
    )
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result.dtype.names is not None
    assert "ICaN" in result.dtype.names
    assert "dn" in result.dtype.names
    assert "fn" in result.dtype.names


def test_current_clamp_ican_gating_in_bounds():
    """ICaN gating variables dn and fn stay in [0, 1] during current clamp."""
    from patch_sim.calcium import CalciumDynamics

    neuron = _hh_with(
        make_ican_channel(),
        calcium_dynamics=CalciumDynamics(),
    )
    stim = step_current(
        duration=30.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=20.0,
        sampling_frequency=40000.0,
    )
    result = simulate_current_clamp(neuron, stim)
    assert result["dn"].min() >= 0.0
    assert result["dn"].max() <= 1.0
    assert result["fn"].min() >= 0.0
    assert result["fn"].max() <= 1.0


def test_voltage_clamp_with_ican_extra_columns():
    """Voltage clamp with CaN channel adds ICaN, dn, and fn columns."""
    from patch_sim.calcium import CalciumDynamics

    neuron = _hh_with(
        make_ican_channel(),
        calcium_dynamics=CalciumDynamics(),
    )
    prot = step_voltage(
        duration=20.0,
        voltage_amplitude=0.0,
        step_start=5.0,
        step_duration=10.0,
        holding_voltage=-70.0,
        sampling_frequency=40000.0,
    )
    result = simulate_voltage_clamp(neuron, prot)
    assert result.dtype.names is not None
    assert "ICaN" in result.dtype.names
    assert "dn" in result.dtype.names
    assert "fn" in result.dtype.names


# ---------------------------------------------------------------------------
# Core (HH-style) channels — merged from the former test_core_channels.py.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Rate function positivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_alpha_n_positive(V: float) -> None:
    """alpha_n is positive at physiological voltages."""
    assert alpha_n(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_beta_n_positive(V: float) -> None:
    """beta_n is positive at physiological voltages."""
    assert beta_n(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_alpha_m_positive(V: float) -> None:
    """alpha_m is positive at physiological voltages."""
    assert alpha_m(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_beta_m_positive(V: float) -> None:
    """beta_m is positive at physiological voltages."""
    assert beta_m(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_alpha_h_positive(V: float) -> None:
    """alpha_h is positive at physiological voltages."""
    assert alpha_h(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_beta_h_positive(V: float) -> None:
    """beta_h is positive at physiological voltages."""
    assert beta_h(V, 0.0) > 0


# ---------------------------------------------------------------------------
# Singularity guards
# ---------------------------------------------------------------------------


def test_alpha_n_singularity_guard() -> None:
    """alpha_n is finite and positive at the singularity V = −55 mV."""
    val = alpha_n(-55.0, 0.0)
    assert math.isfinite(val)
    assert val > 0.0


def test_alpha_n_near_singularity_continuous_above() -> None:
    """alpha_n is continuous approaching −55 mV from above."""
    assert alpha_n(-55.0 + 1e-5, 0.0) == pytest.approx(0.1, rel=1e-3)


def test_alpha_n_near_singularity_continuous_below() -> None:
    """alpha_n is continuous approaching −55 mV from below."""
    assert alpha_n(-55.0 - 1e-5, 0.0) == pytest.approx(0.1, rel=1e-3)


def test_alpha_m_singularity_guard() -> None:
    """alpha_m is finite and positive at the singularity V = −40 mV."""
    val = alpha_m(-40.0, 0.0)
    assert math.isfinite(val)
    assert val > 0.0


def test_alpha_m_near_singularity_continuous_above() -> None:
    """alpha_m is continuous approaching −40 mV from above."""
    assert alpha_m(-40.0 + 1e-5, 0.0) == pytest.approx(1.0, rel=1e-3)


def test_alpha_m_near_singularity_continuous_below() -> None:
    """alpha_m is continuous approaching −40 mV from below."""
    assert alpha_m(-40.0 - 1e-5, 0.0) == pytest.approx(1.0, rel=1e-3)


# ---------------------------------------------------------------------------
# ca_i independence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-65.0, 0.0])
@pytest.mark.parametrize("fn", [alpha_n, beta_n, alpha_m, beta_m, alpha_h, beta_h])
def test_rate_functions_ignore_ca_i(V: float, fn: Rate) -> None:
    """All rate functions return the same value regardless of ca_i."""
    assert fn(V, 0.0) == pytest.approx(fn(V, 1.0))


# ---------------------------------------------------------------------------
# Steady-state bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_steady_state_gating_bounds(V: float) -> None:
    """Steady-state gating variables computed from module-level rates are in [0,1]."""
    n_inf = alpha_n(V, 0.0) / (alpha_n(V, 0.0) + beta_n(V, 0.0))
    m_inf = alpha_m(V, 0.0) / (alpha_m(V, 0.0) + beta_m(V, 0.0))
    h_inf = alpha_h(V, 0.0) / (alpha_h(V, 0.0) + beta_h(V, 0.0))
    assert 0.0 <= n_inf <= 1.0
    assert 0.0 <= m_inf <= 1.0
    assert 0.0 <= h_inf <= 1.0


# ---------------------------------------------------------------------------
# Factory function structure
# ---------------------------------------------------------------------------


def test_make_na_channel_structure() -> None:
    """make_na_channel returns a channel with correct name, gates, and reversal spec."""
    ch = make_na_channel(g_max=120.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "Na"
    assert ch.g_max == pytest.approx(120.0)
    assert len(ch.gating_variables) == 2
    assert ch.gating_variables[0].name == "m"
    assert ch.gating_variables[0].power == 3
    assert ch.gating_variables[1].name == "h"
    assert ch.gating_variables[1].power == 1
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.SODIUM
    assert not ch.carries_calcium


def test_make_k_channel_structure() -> None:
    """make_k_channel returns a channel with correct name, gate, and reversal spec."""
    ch = make_k_channel(g_max=36.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "K"
    assert ch.g_max == pytest.approx(36.0)
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "n"
    assert ch.gating_variables[0].power == 4
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM
    assert not ch.carries_calcium


def test_make_na_leak_channel_structure() -> None:
    """make_na_leak_channel returns a channel with no gates and Na⁺ reversal spec."""
    ch = make_na_leak_channel(g_max=0.054)
    assert isinstance(ch, IonChannel)
    assert ch.name == "NaL"
    assert ch.g_max == pytest.approx(0.054)
    assert len(ch.gating_variables) == 0
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.SODIUM
    assert not ch.carries_calcium


def test_make_k_leak_channel_structure() -> None:
    """make_k_leak_channel returns a channel with no gates and K⁺ reversal spec."""
    ch = make_k_leak_channel(g_max=0.246)
    assert isinstance(ch, IonChannel)
    assert ch.name == "KL"
    assert ch.g_max == pytest.approx(0.246)
    assert len(ch.gating_variables) == 0
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM
    assert not ch.carries_calcium


# ---------------------------------------------------------------------------
# Numerical equivalence with old inline formulas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -80.0, -65.0, -55.0, -40.0, 0.0, 20.0, 40.0])
def test_na_channel_current_matches_inline(V: float) -> None:
    """Na channel compute_current equals g_Na * m³ * h * (V − E_Na)."""
    neuron = Neuron()
    g_Na = 120.0
    ch = make_na_channel(g_max=g_Na)

    m = alpha_m(V, 0.0) / (alpha_m(V, 0.0) + beta_m(V, 0.0))
    h = alpha_h(V, 0.0) / (alpha_h(V, 0.0) + beta_h(V, 0.0))
    gating_state = {"m": m, "h": h}

    result = ch.compute_current(V, gating_state, neuron)
    E_Na = ch.reversal_potential(neuron)
    expected = g_Na * (m**3) * h * (V - E_Na)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("V", [-100.0, -80.0, -65.0, -55.0, 0.0, 20.0, 40.0])
def test_k_channel_current_matches_inline(V: float) -> None:
    """K channel compute_current equals g_K * n⁴ * (V − E_K)."""
    neuron = Neuron()
    g_K = 36.0
    ch = make_k_channel(g_max=g_K)

    n = alpha_n(V, 0.0) / (alpha_n(V, 0.0) + beta_n(V, 0.0))
    gating_state = {"n": n}

    result = ch.compute_current(V, gating_state, neuron)
    E_K = ch.reversal_potential(neuron)
    expected = g_K * (n**4) * (V - E_K)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_na_leak_channel_current_matches_inline(V: float) -> None:
    """Na leak channel compute_current equals g_NaL * (V − E_Na)."""
    neuron = Neuron()
    g_NaL = 0.054
    ch = make_na_leak_channel(g_max=g_NaL)

    result = ch.compute_current(V, {}, neuron)
    E_Na = ch.reversal_potential(neuron)
    expected = g_NaL * (V - E_Na)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_k_leak_channel_current_matches_inline(V: float) -> None:
    """K leak channel compute_current equals g_KL * (V − E_K)."""
    neuron = Neuron()
    g_KL = 0.246
    ch = make_k_leak_channel(g_max=g_KL)

    result = ch.compute_current(V, {}, neuron)
    E_K = ch.reversal_potential(neuron)
    expected = g_KL * (V - E_K)
    assert result == pytest.approx(expected)


# ===========================================================================
# Pospischil et al. (2008) cortical RS kinetics
# ===========================================================================

# ---------------------------------------------------------------------------
# Rate function positivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_pospischil_alpha_m_positive(V: float) -> None:
    """pospischil_alpha_m is positive at physiological voltages."""
    assert pospischil_alpha_m(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_pospischil_beta_m_positive(V: float) -> None:
    """pospischil_beta_m is positive at physiological voltages."""
    assert pospischil_beta_m(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_pospischil_alpha_h_positive(V: float) -> None:
    """pospischil_alpha_h is positive at physiological voltages."""
    assert pospischil_alpha_h(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_pospischil_beta_h_positive(V: float) -> None:
    """pospischil_beta_h is positive at physiological voltages."""
    assert pospischil_beta_h(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_pospischil_alpha_n_positive(V: float) -> None:
    """pospischil_alpha_n is positive at physiological voltages."""
    assert pospischil_alpha_n(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_pospischil_beta_n_positive(V: float) -> None:
    """pospischil_beta_n is positive at physiological voltages."""
    assert pospischil_beta_n(V, 0.0) > 0


# ---------------------------------------------------------------------------
# Singularity guards
# ---------------------------------------------------------------------------

# Singularity voltages derived from VT = -56.2 mV:
#   pospischil_alpha_m: V = VT + 13 = -43.2 mV, limit = 0.32 * 4 = 1.28
#   pospischil_beta_m:  V = VT + 40 = -16.2 mV, limit = 0.28 * 5 = 1.4
#   pospischil_alpha_n: V = VT + 15 = -41.2 mV, limit = 0.032 * 5 = 0.16

_ALPHA_M_SINGULARITY = POSPISCHIL_VT + 13  # -43.2 mV
_BETA_M_SINGULARITY = POSPISCHIL_VT + 40  # -16.2 mV
_ALPHA_N_SINGULARITY = POSPISCHIL_VT + 15  # -41.2 mV


def test_pospischil_alpha_m_singularity_guard() -> None:
    """pospischil_alpha_m returns L'Hôpital limit 1.28 at V = VT + 13."""
    assert pospischil_alpha_m(_ALPHA_M_SINGULARITY, 0.0) == pytest.approx(1.28)


def test_pospischil_alpha_m_near_singularity_continuous_above() -> None:
    """pospischil_alpha_m is continuous approaching the singularity from above."""
    assert pospischil_alpha_m(_ALPHA_M_SINGULARITY + 1e-5, 0.0) == pytest.approx(
        1.28, rel=1e-3
    )


def test_pospischil_alpha_m_near_singularity_continuous_below() -> None:
    """pospischil_alpha_m is continuous approaching the singularity from below."""
    assert pospischil_alpha_m(_ALPHA_M_SINGULARITY - 1e-5, 0.0) == pytest.approx(
        1.28, rel=1e-3
    )


def test_pospischil_beta_m_singularity_guard() -> None:
    """pospischil_beta_m returns L'Hôpital limit 1.4 at V = VT + 40."""
    assert pospischil_beta_m(_BETA_M_SINGULARITY, 0.0) == pytest.approx(1.4)


def test_pospischil_beta_m_near_singularity_continuous_above() -> None:
    """pospischil_beta_m is continuous approaching the singularity from above."""
    assert pospischil_beta_m(_BETA_M_SINGULARITY + 1e-5, 0.0) == pytest.approx(
        1.4, rel=1e-3
    )


def test_pospischil_beta_m_near_singularity_continuous_below() -> None:
    """pospischil_beta_m is continuous approaching the singularity from below."""
    assert pospischil_beta_m(_BETA_M_SINGULARITY - 1e-5, 0.0) == pytest.approx(
        1.4, rel=1e-3
    )


def test_pospischil_alpha_n_singularity_guard() -> None:
    """pospischil_alpha_n returns L'Hôpital limit 0.16 at V = VT + 15."""
    assert pospischil_alpha_n(_ALPHA_N_SINGULARITY, 0.0) == pytest.approx(0.16)


def test_pospischil_alpha_n_near_singularity_continuous_above() -> None:
    """pospischil_alpha_n is continuous approaching the singularity from above."""
    assert pospischil_alpha_n(_ALPHA_N_SINGULARITY + 1e-5, 0.0) == pytest.approx(
        0.16, rel=1e-3
    )


def test_pospischil_alpha_n_near_singularity_continuous_below() -> None:
    """pospischil_alpha_n is continuous approaching the singularity from below."""
    assert pospischil_alpha_n(_ALPHA_N_SINGULARITY - 1e-5, 0.0) == pytest.approx(
        0.16, rel=1e-3
    )


# ---------------------------------------------------------------------------
# ca_i independence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-65.0, 0.0])
@pytest.mark.parametrize(
    "fn",
    [
        pospischil_alpha_m,
        pospischil_beta_m,
        pospischil_alpha_h,
        pospischil_beta_h,
        pospischil_alpha_n,
        pospischil_beta_n,
    ],
)
def test_pospischil_rate_functions_ignore_ca_i(V: float, fn: Rate) -> None:
    """All Pospischil rate functions return the same value regardless of ca_i."""
    assert fn(V, 0.0) == pytest.approx(fn(V, 1.0))


# ---------------------------------------------------------------------------
# Steady-state bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_pospischil_steady_state_gating_bounds(V: float) -> None:
    """Pospischil steady-state gating variables are in [0, 1]."""
    n_inf = pospischil_alpha_n(V, 0.0) / (
        pospischil_alpha_n(V, 0.0) + pospischil_beta_n(V, 0.0)
    )
    m_inf = pospischil_alpha_m(V, 0.0) / (
        pospischil_alpha_m(V, 0.0) + pospischil_beta_m(V, 0.0)
    )
    h_inf = pospischil_alpha_h(V, 0.0) / (
        pospischil_alpha_h(V, 0.0) + pospischil_beta_h(V, 0.0)
    )
    assert 0.0 <= n_inf <= 1.0
    assert 0.0 <= m_inf <= 1.0
    assert 0.0 <= h_inf <= 1.0


# ---------------------------------------------------------------------------
# Factory function structure
# ---------------------------------------------------------------------------


def test_make_nav12_channel_structure() -> None:
    """make_nav12_channel returns a (m, h, sNa12) Nav1.2-flavoured Na channel."""
    ch = make_nav12_channel(g_max=50.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "Na"
    assert ch.g_max == pytest.approx(50.0)
    assert len(ch.gating_variables) == 3
    assert ch.gating_variables[0].name == "m"
    assert ch.gating_variables[0].power == 3
    assert ch.gating_variables[1].name == "h"
    assert ch.gating_variables[1].power == 1
    assert ch.gating_variables[2].name == "sNa12"
    assert ch.gating_variables[2].power == 1
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.SODIUM
    assert not ch.carries_calcium


def test_make_nav11_channel_structure() -> None:
    """make_nav11_channel returns a (m, h, sNa11) Nav1.1-flavoured Na channel."""
    ch = make_nav11_channel(g_max=80.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "Na"
    assert ch.g_max == pytest.approx(80.0)
    assert len(ch.gating_variables) == 3
    assert ch.gating_variables[0].name == "m"
    assert ch.gating_variables[0].power == 3
    assert ch.gating_variables[1].name == "h"
    assert ch.gating_variables[1].power == 1
    assert ch.gating_variables[2].name == "sNa11"
    assert ch.gating_variables[2].power == 1
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.SODIUM
    assert not ch.carries_calcium


# ---------------------------------------------------------------------------
# Nav1.2 slow Na inactivation rate functions (sNa12 gate; Fleidervish &
# Gutnick 1996; Mickus, Jung & Spruston 1999).  Always-on gate baked into
# make_nav12_channel to abolish the residual depol-block plateau that
# single-gate Pospischil kinetics leave open under sustained suprathreshold
# drive.
# ---------------------------------------------------------------------------


def _nav12_sNa_inf(V: float) -> float:
    """Steady-state availability of the Nav1.2 slow Na inactivation gate.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Steady-state availability of the sNa12 gate at voltage V, in [0, 1].
    """
    a, b = _nav12_alpha_sNa(V, 0.0), _nav12_beta_sNa(V, 0.0)
    return a / (a + b)


def test_nav12_slow_na_inactivation_steady_state_in_bounds() -> None:
    """The sNa12 rates are positive and steady state in [0, 1] across V."""
    for V in (-120.0, -100.0, -75.0, -65.0, -50.0, -30.0, -15.0, 0.0, 30.0):
        a = _nav12_alpha_sNa(V, 0.0)
        b = _nav12_beta_sNa(V, 0.0)
        assert a >= 0, f"alpha_sNa12 negative at V={V}"
        assert b >= 0, f"beta_sNa12 negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"sNa12 steady state {ss} out of [0,1] at V={V}"


def test_nav12_slow_na_inactivation_decreases_with_depolarization() -> None:
    """The sNa12 availability decreases monotonically with depolarization."""
    assert _nav12_sNa_inf(-80.0) > _nav12_sNa_inf(-50.0) > _nav12_sNa_inf(-15.0)


def test_nav12_slow_na_inactivation_half_voltage() -> None:
    """V½ for sNa12 sits at -50 mV (Fleidervish & Gutnick 1996 mid-range)."""
    assert _nav12_sNa_inf(-50.0) == pytest.approx(0.5, abs=0.01)


def test_nav12_slow_na_inactivation_resting_availability() -> None:
    """Cortical pyramidal rests at -70 mV — sNa12 must remain near-fully open."""
    # Cortical pyramidal v_rest = -70 mV (deeper than STN's -60 mV cycle), so
    # subthreshold sNa12 availability should be even higher than the STN
    # gate's rest value.  Loss of >10 % rest availability would noticeably
    # suppress AP amplitude on every step from rest.
    assert _nav12_sNa_inf(-70.0) > 0.9


def test_nav12_slow_na_inactivation_blocks_depol_plateau() -> None:
    """At depolarized plateau voltages sNa12 closes, abolishing the residual h-tail."""
    # The depol-block plateau the gate must escape (mirroring #324) hangs at
    # ≈ −15 mV; sNa12 must close firmly there so g_Na_eff = g_max * m^3 * h
    # * sNa12 collapses below the leak + IM outward drive.
    assert _nav12_sNa_inf(-15.0) < 0.05


def test_nav12_slow_na_inactivation_tau_is_slow() -> None:
    """τ_sNa12 at V½ stays distinctly slow vs the fast m, h gates."""
    a = _nav12_alpha_sNa(-50.0, 0.0)
    b = _nav12_beta_sNa(-50.0, 0.0)
    tau = 1.0 / (a + b)
    assert tau > 100.0, f"sNa12 tau at V½ is {tau:.1f} ms, expected > 100 ms"


# ---------------------------------------------------------------------------
# Nav1.1 slow Na inactivation rate functions (sNa11 gate; Patel et al. 2015
# Nav1.1 vs Nav1.6 comparison).  Weak gate baked into make_nav11_channel:
# V½ = −45 mV with much slower kinetics (τ_scale = 50000 ms,
# τ_floor = 5000 ms) so the gate barely engages at the 100–500 Hz firing
# rates typical of FSI.
# ---------------------------------------------------------------------------


def _nav11_sNa_inf(V: float) -> float:
    """Steady-state availability of the Nav1.1 slow Na inactivation gate.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Steady-state availability of the sNa11 gate at voltage V, in [0, 1].
    """
    a, b = _nav11_alpha_sNa(V, 0.0), _nav11_beta_sNa(V, 0.0)
    return a / (a + b)


def test_nav11_slow_na_inactivation_steady_state_in_bounds() -> None:
    """The sNa11 rates are positive and steady state in [0, 1] across V."""
    for V in (-120.0, -100.0, -75.0, -65.0, -50.0, -30.0, -15.0, 0.0, 30.0):
        a = _nav11_alpha_sNa(V, 0.0)
        b = _nav11_beta_sNa(V, 0.0)
        assert a >= 0, f"alpha_sNa11 negative at V={V}"
        assert b >= 0, f"beta_sNa11 negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"sNa11 steady state {ss} out of [0,1] at V={V}"


def test_nav11_slow_na_inactivation_half_voltage() -> None:
    """V½ for sNa11 sits at -45 mV (native β-subunit-shifted Nav1.1 estimate)."""
    assert _nav11_sNa_inf(-45.0) == pytest.approx(0.5, abs=0.01)


def test_nav11_slow_na_inactivation_resting_availability() -> None:
    """At FSI v_rest = −65 mV the sNa11 gate must be near-fully open.

    A meaningful reduction at rest would suppress Na availability and
    break the high-frequency firing phenotype FSI relies on.  V½ = −45 mV
    keeps sNa11_inf above 0.9 at v_rest = −65 mV.
    """
    assert _nav11_sNa_inf(-65.0) > 0.9


def test_nav11_slow_na_inactivation_tau_floor_is_seconds() -> None:
    """τ_sNa11 must be in seconds at every voltage so the gate barely moves over 1 s.

    The kinetic separation is the whole reason the gate survives on FSI:
    even at AP peak voltages (~+30 mV) the gate must move slowly enough
    that 250+ APs over 1 s do not collapse FSI Na availability.
    τ_floor = 5000 ms enforces this.
    """
    for V in (-65.0, -45.0, -15.0, 0.0, 30.0):
        a = _nav11_alpha_sNa(V, 0.0)
        b = _nav11_beta_sNa(V, 0.0)
        tau = 1.0 / (a + b)
        assert tau >= 4990.0, (
            f"sNa11 τ at V={V} mV is {tau:.1f} ms — must be at the 5000 ms "
            f"floor or above so per-spike inactivation closure is negligible."
        )


def test_make_pospischil_k_channel_structure() -> None:
    """make_pospischil_k_channel returns a channel with correct structure."""
    ch = make_pospischil_k_channel(g_max=5.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "K"
    assert ch.g_max == pytest.approx(5.0)
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "n"
    assert ch.gating_variables[0].power == 4
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM
    assert not ch.carries_calcium


# ---------------------------------------------------------------------------
# Numerical equivalence with inline formulas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -80.0, -65.0, -43.2, -16.2, 0.0, 20.0, 40.0])
def test_nav12_channel_current_matches_inline(V: float) -> None:
    """nav12 Na channel compute_current equals g_Na * m³ * h * sNa12 * (V − E_Na)."""
    neuron = Neuron()
    g_Na = 120.0
    ch = make_nav12_channel(g_max=g_Na)

    m = pospischil_alpha_m(V, 0.0) / (
        pospischil_alpha_m(V, 0.0) + pospischil_beta_m(V, 0.0)
    )
    h = pospischil_alpha_h(V, 0.0) / (
        pospischil_alpha_h(V, 0.0) + pospischil_beta_h(V, 0.0)
    )
    s = _nav12_alpha_sNa(V, 0.0) / (_nav12_alpha_sNa(V, 0.0) + _nav12_beta_sNa(V, 0.0))
    gating_state = {"m": m, "h": h, "sNa12": s}

    result = ch.compute_current(V, gating_state, neuron)
    E_Na = ch.reversal_potential(neuron)
    expected = g_Na * (m**3) * h * s * (V - E_Na)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("V", [-100.0, -80.0, -65.0, -41.2, 0.0, 20.0, 40.0])
def test_pospischil_k_channel_current_matches_inline(V: float) -> None:
    """Pospischil K channel compute_current equals g_K * n⁴ * (V − E_K)."""
    neuron = Neuron()
    g_K = 36.0
    ch = make_pospischil_k_channel(g_max=g_K)

    n = pospischil_alpha_n(V, 0.0) / (
        pospischil_alpha_n(V, 0.0) + pospischil_beta_n(V, 0.0)
    )
    gating_state = {"n": n}

    result = ch.compute_current(V, gating_state, neuron)
    E_K = ch.reversal_potential(neuron)
    expected = g_K * (n**4) * (V - E_K)
    assert result == pytest.approx(expected)


# ===========================================================================
# Mainen & Sejnowski (1996) cortical pyramidal Kv kinetics
# ===========================================================================

# ---------------------------------------------------------------------------
# Rate function positivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 25.0, 40.0])
def test_mainen_sejnowski_alpha_n_positive(V: float) -> None:
    """mainen_sejnowski_alpha_n is positive at physiological voltages."""
    assert mainen_sejnowski_alpha_n(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 25.0, 40.0])
def test_mainen_sejnowski_beta_n_positive(V: float) -> None:
    """mainen_sejnowski_beta_n is positive at physiological voltages."""
    assert mainen_sejnowski_beta_n(V, 0.0) > 0


# ---------------------------------------------------------------------------
# Singularity guards at V = V_½ = +25 mV
# ---------------------------------------------------------------------------

# Expected L'Hôpital limits with the 23 → 34 °C Q10 = 2.3 pre-scale baked in:
#   alpha_n(25) = 0.02 * 2.55 * 9 = 0.459 / ms
#   beta_n(25)  = 0.002 * 2.55 * 9 = 0.0459 / ms
_MS_ALPHA_N_LIMIT = 0.02 * MAINEN_SEJNOWSKI_KV_PRESCALE * 9.0
_MS_BETA_N_LIMIT = 0.002 * MAINEN_SEJNOWSKI_KV_PRESCALE * 9.0


def test_mainen_sejnowski_alpha_n_singularity_guard() -> None:
    """mainen_sejnowski_alpha_n returns the L'Hôpital limit at V = 25 mV."""
    assert mainen_sejnowski_alpha_n(MAINEN_SEJNOWSKI_KV_VHALF, 0.0) == pytest.approx(
        _MS_ALPHA_N_LIMIT
    )


def test_mainen_sejnowski_alpha_n_continuous_above() -> None:
    """mainen_sejnowski_alpha_n is continuous approaching the singularity from above."""
    assert mainen_sejnowski_alpha_n(
        MAINEN_SEJNOWSKI_KV_VHALF + 1e-5, 0.0
    ) == pytest.approx(_MS_ALPHA_N_LIMIT, rel=1e-3)


def test_mainen_sejnowski_alpha_n_continuous_below() -> None:
    """mainen_sejnowski_alpha_n is continuous approaching the singularity from below."""
    assert mainen_sejnowski_alpha_n(
        MAINEN_SEJNOWSKI_KV_VHALF - 1e-5, 0.0
    ) == pytest.approx(_MS_ALPHA_N_LIMIT, rel=1e-3)


def test_mainen_sejnowski_beta_n_singularity_guard() -> None:
    """mainen_sejnowski_beta_n returns the L'Hôpital limit at V = 25 mV."""
    assert mainen_sejnowski_beta_n(MAINEN_SEJNOWSKI_KV_VHALF, 0.0) == pytest.approx(
        _MS_BETA_N_LIMIT
    )


def test_mainen_sejnowski_beta_n_continuous_above() -> None:
    """mainen_sejnowski_beta_n is continuous approaching the singularity from above."""
    assert mainen_sejnowski_beta_n(
        MAINEN_SEJNOWSKI_KV_VHALF + 1e-5, 0.0
    ) == pytest.approx(_MS_BETA_N_LIMIT, rel=1e-3)


def test_mainen_sejnowski_beta_n_continuous_below() -> None:
    """mainen_sejnowski_beta_n is continuous approaching the singularity from below."""
    assert mainen_sejnowski_beta_n(
        MAINEN_SEJNOWSKI_KV_VHALF - 1e-5, 0.0
    ) == pytest.approx(_MS_BETA_N_LIMIT, rel=1e-3)


# ---------------------------------------------------------------------------
# Steady-state shape: high-threshold, mostly closed at rest
# ---------------------------------------------------------------------------


def test_mainen_sejnowski_n_inf_closed_at_rest() -> None:
    """n_inf is < 0.05 at v_rest = -70 mV (channel almost fully closed)."""
    a = mainen_sejnowski_alpha_n(-70.0, 0.0)
    b = mainen_sejnowski_beta_n(-70.0, 0.0)
    n_inf = a / (a + b)
    assert n_inf < 0.05


def test_mainen_sejnowski_n_inf_strongly_open_at_peak() -> None:
    """n_inf > 0.9 above the AP peak voltage so K⁺ repolarizes after the spike.

    The published M-S Kv has α/β = 10 at V = ``MAINEN_SEJNOWSKI_KV_VHALF`` (the
    rate-function singularity), so n_inf ≈ 0.91 there.  This is intentional —
    the channel is essentially fully open above ~+10 mV, providing strong
    repolarizing drive once the AP threshold is crossed.
    """
    a = mainen_sejnowski_alpha_n(MAINEN_SEJNOWSKI_KV_VHALF, 0.0)
    b = mainen_sejnowski_beta_n(MAINEN_SEJNOWSKI_KV_VHALF, 0.0)
    n_inf = a / (a + b)
    assert n_inf > 0.9


def test_mainen_sejnowski_n_inf_half_activated_near_zero() -> None:
    """n_inf passes through 0.5 between -5 and +10 mV (true activation V_½).

    Solving α(V) = β(V) for the published prefactors gives V_½ ≈ +4.3 mV.
    The activation V_½ is *not* the same as MAINEN_SEJNOWSKI_KV_VHALF — that
    constant marks the rate-function singularity at V = +25 mV.
    """
    n_inf_minus5 = mainen_sejnowski_alpha_n(-5.0, 0.0) / (
        mainen_sejnowski_alpha_n(-5.0, 0.0) + mainen_sejnowski_beta_n(-5.0, 0.0)
    )
    n_inf_plus10 = mainen_sejnowski_alpha_n(10.0, 0.0) / (
        mainen_sejnowski_alpha_n(10.0, 0.0) + mainen_sejnowski_beta_n(10.0, 0.0)
    )
    assert n_inf_minus5 < 0.5 < n_inf_plus10


# ---------------------------------------------------------------------------
# ca_i independence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-65.0, 0.0, 30.0])
@pytest.mark.parametrize(
    "fn",
    [mainen_sejnowski_alpha_n, mainen_sejnowski_beta_n],
)
def test_mainen_sejnowski_rate_functions_ignore_ca_i(V: float, fn: Rate) -> None:
    """Mainen-Sejnowski Kv rates return the same value regardless of ca_i."""
    assert fn(V, 0.0) == pytest.approx(fn(V, 1.0))


# ---------------------------------------------------------------------------
# Factory function structure
# ---------------------------------------------------------------------------


def test_make_mainen_sejnowski_kv_channel_structure() -> None:
    """make_mainen_sejnowski_kv_channel returns a channel with correct structure.

    Uses channel name ``"Kv"`` (not ``"K"``) and gating-variable name ``"nKv"``
    so it can coexist with a Pospischil delayed rectifier in the same neuron
    if a future preset wants a dual-K architecture.
    """
    ch = make_mainen_sejnowski_kv_channel(g_max=30.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "Kv"
    assert ch.g_max == pytest.approx(30.0)
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "nKv"
    assert ch.gating_variables[0].power == 1
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM
    assert not ch.carries_calcium


# ---------------------------------------------------------------------------
# Mainen-Sejnowski Na rate functions and factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -65.0, -35.0, 0.0, 40.0])
def test_mainen_sejnowski_na_rates_positive(V: float) -> None:
    """All M-S Na rate functions are positive at physiological voltages."""
    assert mainen_sejnowski_alpha_m(V, 0.0) > 0
    assert mainen_sejnowski_beta_m(V, 0.0) > 0
    assert mainen_sejnowski_alpha_h(V, 0.0) > 0
    assert mainen_sejnowski_beta_h(V, 0.0) > 0


def test_mainen_sejnowski_alpha_m_singularity_guard() -> None:
    """alpha_m returns the trap0 L'Hôpital limit at V = tha = -35 mV."""
    expected = 0.182 * MAINEN_SEJNOWSKI_KV_PRESCALE * 9.0
    assert mainen_sejnowski_alpha_m(-35.0, 0.0) == pytest.approx(expected)


def test_mainen_sejnowski_beta_m_singularity_guard() -> None:
    """beta_m hits its singularity at the same V = -35 mV (-V = -tha branch).

    The na.mod ``trap0(-v, -tha, Rb, qa)`` form has its removable singularity
    at -V = -tha = +35 mV, i.e. V = -35 mV — the same physical voltage as
    the α_m singularity.
    """
    expected = 0.124 * MAINEN_SEJNOWSKI_KV_PRESCALE * 9.0
    assert mainen_sejnowski_beta_m(-35.0, 0.0) == pytest.approx(expected)


def test_mainen_sejnowski_alpha_m_continuous_around_singularity() -> None:
    """alpha_m is continuous approaching the singularity from either side."""
    expected = 0.182 * MAINEN_SEJNOWSKI_KV_PRESCALE * 9.0
    above = mainen_sejnowski_alpha_m(-35.0 + 1e-5, 0.0)
    below = mainen_sejnowski_alpha_m(-35.0 - 1e-5, 0.0)
    assert above == pytest.approx(expected, rel=1e-3)
    assert below == pytest.approx(expected, rel=1e-3)


def test_mainen_sejnowski_h_inf_open_at_rest() -> None:
    """h_inf > 0.5 at v_rest = -70 mV (Na channels mostly available for the next AP).

    Boltzmann form gives h_inf(-70) = 1/(1 + exp(-5/6.2)) ≈ 0.69 — just
    above half-availability, consistent with most Na channels being open
    and ready to fire.
    """
    a = mainen_sejnowski_alpha_h(-70.0, 0.0)
    b = mainen_sejnowski_beta_h(-70.0, 0.0)
    h_inf = a / (a + b)
    assert h_inf > 0.5


def test_mainen_sejnowski_h_inf_strongly_inactivated_at_minus_20() -> None:
    """h_inf < 0.001 at V = -20 mV (steady-state Na window current is tiny).

    Critical for the cortical pyramidal preset's equilibrium analysis:
    Pospischil Na has h_inf ≈ 0.034 at -20 mV, producing a large Na window
    current that prevents the bracket [-100, -20] from finding a unique
    zero crossing once Kv kinetics are slowed.  M-S Na's stronger
    inactivation (h_inf ≈ 7e-4 at -20 mV) eliminates that current.
    """
    a = mainen_sejnowski_alpha_h(-20.0, 0.0)
    b = mainen_sejnowski_beta_h(-20.0, 0.0)
    h_inf = a / (a + b)
    assert h_inf < 0.001


def test_mainen_sejnowski_m_inf_strongly_open_at_peak() -> None:
    """m_inf > 0.95 at V = +30 mV (full activation during AP peak)."""
    a = mainen_sejnowski_alpha_m(30.0, 0.0)
    b = mainen_sejnowski_beta_m(30.0, 0.0)
    m_inf = a / (a + b)
    assert m_inf > 0.95


@pytest.mark.parametrize("V", [-65.0, 0.0, 30.0])
@pytest.mark.parametrize(
    "fn",
    [
        mainen_sejnowski_alpha_m,
        mainen_sejnowski_beta_m,
        mainen_sejnowski_alpha_h,
        mainen_sejnowski_beta_h,
    ],
)
def test_mainen_sejnowski_na_rates_ignore_ca_i(V: float, fn: Rate) -> None:
    """M-S Na rate functions return the same value regardless of ca_i."""
    assert fn(V, 0.0) == pytest.approx(fn(V, 1.0))


def test_make_mainen_sejnowski_na_channel_structure() -> None:
    """make_mainen_sejnowski_na_channel returns a channel with correct structure."""
    ch = make_mainen_sejnowski_na_channel(g_max=50.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "Na"
    assert ch.g_max == pytest.approx(50.0)
    assert len(ch.gating_variables) == 2
    assert ch.gating_variables[0].name == "m"
    assert ch.gating_variables[0].power == 3
    assert ch.gating_variables[1].name == "h"
    assert ch.gating_variables[1].power == 1
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.SODIUM
    assert not ch.carries_calcium


# ---------------------------------------------------------------------------
# Otsuka et al. (2004) STN channel kinetics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -65.0, -40.0, 0.0, 40.0])
def test_stn_na_rate_functions_positive(V: float) -> None:
    """All STN Na⁺ rate functions are strictly positive at physiological voltages."""
    assert _stn_alpha_m(V, 0.0) > 0
    assert _stn_beta_m(V, 0.0) > 0
    assert _stn_alpha_h(V, 0.0) > 0
    assert _stn_beta_h(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, -40.0, 0.0, 40.0])
def test_stn_na_steady_state_bounds(V: float) -> None:
    """STN Na⁺ steady-state gating variables are in [0, 1]."""
    m_inf = _stn_alpha_m(V, 0.0) / (_stn_alpha_m(V, 0.0) + _stn_beta_m(V, 0.0))
    h_inf = _stn_alpha_h(V, 0.0) / (_stn_alpha_h(V, 0.0) + _stn_beta_h(V, 0.0))
    assert 0.0 <= m_inf <= 1.0
    assert 0.0 <= h_inf <= 1.0


@pytest.mark.parametrize(
    "V, expected_m_inf",
    [
        (-80.0, 1.0 / (1.0 + math.exp(-(-80.0 + 40.0) / 8.0))),
        (-40.0, 0.5),  # V_half of m
        (0.0, 1.0 / (1.0 + math.exp(-(0.0 + 40.0) / 8.0))),
    ],
)
def test_stn_na_m_inf_matches_boltzmann(V: float, expected_m_inf: float) -> None:
    """STN Na⁺ m steady-state matches 1/(1+exp(-(V+40)/8))."""
    m_inf = _stn_alpha_m(V, 0.0) / (_stn_alpha_m(V, 0.0) + _stn_beta_m(V, 0.0))
    assert m_inf == pytest.approx(expected_m_inf, rel=1e-6)


@pytest.mark.parametrize(
    "V, expected_h_inf",
    [
        (-80.0, 1.0 / (1.0 + math.exp((-80.0 + 45.5) / 6.4))),
        (-45.5, 0.5),  # V_half of h
        (-20.0, 1.0 / (1.0 + math.exp((-20.0 + 45.5) / 6.4))),
    ],
)
def test_stn_na_h_inf_matches_boltzmann(V: float, expected_h_inf: float) -> None:
    """STN Na⁺ h steady-state matches 1/(1+exp((V+45.5)/6.4))."""
    h_inf = _stn_alpha_h(V, 0.0) / (_stn_alpha_h(V, 0.0) + _stn_beta_h(V, 0.0))
    assert h_inf == pytest.approx(expected_h_inf, rel=1e-6)


def test_stn_na_m_tau_is_voltage_independent() -> None:
    """STN Na⁺ activation time constant is positive and voltage-independent."""
    voltages = (-80.0, -40.0, 0.0, 40.0)
    taus = [1.0 / (_stn_alpha_m(V, 0.0) + _stn_beta_m(V, 0.0)) for V in voltages]
    assert all(tau > 0.0 for tau in taus)
    assert all(tau == pytest.approx(taus[0], rel=1e-6) for tau in taus)


def test_make_stn_na_channel_structure() -> None:
    """make_stn_na_channel returns the (m, h, sNa) three-gate Na topology."""
    ch = make_stn_na_channel(g_max=49.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "Na"
    assert ch.g_max == pytest.approx(49.0)
    assert len(ch.gating_variables) == 3
    assert ch.gating_variables[0].name == "m"
    assert ch.gating_variables[0].power == 3
    assert ch.gating_variables[1].name == "h"
    assert ch.gating_variables[1].power == 1
    assert ch.gating_variables[2].name == "sNa"
    assert ch.gating_variables[2].power == 1
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.SODIUM
    assert not ch.carries_calcium


# ---------------------------------------------------------------------------
# STN slow Na inactivation rate functions (sNa gate; Fleidervish & Gutnick
# 1996; Mickus, Jung & Spruston 1999; Do & Bean 2003).  Always-on gate
# baked into make_stn_na_channel to abolish the residual −15 mV plateau
# the Otsuka 2004 fast h-tail leaves open.
# ---------------------------------------------------------------------------


def _sNa_inf(V: float) -> float:
    """Steady-state availability of the STN slow Na inactivation gate."""
    a, b = _stn_alpha_sNa(V, 0.0), _stn_beta_sNa(V, 0.0)
    return a / (a + b)


def test_stn_slow_na_inactivation_steady_state_in_bounds() -> None:
    """The sNa rates are positive and steady state in [0, 1] across V."""
    for V in (-120.0, -100.0, -75.0, -65.0, -50.0, -30.0, -15.0, 0.0, 30.0):
        a = _stn_alpha_sNa(V, 0.0)
        b = _stn_beta_sNa(V, 0.0)
        assert a >= 0, f"alpha_sNa negative at V={V}"
        assert b >= 0, f"beta_sNa negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"sNa steady state {ss} out of [0,1] at V={V}"


def test_stn_slow_na_inactivation_decreases_with_depolarization() -> None:
    """The sNa availability decreases monotonically with depolarization."""
    assert _sNa_inf(-80.0) > _sNa_inf(-50.0) > _sNa_inf(-15.0)


def test_stn_slow_na_inactivation_half_voltage() -> None:
    """V½ for sNa sits at -50 mV (Fleidervish & Gutnick 1996 mid-range)."""
    assert _sNa_inf(-50.0) == pytest.approx(0.5, abs=0.01)


def test_stn_slow_na_inactivation_resting_availability() -> None:
    """Near-rest sNa availability stays high to preserve autonomous pacemaking."""
    # STN cycles around -60 mV at the autonomous rate; sNa must remain high
    # there or the spike train would lose Na availability.
    assert _sNa_inf(-65.0) > 0.85
    assert _sNa_inf(-75.0) > 0.93


def test_stn_slow_na_inactivation_blocks_depol_plateau() -> None:
    """At depolarized plateau voltages sNa closes, abolishing the residual h-tail."""
    # The depol-block plateau in #324 hung at ≈ −15 mV; sNa must close
    # firmly there so g_Na_eff = g_max * m^3 * h * sNa collapses.
    assert _sNa_inf(-15.0) < 0.05


def test_stn_slow_na_inactivation_tau_is_slow() -> None:
    """τ_sNa at V½ stays distinctly slow vs the fast m, h gates."""
    a, b = _stn_alpha_sNa(-50.0, 0.0), _stn_beta_sNa(-50.0, 0.0)
    tau = 1.0 / (a + b)
    assert tau > 100.0, f"sNa tau at V½ is {tau:.1f} ms, expected > 100 ms"


def test_stn_preset_uses_otsuka_na_kinetics() -> None:
    """STN preset's Na channel carries the Otsuka sNa slow-inactivation gate.

    Replaces the pre-#320 factory-identity check; STN's K is now Kv3.1
    (no HH-style core K), so we only assert the Na kinetics structurally.
    """
    from patch_sim.constants import STN
    from patch_sim.presets import NEURON_PRESETS

    neuron = NEURON_PRESETS[STN]()
    by_name = {ch.name: ch for ch in neuron.channels}
    na = by_name["Na"]
    gate_names = {gv.name for gv in na.gating_variables}
    # sNa is the Otsuka slow-inactivation gate (#324) — unique to make_stn_na_channel
    assert "sNa" in gate_names
    # Stock Otsuka kinetics also expose m and h
    assert {"m", "h"} <= gate_names


# ---------------------------------------------------------------------------
# Purkinje (De Schutter & Bower 1994) rate functions
# ---------------------------------------------------------------------------

_PURKINJE_ALPHA_M_SINGULARITY = PURKINJE_VT + 13  # -45 mV
_PURKINJE_BETA_M_SINGULARITY = PURKINJE_VT + 40  # -18 mV
_PURKINJE_ALPHA_N_SINGULARITY = PURKINJE_VT + 15  # -43 mV


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_purkinje_na_rate_functions_positive(V: float) -> None:
    """All Purkinje Na⁺ rate functions are positive at physiological voltages."""
    assert purkinje_alpha_m(V, 0.0) > 0
    assert purkinje_beta_m(V, 0.0) > 0
    assert purkinje_alpha_h(V, 0.0) > 0
    assert purkinje_beta_h(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_purkinje_k_rate_functions_positive(V: float) -> None:
    """All Purkinje K⁺ rate functions are positive at physiological voltages."""
    assert purkinje_alpha_n(V, 0.0) > 0
    assert purkinje_beta_n(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_purkinje_steady_state_gating_bounds(V: float) -> None:
    """Purkinje steady-state gating variables are in [0, 1]."""
    m_inf = purkinje_alpha_m(V, 0.0) / (
        purkinje_alpha_m(V, 0.0) + purkinje_beta_m(V, 0.0)
    )
    h_inf = purkinje_alpha_h(V, 0.0) / (
        purkinje_alpha_h(V, 0.0) + purkinje_beta_h(V, 0.0)
    )
    n_inf = purkinje_alpha_n(V, 0.0) / (
        purkinje_alpha_n(V, 0.0) + purkinje_beta_n(V, 0.0)
    )
    assert 0.0 <= m_inf <= 1.0
    assert 0.0 <= h_inf <= 1.0
    assert 0.0 <= n_inf <= 1.0


def test_purkinje_alpha_m_singularity_guard() -> None:
    """purkinje_alpha_m returns L'Hôpital limit 1.28 at V = VT + 13."""
    assert purkinje_alpha_m(_PURKINJE_ALPHA_M_SINGULARITY, 0.0) == pytest.approx(1.28)


def test_purkinje_alpha_m_near_singularity_continuous_above() -> None:
    """purkinje_alpha_m is continuous approaching the singularity from above."""
    assert purkinje_alpha_m(_PURKINJE_ALPHA_M_SINGULARITY + 1e-5, 0.0) == pytest.approx(
        1.28, rel=1e-3
    )


def test_purkinje_alpha_m_near_singularity_continuous_below() -> None:
    """purkinje_alpha_m is continuous approaching the singularity from below."""
    assert purkinje_alpha_m(_PURKINJE_ALPHA_M_SINGULARITY - 1e-5, 0.0) == pytest.approx(
        1.28, rel=1e-3
    )


def test_purkinje_beta_m_singularity_guard() -> None:
    """purkinje_beta_m returns L'Hôpital limit 1.4 at V = VT + 40."""
    assert purkinje_beta_m(_PURKINJE_BETA_M_SINGULARITY, 0.0) == pytest.approx(1.4)


def test_purkinje_beta_m_near_singularity_continuous_above() -> None:
    """purkinje_beta_m is continuous approaching the singularity from above."""
    assert purkinje_beta_m(_PURKINJE_BETA_M_SINGULARITY + 1e-5, 0.0) == pytest.approx(
        1.4, rel=1e-3
    )


def test_purkinje_beta_m_near_singularity_continuous_below() -> None:
    """purkinje_beta_m is continuous approaching the singularity from below."""
    assert purkinje_beta_m(_PURKINJE_BETA_M_SINGULARITY - 1e-5, 0.0) == pytest.approx(
        1.4, rel=1e-3
    )


def test_purkinje_alpha_n_singularity_guard() -> None:
    """purkinje_alpha_n returns L'Hôpital limit 0.16 at V = VT + 15."""
    assert purkinje_alpha_n(_PURKINJE_ALPHA_N_SINGULARITY, 0.0) == pytest.approx(0.16)


def test_purkinje_alpha_n_near_singularity_continuous_above() -> None:
    """purkinje_alpha_n is continuous approaching the singularity from above."""
    assert purkinje_alpha_n(_PURKINJE_ALPHA_N_SINGULARITY + 1e-5, 0.0) == pytest.approx(
        0.16, rel=1e-3
    )


def test_purkinje_alpha_n_near_singularity_continuous_below() -> None:
    """purkinje_alpha_n is continuous approaching the singularity from below."""
    assert purkinje_alpha_n(_PURKINJE_ALPHA_N_SINGULARITY - 1e-5, 0.0) == pytest.approx(
        0.16, rel=1e-3
    )


@pytest.mark.parametrize("V", [-65.0, 0.0])
@pytest.mark.parametrize(
    "fn",
    [
        purkinje_alpha_m,
        purkinje_beta_m,
        purkinje_alpha_h,
        purkinje_beta_h,
        purkinje_alpha_n,
        purkinje_beta_n,
    ],
)
def test_purkinje_rate_functions_ignore_ca_i(V: float, fn: Rate) -> None:
    """All Purkinje rate functions return the same value regardless of ca_i."""
    assert fn(V, 0.0) == pytest.approx(fn(V, 1.0))


def test_make_purkinje_na_channel_structure() -> None:
    """make_purkinje_na_channel returns m³, h, and sNa gates with Na⁺ reversal."""
    ch = make_purkinje_na_channel(g_max=120.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "Na"
    assert ch.g_max == pytest.approx(120.0)
    assert len(ch.gating_variables) == 3
    assert ch.gating_variables[0].name == "m"
    assert ch.gating_variables[0].power == 3
    assert ch.gating_variables[1].name == "h"
    assert ch.gating_variables[1].power == 1
    assert ch.gating_variables[2].name == "sNa"
    assert ch.gating_variables[2].power == 1
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.SODIUM
    assert not ch.carries_calcium


# ---------------------------------------------------------------------------
# Purkinje slow Na inactivation rate functions (sNa gate; Carter & Bean
# 2009).  Always-on gate added in #329 to abolish the residual depol-block
# plateau under sustained climbing-fiber-style drive.
# ---------------------------------------------------------------------------


def _purkinje_sNa_inf(V: float) -> float:
    """Steady-state availability of the Purkinje slow Na inactivation gate.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Steady-state availability of the sNa gate at voltage V, in [0, 1].
    """
    a, b = _purkinje_alpha_sNa(V, 0.0), _purkinje_beta_sNa(V, 0.0)
    return a / (a + b)


def test_purkinje_slow_na_inactivation_steady_state_in_bounds() -> None:
    """The sNa rates are positive and steady state in [0, 1] across V."""
    for V in (-120.0, -100.0, -75.0, -65.0, -50.0, -30.0, -15.0, 0.0, 30.0):
        a = _purkinje_alpha_sNa(V, 0.0)
        b = _purkinje_beta_sNa(V, 0.0)
        assert a >= 0, f"alpha_sNa negative at V={V}"
        assert b >= 0, f"beta_sNa negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"sNa steady state {ss} out of [0,1] at V={V}"


def test_purkinje_slow_na_inactivation_decreases_with_depolarization() -> None:
    """The sNa availability decreases monotonically with depolarization."""
    assert (
        _purkinje_sNa_inf(-80.0) > _purkinje_sNa_inf(-50.0) > _purkinje_sNa_inf(-15.0)
    )


def test_purkinje_slow_na_inactivation_half_voltage() -> None:
    """V½ for sNa sits at -50 mV (mirrors STN / Pospischil mid-range)."""
    assert _purkinje_sNa_inf(-50.0) == pytest.approx(0.5, abs=0.01)


def test_purkinje_slow_na_inactivation_resting_availability() -> None:
    """Purkinje rests near -65 mV — sNa must remain mostly open."""
    # Purkinje v_rest = -65 mV (matches the STN cycle hyperpolarized end).
    # Loss of >15 % rest availability would noticeably suppress AP amplitude
    # on every spontaneous beat.
    assert _purkinje_sNa_inf(-65.0) > 0.85


def test_purkinje_slow_na_inactivation_blocks_depol_plateau() -> None:
    """At depolarized plateau voltages sNa closes, abolishing the residual h-tail."""
    # The depol-block plateau the new gate must escape (mirroring #324) hangs
    # at ≈ −15 mV; sNa must close firmly there so g_Na_eff = g_max * m^3 * h
    # * sNa collapses below the leak + IK outward drive.
    assert _purkinje_sNa_inf(-15.0) < 0.05


def test_purkinje_slow_na_inactivation_tau_is_slow() -> None:
    """τ_sNa at V½ stays distinctly slow vs the fast m, h gates."""
    a = _purkinje_alpha_sNa(-50.0, 0.0)
    b = _purkinje_beta_sNa(-50.0, 0.0)
    tau = 1.0 / (a + b)
    assert tau > 100.0, f"sNa tau at V½ is {tau:.1f} ms, expected > 100 ms"


def test_make_purkinje_k_channel_structure() -> None:
    """make_purkinje_k_channel returns a channel with n⁴ gating and K⁺ reversal."""
    ch = make_purkinje_k_channel(g_max=36.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "K"
    assert ch.g_max == pytest.approx(36.0)
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "n"
    assert ch.gating_variables[0].power == 4
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM
    assert not ch.carries_calcium


# ===========================================================================
# Komendantov et al. (2004) SNc dopaminergic (VT = -67 mV) kinetics
# ===========================================================================

# ---------------------------------------------------------------------------
# Rate function positivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -65.0, -54.0, 0.0, 40.0])
def test_dopaminergic_alpha_m_positive(V: float) -> None:
    """dopaminergic_alpha_m is positive at physiological voltages and singularity."""
    assert dopaminergic_alpha_m(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, -27.0, 0.0, 40.0])
def test_dopaminergic_beta_m_positive(V: float) -> None:
    """dopaminergic_beta_m is positive at physiological voltages and singularity."""
    assert dopaminergic_beta_m(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_dopaminergic_alpha_h_positive(V: float) -> None:
    """dopaminergic_alpha_h is positive at physiological voltages."""
    assert dopaminergic_alpha_h(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_dopaminergic_beta_h_positive(V: float) -> None:
    """dopaminergic_beta_h is positive at physiological voltages."""
    assert dopaminergic_beta_h(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, -52.0, 0.0, 40.0])
def test_dopaminergic_alpha_n_positive(V: float) -> None:
    """dopaminergic_alpha_n is positive at physiological voltages and singularity."""
    assert dopaminergic_alpha_n(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_dopaminergic_beta_n_positive(V: float) -> None:
    """dopaminergic_beta_n is positive at physiological voltages."""
    assert dopaminergic_beta_n(V, 0.0) > 0


# ---------------------------------------------------------------------------
# Singularity guards  (VT = -67 mV → singularities at -54, -27, -52 mV)
# ---------------------------------------------------------------------------

_DA_ALPHA_M_SINGULARITY = DOPAMINERGIC_VT + 13  # -54 mV; limit = 0.32 * 4 = 1.28
_DA_BETA_M_SINGULARITY = DOPAMINERGIC_VT + 40  # -27 mV; limit = 0.28 * 5 = 1.4
_DA_ALPHA_N_SINGULARITY = DOPAMINERGIC_VT + 15  # -52 mV; limit = 0.032 * 5 = 0.16


def test_dopaminergic_alpha_m_singularity_guard() -> None:
    """dopaminergic_alpha_m returns L'Hôpital limit 1.28 at V = VT + 13 = -54 mV."""
    assert dopaminergic_alpha_m(_DA_ALPHA_M_SINGULARITY, 0.0) == pytest.approx(1.28)


def test_dopaminergic_alpha_m_near_singularity_continuous_above() -> None:
    """dopaminergic_alpha_m is continuous approaching -54 mV from above."""
    assert dopaminergic_alpha_m(_DA_ALPHA_M_SINGULARITY + 1e-5, 0.0) == pytest.approx(
        1.28, rel=1e-3
    )


def test_dopaminergic_alpha_m_near_singularity_continuous_below() -> None:
    """dopaminergic_alpha_m is continuous approaching -54 mV from below."""
    assert dopaminergic_alpha_m(_DA_ALPHA_M_SINGULARITY - 1e-5, 0.0) == pytest.approx(
        1.28, rel=1e-3
    )


def test_dopaminergic_beta_m_singularity_guard() -> None:
    """dopaminergic_beta_m returns L'Hôpital limit 1.4 at V = VT + 40 = -27 mV."""
    assert dopaminergic_beta_m(_DA_BETA_M_SINGULARITY, 0.0) == pytest.approx(1.4)


def test_dopaminergic_beta_m_near_singularity_continuous_above() -> None:
    """dopaminergic_beta_m is continuous approaching -27 mV from above."""
    assert dopaminergic_beta_m(_DA_BETA_M_SINGULARITY + 1e-5, 0.0) == pytest.approx(
        1.4, rel=1e-3
    )


def test_dopaminergic_beta_m_near_singularity_continuous_below() -> None:
    """dopaminergic_beta_m is continuous approaching -27 mV from below."""
    assert dopaminergic_beta_m(_DA_BETA_M_SINGULARITY - 1e-5, 0.0) == pytest.approx(
        1.4, rel=1e-3
    )


def test_dopaminergic_alpha_n_singularity_guard() -> None:
    """dopaminergic_alpha_n returns L'Hôpital limit 0.16 at V = VT + 15 = -52 mV."""
    assert dopaminergic_alpha_n(_DA_ALPHA_N_SINGULARITY, 0.0) == pytest.approx(0.16)


def test_dopaminergic_alpha_n_near_singularity_continuous_above() -> None:
    """dopaminergic_alpha_n is continuous approaching -52 mV from above."""
    assert dopaminergic_alpha_n(_DA_ALPHA_N_SINGULARITY + 1e-5, 0.0) == pytest.approx(
        0.16, rel=1e-3
    )


def test_dopaminergic_alpha_n_near_singularity_continuous_below() -> None:
    """dopaminergic_alpha_n is continuous approaching -52 mV from below."""
    assert dopaminergic_alpha_n(_DA_ALPHA_N_SINGULARITY - 1e-5, 0.0) == pytest.approx(
        0.16, rel=1e-3
    )


# ---------------------------------------------------------------------------
# ca_i independence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-65.0, 0.0])
@pytest.mark.parametrize(
    "fn",
    [
        dopaminergic_alpha_m,
        dopaminergic_beta_m,
        dopaminergic_alpha_h,
        dopaminergic_beta_h,
        dopaminergic_alpha_n,
        dopaminergic_beta_n,
    ],
)
def test_dopaminergic_rate_functions_ignore_ca_i(V: float, fn: Rate) -> None:
    """All dopaminergic rate functions return the same value regardless of ca_i."""
    assert fn(V, 0.0) == pytest.approx(fn(V, 1.0))


# ---------------------------------------------------------------------------
# Steady-state bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_dopaminergic_steady_state_gating_bounds(V: float) -> None:
    """Dopaminergic steady-state gating variables are in [0, 1]."""
    n_inf = dopaminergic_alpha_n(V, 0.0) / (
        dopaminergic_alpha_n(V, 0.0) + dopaminergic_beta_n(V, 0.0)
    )
    m_inf = dopaminergic_alpha_m(V, 0.0) / (
        dopaminergic_alpha_m(V, 0.0) + dopaminergic_beta_m(V, 0.0)
    )
    h_inf = dopaminergic_alpha_h(V, 0.0) / (
        dopaminergic_alpha_h(V, 0.0) + dopaminergic_beta_h(V, 0.0)
    )
    assert 0.0 <= n_inf <= 1.0
    assert 0.0 <= m_inf <= 1.0
    assert 0.0 <= h_inf <= 1.0


# ---------------------------------------------------------------------------
# Factory function structure
# ---------------------------------------------------------------------------


def test_make_dopaminergic_na_channel_structure() -> None:
    """make_dopaminergic_na_channel returns m³, h, sNa_da gates with Na⁺ reversal."""
    ch = make_dopaminergic_na_channel(g_max=120.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "Na"
    assert ch.g_max == pytest.approx(120.0)
    assert len(ch.gating_variables) == 3
    assert ch.gating_variables[0].name == "m"
    assert ch.gating_variables[0].power == 3
    assert ch.gating_variables[1].name == "h"
    assert ch.gating_variables[1].power == 1
    assert ch.gating_variables[2].name == "sNa_da"
    assert ch.gating_variables[2].power == 1
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.SODIUM
    assert not ch.carries_calcium


# ---------------------------------------------------------------------------
# Dopaminergic SNc slow Na inactivation rate functions (sNa_da gate;
# Khaliq & Bean 2010).  Always-on gate added in #330.
# ---------------------------------------------------------------------------


def _dopaminergic_sNa_inf(V: float) -> float:
    """Steady-state availability of the dopaminergic slow Na inactivation gate.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Steady-state availability of the sNa_da gate at voltage V, in [0, 1].
    """
    a, b = _dopaminergic_alpha_sNa(V, 0.0), _dopaminergic_beta_sNa(V, 0.0)
    return a / (a + b)


def test_dopaminergic_slow_na_inactivation_steady_state_in_bounds() -> None:
    """The sNa_da rates are positive and steady state in [0, 1] across V."""
    for V in (-120.0, -100.0, -75.0, -55.0, -50.0, -30.0, -15.0, 0.0, 30.0):
        a = _dopaminergic_alpha_sNa(V, 0.0)
        b = _dopaminergic_beta_sNa(V, 0.0)
        assert a >= 0, f"alpha_sNa_da negative at V={V}"
        assert b >= 0, f"beta_sNa_da negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"sNa_da steady state {ss} out of [0,1] at V={V}"


def test_dopaminergic_slow_na_inactivation_decreases_with_depolarization() -> None:
    """The sNa_da availability decreases monotonically with depolarization."""
    assert (
        _dopaminergic_sNa_inf(-80.0)
        > _dopaminergic_sNa_inf(-50.0)
        > _dopaminergic_sNa_inf(-15.0)
    )


def test_dopaminergic_slow_na_inactivation_half_voltage() -> None:
    """V½ for sNa_da sits at -50 mV (mirrors STN / Pospischil / Purkinje sNa)."""
    assert _dopaminergic_sNa_inf(-50.0) == pytest.approx(0.5, abs=0.01)


def test_dopaminergic_slow_na_inactivation_resting_availability() -> None:
    """SNc DA cycles through −90 to −55 mV; sNa_da must stay open in the trough.

    The looser lower bound at v_rest = −55 mV (vs Purkinje 0.85 at −65 mV) is
    expected: SNc rests more depolarized, so the slow gate sits closer to V½
    at rest.  The cycle hyperpolarized end (≈ −75 mV) is what matters for
    recovery between spikes, and there sNa_da > 0.93.
    """
    assert _dopaminergic_sNa_inf(-75.0) > 0.93
    assert _dopaminergic_sNa_inf(-55.0) > 0.6


def test_dopaminergic_slow_na_inactivation_blocks_depol_plateau() -> None:
    """At depolarized plateau voltages sNa_da closes, abolishing the residual h-tail."""
    # The depol-block plateau the new gate guards against (mirroring STN/
    # Purkinje rationale) hangs at ≈ −15 mV; sNa_da must close firmly there
    # so g_Na_eff = g_max * m^3 * h * sNa_da collapses below leak + IK.
    assert _dopaminergic_sNa_inf(-15.0) < 0.05


def test_dopaminergic_slow_na_inactivation_tau_is_slow() -> None:
    """τ_sNa_da at V½ stays distinctly slow vs the fast m, h gates."""
    a = _dopaminergic_alpha_sNa(-50.0, 0.0)
    b = _dopaminergic_beta_sNa(-50.0, 0.0)
    tau = 1.0 / (a + b)
    assert tau > 100.0, f"sNa_da tau at V½ is {tau:.1f} ms, expected > 100 ms"


def test_make_dopaminergic_k_channel_structure() -> None:
    """make_dopaminergic_k_channel returns a channel with n⁴ gating and K⁺ reversal."""
    ch = make_dopaminergic_k_channel(g_max=36.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "K"
    assert ch.g_max == pytest.approx(36.0)
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "n"
    assert ch.gating_variables[0].power == 4
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM
    assert not ch.carries_calcium
