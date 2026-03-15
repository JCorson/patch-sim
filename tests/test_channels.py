"""Tests for the optional ion channel framework.

Covers BaseIonChannel math, GatingVariable steady states, Ih kinetics,
backward compatibility with no optional channels, and validation errors.
"""

import numpy as np
import pytest

import ap_sim
from ap_sim.channels import BaseIonChannel, GatingVariable, IonChannel
from ap_sim.clamp_simulations import simulate_current_clamp, simulate_voltage_clamp
from ap_sim.hodgkin_huxley import HodgkinHuxley
from ap_sim.optional_channels import (
    _alpha_a,
    _alpha_b,
    _alpha_r,
    _beta_a,
    _beta_b,
    _beta_r,
    make_ih_channel,
    make_ika_channel,
)
from ap_sim.protocols import step_current, step_voltage


# ---------------------------------------------------------------------------
# GatingVariable
# ---------------------------------------------------------------------------


def test_gating_variable_steady_state_in_bounds():
    """Ih gating variable steady state is in [0, 1] for all physiological voltages."""
    for V in np.linspace(-120.0, 0.0, 50):
        a = _alpha_r(V)
        b = _beta_r(V)
        assert a >= 0, f"alpha_r negative at V={V}"
        assert b >= 0, f"beta_r negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"steady state {ss} out of [0,1] at V={V}"


def test_ih_kinetics_alpha_increases_with_hyperpolarisation():
    """alpha_r should increase as voltage becomes more negative (Ih is HCN-type)."""
    alpha_at_minus100 = _alpha_r(-100.0)
    alpha_at_minus65 = _alpha_r(-65.0)
    alpha_at_minus40 = _alpha_r(-40.0)
    assert alpha_at_minus100 > alpha_at_minus65 > alpha_at_minus40


def test_ih_kinetics_steady_state_higher_at_hyperpolarised():
    """Ih r steady state is higher at -100 mV than at -65 mV."""
    a65, b65 = _alpha_r(-65.0), _beta_r(-65.0)
    a100, b100 = _alpha_r(-100.0), _beta_r(-100.0)
    ss65 = a65 / (a65 + b65)
    ss100 = a100 / (a100 + b100)
    assert ss100 > ss65


# ---------------------------------------------------------------------------
# BaseIonChannel
# ---------------------------------------------------------------------------


def _make_simple_channel(g_max: float = 1.0, e_rev: float = 0.0) -> BaseIonChannel:
    """Helper: create a channel with a single linear gating variable (power=1)."""
    gv = GatingVariable(
        name="x",
        power=1,
        alpha=lambda V: 0.1,
        beta=lambda V: 0.1,
    )
    return BaseIonChannel(name="test", g_max=g_max, gating_variables=(gv,), e_rev=e_rev)


def test_base_ion_channel_compute_current_math():
    """compute_current returns g_max * gate^power * (V - e_rev)."""
    ch = _make_simple_channel(g_max=2.0, e_rev=-10.0)
    # gate value = 0.5, power = 1 → g = 2.0 * 0.5^1 = 1.0
    # current = 1.0 * (V - (-10)) = 1.0 * 10 = 10.0
    result = ch.compute_current(V=0.0, gating_state={"x": 0.5})
    assert result == pytest.approx(2.0 * 0.5 * (0.0 - (-10.0)))


def test_base_ion_channel_power_two():
    """compute_current correctly raises the gate to its power."""
    gv = GatingVariable(name="y", power=2, alpha=lambda V: 0.1, beta=lambda V: 0.1)
    ch = BaseIonChannel(name="pow2", g_max=1.0, gating_variables=(gv,), e_rev=0.0)
    # gate=0.5, power=2 → g = 1.0 * 0.5^2 = 0.25
    result = ch.compute_current(V=10.0, gating_state={"y": 0.5})
    assert result == pytest.approx(1.0 * (0.5**2) * (10.0 - 0.0))


def test_base_ion_channel_reversal_potential_returns_e_rev():
    """reversal_potential() returns the fixed e_rev value."""
    ch = _make_simple_channel(e_rev=-30.0)
    assert ch.reversal_potential(neuron=None) == pytest.approx(-30.0)


def test_base_ion_channel_zero_current_at_reversal():
    """Current is zero when V equals e_rev."""
    ch = _make_simple_channel(g_max=1.0, e_rev=-40.0)
    result = ch.compute_current(V=-40.0, gating_state={"x": 0.8})
    assert result == pytest.approx(0.0)


def test_base_ion_channel_satisfies_protocol():
    """BaseIonChannel is an instance of the IonChannel protocol."""
    ch = _make_simple_channel()
    assert isinstance(ch, IonChannel)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_base_ion_channel_negative_gmax_raises():
    """Negative g_max raises ValueError."""
    with pytest.raises(ValueError, match="g_max must be non-negative"):
        _make_simple_channel(g_max=-1.0)


def test_base_ion_channel_duplicate_gating_names_raises():
    """Duplicate gating variable names within a channel raise ValueError."""
    gv1 = GatingVariable(name="x", power=1, alpha=lambda V: 0.1, beta=lambda V: 0.1)
    gv2 = GatingVariable(name="x", power=2, alpha=lambda V: 0.2, beta=lambda V: 0.2)
    with pytest.raises(ValueError, match="names must be unique"):
        BaseIonChannel(name="dup", g_max=1.0, gating_variables=(gv1, gv2), e_rev=0.0)


def test_hh_duplicate_optional_channel_names_raises():
    """Duplicate optional channel names on HodgkinHuxley raise ValueError."""
    ch = make_ih_channel()
    with pytest.raises(ValueError, match="names must be unique"):
        HodgkinHuxley(optional_channels=(ch, ch))


def test_hh_builtin_channel_name_collision_raises():
    """Optional channel named 'Na' collides with built-in and raises ValueError."""
    gv = GatingVariable(name="r", power=1, alpha=lambda V: 0.1, beta=lambda V: 0.1)
    ch = BaseIonChannel(name="Na", g_max=0.1, gating_variables=(gv,), e_rev=-30.0)
    with pytest.raises(ValueError, match="collides with a built-in"):
        HodgkinHuxley(optional_channels=(ch,))


# ---------------------------------------------------------------------------
# make_ih_channel
# ---------------------------------------------------------------------------


def test_make_ih_channel_defaults():
    """make_ih_channel() produces a channel with the expected defaults."""
    from ap_sim.constants import DEFAULT_E_IH, DEFAULT_G_IH

    ch = make_ih_channel()
    assert ch.name == "Ih"
    assert ch.g_max == pytest.approx(DEFAULT_G_IH)
    assert ch.e_rev == pytest.approx(DEFAULT_E_IH)
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "r"
    assert ch.gating_variables[0].power == 1


def test_make_ih_channel_custom_params():
    """make_ih_channel accepts custom g_max and e_rev."""
    ch = make_ih_channel(g_max=0.5, e_rev=-25.0)
    assert ch.g_max == pytest.approx(0.5)
    assert ch.e_rev == pytest.approx(-25.0)


# ---------------------------------------------------------------------------
# Backward compatibility — no optional channels
# ---------------------------------------------------------------------------


def test_current_clamp_no_optional_channels_identical_columns(hh_model):
    """simulate_current_clamp with no optional channels has exact classic columns."""
    stim = step_current(
        duration=10.0,
        current_amplitude=10.0,
        step_start=2.0,
        step_duration=5.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(hh_model, stim)
    expected = {
        "voltage",
        "potassium_activation",
        "sodium_activation",
        "sodium_inactivation",
    }
    assert set(df.columns) == expected


def test_voltage_clamp_no_optional_channels_identical_columns(hh_model):
    """simulate_voltage_clamp with no optional channels has exact classic columns."""
    prot = step_voltage(
        duration=10.0,
        voltage_amplitude=0.0,
        step_start=2.0,
        step_duration=5.0,
        holding_voltage=-70.0,
        sampling_frequency=40000.0,
    )
    df = simulate_voltage_clamp(hh_model, prot)
    expected = {
        "voltage",
        "total_current",
        "sodium_current",
        "potassium_current",
        "leak_current",
        "potassium_activation",
        "sodium_activation",
        "sodium_inactivation",
    }
    assert set(df.columns) == expected


def test_current_clamp_no_optional_channels_values_unchanged():
    """Voltage trace is unchanged when optional_channels is empty vs. default."""
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    df_default = simulate_current_clamp(HodgkinHuxley(), stim)
    df_empty = simulate_current_clamp(HodgkinHuxley(optional_channels=()), stim)
    np.testing.assert_array_equal(
        df_default["voltage"].values, df_empty["voltage"].values
    )


# ---------------------------------------------------------------------------
# Ih channel integration tests
# ---------------------------------------------------------------------------


def test_current_clamp_with_ih_extra_columns():
    """Current clamp with Ih channel adds Ih_current and r columns."""
    neuron = HodgkinHuxley(optional_channels=(make_ih_channel(),))
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(neuron, stim)
    assert "Ih_current" in df.columns
    assert "r" in df.columns


def test_current_clamp_ih_gating_variable_in_bounds():
    """Ih gating variable r stays in [0, 1] during current clamp."""
    neuron = HodgkinHuxley(optional_channels=(make_ih_channel(),))
    stim = step_current(
        duration=30.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=20.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(neuron, stim)
    assert df["r"].min() >= 0.0
    assert df["r"].max() <= 1.0


def test_voltage_clamp_with_ih_extra_columns():
    """Voltage clamp with Ih channel adds Ih_current and r columns."""
    neuron = HodgkinHuxley(optional_channels=(make_ih_channel(),))
    prot = step_voltage(
        duration=20.0,
        voltage_amplitude=-40.0,
        step_start=5.0,
        step_duration=10.0,
        holding_voltage=-70.0,
        sampling_frequency=40000.0,
    )
    df = simulate_voltage_clamp(neuron, prot)
    assert "Ih_current" in df.columns
    assert "r" in df.columns


def test_voltage_clamp_total_current_includes_ih():
    """total_current includes Ih contribution: I_total == I_Na + I_K + I_L + I_Ih."""
    neuron = HodgkinHuxley(optional_channels=(make_ih_channel(),))
    prot = step_voltage(
        duration=10.0,
        voltage_amplitude=-40.0,
        step_start=2.0,
        step_duration=5.0,
        holding_voltage=-70.0,
        sampling_frequency=40000.0,
    )
    df = simulate_voltage_clamp(neuron, prot)
    expected = (
        df["sodium_current"]
        + df["potassium_current"]
        + df["leak_current"]
        + df["Ih_current"]
    )
    np.testing.assert_allclose(df["total_current"].values, expected.values, rtol=1e-10)


def test_multiple_optional_channels_coexist():
    """Two distinct optional channels can coexist and each contributes columns."""
    ch1 = make_ih_channel(g_max=0.1)
    gv2 = GatingVariable(name="q", power=1, alpha=lambda V: 0.05, beta=lambda V: 0.05)
    ch2 = BaseIonChannel(name="Iq", g_max=0.05, gating_variables=(gv2,), e_rev=-80.0)
    neuron = HodgkinHuxley(optional_channels=(ch1, ch2))
    stim = step_current(
        duration=10.0,
        current_amplitude=5.0,
        step_start=2.0,
        step_duration=5.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(neuron, stim)
    assert "Ih_current" in df.columns
    assert "Iq_current" in df.columns
    assert "r" in df.columns
    assert "q" in df.columns


def test_public_api_exports():
    """GatingVariable, BaseIonChannel, IonChannel, and make_ih_channel are exported."""
    assert hasattr(ap_sim, "GatingVariable")
    assert hasattr(ap_sim, "BaseIonChannel")
    assert hasattr(ap_sim, "IonChannel")
    assert hasattr(ap_sim, "make_ih_channel")


# ---------------------------------------------------------------------------
# IKa rate functions
# ---------------------------------------------------------------------------


def test_ika_gating_variable_steady_state_in_bounds():
    """IKa gating variables a_inf and b_inf are in [0, 1] for physiological voltages."""
    for V in np.linspace(-120.0, 0.0, 50):
        for alpha_fn, beta_fn in ((_alpha_a, _beta_a), (_alpha_b, _beta_b)):
            a = alpha_fn(V)
            b = beta_fn(V)
            assert a >= 0, f"alpha negative at V={V}"
            assert b >= 0, f"beta negative at V={V}"
            ss = a / (a + b)
            assert 0.0 <= ss <= 1.0, f"steady state {ss} out of [0,1] at V={V}"


def test_ika_kinetics_activation_increases_with_depolarisation():
    """IKa a_inf (activation) is higher at depolarised voltages."""

    def a_inf(V: float) -> float:
        """Activation steady-state at voltage V."""
        a, b = _alpha_a(V), _beta_a(V)
        return a / (a + b)

    assert a_inf(-20.0) > a_inf(-65.0) > a_inf(-100.0)


def test_ika_kinetics_inactivation_decreases_with_depolarisation():
    """IKa b_inf (inactivation) is lower at depolarised voltages."""

    def b_inf(V: float) -> float:
        """Inactivation steady-state at voltage V."""
        a, b = _alpha_b(V), _beta_b(V)
        return a / (a + b)

    assert b_inf(-100.0) > b_inf(-65.0) > b_inf(-20.0)


# ---------------------------------------------------------------------------
# make_ika_channel
# ---------------------------------------------------------------------------


def test_make_ika_channel_defaults():
    """make_ika_channel() produces a channel with the expected defaults."""
    from ap_sim.constants import DEFAULT_E_IKA, DEFAULT_G_IKA

    ch = make_ika_channel()
    assert ch.name == "IKa"
    assert ch.g_max == pytest.approx(DEFAULT_G_IKA)
    assert ch.e_rev == pytest.approx(DEFAULT_E_IKA)
    assert len(ch.gating_variables) == 2
    assert ch.gating_variables[0].name == "a"
    assert ch.gating_variables[0].power == 1
    assert ch.gating_variables[1].name == "b"
    assert ch.gating_variables[1].power == 1


def test_make_ika_channel_custom_params():
    """make_ika_channel accepts custom g_max and e_rev."""
    ch = make_ika_channel(g_max=10.0, e_rev=-80.0)
    assert ch.g_max == pytest.approx(10.0)
    assert ch.e_rev == pytest.approx(-80.0)


# ---------------------------------------------------------------------------
# IKa integration tests
# ---------------------------------------------------------------------------


def test_current_clamp_with_ika_extra_columns():
    """Current clamp with IKa channel adds IKa_current, a, and b columns."""
    neuron = HodgkinHuxley(optional_channels=(make_ika_channel(),))
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(neuron, stim)
    assert "IKa_current" in df.columns
    assert "a" in df.columns
    assert "b" in df.columns


def test_current_clamp_ika_gating_in_bounds():
    """IKa gating variables a and b stay in [0, 1] during current clamp."""
    neuron = HodgkinHuxley(optional_channels=(make_ika_channel(),))
    stim = step_current(
        duration=30.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=20.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(neuron, stim)
    assert df["a"].min() >= 0.0
    assert df["a"].max() <= 1.0
    assert df["b"].min() >= 0.0
    assert df["b"].max() <= 1.0


def test_voltage_clamp_with_ika_extra_columns():
    """Voltage clamp with IKa channel adds IKa_current, a, and b columns."""
    neuron = HodgkinHuxley(optional_channels=(make_ika_channel(),))
    prot = step_voltage(
        duration=20.0,
        voltage_amplitude=0.0,
        step_start=5.0,
        step_duration=10.0,
        holding_voltage=-70.0,
        sampling_frequency=40000.0,
    )
    df = simulate_voltage_clamp(neuron, prot)
    assert "IKa_current" in df.columns
    assert "a" in df.columns
    assert "b" in df.columns


def test_ika_and_ih_coexist():
    """IKa and Ih channels can coexist and each contributes its columns."""
    neuron = HodgkinHuxley(optional_channels=(make_ika_channel(), make_ih_channel()))
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(neuron, stim)
    assert "IKa_current" in df.columns
    assert "Ih_current" in df.columns
    assert "a" in df.columns
    assert "b" in df.columns
    assert "r" in df.columns


def test_public_api_exports_ika():
    """make_ika_channel is exported from the ap_sim public API."""
    assert hasattr(ap_sim, "make_ika_channel")
