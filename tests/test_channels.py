"""Tests for the additional ion channel framework.

Covers BaseIonChannel math, GatingVariable steady states, Ih kinetics,
backward compatibility with no additional channels, and validation errors.
"""

import numpy as np
import pytest

import ap_sim
from ap_sim.channels import (
    BaseIonChannel,
    CalciumGatingVariable,
    GatingVariable,
    IonChannel,
)
from ap_sim.clamp_simulations import simulate_current_clamp, simulate_voltage_clamp
from ap_sim.hodgkin_huxley import HodgkinHuxley
from ap_sim.additional_channels import (
    _alpha_a,
    _alpha_b,
    _alpha_hr,
    _alpha_kir,
    _alpha_p,
    _alpha_q,
    _alpha_r,
    _alpha_s,
    _alpha_w,
    _beta_a,
    _beta_b,
    _beta_hr,
    _beta_kir,
    _beta_p,
    _beta_q,
    _beta_r,
    _beta_s,
    _beta_w,
    make_ih_channel,
    make_ika_channel,
    make_ikca_channel,
    make_ikir_channel,
    make_im_channel,
    make_inar_channel,
    make_inap_channel,
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


def test_hh_duplicate_additional_channel_names_raises():
    """Duplicate additional channel names on HodgkinHuxley raise ValueError."""
    ch = make_ih_channel()
    with pytest.raises(ValueError, match="names must be unique"):
        HodgkinHuxley(additional_channels=(ch, ch))


def test_hh_builtin_channel_name_collision_raises():
    """Additional channel named 'Na' collides with built-in and raises ValueError."""
    gv = GatingVariable(name="r", power=1, alpha=lambda V: 0.1, beta=lambda V: 0.1)
    ch = BaseIonChannel(name="Na", g_max=0.1, gating_variables=(gv,), e_rev=-30.0)
    with pytest.raises(ValueError, match="collides with a built-in"):
        HodgkinHuxley(additional_channels=(ch,))


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


def test_current_clamp_no_additional_channels_identical_columns(hh_model):
    """simulate_current_clamp with no additional channels has exact classic columns."""
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


def test_current_clamp_no_additional_channels_values_unchanged():
    """Voltage trace is unchanged when additional_channels is empty vs. default."""
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    df_default = simulate_current_clamp(HodgkinHuxley(), stim)
    df_empty = simulate_current_clamp(HodgkinHuxley(additional_channels=()), stim)
    np.testing.assert_array_equal(
        df_default["voltage"].values, df_empty["voltage"].values
    )


# ---------------------------------------------------------------------------
# Ih channel integration tests
# ---------------------------------------------------------------------------


def test_current_clamp_with_ih_extra_columns():
    """Current clamp with Ih channel adds Ih_current and r columns."""
    neuron = HodgkinHuxley(additional_channels=(make_ih_channel(),))
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
    neuron = HodgkinHuxley(additional_channels=(make_ih_channel(),))
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
    neuron = HodgkinHuxley(additional_channels=(make_ih_channel(),))
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
    neuron = HodgkinHuxley(additional_channels=(make_ih_channel(),))
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
    neuron = HodgkinHuxley(additional_channels=(ch1, ch2))
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
    neuron = HodgkinHuxley(additional_channels=(make_ika_channel(),))
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
    neuron = HodgkinHuxley(additional_channels=(make_ika_channel(),))
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
    neuron = HodgkinHuxley(additional_channels=(make_ika_channel(),))
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
    neuron = HodgkinHuxley(additional_channels=(make_ika_channel(), make_ih_channel()))
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


# ---------------------------------------------------------------------------
# INaP rate functions
# ---------------------------------------------------------------------------


def test_inap_gating_variable_steady_state_in_bounds():
    """INaP gating variable p_inf is in [0, 1] for physiological voltages."""
    for V in np.linspace(-120.0, 0.0, 50):
        a = _alpha_p(V)
        b = _beta_p(V)
        assert a >= 0, f"alpha_p negative at V={V}"
        assert b >= 0, f"beta_p negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"steady state {ss} out of [0,1] at V={V}"


def test_inap_activation_increases_with_depolarisation():
    """INaP p_inf (activation) is higher at depolarised voltages."""

    def p_inf(V: float) -> float:
        """Activation steady-state at voltage V."""
        a, b = _alpha_p(V), _beta_p(V)
        return a / (a + b)

    assert p_inf(-20.0) > p_inf(-53.0) > p_inf(-100.0)


def test_inap_subthreshold_activation():
    """INaP p_inf is substantially activated below spike threshold (-52.6 mV half)."""

    def p_inf(V: float) -> float:
        """Activation steady-state at voltage V."""
        a, b = _alpha_p(V), _beta_p(V)
        return a / (a + b)

    # At half-activation voltage p_inf should be ~0.5
    assert p_inf(-52.6) == pytest.approx(0.5, abs=0.01)
    # Subthreshold range (-55 to -40 mV) should have partial activation
    assert 0.0 < p_inf(-55.0) < 0.5
    assert 0.5 < p_inf(-40.0) < 1.0


# ---------------------------------------------------------------------------
# make_inap_channel
# ---------------------------------------------------------------------------


def test_make_inap_channel_defaults():
    """make_inap_channel() produces a channel with the expected defaults."""
    from ap_sim.constants import DEFAULT_E_NAP, DEFAULT_G_NAP

    ch = make_inap_channel()
    assert ch.name == "INaP"
    assert ch.g_max == pytest.approx(DEFAULT_G_NAP)
    assert ch.e_rev == pytest.approx(DEFAULT_E_NAP)
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "p"
    assert ch.gating_variables[0].power == 1


def test_make_inap_channel_custom_params():
    """make_inap_channel accepts custom g_max and e_rev."""
    ch = make_inap_channel(g_max=1.0, e_rev=55.0)
    assert ch.g_max == pytest.approx(1.0)
    assert ch.e_rev == pytest.approx(55.0)


# ---------------------------------------------------------------------------
# INaP integration tests
# ---------------------------------------------------------------------------


def test_current_clamp_with_inap_extra_columns():
    """Current clamp with INaP channel adds INaP_current and p columns."""
    neuron = HodgkinHuxley(additional_channels=(make_inap_channel(),))
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(neuron, stim)
    assert "INaP_current" in df.columns
    assert "p" in df.columns


def test_voltage_clamp_with_inap_extra_columns():
    """Voltage clamp with INaP channel adds INaP_current and p columns."""
    neuron = HodgkinHuxley(additional_channels=(make_inap_channel(),))
    prot = step_voltage(
        duration=20.0,
        voltage_amplitude=0.0,
        step_start=5.0,
        step_duration=10.0,
        holding_voltage=-70.0,
        sampling_frequency=40000.0,
    )
    df = simulate_voltage_clamp(neuron, prot)
    assert "INaP_current" in df.columns
    assert "p" in df.columns


def test_current_clamp_inap_gating_in_bounds():
    """INaP gating variable p stays in [0, 1] during current clamp."""
    neuron = HodgkinHuxley(additional_channels=(make_inap_channel(),))
    stim = step_current(
        duration=30.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=20.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(neuron, stim)
    assert df["p"].min() >= 0.0
    assert df["p"].max() <= 1.0


def test_public_api_exports_inap():
    """make_inap_channel is exported from the ap_sim public API."""
    assert hasattr(ap_sim, "make_inap_channel")


# ---------------------------------------------------------------------------
# INaR rate functions
# ---------------------------------------------------------------------------


def test_inar_s_gating_steady_state_in_bounds():
    """INaR activation variable s_inf is in [0, 1] for physiological voltages."""
    for V in np.linspace(-120.0, 0.0, 50):
        a = _alpha_s(V)
        b = _beta_s(V)
        assert a >= 0, f"alpha_s negative at V={V}"
        assert b >= 0, f"beta_s negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"s steady state {ss} out of [0,1] at V={V}"


def test_inar_hr_gating_steady_state_in_bounds():
    """INaR unblocking variable hr_inf is in [0, 1] for physiological voltages."""
    for V in np.linspace(-120.0, 0.0, 50):
        a = _alpha_hr(V)
        b = _beta_hr(V)
        assert a >= 0, f"alpha_hr negative at V={V}"
        assert b >= 0, f"beta_hr negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"hr steady state {ss} out of [0,1] at V={V}"


def test_inar_activation_increases_with_depolarisation():
    """INaR s_inf (activation) is higher at depolarised voltages."""

    def s_inf(V: float) -> float:
        """Activation steady-state at voltage V."""
        a, b = _alpha_s(V), _beta_s(V)
        return a / (a + b)

    assert s_inf(-20.0) > s_inf(-42.0) > s_inf(-100.0)


def test_inar_unblocking_decreases_with_depolarisation():
    """INaR hr_inf (unblocking) is lower at depolarised voltages (more blocked)."""

    def hr_inf(V: float) -> float:
        """Unblocking steady-state at voltage V."""
        a, b = _alpha_hr(V), _beta_hr(V)
        return a / (a + b)

    assert hr_inf(-100.0) > hr_inf(-55.0) > hr_inf(-20.0)


def test_inar_rates_non_negative():
    """All four INaR rate functions are non-negative across physiological voltages."""
    for V in np.linspace(-120.0, 60.0, 100):
        assert _alpha_s(V) >= 0, f"alpha_s negative at V={V}"
        assert _beta_s(V) >= 0, f"beta_s negative at V={V}"
        assert _alpha_hr(V) >= 0, f"alpha_hr negative at V={V}"
        assert _beta_hr(V) >= 0, f"beta_hr negative at V={V}"


# ---------------------------------------------------------------------------
# make_inar_channel
# ---------------------------------------------------------------------------


def test_make_inar_channel_defaults():
    """make_inar_channel() produces a channel with the expected defaults."""
    from ap_sim.constants import DEFAULT_E_NAR, DEFAULT_G_NAR

    ch = make_inar_channel()
    assert ch.name == "INaR"
    assert ch.g_max == pytest.approx(DEFAULT_G_NAR)
    assert ch.e_rev == pytest.approx(DEFAULT_E_NAR)
    assert len(ch.gating_variables) == 2
    assert ch.gating_variables[0].name == "s"
    assert ch.gating_variables[0].power == 1
    assert ch.gating_variables[1].name == "hr"
    assert ch.gating_variables[1].power == 1


def test_make_inar_channel_custom_params():
    """make_inar_channel accepts custom g_max and e_rev."""
    ch = make_inar_channel(g_max=0.5, e_rev=55.0)
    assert ch.g_max == pytest.approx(0.5)
    assert ch.e_rev == pytest.approx(55.0)


# ---------------------------------------------------------------------------
# INaR integration tests
# ---------------------------------------------------------------------------


def test_current_clamp_with_inar_extra_columns():
    """Current clamp with INaR channel adds INaR_current, s, and hr columns."""
    neuron = HodgkinHuxley(additional_channels=(make_inar_channel(),))
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(neuron, stim)
    assert "INaR_current" in df.columns
    assert "s" in df.columns
    assert "hr" in df.columns


def test_voltage_clamp_with_inar_extra_columns():
    """Voltage clamp with INaR channel adds INaR_current, s, and hr columns."""
    neuron = HodgkinHuxley(additional_channels=(make_inar_channel(),))
    prot = step_voltage(
        duration=20.0,
        voltage_amplitude=0.0,
        step_start=5.0,
        step_duration=10.0,
        holding_voltage=-70.0,
        sampling_frequency=40000.0,
    )
    df = simulate_voltage_clamp(neuron, prot)
    assert "INaR_current" in df.columns
    assert "s" in df.columns
    assert "hr" in df.columns


def test_current_clamp_inar_gating_in_bounds():
    """INaR gating variables s and hr stay in [0, 1] during current clamp."""
    neuron = HodgkinHuxley(additional_channels=(make_inar_channel(),))
    stim = step_current(
        duration=30.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=20.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(neuron, stim)
    assert df["s"].min() >= 0.0
    assert df["s"].max() <= 1.0
    assert df["hr"].min() >= 0.0
    assert df["hr"].max() <= 1.0


def test_inap_and_inar_coexist():
    """INaP and INaR channels can coexist and each contributes columns."""
    neuron = HodgkinHuxley(
        additional_channels=(make_inap_channel(), make_inar_channel())
    )
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(neuron, stim)
    assert "INaP_current" in df.columns
    assert "INaR_current" in df.columns
    assert "p" in df.columns
    assert "s" in df.columns
    assert "hr" in df.columns


def test_all_additional_channels_coexist():
    """All seven additional channels (Ih, IKa, INaP, INaR, IM, IKir, IKCa) coexist."""
    from ap_sim.calcium import CalciumDynamics

    neuron = HodgkinHuxley(
        additional_channels=(
            make_ih_channel(),
            make_ika_channel(),
            make_inap_channel(),
            make_inar_channel(),
            make_im_channel(),
            make_ikir_channel(),
            make_ikca_channel(),
        ),
        calcium_dynamics=CalciumDynamics(),
    )
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(neuron, stim)
    for col in (
        "Ih_current",
        "IKa_current",
        "INaP_current",
        "INaR_current",
        "IM_current",
        "IKir_current",
        "IKCa_current",
    ):
        assert col in df.columns
    for gate in ("r", "a", "b", "p", "s", "hr", "w", "kir", "q"):
        assert gate in df.columns


def test_public_api_exports_inar():
    """make_inar_channel is exported from the ap_sim public API."""
    assert hasattr(ap_sim, "make_inar_channel")


# ---------------------------------------------------------------------------
# I_M rate functions
# ---------------------------------------------------------------------------


def test_im_gating_variable_steady_state_in_bounds():
    """IM gating variable w_inf is in [0, 1] for physiological voltages."""
    for V in np.linspace(-120.0, 0.0, 50):
        a = _alpha_w(V)
        b = _beta_w(V)
        assert a >= 0, f"alpha_w negative at V={V}"
        assert b >= 0, f"beta_w negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"steady state {ss} out of [0,1] at V={V}"


def test_im_activation_increases_with_depolarisation():
    """IM w_inf (activation) is higher at depolarised voltages."""

    def w_inf(V: float) -> float:
        """Activation steady-state at voltage V."""
        a, b = _alpha_w(V), _beta_w(V)
        return a / (a + b)

    assert w_inf(-20.0) > w_inf(-35.0) > w_inf(-100.0)


def test_im_slow_kinetics():
    """IM tau_w is greater than 50 ms near the half-activation voltage (-35 mV)."""
    tau = 1.0 / (_alpha_w(-35.0) + _beta_w(-35.0))
    assert tau > 50.0


# ---------------------------------------------------------------------------
# make_im_channel
# ---------------------------------------------------------------------------


def test_make_im_channel_defaults():
    """make_im_channel() produces a channel with the expected defaults."""
    from ap_sim.constants import DEFAULT_E_IM, DEFAULT_G_IM

    ch = make_im_channel()
    assert ch.name == "IM"
    assert ch.g_max == pytest.approx(DEFAULT_G_IM)
    assert ch.e_rev == pytest.approx(DEFAULT_E_IM)
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "w"
    assert ch.gating_variables[0].power == 1


def test_make_im_channel_custom_params():
    """make_im_channel accepts custom g_max and e_rev."""
    ch = make_im_channel(g_max=1.0, e_rev=-80.0)
    assert ch.g_max == pytest.approx(1.0)
    assert ch.e_rev == pytest.approx(-80.0)


# ---------------------------------------------------------------------------
# I_M integration tests
# ---------------------------------------------------------------------------


def test_current_clamp_with_im_extra_columns():
    """Current clamp with IM channel adds IM_current and w columns."""
    neuron = HodgkinHuxley(additional_channels=(make_im_channel(),))
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(neuron, stim)
    assert "IM_current" in df.columns
    assert "w" in df.columns


def test_voltage_clamp_with_im_extra_columns():
    """Voltage clamp with IM channel adds IM_current and w columns."""
    neuron = HodgkinHuxley(additional_channels=(make_im_channel(),))
    prot = step_voltage(
        duration=20.0,
        voltage_amplitude=0.0,
        step_start=5.0,
        step_duration=10.0,
        holding_voltage=-70.0,
        sampling_frequency=40000.0,
    )
    df = simulate_voltage_clamp(neuron, prot)
    assert "IM_current" in df.columns
    assert "w" in df.columns


def test_current_clamp_im_gating_in_bounds():
    """IM gating variable w stays in [0, 1] during current clamp."""
    neuron = HodgkinHuxley(additional_channels=(make_im_channel(),))
    stim = step_current(
        duration=30.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=20.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(neuron, stim)
    assert df["w"].min() >= 0.0
    assert df["w"].max() <= 1.0


def test_public_api_exports_im():
    """make_im_channel is exported from the ap_sim public API."""
    assert hasattr(ap_sim, "make_im_channel")


# ---------------------------------------------------------------------------
# I_Kir rate functions
# ---------------------------------------------------------------------------


def test_ikir_gating_variable_steady_state_in_bounds():
    """IKir gating variable kir_inf is in [0, 1] for physiological voltages."""
    for V in np.linspace(-120.0, 0.0, 50):
        a = _alpha_kir(V)
        b = _beta_kir(V)
        assert a >= 0, f"alpha_kir negative at V={V}"
        assert b >= 0, f"beta_kir negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"steady state {ss} out of [0,1] at V={V}"


def test_ikir_activation_increases_with_hyperpolarisation():
    """IKir kir_inf is higher at hyperpolarised voltages (inverted rectifier)."""

    def kir_inf(V: float) -> float:
        """Activation steady-state at voltage V."""
        a, b = _alpha_kir(V), _beta_kir(V)
        return a / (a + b)

    assert kir_inf(-100.0) > kir_inf(-80.0) > kir_inf(-40.0)


def test_ikir_fast_kinetics():
    """IKir tau_kir is at most 10 ms across physiological voltages."""
    for V in np.linspace(-120.0, 0.0, 50):
        tau = 1.0 / (_alpha_kir(V) + _beta_kir(V))
        assert tau <= 10.0, f"tau_kir too slow at V={V}"


# ---------------------------------------------------------------------------
# make_ikir_channel
# ---------------------------------------------------------------------------


def test_make_ikir_channel_defaults():
    """make_ikir_channel() produces a channel with the expected defaults."""
    from ap_sim.constants import DEFAULT_E_IKIR, DEFAULT_G_IKIR

    ch = make_ikir_channel()
    assert ch.name == "IKir"
    assert ch.g_max == pytest.approx(DEFAULT_G_IKIR)
    assert ch.e_rev == pytest.approx(DEFAULT_E_IKIR)
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "kir"
    assert ch.gating_variables[0].power == 1


def test_make_ikir_channel_custom_params():
    """make_ikir_channel accepts custom g_max and e_rev."""
    ch = make_ikir_channel(g_max=0.5, e_rev=-80.0)
    assert ch.g_max == pytest.approx(0.5)
    assert ch.e_rev == pytest.approx(-80.0)


# ---------------------------------------------------------------------------
# I_Kir integration tests
# ---------------------------------------------------------------------------


def test_current_clamp_with_ikir_extra_columns():
    """Current clamp with IKir channel adds IKir_current and kir columns."""
    neuron = HodgkinHuxley(additional_channels=(make_ikir_channel(),))
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(neuron, stim)
    assert "IKir_current" in df.columns
    assert "kir" in df.columns


def test_voltage_clamp_with_ikir_extra_columns():
    """Voltage clamp with IKir channel adds IKir_current and kir columns."""
    neuron = HodgkinHuxley(additional_channels=(make_ikir_channel(),))
    prot = step_voltage(
        duration=20.0,
        voltage_amplitude=0.0,
        step_start=5.0,
        step_duration=10.0,
        holding_voltage=-70.0,
        sampling_frequency=40000.0,
    )
    df = simulate_voltage_clamp(neuron, prot)
    assert "IKir_current" in df.columns
    assert "kir" in df.columns


def test_current_clamp_ikir_gating_in_bounds():
    """IKir gating variable kir stays in [0, 1] during current clamp."""
    neuron = HodgkinHuxley(additional_channels=(make_ikir_channel(),))
    stim = step_current(
        duration=30.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=20.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(neuron, stim)
    assert df["kir"].min() >= 0.0
    assert df["kir"].max() <= 1.0


def test_public_api_exports_ikir():
    """make_ikir_channel is exported from the ap_sim public API."""
    assert hasattr(ap_sim, "make_ikir_channel")


# ---------------------------------------------------------------------------
# CalciumGatingVariable infrastructure (Step 4)
# ---------------------------------------------------------------------------


def test_calcium_gating_variable_in_integrator():
    """A channel with CalciumGatingVariable initializes and runs without error."""
    cg = CalciumGatingVariable(
        name="q_test",
        power=1,
        alpha=lambda V, ca: 0.1 * ca if ca > 0 else 0.0,
        beta=lambda V, ca: 0.1,
    )
    ch = BaseIonChannel(
        name="ITest",
        g_max=0.5,
        gating_variables=(cg,),
        e_rev=-77.0,
    )
    from ap_sim.calcium import CalciumDynamics

    neuron = HodgkinHuxley(
        additional_channels=(ch,),
        calcium_dynamics=CalciumDynamics(),
    )
    stim = step_current(
        duration=10.0,
        current_amplitude=0.0,
        step_start=2.0,
        step_duration=5.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(neuron, stim)
    assert "ITest_current" in df.columns
    assert "q_test" in df.columns


def test_calcium_gating_variable_steady_state_depends_on_ca():
    """CalciumGatingVariable steady state differs for different ca_i values."""
    cg = CalciumGatingVariable(
        name="q_test2",
        power=1,
        alpha=lambda V, ca: ca / (ca + 0.001),
        beta=lambda V, ca: 1.0 - ca / (ca + 0.001),
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
    """Voltage-only channels still work after CalciumGatingVariable addition."""
    neuron = HodgkinHuxley(additional_channels=(make_ih_channel(), make_ika_channel()))
    stim = step_current(
        duration=10.0,
        current_amplitude=10.0,
        step_start=2.0,
        step_duration=5.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(neuron, stim)
    assert "Ih_current" in df.columns
    assert "IKa_current" in df.columns
    assert df["r"].min() >= 0.0
    assert df["r"].max() <= 1.0


def test_calcium_gating_variable_exported():
    """CalciumGatingVariable and AnyGatingVariable are in the ap_sim public API."""
    assert hasattr(ap_sim, "CalciumGatingVariable")
    assert hasattr(ap_sim, "AnyGatingVariable")


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
    from ap_sim.additional_channels import _ikca_q_inf

    V = -20.0
    assert _ikca_q_inf(V, 1e-2) > _ikca_q_inf(V, 1e-3) > _ikca_q_inf(V, 1e-4)


def test_ikca_activation_increases_with_depolarisation():
    """IKCa q_inf is higher at depolarised voltages at fixed [Ca²⁺]ᵢ."""
    from ap_sim.additional_channels import _ikca_q_inf

    ca = 1e-3
    assert _ikca_q_inf(20.0, ca) > _ikca_q_inf(-20.0, ca) > _ikca_q_inf(-80.0, ca)


def test_ikca_zero_calcium_gives_zero_activation():
    """IKCa q_inf is zero when [Ca²⁺]ᵢ is zero, regardless of voltage."""
    from ap_sim.additional_channels import _ikca_q_inf

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
    from ap_sim.constants import DEFAULT_E_IKCA, DEFAULT_G_IKCA

    ch = make_ikca_channel()
    assert ch.name == "IKCa"
    assert ch.g_max == pytest.approx(DEFAULT_G_IKCA)
    assert ch.e_rev == pytest.approx(DEFAULT_E_IKCA)
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "q"
    assert ch.gating_variables[0].power == 1


def test_make_ikca_channel_custom_params():
    """make_ikca_channel accepts custom g_max and e_rev."""
    ch = make_ikca_channel(g_max=2.0, e_rev=-80.0)
    assert ch.g_max == pytest.approx(2.0)
    assert ch.e_rev == pytest.approx(-80.0)


def test_ikca_is_not_calcium_ion_channel():
    """IKCa does not inherit CalciumIonChannel — it carries K⁺, not Ca²⁺."""
    from ap_sim.channels import CalciumIonChannel

    ch = make_ikca_channel()
    assert not isinstance(ch, CalciumIonChannel)


# ---------------------------------------------------------------------------
# I_KCa integration tests
# ---------------------------------------------------------------------------


def test_current_clamp_with_ikca():
    """Current clamp with IKCa channel adds IKCa_current and q columns."""
    from ap_sim.calcium import CalciumDynamics

    neuron = HodgkinHuxley(
        additional_channels=(make_ikca_channel(),),
        calcium_dynamics=CalciumDynamics(),
    )
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(neuron, stim)
    assert "IKCa_current" in df.columns
    assert "q" in df.columns


def test_current_clamp_ikca_gating_in_bounds():
    """IKCa gating variable q stays in [0, 1] during current clamp."""
    from ap_sim.calcium import CalciumDynamics

    neuron = HodgkinHuxley(
        additional_channels=(make_ikca_channel(),),
        calcium_dynamics=CalciumDynamics(),
    )
    stim = step_current(
        duration=30.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=20.0,
        sampling_frequency=40000.0,
    )
    df = simulate_current_clamp(neuron, stim)
    assert df["q"].min() >= 0.0
    assert df["q"].max() <= 1.0


def test_public_api_exports_ikca():
    """make_ikca_channel is exported from the ap_sim public API."""
    assert hasattr(ap_sim, "make_ikca_channel")
