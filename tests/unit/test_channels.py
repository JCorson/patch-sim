"""Tests for the additional ion channel framework.

Covers IonChannel math, GatingVariable steady states, Ih kinetics,
backward compatibility with no additional channels, and validation errors.
"""

import numpy as np
import pytest

import patch_sim
from patch_sim.additional_channels import (
    _alpha_a,
    _alpha_b,
    _alpha_d,
    _alpha_dn,
    _alpha_dt,
    _alpha_f,
    _alpha_fn,
    _alpha_ft,
    _alpha_hr,
    _alpha_kir,
    _alpha_p,
    _alpha_q,
    _alpha_r,
    _alpha_s,
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
    _beta_kir,
    _beta_p,
    _beta_q,
    _beta_r,
    _beta_s,
    _beta_w,
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
)
from patch_sim.channels import (
    GatingVariable,
    GoldmanSpec,
    IonChannel,
    IonSpecies,
    NernstSpec,
)
from patch_sim.clamp_simulations import simulate_current_clamp, simulate_voltage_clamp
from patch_sim.neuron import Neuron
from patch_sim.protocols import step_current, step_voltage
from patch_sim.rates import CalciumDependentFn, VoltageOnlyFn

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


def test_ih_kinetics_alpha_increases_with_hyperpolarisation():
    """alpha_r should increase as voltage becomes more negative (Ih is HCN-type)."""
    alpha_at_minus100 = _alpha_r(-100.0, 0.0)
    alpha_at_minus65 = _alpha_r(-65.0, 0.0)
    alpha_at_minus40 = _alpha_r(-40.0, 0.0)
    assert alpha_at_minus100 > alpha_at_minus65 > alpha_at_minus40


def test_ih_kinetics_steady_state_higher_at_hyperpolarised():
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
        Neuron(additional_channels=(ch, ch))


def test_hh_builtin_channel_name_collision_raises():
    """Additional channel named 'Na' collides with built-in and raises ValueError."""
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
    with pytest.raises(ValueError, match="collides with a built-in"):
        Neuron(additional_channels=(ch,))


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


def test_current_clamp_no_additional_channels_values_unchanged():
    """Voltage trace is unchanged when additional_channels is empty vs. default."""
    stim = step_current(
        duration=20.0,
        current_amplitude=10.0,
        step_start=5.0,
        step_duration=10.0,
        sampling_frequency=40000.0,
    )
    result_default = simulate_current_clamp(Neuron(), stim)
    result_empty = simulate_current_clamp(Neuron(additional_channels=()), stim)
    np.testing.assert_array_equal(result_default["voltage"], result_empty["voltage"])


# ---------------------------------------------------------------------------
# h channel integration tests
# ---------------------------------------------------------------------------


def test_current_clamp_with_ih_extra_columns():
    """Current clamp with h channel adds Ih and r columns."""
    neuron = Neuron(additional_channels=(make_ih_channel(),))
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
    neuron = Neuron(additional_channels=(make_ih_channel(),))
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
    neuron = Neuron(additional_channels=(make_ih_channel(),))
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
    neuron = Neuron(additional_channels=(make_ih_channel(),))
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
    neuron = Neuron(additional_channels=(ch1, ch2))
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


def test_public_api_exports():
    """GatingVariable and IonChannel and make_ih_channel are exported."""
    assert hasattr(patch_sim, "GatingVariable")
    assert hasattr(patch_sim, "IonChannel")
    assert hasattr(patch_sim, "make_ih_channel")


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


def test_ika_kinetics_activation_increases_with_depolarisation():
    """IKa a_inf (activation) is higher at depolarised voltages."""

    def a_inf(V: float) -> float:
        """Activation steady-state at voltage V."""
        a, b = _alpha_a(V, 0.0), _beta_a(V, 0.0)
        return a / (a + b)

    assert a_inf(-20.0) > a_inf(-65.0) > a_inf(-100.0)


def test_ika_kinetics_inactivation_decreases_with_depolarisation():
    """IKa b_inf (inactivation) is lower at depolarised voltages."""

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
    neuron = Neuron(additional_channels=(make_ika_channel(),))
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
    neuron = Neuron(additional_channels=(make_ika_channel(),))
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
    neuron = Neuron(additional_channels=(make_ika_channel(),))
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
    neuron = Neuron(additional_channels=(make_ika_channel(), make_ih_channel()))
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


def test_public_api_exports_ika():
    """make_ika_channel is exported from the patch_sim public API."""
    assert hasattr(patch_sim, "make_ika_channel")


# ---------------------------------------------------------------------------
# IKv31 (Kv3.1-type K+)
# ---------------------------------------------------------------------------


def test_ikv31_gating_steady_state_in_bounds():
    """IKv31 gating variable nk_inf is in [0, 1] for physiological voltages."""
    from patch_sim.additional_channels import _ikv31_alpha_nk, _ikv31_beta_nk

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
    from patch_sim.additional_channels import _ikv31_alpha_nk, _ikv31_beta_nk

    alpha = _ikv31_alpha_nk(-65.0, 0.0)
    beta = _ikv31_beta_nk(-65.0, 0.0)
    nk_inf = alpha / (alpha + beta)
    assert nk_inf < 0.02, f"nk_inf={nk_inf} should be near zero at -65 mV"


def test_ikv31_strong_activation_depolarized():
    """IKv31 nk_inf is well above 0.5 at 0 mV (depolarized)."""
    from patch_sim.additional_channels import _ikv31_alpha_nk, _ikv31_beta_nk

    alpha = _ikv31_alpha_nk(0.0, 0.0)
    beta = _ikv31_beta_nk(0.0, 0.0)
    nk_inf = alpha / (alpha + beta)
    assert nk_inf > 0.5, f"nk_inf={nk_inf} should be above 0.5 at 0 mV"


def test_make_ikv31_channel_defaults():
    """make_ikv31_channel() produces a channel with the expected defaults."""
    from patch_sim.additional_channels import make_ikv31_channel
    from patch_sim.constants import DEFAULT_G_IKV31

    ch = make_ikv31_channel()
    assert ch.name == "Kv31"
    assert ch.g_max == pytest.approx(DEFAULT_G_IKV31)
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "nk"
    assert ch.gating_variables[0].power == 2


def test_current_clamp_with_ikv31():
    """Current clamp with IKv31 channel adds Kv31 and nk columns."""
    from patch_sim.additional_channels import make_ikv31_channel

    neuron = Neuron(additional_channels=(make_ikv31_channel(),))
    stimulus = np.zeros(int(40_000 * 0.05))
    result = simulate_current_clamp(neuron=neuron, current_external=stimulus)
    assert "IKv31" in result.dtype.names
    assert "nk" in result.dtype.names


def test_public_api_exports_ikv31():
    """make_ikv31_channel is exported from the patch_sim public API."""
    assert hasattr(patch_sim, "make_ikv31_channel")


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


def test_inap_activation_increases_with_depolarisation():
    """INaP p_inf (activation) is higher at depolarised voltages."""

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
# make_inap_channel
# ---------------------------------------------------------------------------


def test_make_inap_channel_defaults():
    """make_inap_channel() produces a channel with the expected defaults."""
    from patch_sim.constants import DEFAULT_G_NAP

    ch = make_inap_channel()
    assert ch.name == "NaP"
    assert ch.g_max == pytest.approx(DEFAULT_G_NAP)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.SODIUM
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "p"
    assert ch.gating_variables[0].power == 1


def test_make_inap_channel_custom_params():
    """make_inap_channel accepts custom g_max."""
    ch = make_inap_channel(g_max=1.0)
    assert ch.g_max == pytest.approx(1.0)
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.SODIUM


# ---------------------------------------------------------------------------
# INaP integration tests
# ---------------------------------------------------------------------------


def test_current_clamp_with_inap_extra_columns():
    """Current clamp with NaP channel adds INaP and p columns."""
    neuron = Neuron(additional_channels=(make_inap_channel(),))
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


def test_voltage_clamp_with_inap_extra_columns():
    """Voltage clamp with NaP channel adds INaP and p columns."""
    neuron = Neuron(additional_channels=(make_inap_channel(),))
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
    neuron = Neuron(additional_channels=(make_inap_channel(),))
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


def test_public_api_exports_inap():
    """make_inap_channel is exported from the patch_sim public API."""
    assert hasattr(patch_sim, "make_inap_channel")


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


def test_inar_activation_increases_with_depolarisation():
    """INaR s_inf (activation) is higher at depolarised voltages."""

    def s_inf(V: float) -> float:
        """Activation steady-state at voltage V."""
        a, b = _alpha_s(V, 0.0), _beta_s(V, 0.0)
        return a / (a + b)

    assert s_inf(-20.0) > s_inf(-42.0) > s_inf(-100.0)


def test_inar_unblocking_decreases_with_depolarisation():
    """INaR hr_inf (unblocking) is lower at depolarised voltages (more blocked)."""

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
    neuron = Neuron(additional_channels=(make_inar_channel(),))
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
    neuron = Neuron(additional_channels=(make_inar_channel(),))
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
    neuron = Neuron(additional_channels=(make_inar_channel(),))
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
    neuron = Neuron(additional_channels=(make_inap_channel(), make_inar_channel()))
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


def test_all_additional_channels_coexist():
    """All seven additional channels (Ih, IKa, INaP, INaR, IM, IKir, IKCa) coexist."""
    from patch_sim.calcium import CalciumDynamics

    neuron = Neuron(
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


def test_public_api_exports_inar():
    """make_inar_channel is exported from the patch_sim public API."""
    assert hasattr(patch_sim, "make_inar_channel")


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


def test_im_activation_increases_with_depolarisation():
    """IM w_inf (activation) is higher at depolarised voltages."""

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
    neuron = Neuron(additional_channels=(make_im_channel(),))
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
    neuron = Neuron(additional_channels=(make_im_channel(),))
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
    neuron = Neuron(additional_channels=(make_im_channel(),))
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


def test_public_api_exports_im():
    """make_im_channel is exported from the patch_sim public API."""
    assert hasattr(patch_sim, "make_im_channel")


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


def test_ikir_activation_increases_with_hyperpolarisation():
    """IKir kir_inf is higher at hyperpolarised voltages (inverted rectifier)."""

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
    neuron = Neuron(additional_channels=(make_ikir_channel(),))
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
    neuron = Neuron(additional_channels=(make_ikir_channel(),))
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
    neuron = Neuron(additional_channels=(make_ikir_channel(),))
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


def test_public_api_exports_ikir():
    """make_ikir_channel is exported from the patch_sim public API."""
    assert hasattr(patch_sim, "make_ikir_channel")


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

    neuron = Neuron(
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
    neuron = Neuron(additional_channels=(make_ih_channel(), make_ika_channel()))
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


def test_calcium_gating_variable_exported():
    """GatingVariable is in the patch_sim public API."""
    assert hasattr(patch_sim, "GatingVariable")


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
    from patch_sim.additional_channels import _ikca_q_inf

    V = -20.0
    assert _ikca_q_inf(V, 1e-2) > _ikca_q_inf(V, 1e-3) > _ikca_q_inf(V, 1e-4)


def test_ikca_activation_increases_with_depolarisation():
    """IKCa q_inf is higher at depolarised voltages at fixed [Ca²⁺]ᵢ."""
    from patch_sim.additional_channels import _ikca_q_inf

    ca = 1e-3
    assert _ikca_q_inf(20.0, ca) > _ikca_q_inf(-20.0, ca) > _ikca_q_inf(-80.0, ca)


def test_ikca_zero_calcium_gives_zero_activation():
    """IKCa q_inf is zero when [Ca²⁺]ᵢ is zero, regardless of voltage."""
    from patch_sim.additional_channels import _ikca_q_inf

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

    neuron = Neuron(
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
    result = simulate_current_clamp(neuron, stim)
    assert result.dtype.names is not None
    assert "IKCa" in result.dtype.names
    assert "q" in result.dtype.names


def test_current_clamp_ikca_gating_in_bounds():
    """IKCa gating variable q stays in [0, 1] during current clamp."""
    from patch_sim.calcium import CalciumDynamics

    neuron = Neuron(
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
    result = simulate_current_clamp(neuron, stim)
    assert result["q"].min() >= 0.0
    assert result["q"].max() <= 1.0


def test_public_api_exports_ikca():
    """make_ikca_channel is exported from the patch_sim public API."""
    assert hasattr(patch_sim, "make_ikca_channel")


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


def test_ical_activation_increases_with_depolarisation():
    """ICaL d_inf (activation) is higher at depolarised voltages."""

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

    neuron = Neuron(
        additional_channels=(make_ical_channel(),),
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

    neuron = Neuron(
        additional_channels=(make_ical_channel(),),
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

    neuron = Neuron(
        additional_channels=(make_ical_channel(),),
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


def test_public_api_exports_ical():
    """make_ical_channel is exported from patch_sim."""
    assert hasattr(patch_sim, "make_ical_channel")


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


def test_icat_activation_increases_with_depolarisation():
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

    neuron = Neuron(
        additional_channels=(make_icat_channel(),),
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

    neuron = Neuron(
        additional_channels=(make_icat_channel(),),
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

    neuron = Neuron(
        additional_channels=(make_icat_channel(),),
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


def test_public_api_exports_icat():
    """make_icat_channel is exported from the patch_sim public API."""
    assert hasattr(patch_sim, "make_icat_channel")


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


def test_ican_activation_increases_with_depolarisation():
    """ICaN dn_inf (activation) is higher at depolarised voltages."""

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

    neuron = Neuron(
        additional_channels=(make_ican_channel(),),
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

    neuron = Neuron(
        additional_channels=(make_ican_channel(),),
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

    neuron = Neuron(
        additional_channels=(make_ican_channel(),),
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


def test_public_api_exports_ican():
    """make_ican_channel is exported from the patch_sim public API."""
    assert hasattr(patch_sim, "make_ican_channel")
