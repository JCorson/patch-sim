"""Unit tests for array-backed gating state primitives.

Covers _initialize_gating_array, _dict_to_array, and the array-based
derivative/advance/clip helpers added for issue #262.
"""

import numpy as np
import pytest

from patch_sim.channels import GatingVariable, IonChannel, IonSpecies, NernstSpec
from patch_sim.clamp_simulations import (
    _dict_to_array,
    _gating_derivatives_arr,
    _initialize_gating_array,
    _initialize_gating_variables,
    _precompute_rate_tables,
    _precompute_rate_tables_arr,
)
from patch_sim.neuron import Neuron
from patch_sim.rates import CalciumDependentFn, VoltageOnlyFn

# ---------------------------------------------------------------------------
# _initialize_gating_array
# ---------------------------------------------------------------------------


def test_initialize_gating_array_matches_dict(hh_model):
    """Array initialisation agrees element-wise with the dict path."""
    V0 = -65.0
    ca_i = 0.0
    dict_state = _initialize_gating_variables(hh_model, V0, ca_i)
    arr_state = _initialize_gating_array(hh_model, V0, ca_i)

    assert arr_state.dtype == np.float64
    assert arr_state.shape == (len(hh_model.all_gating_variables),)
    for i, gv in enumerate(hh_model.all_gating_variables):
        assert arr_state[i] == pytest.approx(dict_state[gv.name])


def test_initialize_gating_array_values_in_unit_interval(hh_model):
    """All steady-state gating values are in [0, 1]."""
    arr = _initialize_gating_array(hh_model, -65.0)
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


# ---------------------------------------------------------------------------
# _dict_to_array
# ---------------------------------------------------------------------------


def test_dict_to_array_roundtrip(hh_model):
    """Converting dict → array preserves all gate values."""
    dict_state = _initialize_gating_variables(hh_model, -65.0)
    arr = _dict_to_array(dict_state, hh_model)

    assert arr.dtype == np.float64
    for i, gv in enumerate(hh_model.all_gating_variables):
        assert arr[i] == pytest.approx(dict_state[gv.name])


def test_dict_to_array_ordering_follows_gating_index(hh_model):
    """Element i of the array corresponds to gating_index[gv.name] == i."""
    dict_state = _initialize_gating_variables(hh_model, -65.0)
    arr = _dict_to_array(dict_state, hh_model)
    idx = hh_model.gating_index

    for name, i in idx.items():
        assert arr[i] == pytest.approx(dict_state[name])


# ---------------------------------------------------------------------------
# compute_current_array — fast paths
# ---------------------------------------------------------------------------


def _make_zero_gate_channel() -> IonChannel:
    """Build a zero-gate Na leak channel for testing."""
    return IonChannel(
        name="NaLeak",
        g_max=0.5,
        gating_variables=(),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )


def _make_single_power1_channel() -> IonChannel:
    """Build a single-gate power-1 channel for testing."""
    gv = GatingVariable(
        name="x",
        power=1,
        alpha=VoltageOnlyFn(lambda _v, _c: 0.1),
        beta=VoltageOnlyFn(lambda _v, _c: 0.1),
    )
    return IonChannel(
        name="Xch",
        g_max=1.0,
        gating_variables=(gv,),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )


def _make_two_gate_channel() -> IonChannel:
    """Build a two-gate channel (power 3, power 1) for testing."""
    gv1 = GatingVariable(
        name="m2",
        power=3,
        alpha=VoltageOnlyFn(lambda _v, _c: 0.2),
        beta=VoltageOnlyFn(lambda _v, _c: 0.1),
    )
    gv2 = GatingVariable(
        name="h2",
        power=1,
        alpha=VoltageOnlyFn(lambda _v, _c: 0.05),
        beta=VoltageOnlyFn(lambda _v, _c: 0.15),
    )
    return IonChannel(
        name="Mch",
        g_max=2.0,
        gating_variables=(gv1, gv2),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )


def test_compute_current_array_zero_gate_fast_path():
    """Zero-gate channel returns g_max * (V - e_rev) without touching state."""
    neuron = Neuron()
    ch = _make_zero_gate_channel()
    state_array = np.array([], dtype=np.float64)
    gate_indices = np.empty(0, dtype=np.intp)
    gate_powers = np.empty(0, dtype=np.intp)
    e_rev = ch.reversal_potential(neuron)

    result = ch.compute_current_array(
        0.0, state_array, gate_indices, gate_powers, e_rev
    )
    assert result == pytest.approx(ch.g_max * (0.0 - e_rev))


def test_compute_current_array_single_power1_fast_path():
    """Single-gate power-1 channel matches compute_current on identical state."""
    neuron = Neuron(additional_channels=(_make_single_power1_channel(),))
    ch = neuron.additional_channels[0]
    e_rev = neuron.reversal_potentials[ch.name]

    dict_state = _initialize_gating_variables(neuron, -65.0)
    arr_state = _dict_to_array(dict_state, neuron)
    specs = neuron.channel_gate_specs
    ch_idx = list(neuron.all_channels).index(ch)
    idxs, pows = specs[ch_idx]

    V = -30.0
    dict_result = ch.compute_current(V, dict_state, neuron)
    arr_result = ch.compute_current_array(V, arr_state, idxs, pows, e_rev)
    assert arr_result == pytest.approx(dict_result)


def test_compute_current_array_general_path_matches_dict():
    """Multi-gate channel with mixed powers matches compute_current."""
    neuron = Neuron(additional_channels=(_make_two_gate_channel(),))
    ch = neuron.additional_channels[0]
    e_rev = neuron.reversal_potentials[ch.name]

    dict_state = _initialize_gating_variables(neuron, -65.0)
    arr_state = _dict_to_array(dict_state, neuron)
    specs = neuron.channel_gate_specs
    ch_idx = list(neuron.all_channels).index(ch)
    idxs, pows = specs[ch_idx]

    for V in (-80.0, -40.0, 0.0, 20.0):
        dict_result = ch.compute_current(V, dict_state, neuron)
        arr_result = ch.compute_current_array(V, arr_state, idxs, pows, e_rev)
        assert arr_result == pytest.approx(dict_result)


# ---------------------------------------------------------------------------
# _gating_derivatives_arr
# ---------------------------------------------------------------------------


def test_gating_derivatives_arr_matches_dict(hh_model):
    """Array-based gating derivatives match the dict path element-wise."""
    from patch_sim.clamp_simulations import _gating_derivatives

    V = -55.0
    ca_i = 0.0
    dict_state = _initialize_gating_variables(hh_model, -65.0, ca_i)
    arr_state = _dict_to_array(dict_state, hh_model)

    dict_derivs, dict_dca = _gating_derivatives(hh_model, V, dict_state, ca_i)
    arr_derivs, arr_dca = _gating_derivatives_arr(hh_model, V, arr_state, ca_i)

    assert arr_dca == pytest.approx(dict_dca)
    for i, gv in enumerate(hh_model.all_gating_variables):
        assert float(arr_derivs[i]) == pytest.approx(dict_derivs[gv.name])


# ---------------------------------------------------------------------------
# _precompute_rate_tables_arr
# ---------------------------------------------------------------------------


def test_precompute_rate_tables_arr_shape(hh_model):
    """2-D table has shape (n_gates, n_steps) for an HH model."""
    voltage_protocol = np.linspace(-70.0, 0.0, 200)
    alpha_arr, beta_arr, ca_dep = _precompute_rate_tables_arr(
        hh_model, voltage_protocol
    )

    n_gates = len(hh_model.all_gating_variables)
    n_steps = len(voltage_protocol)
    assert alpha_arr.shape == (n_gates, n_steps)
    assert beta_arr.shape == (n_gates, n_steps)
    assert ca_dep.shape == (n_gates,)
    assert ca_dep.dtype == bool


def test_precompute_rate_tables_arr_no_ca_dep_for_hh(hh_model):
    """HH model has no Ca²⁺-dependent gates — all entries should be False."""
    voltage_protocol = np.linspace(-70.0, 0.0, 100)
    _, _, ca_dep = _precompute_rate_tables_arr(hh_model, voltage_protocol)
    assert not ca_dep.any()


def test_precompute_rate_tables_arr_matches_dict_tables(hh_model):
    """Array tables agree with per-name dict tables from _precompute_rate_tables."""
    voltage_protocol = np.linspace(-70.0, 0.0, 100)
    alpha_dict, beta_dict = _precompute_rate_tables(hh_model, voltage_protocol)
    alpha_arr, beta_arr, _ = _precompute_rate_tables_arr(hh_model, voltage_protocol)

    for i, gv in enumerate(hh_model.all_gating_variables):
        a_expected = alpha_dict[gv.name]
        b_expected = beta_dict[gv.name]
        if a_expected is not None and b_expected is not None:
            np.testing.assert_allclose(alpha_arr[i], a_expected)
            np.testing.assert_allclose(beta_arr[i], b_expected)


def test_precompute_rate_tables_arr_ca_dep_gate_marked():
    """A channel with a Ca²⁺-dependent gate is marked True in ca_dep_mask."""
    from patch_sim.calcium import CalciumDynamics

    gv = GatingVariable(
        name="ca_gate",
        power=1,
        alpha=CalciumDependentFn(lambda _v, ca_i: 0.1 * ca_i),
        beta=VoltageOnlyFn(lambda _v, _c: 0.1),
    )
    ch = IonChannel(
        name="Test",
        g_max=0.5,
        gating_variables=(gv,),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )
    neuron = Neuron(
        additional_channels=(ch,),
        calcium_dynamics=CalciumDynamics(),
    )
    voltage_protocol = np.linspace(-70.0, 0.0, 50)
    _, _, ca_dep = _precompute_rate_tables_arr(neuron, voltage_protocol)
    idx = neuron.gating_index["ca_gate"]
    assert ca_dep[idx]
