"""Integration tests for the array-backed gating state path (issue #262).

Verifies that use_array_state=True produces results numerically equivalent
to the dict path for HH, multi-channel, and Ca²⁺-dependent presets.
"""

import numpy as np
import pytest

from patch_sim.calcium import CalciumDynamics
from patch_sim.clamp_simulations import (
    simulate_current_clamp,
    simulate_current_clamp_from_state,
    simulate_voltage_clamp,
    simulate_voltage_clamp_from_state,
)
from patch_sim.neuron import Neuron
from patch_sim.protocols import step_current, step_voltage

# Short protocols for test speed (5 ms at 40 kHz = 200 steps)
_VC_PROT = step_voltage(
    duration=5.0,
    voltage_amplitude=0.0,
    step_start=1.0,
    step_duration=3.0,
    holding_voltage=-65.0,
    sampling_frequency=40000.0,
)
_CC_PROT = step_current(
    duration=5.0,
    current_amplitude=10.0,
    step_start=1.0,
    step_duration=3.0,
    sampling_frequency=40000.0,
)


def _assert_results_match(dict_result, arr_result):
    """Assert all structured-array fields agree to rtol=1e-12."""
    assert dict_result.dtype.names == arr_result.dtype.names
    for field in dict_result.dtype.names:
        np.testing.assert_allclose(
            dict_result[field],
            arr_result[field],
            rtol=1e-12,
            err_msg=f"Field '{field}' differs between dict and array paths",
        )


# ---------------------------------------------------------------------------
# HH model
# ---------------------------------------------------------------------------


def test_vc_array_matches_dict_hh(hh_model):
    """Voltage clamp with use_array_state=True matches dict path for HH model."""
    dict_result = simulate_voltage_clamp(hh_model, _VC_PROT)
    arr_result = simulate_voltage_clamp(hh_model, _VC_PROT, use_array_state=True)
    _assert_results_match(dict_result, arr_result)


def test_cc_array_matches_dict_hh(hh_model):
    """Current clamp with use_array_state=True matches dict path for HH model."""
    dict_result = simulate_current_clamp(hh_model, _CC_PROT)
    arr_result = simulate_current_clamp(hh_model, _CC_PROT, use_array_state=True)
    _assert_results_match(dict_result, arr_result)


# ---------------------------------------------------------------------------
# Multi-channel model (HH + Ih + IKa)
# ---------------------------------------------------------------------------


def _make_multi_channel_neuron() -> Neuron:
    """HH + Ih + IKa to test multi-gate and GoldmanSpec channels."""
    from patch_sim.additional_channels import make_ih_channel, make_ika_channel

    return Neuron(additional_channels=(make_ih_channel(), make_ika_channel()))


def test_vc_array_matches_dict_multi_channel():
    """Voltage clamp matches with multiple additional channels."""
    neuron = _make_multi_channel_neuron()
    dict_result = simulate_voltage_clamp(neuron, _VC_PROT)
    arr_result = simulate_voltage_clamp(neuron, _VC_PROT, use_array_state=True)
    _assert_results_match(dict_result, arr_result)


def test_cc_array_matches_dict_multi_channel():
    """Current clamp matches with multiple additional channels."""
    neuron = _make_multi_channel_neuron()
    dict_result = simulate_current_clamp(neuron, _CC_PROT)
    arr_result = simulate_current_clamp(neuron, _CC_PROT, use_array_state=True)
    _assert_results_match(dict_result, arr_result)


# ---------------------------------------------------------------------------
# Ca²⁺-dependent channel (IKCa) — exercises the ca_dep_mask fallback
# ---------------------------------------------------------------------------


def _make_ikca_neuron() -> Neuron:
    """HH + IKCa + CalciumDynamics to exercise Ca²⁺-dependent rate fallback."""
    from patch_sim.additional_channels import make_ikca_channel

    return Neuron(
        additional_channels=(make_ikca_channel(),),
        calcium_dynamics=CalciumDynamics(),
    )


def test_vc_array_matches_dict_ikca():
    """Voltage clamp matches with Ca²⁺-dependent IKCa channel."""
    neuron = _make_ikca_neuron()
    dict_result = simulate_voltage_clamp(neuron, _VC_PROT)
    arr_result = simulate_voltage_clamp(neuron, _VC_PROT, use_array_state=True)
    _assert_results_match(dict_result, arr_result)


def test_cc_array_matches_dict_ikca():
    """Current clamp matches with Ca²⁺-dependent IKCa channel."""
    neuron = _make_ikca_neuron()
    dict_result = simulate_current_clamp(neuron, _CC_PROT)
    arr_result = simulate_current_clamp(neuron, _CC_PROT, use_array_state=True)
    _assert_results_match(dict_result, arr_result)


# ---------------------------------------------------------------------------
# from_state variants — key-validation contract is preserved
# ---------------------------------------------------------------------------


def test_vc_from_state_array_matches_dict(hh_model):
    """simulate_voltage_clamp_from_state array path matches dict path."""
    from patch_sim.clamp_simulations import _initialize_gating_variables

    ca_i = 0.0
    gating_state = _initialize_gating_variables(hh_model, _VC_PROT[0], ca_i)

    dict_result = simulate_voltage_clamp_from_state(hh_model, _VC_PROT, gating_state)
    arr_result = simulate_voltage_clamp_from_state(
        hh_model, _VC_PROT, gating_state, use_array_state=True
    )
    _assert_results_match(dict_result, arr_result)


def test_cc_from_state_array_matches_dict(hh_model):
    """simulate_current_clamp_from_state array path matches dict path."""
    from patch_sim.clamp_simulations import _initialize_gating_variables

    ca_i = 0.0
    gating_state = _initialize_gating_variables(hh_model, hh_model.v_rest, ca_i)

    dict_result = simulate_current_clamp_from_state(
        hh_model, _CC_PROT, hh_model.v_rest, gating_state
    )
    arr_result = simulate_current_clamp_from_state(
        hh_model, _CC_PROT, hh_model.v_rest, gating_state, use_array_state=True
    )
    _assert_results_match(dict_result, arr_result)


def test_vc_from_state_array_mismatched_keys_raises(hh_model):
    """Array path raises with same message as dict path on bad keys."""
    bad_state = {"bad_key": 0.5}
    with pytest.raises(ValueError, match="gating_state keys do not match"):
        simulate_voltage_clamp_from_state(
            hh_model, _VC_PROT, bad_state, use_array_state=True
        )


def test_cc_from_state_array_mismatched_keys_raises(hh_model):
    """Array path raises with same message as dict path on bad keys."""
    bad_state = {"bad_key": 0.5}
    with pytest.raises(ValueError, match="gating_state keys do not match"):
        simulate_current_clamp_from_state(
            hh_model, _CC_PROT, hh_model.v_rest, bad_state, use_array_state=True
        )


# ---------------------------------------------------------------------------
# Output structure — column names and dtypes are unchanged
# ---------------------------------------------------------------------------


def test_vc_array_output_fields_identical_to_dict(hh_model):
    """Array path produces the same structured-array fields as the dict path."""
    dict_result = simulate_voltage_clamp(hh_model, _VC_PROT)
    arr_result = simulate_voltage_clamp(hh_model, _VC_PROT, use_array_state=True)
    assert dict_result.dtype == arr_result.dtype


def test_cc_array_output_fields_identical_to_dict(hh_model):
    """Array path produces the same structured-array fields as the dict path."""
    dict_result = simulate_current_clamp(hh_model, _CC_PROT)
    arr_result = simulate_current_clamp(hh_model, _CC_PROT, use_array_state=True)
    assert dict_result.dtype == arr_result.dtype
