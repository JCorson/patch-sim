"""Unit tests for pure logic in the patch_sim_ui state package.

These tests exercise substate methods that contain no Reflex event-loop
machinery.  Instantiation uses ``_reflex_internal_init=True`` (the same
flag the framework passes internally) after setting ``PYTEST_CURRENT_TEST``
so that Reflex's ``is_testing_env()`` guard also passes.

The environment variable is set at import time so the substate metaclass
registration does not see a non-testing environment.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Must be set before importing Reflex/SimulationState so the metaclass and init
# guard both see a testing environment.
os.environ.setdefault("PYTEST_CURRENT_TEST", "test_state.py::setup")

pytest.importorskip("reflex")

from patch_sim.constants import (
    CORTICAL_PYRAMIDAL,
    DOPAMINERGIC,
    FAST_SPIKING_INTERNEURON,
    PURKINJE,
    SQUID_GIANT_AXON,
    THALAMIC_RELAY,
)
from patch_sim_ui import constants  # noqa: E402
from patch_sim_ui.log_handler import UILogRecord  # noqa: E402
from patch_sim_ui.plotting import Sweep  # noqa: E402
from patch_sim_ui.state import SimulationState  # noqa: E402
from patch_sim_ui.state.log import LogState  # noqa: E402
from patch_sim_ui.state.neuron import NeuronState  # noqa: E402
from patch_sim_ui.state.protocol import ProtocolState  # noqa: E402
from patch_sim_ui.state.visibility import VisibilityState  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state() -> SimulationState:
    """Return a fresh SimulationState instance bypassing the Reflex runtime guard."""
    return SimulationState(_reflex_internal_init=True)


def _make_log_state() -> LogState:
    """Return a fresh LogState instance bypassing the Reflex runtime guard."""
    return LogState(_reflex_internal_init=True)


def _make_neuron_state() -> NeuronState:
    """Return a fresh NeuronState instance bypassing the Reflex runtime guard."""
    return NeuronState(_reflex_internal_init=True)


def _make_protocol_state() -> ProtocolState:
    """Return a fresh ProtocolState instance bypassing the Reflex runtime guard."""
    return ProtocolState(_reflex_internal_init=True)


def _make_visibility_state() -> VisibilityState:
    """Return a fresh VisibilityState instance bypassing the Reflex runtime guard."""
    return VisibilityState(_reflex_internal_init=True)


def _make_sweep(label: str = "test", color: str = "#000000") -> Sweep:
    """Return a minimal Sweep suitable for testing simulation state.

    Args:
        label: Human-readable label for the sweep.
        color: Hex color string.

    Returns:
        A Sweep with all-zero data for 100 time points over 50 ms.
    """
    n = 100
    t = np.linspace(0, 50, n)
    zeros = np.zeros(n)
    fields = [
        ("time", np.float64),
        ("voltage", np.float64),
        ("Itotal", np.float64),
        ("INa", np.float64),
        ("IK", np.float64),
        ("Ileak", np.float64),
        ("n", np.float64),
        ("m", np.float64),
        ("h", np.float64),
    ]
    result = np.empty(n, dtype=np.dtype(fields))
    result["time"] = t
    for name in (
        "voltage",
        "Itotal",
        "INa",
        "IK",
        "Ileak",
        "n",
        "m",
        "h",
    ):
        result[name] = zeros
    return Sweep.from_result(result, zeros, label, color, "Current Clamp")


def _make_get_state_fn(class_returns: dict):
    """Build an async get_state replacement for use with ``patch.object``.

    Returns a fresh ``MagicMock`` for any class not in ``class_returns``.

    Args:
        class_returns: Mapping from state class to the instance to return.

    Returns:
        An async method suitable for ``patch.object(StateClass, 'get_state', new=...)``.
    """

    async def _get_state(_self, cls):
        """Return the mapped instance or a fresh MagicMock."""
        return class_returns.get(cls, MagicMock())

    return _get_state


# ---------------------------------------------------------------------------
# ProtocolState._set_float
# ---------------------------------------------------------------------------


def test_set_float_accepts_plain_float() -> None:
    """_set_float stores a plain float value as-is."""
    ps = _make_protocol_state()
    ps._set_float("stimulus_duration", 99.5)
    assert ps.stimulus_duration == pytest.approx(99.5)


def test_set_float_accepts_string() -> None:
    """_set_float parses a string to float and stores it."""
    ps = _make_protocol_state()
    ps._set_float("stimulus_duration", "77.25")
    assert ps.stimulus_duration == pytest.approx(77.25)


def test_set_float_accepts_list_uses_first_element() -> None:
    """_set_float uses the first element when given a list (slider events)."""
    ps = _make_protocol_state()
    ps._set_float("stimulus_duration", [42.0, 50.0])
    assert ps.stimulus_duration == pytest.approx(42.0)


def test_set_float_ignores_unparseable_string() -> None:
    """_set_float silently ignores values that cannot be converted to float."""
    ps = _make_protocol_state()
    original = ps.stimulus_duration
    ps._set_float("stimulus_duration", "not_a_number")
    assert ps.stimulus_duration == pytest.approx(original)


def test_set_float_ignores_none() -> None:
    """_set_float silently ignores None without raising."""
    ps = _make_protocol_state()
    original = ps.stimulus_duration
    ps._set_float("stimulus_duration", None)
    assert ps.stimulus_duration == pytest.approx(original)


# ---------------------------------------------------------------------------
# Generated float setter
# ---------------------------------------------------------------------------


def test_generated_float_setter_stores_value() -> None:
    """set_stimulus_duration(50.0) stores 50.0 in self.stimulus_duration."""
    ps = _make_protocol_state()
    ps.set_stimulus_duration(50.0)
    assert ps.stimulus_duration == pytest.approx(50.0)


def test_generated_float_setter_accepts_string() -> None:
    """Generated float setter parses a string value via _set_float."""
    ps = _make_protocol_state()
    ps.set_stimulus_duration("123.4")
    assert ps.stimulus_duration == pytest.approx(123.4)


def test_generated_float_setter_accepts_list() -> None:
    """Generated float setter accepts a slider list and uses the first element."""
    ps = _make_protocol_state()
    ps.set_stimulus_duration([25.0, 50.0])
    assert ps.stimulus_duration == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Generated bool setter
# ---------------------------------------------------------------------------


async def test_generated_bool_setter_stores_false() -> None:
    """set_show_voltage(False) stores False in VisibilityState.show_voltage."""
    vs = _make_visibility_state()
    vs.show_voltage = True
    with patch.object(VisibilityState, "get_state", new=_make_get_state_fn({})):
        await vs.set_show_voltage(False)
    assert vs.show_voltage is False


async def test_generated_bool_setter_stores_true() -> None:
    """set_show_voltage(True) stores True in VisibilityState.show_voltage."""
    vs = _make_visibility_state()
    vs.show_voltage = False
    with patch.object(VisibilityState, "get_state", new=_make_get_state_fn({})):
        await vs.set_show_voltage(True)
    assert vs.show_voltage is True


# ---------------------------------------------------------------------------
# is_multi_sweep
# ---------------------------------------------------------------------------


def test_is_multi_sweep_false_when_no_sweeps() -> None:
    """is_multi_sweep is False when current_sweeps is empty."""
    s = _make_state()
    assert s.is_multi_sweep is False


def test_is_multi_sweep_false_for_single_sweep() -> None:
    """is_multi_sweep is False when there is exactly one current sweep."""
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    assert s.is_multi_sweep is False


def test_is_multi_sweep_true_for_multiple_sweeps() -> None:
    """is_multi_sweep is True when current_sweeps has more than one entry."""
    s = _make_state()
    s.current_sweeps = [_make_sweep(), _make_sweep()]
    assert s.is_multi_sweep is True


# ---------------------------------------------------------------------------
# load_protocol_preset
# ---------------------------------------------------------------------------


async def test_load_protocol_preset_action_potential_sets_stimulus_duration() -> None:
    """load_protocol_preset('Action Potential') sets stimulus_duration to 30.0."""
    ps = _make_protocol_state()
    with patch.object(ProtocolState, "get_state", new=_make_get_state_fn({})):
        await ps.load_protocol_preset("Action Potential")
    assert ps.stimulus_duration == pytest.approx(30.0)


async def test_load_protocol_preset_action_potential_sets_clamp_mode() -> None:
    """load_protocol_preset('Action Potential') sets clamp_mode to 'Current Clamp'."""
    ps = _make_protocol_state()
    with patch.object(ProtocolState, "get_state", new=_make_get_state_fn({})):
        await ps.load_protocol_preset("Action Potential")
    assert ps.clamp_mode == "Current Clamp"


async def test_load_protocol_preset_repetitive_firing_sets_stimulus_duration() -> None:
    """load_protocol_preset('Repetitive Firing') sets stimulus_duration to 180.0."""
    ps = _make_protocol_state()
    with patch.object(ProtocolState, "get_state", new=_make_get_state_fn({})):
        await ps.load_protocol_preset("Repetitive Firing")
    assert ps.stimulus_duration == pytest.approx(180.0)


async def test_load_protocol_preset_iv_curve_sets_voltage_clamp_mode() -> None:
    """load_protocol_preset('I-V Curve') sets clamp_mode to 'Voltage Clamp'."""
    ps = _make_protocol_state()
    with patch.object(ProtocolState, "get_state", new=_make_get_state_fn({})):
        await ps.load_protocol_preset("I-V Curve")
    assert ps.clamp_mode == "Voltage Clamp"


async def test_set_clamp_mode_clears_current_sweeps() -> None:
    """set_clamp_mode resets current_sweeps on the SimulationState."""
    ps = _make_protocol_state()
    sim_st = _make_state()
    sim_st.current_sweeps = [_make_sweep()]
    with patch.object(
        ProtocolState, "get_state", new=_make_get_state_fn({SimulationState: sim_st})
    ):
        await ps.set_clamp_mode("Voltage Clamp")
    assert len(sim_st.current_sweeps) == 0


async def test_set_clamp_mode_clears_stored_traces() -> None:
    """set_clamp_mode resets stored_traces on the SimulationState."""
    ps = _make_protocol_state()
    sim_st = _make_state()
    sim_st.stored_traces = [_make_sweep()]
    with patch.object(
        ProtocolState, "get_state", new=_make_get_state_fn({SimulationState: sim_st})
    ):
        await ps.set_clamp_mode("Voltage Clamp")
    assert len(sim_st.stored_traces) == 0


async def test_set_clamp_mode_resets_cont_has_state() -> None:
    """set_clamp_mode resets _cont_has_state on the SimulationState."""
    ps = _make_protocol_state()
    sim_st = _make_state()
    sim_st._cont_has_state = True
    with patch.object(
        ProtocolState, "get_state", new=_make_get_state_fn({SimulationState: sim_st})
    ):
        await ps.set_clamp_mode("Voltage Clamp")
    assert sim_st._cont_has_state is False


async def test_load_protocol_preset_unknown_name_is_ignored() -> None:
    """load_protocol_preset silently ignores an unknown preset name."""
    ps = _make_protocol_state()
    original = ps.stimulus_duration
    await ps.load_protocol_preset("NonExistentPreset")
    assert ps.stimulus_duration == pytest.approx(original)


# ---------------------------------------------------------------------------
# load_neuron_preset
# ---------------------------------------------------------------------------


async def test_load_neuron_preset_fast_spiking_interneuron() -> None:
    """load_neuron_preset enables IKv31 and leaves IKa disabled for Fast-Spiking."""
    ns = _make_neuron_state()
    with patch.object(NeuronState, "get_state", new=_make_get_state_fn({})):
        await ns.load_neuron_preset(FAST_SPIKING_INTERNEURON)
    assert ns.ikv31_enabled is True
    assert ns.ika_enabled is False


async def test_load_neuron_preset_sets_active_neuron_type() -> None:
    """load_neuron_preset records the selected neuron type on NeuronState."""
    ns = _make_neuron_state()
    with patch.object(NeuronState, "get_state", new=_make_get_state_fn({})):
        await ns.load_neuron_preset(CORTICAL_PYRAMIDAL)
    assert ns.active_neuron_type == CORTICAL_PYRAMIDAL


async def test_load_neuron_preset_resets_previously_enabled_channels() -> None:
    """Loading a second neuron preset disables channels from the first."""
    ns = _make_neuron_state()
    with patch.object(NeuronState, "get_state", new=_make_get_state_fn({})):
        await ns.load_neuron_preset(FAST_SPIKING_INTERNEURON)
        assert ns.ikv31_enabled is True
        await ns.load_neuron_preset(CORTICAL_PYRAMIDAL)
    assert ns.ikv31_enabled is False


async def test_load_neuron_preset_pyramidal_neuron() -> None:
    """load_neuron_preset enables Ih, INaP, and IM channels for Cortical Pyramidal."""
    ns = _make_neuron_state()
    with patch.object(NeuronState, "get_state", new=_make_get_state_fn({})):
        await ns.load_neuron_preset(CORTICAL_PYRAMIDAL)
    assert ns.ih_enabled is True
    assert ns.inap_enabled is True
    assert ns.im_enabled is True


async def test_load_neuron_preset_purkinje_cell() -> None:
    """load_neuron_preset enables ICaL, ICaT, and IKCa channels for Purkinje Cell."""
    ns = _make_neuron_state()
    with patch.object(NeuronState, "get_state", new=_make_get_state_fn({})):
        await ns.load_neuron_preset(PURKINJE)
    assert ns.ical_enabled is True
    assert ns.icat_enabled is True
    assert ns.ikca_enabled is True


async def test_load_neuron_preset_dopaminergic_neuron() -> None:
    """load_neuron_preset enables Ih and IM channels for Dopaminergic Neuron."""
    ns = _make_neuron_state()
    with patch.object(NeuronState, "get_state", new=_make_get_state_fn({})):
        await ns.load_neuron_preset(DOPAMINERGIC)
    assert ns.ih_enabled is True
    assert ns.im_enabled is True


async def test_load_neuron_preset_thalamic_relay() -> None:
    """load_neuron_preset enables ICaT and Ih channels for Thalamic Relay."""
    ns = _make_neuron_state()
    with patch.object(NeuronState, "get_state", new=_make_get_state_fn({})):
        await ns.load_neuron_preset(THALAMIC_RELAY)
    assert ns.icat_enabled is True
    assert ns.ih_enabled is True


async def test_load_neuron_preset_unknown_name_is_ignored() -> None:
    """load_neuron_preset silently ignores an unknown preset name."""
    ns = _make_neuron_state()
    before = ns.active_neuron_type
    await ns.load_neuron_preset("NonExistentNeuron")
    assert ns.active_neuron_type == before


async def test_load_neuron_preset_preserves_stored_traces() -> None:
    """load_neuron_preset retains stored traces so neuron types can be compared."""
    sim_st = _make_state()
    sim_st.stored_traces = [_make_sweep()]
    ns = _make_neuron_state()
    with patch.object(
        NeuronState,
        "get_state",
        new=_make_get_state_fn({SimulationState: sim_st}),
    ):
        await ns.load_neuron_preset(CORTICAL_PYRAMIDAL)
    assert len(sim_st.stored_traces) == 1


async def test_load_neuron_preset_keeps_sweeps_when_stored() -> None:
    """load_neuron_preset preserves current_sweeps when stored traces exist.

    When the user stores a trace and then changes neuron type, the previous
    simulation should remain visible in the figure alongside the stored trace.
    """
    sim_st = _make_state()
    sim_st.stored_traces = [_make_sweep()]
    sim_st.current_sweeps = [_make_sweep()]
    ns = _make_neuron_state()
    with patch.object(
        NeuronState,
        "get_state",
        new=_make_get_state_fn({SimulationState: sim_st}),
    ):
        await ns.load_neuron_preset(CORTICAL_PYRAMIDAL)
    assert len(sim_st.current_sweeps) == 1


async def test_load_neuron_preset_clears_sweeps_without_stored() -> None:
    """load_neuron_preset clears current_sweeps when no stored traces exist.

    Without stored traces the previous simulation is stale and should be
    removed so the figure does not show results from the wrong neuron type.
    """
    sim_st = _make_state()
    sim_st.stored_traces = []
    sim_st.current_sweeps = [_make_sweep()]
    ns = _make_neuron_state()
    with patch.object(
        NeuronState,
        "get_state",
        new=_make_get_state_fn({SimulationState: sim_st}),
    ):
        await ns.load_neuron_preset(CORTICAL_PYRAMIDAL)
    assert len(sim_st.current_sweeps) == 0


async def test_store_trace_label_includes_neuron_type() -> None:
    """store_trace includes the active neuron type in the stored trace label."""
    s = _make_state()
    s._label_neuron_type = CORTICAL_PYRAMIDAL
    s.current_sweeps = [_make_sweep()]
    with patch.object(SimulationState, "get_state", new=_make_get_state_fn({})):
        await s.store_trace()
    assert CORTICAL_PYRAMIDAL in s.stored_traces[0].label


# ---------------------------------------------------------------------------
# Neuron-type protocol adjustments
# ---------------------------------------------------------------------------


async def test_protocol_preset_with_active_neuron_type_applies_adjustment() -> None:
    """Repetitive Firing with Thalamic Relay applies a depolarizing adjustment."""
    ps = _make_protocol_state()
    mock_neuron = MagicMock()
    mock_neuron.active_neuron_type = THALAMIC_RELAY
    with patch.object(
        ProtocolState, "get_state", new=_make_get_state_fn({NeuronState: mock_neuron})
    ):
        await ps.load_protocol_preset("Repetitive Firing")
    assert ps.min_stimulus > 0.0


async def test_protocol_preset_without_active_neuron_type_uses_base_params() -> None:
    """Repetitive Firing with no active neuron type uses the base stimulus."""
    ps = _make_protocol_state()
    mock_neuron = MagicMock()
    mock_neuron.active_neuron_type = ""
    with patch.object(
        ProtocolState, "get_state", new=_make_get_state_fn({NeuronState: mock_neuron})
    ):
        await ps.load_protocol_preset("Repetitive Firing")
    assert ps.min_stimulus == pytest.approx(15.0)


async def test_protocol_preset_with_no_adjustment_entry_uses_base_params() -> None:
    """Action Potential with Squid Giant Axon active falls through to base duration."""
    ps = _make_protocol_state()
    mock_neuron = MagicMock()
    mock_neuron.active_neuron_type = SQUID_GIANT_AXON
    with patch.object(
        ProtocolState, "get_state", new=_make_get_state_fn({NeuronState: mock_neuron})
    ):
        await ps.load_protocol_preset("Action Potential")
    assert ps.stimulus_duration == pytest.approx(30.0)


async def test_protocol_preset_dopaminergic_repetitive_firing_long_duration() -> None:
    """Dopaminergic Neuron + Repetitive Firing sets stimulus_duration to 480 ms."""
    ps = _make_protocol_state()
    mock_neuron = MagicMock()
    mock_neuron.active_neuron_type = DOPAMINERGIC
    with patch.object(
        ProtocolState, "get_state", new=_make_get_state_fn({NeuronState: mock_neuron})
    ):
        await ps.load_protocol_preset("Repetitive Firing")
    assert ps.stimulus_duration == pytest.approx(480.0)


# ---------------------------------------------------------------------------
# store_trace / clear_stored_traces
# ---------------------------------------------------------------------------


async def test_store_trace_appends_to_stored_traces() -> None:
    """store_trace promotes the current sweep to stored_traces."""
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    assert len(s.stored_traces) == 0
    with patch.object(SimulationState, "get_state", new=_make_get_state_fn({})):
        await s.store_trace()
    assert len(s.stored_traces) == 1


async def test_store_trace_sets_stored_label() -> None:
    """store_trace labels the stored entry with index and neuron type."""
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    with patch.object(SimulationState, "get_state", new=_make_get_state_fn({})):
        await s.store_trace()
    assert s.stored_traces[0].label == f"Stored 1 ({s._label_neuron_type})"


async def test_store_trace_twice_increments_label() -> None:
    """Calling store_trace twice creates sequentially numbered labels."""
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    with patch.object(SimulationState, "get_state", new=_make_get_state_fn({})):
        await s.store_trace()
        await s.store_trace()
    assert len(s.stored_traces) == 2
    assert s.stored_traces[1].label == f"Stored 2 ({s._label_neuron_type})"


async def test_store_trace_does_nothing_when_no_result() -> None:
    """store_trace is a no-op when current_sweeps is empty."""
    s = _make_state()
    assert len(s.current_sweeps) == 0
    with patch.object(SimulationState, "get_state", new=_make_get_state_fn({})):
        await s.store_trace()
    assert len(s.stored_traces) == 0


async def test_clear_stored_traces_empties_list() -> None:
    """clear_stored_traces removes all entries from stored_traces."""
    s = _make_state()
    s.stored_traces = [_make_sweep(), _make_sweep()]
    assert len(s.stored_traces) == 2
    with patch.object(SimulationState, "get_state", new=_make_get_state_fn({})):
        await s.clear_stored_traces()
    assert len(s.stored_traces) == 0


def test_has_stored_traces_false_when_empty() -> None:
    """has_stored_traces is False when no traces have been stored."""
    s = _make_state()
    assert s.has_stored_traces is False


async def test_has_stored_traces_true_after_store() -> None:
    """has_stored_traces is True after store_trace is called."""
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    with patch.object(SimulationState, "get_state", new=_make_get_state_fn({})):
        await s.store_trace()
    assert s.has_stored_traces is True


async def test_set_clamp_mode_to_current_resets_protocol_type() -> None:
    """set_clamp_mode('Current Clamp') resets protocol_type to the first CC option."""
    ps = _make_protocol_state()
    ps.clamp_mode = "Voltage Clamp"
    ps.protocol_type = "Ramp"
    with patch.object(ProtocolState, "get_state", new=_make_get_state_fn({})):
        await ps.set_clamp_mode("Current Clamp")
    assert ps.protocol_type == constants.CURRENT_PROTOCOLS[0]


async def test_set_clamp_mode_to_voltage_resets_protocol_type() -> None:
    """set_clamp_mode('Voltage Clamp') resets protocol_type to the first VC option."""
    ps = _make_protocol_state()
    with patch.object(ProtocolState, "get_state", new=_make_get_state_fn({})):
        await ps.set_clamp_mode("Voltage Clamp")
    assert ps.protocol_type == constants.VOLTAGE_PROTOCOLS[0]


# ---------------------------------------------------------------------------
# protocol_options computed var
# ---------------------------------------------------------------------------


def test_protocol_options_current_clamp() -> None:
    """protocol_options returns the current clamp list when mode is Current Clamp."""
    ps = _make_protocol_state()
    ps.clamp_mode = "Current Clamp"
    assert ps.protocol_options == constants.CURRENT_PROTOCOLS


def test_protocol_options_voltage_clamp() -> None:
    """protocol_options returns the voltage clamp list when mode is Voltage Clamp."""
    ps = _make_protocol_state()
    ps.clamp_mode = "Voltage Clamp"
    assert ps.protocol_options == constants.VOLTAGE_PROTOCOLS


# ---------------------------------------------------------------------------
# can_run_continuous computed var
# ---------------------------------------------------------------------------


def test_can_run_continuous_true_for_step_single_sweep() -> None:
    """can_run_continuous is True for a single-step Step protocol (min == max)."""
    ps = _make_protocol_state()
    ps.protocol_type = "Step"
    ps.min_stimulus = 10.0
    ps.max_stimulus = 10.0
    ps.stimulus_step = 0.0
    assert ps.can_run_continuous is True


def test_can_run_continuous_false_for_step_multi_sweep() -> None:
    """can_run_continuous is False for a multi-sweep Step protocol."""
    ps = _make_protocol_state()
    ps.protocol_type = "Step"
    ps.min_stimulus = -10.0
    ps.max_stimulus = 20.0
    ps.stimulus_step = 2.5
    assert ps.can_run_continuous is False


def test_can_run_continuous_true_for_ramp() -> None:
    """can_run_continuous is True for the Ramp protocol."""
    ps = _make_protocol_state()
    ps.protocol_type = "Ramp"
    assert ps.can_run_continuous is True


# ---------------------------------------------------------------------------
# Stimulus range setters — constraint logic
# ---------------------------------------------------------------------------


def test_set_max_stimulus_auto_sets_step_when_range_opens() -> None:
    """set_max_stimulus auto-sets stimulus_step to 1.0 when min != max and step is 0."""
    ps = _make_protocol_state()
    ps.min_stimulus = 10.0
    ps.max_stimulus = 10.0
    ps.stimulus_step = 0.0
    ps.set_max_stimulus(20.0)
    assert ps.max_stimulus == 20.0
    assert ps.stimulus_step == 1.0


def test_set_min_stimulus_auto_sets_step_when_range_opens() -> None:
    """set_min_stimulus auto-sets stimulus_step to 1.0 when min != max and step is 0."""
    ps = _make_protocol_state()
    ps.min_stimulus = 10.0
    ps.max_stimulus = 10.0
    ps.stimulus_step = 0.0
    ps.set_min_stimulus(0.0)
    assert ps.min_stimulus == 0.0
    assert ps.stimulus_step == 1.0


def test_set_max_stimulus_does_not_change_step_when_already_nonzero() -> None:
    """set_max_stimulus leaves stimulus_step unchanged when it is already non-zero."""
    ps = _make_protocol_state()
    ps.min_stimulus = 0.0
    ps.max_stimulus = 20.0
    ps.stimulus_step = 5.0
    ps.set_max_stimulus(30.0)
    assert ps.stimulus_step == 5.0


def test_set_stimulus_step_zero_rejected_when_range_open() -> None:
    """set_stimulus_step resets to 1.0 when 0 is submitted in multi-sweep mode.

    Resetting to 1.0 (rather than keeping the previous value) guarantees a
    state change, which forces Reflex to emit a delta and snap the controlled
    input back to the validated value.
    """
    ps = _make_protocol_state()
    ps.min_stimulus = 0.0
    ps.max_stimulus = 20.0
    ps.stimulus_step = 5.0
    ps.set_stimulus_step(0.0)
    assert ps.stimulus_step == 1.0


def test_set_stimulus_step_negative_rejected_when_range_open() -> None:
    """set_stimulus_step resets to 1.0 for negative values in multi-sweep mode."""
    ps = _make_protocol_state()
    ps.min_stimulus = 0.0
    ps.max_stimulus = 20.0
    ps.stimulus_step = 5.0
    ps.set_stimulus_step(-1.0)
    assert ps.stimulus_step == 1.0


def test_set_stimulus_step_rejection_always_changes_state() -> None:
    """Rejected step values always mutate stimulus_step so Reflex emits a delta.

    Even when the previous step was already 1.0, a rejected value must still
    produce a state change so the frontend controlled input snaps back.
    """
    ps = _make_protocol_state()
    ps.min_stimulus = 0.0
    ps.max_stimulus = 20.0
    ps.stimulus_step = 1.0
    ps.set_stimulus_step(0.0)
    assert ps.stimulus_step == 1.0


def test_set_stimulus_step_zero_accepted_when_single_sweep() -> None:
    """set_stimulus_step accepts 0 when min_stimulus == max_stimulus."""
    ps = _make_protocol_state()
    ps.min_stimulus = 10.0
    ps.max_stimulus = 10.0
    ps.stimulus_step = 5.0
    ps.set_stimulus_step(0.0)
    assert ps.stimulus_step == 0.0


# ---------------------------------------------------------------------------
# continuous_active computed var
# ---------------------------------------------------------------------------


def test_continuous_active_false_by_default() -> None:
    """continuous_active is False when neither flag is set."""
    s = _make_state()
    assert s.continuous_active is False


def test_continuous_active_false_when_mode_only() -> None:
    """continuous_active is False when continuous_mode is True but loop not running."""
    s = _make_state()
    s.continuous_mode = True
    assert s.continuous_active is False


def test_continuous_active_false_when_loop_only() -> None:
    """continuous_active is False when loop is running but mode is False."""
    s = _make_state()
    s.continuous_loop_running = True
    assert s.continuous_active is False


def test_continuous_active_true_when_both_set() -> None:
    """continuous_active is True when both continuous_mode and the loop are active."""
    s = _make_state()
    s.continuous_mode = True
    s.continuous_loop_running = True
    assert s.continuous_active is True


# ---------------------------------------------------------------------------
# toggle_continuous_mode state transitions
# ---------------------------------------------------------------------------


def test_toggle_continuous_mode_sets_continuous_mode_true() -> None:
    """toggle_continuous_mode enables continuous_mode when the loop is not running."""
    s = _make_state()
    assert s.continuous_loop_running is False
    s.toggle_continuous_mode()
    assert s.continuous_mode is True


def test_toggle_continuous_mode_clears_continuous_mode_when_running() -> None:
    """toggle_continuous_mode disables continuous_mode when the loop is running."""
    s = _make_state()
    s.continuous_loop_running = True
    s.continuous_mode = True
    s.toggle_continuous_mode()
    assert s.continuous_mode is False


# ---------------------------------------------------------------------------
# filtered_log_entries computed var
# ---------------------------------------------------------------------------


def _make_log_record(level: str, message: str) -> UILogRecord:
    """Return a minimal UILogRecord for testing."""
    return UILogRecord(
        timestamp="2026-01-01T00:00:00Z",
        level=level,
        logger_name="test",
        message=message,
    )


def test_filtered_log_entries_returns_all_at_debug() -> None:
    """filtered_log_entries returns all records when filter is DEBUG."""
    s = _make_log_state()
    s.log_level_filter = "DEBUG"
    s.log_entries = [
        _make_log_record("DEBUG", "dbg"),
        _make_log_record("INFO", "info"),
        _make_log_record("WARNING", "warn"),
        _make_log_record("ERROR", "err"),
    ]
    assert len(s.filtered_log_entries) == 4


def test_filtered_log_entries_filters_below_info() -> None:
    """filtered_log_entries omits DEBUG entries when filter is INFO."""
    s = _make_log_state()
    s.log_level_filter = "INFO"
    s.log_entries = [
        _make_log_record("DEBUG", "dbg"),
        _make_log_record("INFO", "info"),
        _make_log_record("WARNING", "warn"),
    ]
    result = s.filtered_log_entries
    assert len(result) == 2
    assert all(e.level != "DEBUG" for e in result)


def test_filtered_log_entries_only_errors_at_error_level() -> None:
    """filtered_log_entries returns only ERROR records when filter is ERROR."""
    s = _make_log_state()
    s.log_level_filter = "ERROR"
    s.log_entries = [
        _make_log_record("DEBUG", "dbg"),
        _make_log_record("INFO", "info"),
        _make_log_record("WARNING", "warn"),
        _make_log_record("ERROR", "err"),
    ]
    result = s.filtered_log_entries
    assert len(result) == 1
    assert result[0].level == "ERROR"


def test_filtered_log_entries_newest_first() -> None:
    """filtered_log_entries returns entries in reverse order (newest first)."""
    s = _make_log_state()
    s.log_level_filter = "DEBUG"
    s.log_entries = [
        _make_log_record("INFO", "first"),
        _make_log_record("INFO", "second"),
        _make_log_record("INFO", "third"),
    ]
    result = s.filtered_log_entries
    assert result[0].message == "third"
    assert result[-1].message == "first"


def test_filtered_log_entries_empty_when_no_entries() -> None:
    """filtered_log_entries returns an empty list when log_entries is empty."""
    s = _make_log_state()
    assert s.filtered_log_entries == []


# ---------------------------------------------------------------------------
# toggle_hover
# ---------------------------------------------------------------------------


def test_show_hover_defaults_to_true() -> None:
    """show_hover is True on a freshly created SimulationState."""
    s = _make_state()
    assert s.show_hover is True


def test_toggle_hover_disables_when_on() -> None:
    """toggle_hover sets show_hover to False when it was True."""
    s = _make_state()
    assert s.show_hover is True
    s.toggle_hover()
    assert s.show_hover is False


def test_toggle_hover_enables_when_off() -> None:
    """toggle_hover sets show_hover to True when it was False."""
    s = _make_state()
    s.show_hover = False
    s.toggle_hover()
    assert s.show_hover is True


def test_toggle_hover_returns_call_script() -> None:
    """toggle_hover returns a non-None value (the rx.call_script event)."""
    s = _make_state()
    result = s.toggle_hover()
    assert result is not None


def test_toggle_hover_twice_restores_original_state() -> None:
    """Calling toggle_hover twice leaves show_hover unchanged."""
    s = _make_state()
    original = s.show_hover
    s.toggle_hover()
    s.toggle_hover()
    assert s.show_hover is original


# ---------------------------------------------------------------------------
# _build_protocols label generation
# ---------------------------------------------------------------------------


def test_build_protocols_single_sweep_current_clamp_has_empty_label() -> None:
    """Single-sweep current clamp protocol returns an empty sweep label."""
    ps = _make_protocol_state()
    ps.clamp_mode = "Current Clamp"
    ps.protocol_type = "Step"
    ps.min_stimulus = 10.0
    ps.max_stimulus = 10.0
    ps.stimulus_step = 0.0
    result = ps._build_protocols()
    assert len(result) == 1
    assert result[0][1] == ""


def test_build_protocols_multi_sweep_current_clamp_labels_contain_unit() -> None:
    """Multi-sweep current clamp Step protocol labels include µA/cm² unit."""
    ps = _make_protocol_state()
    ps.clamp_mode = "Current Clamp"
    ps.protocol_type = "Step"
    ps.min_stimulus = 0.0
    ps.max_stimulus = 10.0
    ps.stimulus_step = 5.0
    result = ps._build_protocols()
    # 0 to 10 in steps of 5 → [0, 5, 10] = 3 sweeps
    assert len(result) == 3
    for _arr, label in result:
        assert label != ""
        assert "µA/cm²" in label


def test_build_protocols_multi_sweep_current_clamp_label_values() -> None:
    """Multi-sweep current clamp labels match the actual stimulus values."""
    ps = _make_protocol_state()
    ps.clamp_mode = "Current Clamp"
    ps.protocol_type = "Step"
    ps.min_stimulus = 0.0
    ps.max_stimulus = 10.0
    ps.stimulus_step = 5.0
    result = ps._build_protocols()
    labels = [label for _arr, label in result]
    assert labels == ["+0.0 µA/cm²", "+5.0 µA/cm²", "+10.0 µA/cm²"]


def test_build_protocols_single_sweep_voltage_clamp_has_empty_label() -> None:
    """Single-sweep voltage clamp protocol returns an empty sweep label."""
    ps = _make_protocol_state()
    ps.clamp_mode = "Voltage Clamp"
    ps.protocol_type = "Step"
    ps.min_stimulus = -70.0
    ps.max_stimulus = -70.0
    ps.stimulus_step = 0.0
    result = ps._build_protocols()
    assert len(result) == 1
    assert result[0][1] == ""


def test_build_protocols_multi_sweep_voltage_clamp_labels_contain_unit() -> None:
    """Multi-sweep voltage clamp Step protocol labels include mV unit."""
    ps = _make_protocol_state()
    ps.clamp_mode = "Voltage Clamp"
    ps.protocol_type = "Step"
    ps.min_stimulus = -40.0
    ps.max_stimulus = 40.0
    ps.stimulus_step = 20.0
    result = ps._build_protocols()
    # -40 to +40 in steps of 20 → 5 sweeps
    assert len(result) == 5
    for _arr, label in result:
        assert label != ""
        assert "mV" in label


def test_build_protocols_multi_sweep_voltage_clamp_label_values() -> None:
    """Multi-sweep voltage clamp labels match the actual voltage step values."""
    ps = _make_protocol_state()
    ps.clamp_mode = "Voltage Clamp"
    ps.protocol_type = "Step"
    ps.min_stimulus = -40.0
    ps.max_stimulus = 40.0
    ps.stimulus_step = 40.0
    result = ps._build_protocols()
    # -40 to +40 in steps of 40 → [-40, 0, +40] = 3 sweeps
    labels = [label for _arr, label in result]
    assert labels == ["-40 mV", "+0 mV", "+40 mV"]
