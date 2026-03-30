"""Unit tests for pure logic in patch_sim_ui/state.py.

These tests exercise AppState methods that contain no Reflex event-loop
machinery.  Instantiation uses ``_reflex_internal_init=True`` (the same
flag the framework passes internally) after setting ``PYTEST_CURRENT_TEST``
so that Reflex's ``is_testing_env()`` guard also passes.

The environment variable is set at import time so the AppState metaclass
registration does not see a non-testing environment.
"""

import os

import numpy as np
import pandas as pd
import pytest

# Must be set before importing Reflex/AppState so the metaclass and init
# guard both see a testing environment.
os.environ.setdefault("PYTEST_CURRENT_TEST", "test_state.py::setup")

pytest.importorskip("reflex")

from patch_sim_ui import constants  # noqa: E402
from patch_sim_ui.log_handler import UILogRecord  # noqa: E402
from patch_sim_ui.plotting import Sweep  # noqa: E402
from patch_sim_ui.state import AppState  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state() -> AppState:
    """Return a fresh AppState instance bypassing the Reflex runtime guard."""
    return AppState(_reflex_internal_init=True)


def _make_sweep(label: str = "test", color: str = "#000000") -> Sweep:
    """Return a minimal Sweep suitable for testing add_sweep.

    Args:
        label: Human-readable label for the sweep.
        color: Hex color string.

    Returns:
        A Sweep with all-zero data for 100 time points over 50 ms.
    """
    n = 100
    t = np.linspace(0, 50, n)
    zeros = np.zeros(n)
    df = pd.DataFrame(
        {
            "time": t,
            "voltage": zeros,
            "total_current": zeros,
            "Na_current": zeros,
            "K_current": zeros,
            "leak_current": zeros,
            "potassium_activation": zeros,
            "sodium_activation": zeros,
            "sodium_inactivation": zeros,
        }
    )
    return Sweep.from_dataframe(df, zeros, label, color, "Current Clamp")


# ---------------------------------------------------------------------------
# AppState._set_float
# ---------------------------------------------------------------------------


def test_set_float_accepts_plain_float() -> None:
    """_set_float stores a plain float value as-is."""
    s = _make_state()
    s._set_float("stimulus_duration", 99.5)
    assert s.stimulus_duration == pytest.approx(99.5)


def test_set_float_accepts_string() -> None:
    """_set_float parses a string to float and stores it."""
    s = _make_state()
    s._set_float("stimulus_duration", "77.25")
    assert s.stimulus_duration == pytest.approx(77.25)


def test_set_float_accepts_list_uses_first_element() -> None:
    """_set_float uses the first element when given a list (slider events)."""
    s = _make_state()
    s._set_float("stimulus_duration", [42.0, 50.0])
    assert s.stimulus_duration == pytest.approx(42.0)


def test_set_float_ignores_unparseable_string() -> None:
    """_set_float silently ignores values that cannot be converted to float."""
    s = _make_state()
    original = s.stimulus_duration
    s._set_float("stimulus_duration", "not_a_number")
    assert s.stimulus_duration == pytest.approx(original)


def test_set_float_ignores_none() -> None:
    """_set_float silently ignores None without raising."""
    s = _make_state()
    original = s.stimulus_duration
    s._set_float("stimulus_duration", None)  # type: ignore[arg-type]
    assert s.stimulus_duration == pytest.approx(original)


# ---------------------------------------------------------------------------
# Generated float setter
# ---------------------------------------------------------------------------


def test_generated_float_setter_stores_value() -> None:
    """set_stimulus_duration(50.0) stores 50.0 in self.stimulus_duration."""
    s = _make_state()
    s.set_stimulus_duration(50.0)
    assert s.stimulus_duration == pytest.approx(50.0)


def test_generated_float_setter_accepts_string() -> None:
    """Generated float setter parses a string value via _set_float."""
    s = _make_state()
    s.set_stimulus_duration("123.4")
    assert s.stimulus_duration == pytest.approx(123.4)


def test_generated_float_setter_accepts_list() -> None:
    """Generated float setter accepts a slider list and uses the first element."""
    s = _make_state()
    s.set_stimulus_duration([25.0, 50.0])
    assert s.stimulus_duration == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Generated bool setter
# ---------------------------------------------------------------------------


def test_generated_bool_setter_stores_false() -> None:
    """set_show_voltage(False) stores False in self.show_voltage."""
    s = _make_state()
    s.set_show_voltage(False)
    assert s.show_voltage is False


def test_generated_bool_setter_stores_true() -> None:
    """set_show_voltage(True) stores True in self.show_voltage."""
    s = _make_state()
    s.show_voltage = False
    s.set_show_voltage(True)
    assert s.show_voltage is True


# ---------------------------------------------------------------------------
# add_sweep / clear_sweeps
# ---------------------------------------------------------------------------


def test_add_sweep_appends_to_saved_sweeps() -> None:
    """add_sweep promotes the current result to saved_sweeps."""
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    assert len(s.saved_sweeps) == 0
    s.add_sweep()
    assert len(s.saved_sweeps) == 1


def test_add_sweep_twice_appends_two_entries() -> None:
    """Calling add_sweep twice creates two saved sweeps."""
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    s.add_sweep()
    s.add_sweep()
    assert len(s.saved_sweeps) == 2


def test_add_sweep_does_nothing_when_no_result() -> None:
    """add_sweep is a no-op when current_sweeps is empty."""
    s = _make_state()
    assert len(s.current_sweeps) == 0
    s.add_sweep()
    assert len(s.saved_sweeps) == 0


def test_clear_sweeps_empties_saved_sweeps() -> None:
    """clear_sweeps removes all entries from saved_sweeps."""
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    s.add_sweep()
    s.add_sweep()
    assert len(s.saved_sweeps) == 2
    s.clear_sweeps()
    assert len(s.saved_sweeps) == 0


def test_clear_sweeps_on_empty_list_does_not_raise() -> None:
    """clear_sweeps on an already-empty list raises no exception."""
    s = _make_state()
    s.clear_sweeps()  # must not raise
    assert len(s.saved_sweeps) == 0


# ---------------------------------------------------------------------------
# load_protocol_preset
# ---------------------------------------------------------------------------


def test_load_protocol_preset_action_potential_sets_stimulus_duration() -> None:
    """Loading 'Action Potential' preset sets stimulus_duration to 30.0."""
    s = _make_state()
    s.load_protocol_preset("Action Potential")
    assert s.stimulus_duration == pytest.approx(30.0)


def test_load_protocol_preset_action_potential_sets_clamp_mode() -> None:
    """Loading 'Action Potential' preset sets clamp_mode to 'Current Clamp'."""
    s = _make_state()
    s.load_protocol_preset("Action Potential")
    assert s.clamp_mode == "Current Clamp"


def test_load_protocol_preset_repetitive_firing_sets_stimulus_duration() -> None:
    """Loading 'Repetitive Firing' preset sets stimulus_duration to 180.0."""
    s = _make_state()
    s.load_protocol_preset("Repetitive Firing")
    assert s.stimulus_duration == pytest.approx(180.0)


def test_load_protocol_preset_iv_curve_sets_voltage_clamp_mode() -> None:
    """Loading 'I-V Curve' preset sets clamp_mode to 'Voltage Clamp'."""
    s = _make_state()
    s.load_protocol_preset("I-V Curve")
    assert s.clamp_mode == "Voltage Clamp"


def test_load_protocol_preset_clears_current_sweeps() -> None:
    """load_protocol_preset resets current_sweeps to an empty list."""
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    s.load_protocol_preset("Action Potential")
    assert len(s.current_sweeps) == 0


def test_load_protocol_preset_clears_saved_sweeps() -> None:
    """load_protocol_preset resets saved_sweeps to an empty list."""
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    s.add_sweep()
    s.load_protocol_preset("Action Potential")
    assert len(s.saved_sweeps) == 0


def test_load_protocol_preset_clears_stored_traces() -> None:
    """load_protocol_preset resets stored_traces to an empty list."""
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    s.store_trace()
    assert len(s.stored_traces) == 1
    s.load_protocol_preset("Action Potential")
    assert len(s.stored_traces) == 0


def test_load_protocol_preset_resets_cont_has_state() -> None:
    """load_protocol_preset resets _cont_has_state before the next continuous run."""
    s = _make_state()
    s._cont_has_state = True
    s.load_protocol_preset("Action Potential")
    assert s._cont_has_state is False


def test_load_protocol_preset_unknown_name_is_ignored() -> None:
    """load_protocol_preset silently ignores an unknown preset name."""
    s = _make_state()
    original = s.stimulus_duration
    s.load_protocol_preset("NonExistentPreset")
    assert s.stimulus_duration == pytest.approx(original)


# ---------------------------------------------------------------------------
# load_neuron_preset
# ---------------------------------------------------------------------------


def test_load_neuron_preset_fast_spiking_interneuron_enables_ika() -> None:
    """Fast-Spiking Interneuron preset enables the IKa channel."""
    s = _make_state()
    s.load_neuron_preset("Fast-Spiking Interneuron")
    assert s.ika_enabled is True


def test_load_neuron_preset_does_not_change_protocol_fields() -> None:
    """load_neuron_preset leaves stimulus_duration and clamp_mode unchanged."""
    s = _make_state()
    original_duration = s.stimulus_duration
    original_clamp = s.clamp_mode
    s.load_neuron_preset("Fast-Spiking Interneuron")
    assert s.stimulus_duration == pytest.approx(original_duration)
    assert s.clamp_mode == original_clamp


def test_load_neuron_preset_sets_active_neuron_type() -> None:
    """load_neuron_preset records the selected neuron type on the state."""
    s = _make_state()
    s.load_neuron_preset("Pyramidal Neuron")
    assert s.active_neuron_type == "Pyramidal Neuron"


def test_load_neuron_preset_resets_previously_enabled_channels() -> None:
    """Loading a second neuron preset disables channels from the first."""
    s = _make_state()
    s.load_neuron_preset("Fast-Spiking Interneuron")
    assert s.ika_enabled is True
    s.load_neuron_preset("Pyramidal Neuron")
    assert s.ika_enabled is False


def test_load_neuron_preset_pyramidal_neuron_enables_ih() -> None:
    """Pyramidal Neuron preset enables the Ih channel."""
    s = _make_state()
    s.load_neuron_preset("Pyramidal Neuron")
    assert s.ih_enabled is True


def test_load_neuron_preset_pyramidal_neuron_enables_inap() -> None:
    """Pyramidal Neuron preset enables the INaP channel."""
    s = _make_state()
    s.load_neuron_preset("Pyramidal Neuron")
    assert s.inap_enabled is True


def test_load_neuron_preset_purkinje_cell_enables_ical() -> None:
    """Purkinje Cell preset enables the ICaL channel."""
    s = _make_state()
    s.load_neuron_preset("Purkinje Cell")
    assert s.ical_enabled is True


def test_load_neuron_preset_purkinje_cell_enables_ikca() -> None:
    """Purkinje Cell preset enables the IKCa channel."""
    s = _make_state()
    s.load_neuron_preset("Purkinje Cell")
    assert s.ikca_enabled is True


def test_load_neuron_preset_dopaminergic_neuron_enables_ih() -> None:
    """Dopaminergic Neuron preset enables the Ih channel."""
    s = _make_state()
    s.load_neuron_preset("Dopaminergic Neuron")
    assert s.ih_enabled is True


def test_load_neuron_preset_dopaminergic_neuron_enables_im() -> None:
    """Dopaminergic Neuron preset enables the IM channel."""
    s = _make_state()
    s.load_neuron_preset("Dopaminergic Neuron")
    assert s.im_enabled is True


def test_load_neuron_preset_thalamic_relay_enables_icat() -> None:
    """Thalamic Relay preset enables the ICaT channel."""
    s = _make_state()
    s.load_neuron_preset("Thalamic Relay")
    assert s.icat_enabled is True


def test_load_neuron_preset_thalamic_relay_enables_ih() -> None:
    """Thalamic Relay preset enables the Ih channel."""
    s = _make_state()
    s.load_neuron_preset("Thalamic Relay")
    assert s.ih_enabled is True


def test_load_neuron_preset_clears_current_sweeps() -> None:
    """load_neuron_preset resets current_sweeps to an empty list."""
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    s.load_neuron_preset("Pyramidal Neuron")
    assert len(s.current_sweeps) == 0


def test_load_neuron_preset_unknown_name_is_ignored() -> None:
    """load_neuron_preset silently ignores an unknown preset name."""
    s = _make_state()
    s.load_neuron_preset("NonExistentNeuron")
    assert s.active_neuron_type == ""


# ---------------------------------------------------------------------------
# Neuron-type protocol adjustments
# ---------------------------------------------------------------------------


def test_protocol_preset_with_active_neuron_type_applies_adjustment() -> None:
    """Repetitive Firing with Thalamic Relay active uses hyperpolarizing current."""
    s = _make_state()
    s.load_neuron_preset("Thalamic Relay")
    s.load_protocol_preset("Repetitive Firing")
    assert s.min_stimulus < 0.0


def test_protocol_preset_without_active_neuron_type_uses_base_params() -> None:
    """Repetitive Firing without an active neuron type uses the base stimulus."""
    s = _make_state()
    s.load_protocol_preset("Repetitive Firing")
    assert s.min_stimulus == pytest.approx(15.0)


def test_protocol_preset_with_no_adjustment_entry_uses_base_params() -> None:
    """Action Potential with Thalamic Relay active falls through to base duration."""
    s = _make_state()
    s.load_neuron_preset("Thalamic Relay")
    s.load_protocol_preset("Action Potential")
    assert s.stimulus_duration == pytest.approx(30.0)


def test_protocol_preset_dopaminergic_repetitive_firing_uses_long_duration() -> None:
    """Dopaminergic Neuron + Repetitive Firing sets stimulus_duration to 480 ms."""
    s = _make_state()
    s.load_neuron_preset("Dopaminergic Neuron")
    s.load_protocol_preset("Repetitive Firing")
    assert s.stimulus_duration == pytest.approx(480.0)


# ---------------------------------------------------------------------------
# store_trace / clear_stored_traces
# ---------------------------------------------------------------------------


def test_store_trace_appends_to_stored_traces() -> None:
    """store_trace promotes the current result to stored_traces."""
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    assert len(s.stored_traces) == 0
    s.store_trace()
    assert len(s.stored_traces) == 1


def test_store_trace_sets_stored_label() -> None:
    """store_trace labels the stored entry 'Stored 1'."""
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    s.store_trace()
    assert s.stored_traces[0].label == "Stored 1"


def test_store_trace_twice_increments_label() -> None:
    """Calling store_trace twice creates 'Stored 1' and 'Stored 2'."""
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    s.store_trace()
    s.store_trace()
    assert len(s.stored_traces) == 2
    assert s.stored_traces[1].label == "Stored 2"


def test_store_trace_does_nothing_when_no_result() -> None:
    """store_trace is a no-op when current_sweeps is empty."""
    s = _make_state()
    assert len(s.current_sweeps) == 0
    s.store_trace()
    assert len(s.stored_traces) == 0


def test_clear_stored_traces_empties_list() -> None:
    """clear_stored_traces removes all entries from stored_traces."""
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    s.store_trace()
    s.store_trace()
    assert len(s.stored_traces) == 2
    s.clear_stored_traces()
    assert len(s.stored_traces) == 0


def test_has_stored_traces_false_when_empty() -> None:
    """has_stored_traces is False when no traces have been stored."""
    s = _make_state()
    assert s.has_stored_traces is False


def test_has_stored_traces_true_after_store() -> None:
    """has_stored_traces is True after store_trace is called."""
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    s.store_trace()
    assert s.has_stored_traces is True


def test_set_clamp_mode_clears_stored_traces() -> None:
    """set_clamp_mode clears stored_traces when switching modes.

    Stored traces from the old mode must not persist — they would have
    incompatible axes alongside new-mode data.
    """
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    s.store_trace()
    assert len(s.stored_traces) == 1
    s.set_clamp_mode("Voltage Clamp")
    assert len(s.stored_traces) == 0


def test_set_clamp_mode_resets_cont_has_state() -> None:
    """set_clamp_mode resets _cont_has_state so the next continuous iter starts fresh.

    This ensures that switching clamp modes during continuous simulation does
    not carry over stale gating state from the previous mode.
    """
    s = _make_state()
    s._cont_has_state = True
    s.set_clamp_mode("Voltage Clamp")
    assert s._cont_has_state is False


def test_set_clamp_mode_to_current_resets_protocol_type() -> None:
    """set_clamp_mode('Current Clamp') resets protocol_type to the first CC option."""
    s = _make_state()
    s.clamp_mode = "Voltage Clamp"
    s.protocol_type = "Ramp"
    s.set_clamp_mode("Current Clamp")
    assert s.protocol_type == constants.CURRENT_PROTOCOLS[0]


def test_set_clamp_mode_to_voltage_resets_protocol_type() -> None:
    """set_clamp_mode('Voltage Clamp') resets protocol_type to the first VC option."""
    s = _make_state()
    s.set_clamp_mode("Voltage Clamp")
    assert s.protocol_type == constants.VOLTAGE_PROTOCOLS[0]


# ---------------------------------------------------------------------------
# protocol_options computed var
# ---------------------------------------------------------------------------


def test_protocol_options_current_clamp() -> None:
    """protocol_options returns the current clamp list when mode is Current Clamp."""
    s = _make_state()
    s.clamp_mode = "Current Clamp"
    assert s.protocol_options == constants.CURRENT_PROTOCOLS


def test_protocol_options_voltage_clamp() -> None:
    """protocol_options returns the voltage clamp list when mode is Voltage Clamp."""
    s = _make_state()
    s.clamp_mode = "Voltage Clamp"
    assert s.protocol_options == constants.VOLTAGE_PROTOCOLS


# ---------------------------------------------------------------------------
# can_run_continuous computed var
# ---------------------------------------------------------------------------


def test_can_run_continuous_true_for_step_single_sweep() -> None:
    """can_run_continuous is True for a single-step Step protocol (min == max)."""
    s = _make_state()
    s.protocol_type = "Step"
    s.min_stimulus = 10.0
    s.max_stimulus = 10.0
    s.stimulus_step = 0.0
    assert s.can_run_continuous is True


def test_can_run_continuous_false_for_step_multi_sweep() -> None:
    """can_run_continuous is False for a multi-sweep Step protocol."""
    s = _make_state()
    s.protocol_type = "Step"
    s.min_stimulus = -10.0
    s.max_stimulus = 20.0
    s.stimulus_step = 2.5
    assert s.can_run_continuous is False


def test_can_run_continuous_true_for_ramp() -> None:
    """can_run_continuous is True for the Ramp protocol."""
    s = _make_state()
    s.protocol_type = "Ramp"
    assert s.can_run_continuous is True


# ---------------------------------------------------------------------------
# Stimulus range setters — constraint logic
# ---------------------------------------------------------------------------


def test_set_max_stimulus_auto_sets_step_when_range_opens() -> None:
    """set_max_stimulus auto-sets stimulus_step to 1.0 when min != max and step is 0."""
    s = _make_state()
    s.min_stimulus = 10.0
    s.max_stimulus = 10.0
    s.stimulus_step = 0.0
    s.set_max_stimulus(20.0)
    assert s.max_stimulus == 20.0
    assert s.stimulus_step == 1.0


def test_set_min_stimulus_auto_sets_step_when_range_opens() -> None:
    """set_min_stimulus auto-sets stimulus_step to 1.0 when min != max and step is 0."""
    s = _make_state()
    s.min_stimulus = 10.0
    s.max_stimulus = 10.0
    s.stimulus_step = 0.0
    s.set_min_stimulus(0.0)
    assert s.min_stimulus == 0.0
    assert s.stimulus_step == 1.0


def test_set_max_stimulus_does_not_change_step_when_already_nonzero() -> None:
    """set_max_stimulus leaves stimulus_step unchanged when it is already non-zero."""
    s = _make_state()
    s.min_stimulus = 0.0
    s.max_stimulus = 20.0
    s.stimulus_step = 5.0
    s.set_max_stimulus(30.0)
    assert s.stimulus_step == 5.0


def test_set_stimulus_step_zero_rejected_when_range_open() -> None:
    """set_stimulus_step resets to 1.0 when 0 is submitted in multi-sweep mode.

    Resetting to 1.0 (rather than keeping the previous value) guarantees a
    state change, which forces Reflex to emit a delta and snap the controlled
    input back to the validated value.
    """
    s = _make_state()
    s.min_stimulus = 0.0
    s.max_stimulus = 20.0
    s.stimulus_step = 5.0
    s.set_stimulus_step(0.0)
    assert s.stimulus_step == 1.0


def test_set_stimulus_step_negative_rejected_when_range_open() -> None:
    """set_stimulus_step resets to 1.0 for negative values in multi-sweep mode."""
    s = _make_state()
    s.min_stimulus = 0.0
    s.max_stimulus = 20.0
    s.stimulus_step = 5.0
    s.set_stimulus_step(-1.0)
    assert s.stimulus_step == 1.0


def test_set_stimulus_step_rejection_always_changes_state() -> None:
    """Rejected step values always mutate stimulus_step so Reflex emits a delta.

    Even when the previous step was already 1.0, a rejected value must still
    produce a state change so the frontend controlled input snaps back.
    """
    s = _make_state()
    s.min_stimulus = 0.0
    s.max_stimulus = 20.0
    s.stimulus_step = 1.0
    s.set_stimulus_step(0.0)
    assert s.stimulus_step == 1.0


def test_set_stimulus_step_zero_accepted_when_single_sweep() -> None:
    """set_stimulus_step accepts 0 when min_stimulus == max_stimulus."""
    s = _make_state()
    s.min_stimulus = 10.0
    s.max_stimulus = 10.0
    s.stimulus_step = 5.0
    s.set_stimulus_step(0.0)
    assert s.stimulus_step == 0.0


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
    s = _make_state()
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
    s = _make_state()
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
    s = _make_state()
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
    s = _make_state()
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
    s = _make_state()
    assert s.filtered_log_entries == []


# ---------------------------------------------------------------------------
# toggle_hover
# ---------------------------------------------------------------------------


def test_show_hover_defaults_to_true() -> None:
    """show_hover is True on a freshly created AppState."""
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
