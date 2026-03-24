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
    s._set_float("duration", 99.5)
    assert s.duration == pytest.approx(99.5)


def test_set_float_accepts_string() -> None:
    """_set_float parses a string to float and stores it."""
    s = _make_state()
    s._set_float("duration", "77.25")
    assert s.duration == pytest.approx(77.25)


def test_set_float_accepts_list_uses_first_element() -> None:
    """_set_float uses the first element when given a list (slider events)."""
    s = _make_state()
    s._set_float("duration", [42.0, 50.0])
    assert s.duration == pytest.approx(42.0)


def test_set_float_ignores_unparseable_string() -> None:
    """_set_float silently ignores values that cannot be converted to float."""
    s = _make_state()
    original = s.duration
    s._set_float("duration", "not_a_number")
    assert s.duration == pytest.approx(original)


def test_set_float_ignores_none() -> None:
    """_set_float silently ignores None without raising."""
    s = _make_state()
    original = s.duration
    s._set_float("duration", None)  # type: ignore[arg-type]
    assert s.duration == pytest.approx(original)


# ---------------------------------------------------------------------------
# Generated float setter
# ---------------------------------------------------------------------------


def test_generated_float_setter_stores_value() -> None:
    """set_duration(50.0) stores 50.0 in self.duration."""
    s = _make_state()
    s.set_duration(50.0)
    assert s.duration == pytest.approx(50.0)


def test_generated_float_setter_accepts_string() -> None:
    """Generated float setter parses a string value via _set_float."""
    s = _make_state()
    s.set_duration("123.4")
    assert s.duration == pytest.approx(123.4)


def test_generated_float_setter_accepts_list() -> None:
    """Generated float setter accepts a slider list and uses the first element."""
    s = _make_state()
    s.set_duration([25.0, 50.0])
    assert s.duration == pytest.approx(25.0)


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
# load_preset
# ---------------------------------------------------------------------------


def test_load_preset_action_potential_sets_duration() -> None:
    """Loading 'Action Potential' preset sets duration to 50.0."""
    s = _make_state()
    s.load_preset("Action Potential")
    assert s.duration == pytest.approx(50.0)


def test_load_preset_action_potential_sets_clamp_mode() -> None:
    """Loading 'Action Potential' preset sets clamp_mode to 'Current Clamp'."""
    s = _make_state()
    s.load_preset("Action Potential")
    assert s.clamp_mode == "Current Clamp"


def test_load_preset_repetitive_firing_sets_duration() -> None:
    """Loading 'Repetitive Firing' preset sets duration to 200.0."""
    s = _make_state()
    s.load_preset("Repetitive Firing")
    assert s.duration == pytest.approx(200.0)


def test_load_preset_iv_curve_sets_voltage_clamp_mode() -> None:
    """Loading 'I-V Curve' preset sets clamp_mode to 'Voltage Clamp'."""
    s = _make_state()
    s.load_preset("I-V Curve")
    assert s.clamp_mode == "Voltage Clamp"


def test_load_preset_clears_current_sweeps() -> None:
    """load_preset resets current_sweeps to an empty list."""
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    s.load_preset("Action Potential")
    assert len(s.current_sweeps) == 0


def test_load_preset_clears_saved_sweeps() -> None:
    """load_preset resets saved_sweeps to an empty list."""
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    s.add_sweep()
    s.load_preset("Action Potential")
    assert len(s.saved_sweeps) == 0


def test_load_preset_clears_stored_traces() -> None:
    """load_preset resets stored_traces to an empty list."""
    s = _make_state()
    s.current_sweeps = [_make_sweep()]
    s.store_trace()
    assert len(s.stored_traces) == 1
    s.load_preset("Action Potential")
    assert len(s.stored_traces) == 0


def test_load_preset_resets_cont_has_state() -> None:
    """load_preset resets _cont_has_state so the next continuous iter starts fresh."""
    s = _make_state()
    s._cont_has_state = True
    s.load_preset("Action Potential")
    assert s._cont_has_state is False


def test_load_preset_unknown_name_is_ignored() -> None:
    """load_preset silently ignores an unknown preset name."""
    s = _make_state()
    original_duration = s.duration
    s.load_preset("NonExistentPreset")
    assert s.duration == pytest.approx(original_duration)


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
    s.protocol_type = "I-V Curve"
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


def test_can_run_continuous_true_for_step() -> None:
    """can_run_continuous is True for the Step protocol."""
    s = _make_state()
    s.protocol_type = "Step"
    assert s.can_run_continuous is True


def test_can_run_continuous_true_for_ramp() -> None:
    """can_run_continuous is True for the Ramp protocol."""
    s = _make_state()
    s.protocol_type = "Ramp"
    assert s.can_run_continuous is True


def test_can_run_continuous_false_for_iv_curve() -> None:
    """can_run_continuous is False for the I-V Curve protocol."""
    s = _make_state()
    s.protocol_type = "I-V Curve"
    assert s.can_run_continuous is False


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
