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


def test_set_clamp_mode_resets_cont_has_state() -> None:
    """set_clamp_mode resets _cont_has_state so the next continuous iter starts fresh.

    This ensures that switching clamp modes during continuous simulation does
    not carry over stale gating state from the previous mode.
    """
    s = _make_state()
    s._cont_has_state = True
    s.set_clamp_mode("Voltage Clamp")
    assert s._cont_has_state is False
