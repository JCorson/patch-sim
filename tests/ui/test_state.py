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

from patch_sim import NEURON_PRESETS  # noqa: E402
from patch_sim.constants import (
    ACTION_POTENTIAL,
    CA1_PYRAMIDAL,
    CORTICAL_PYRAMIDAL,
    DOPAMINERGIC,
    FAST_SPIKING_INTERNEURON,
    HYPERPOLARIZATION_STEPS,
    PURKINJE,
    REPETITIVE_FIRING,
    SQUID_GIANT_AXON,
    STN,
    THALAMIC_RELAY,
    TRN,
)
from patch_sim_ui import constants  # noqa: E402
from patch_sim_ui.log_handler import UILogRecord  # noqa: E402
from patch_sim_ui.presets import (  # noqa: E402
    DEFAULT_NEURON_PRESET,
    DEFAULT_PROTOCOL_PRESET,
    NEURON_CONFIG_SCALAR_DEFAULTS,
    NEURON_CONFIG_SCALAR_FIELDS,
    neuron_config_to_ui_state,
)
from patch_sim_ui.state import SimulationState  # noqa: E402
from patch_sim_ui.state.analysis import AnalysisState  # noqa: E402
from patch_sim_ui.state.log import LogState  # noqa: E402
from patch_sim_ui.state.neuron import NeuronState  # noqa: E402
from patch_sim_ui.state.protocol import ProtocolState  # noqa: E402
from patch_sim_ui.state.visibility import VisibilityState  # noqa: E402
from patch_sim_ui.sweep import Sweep  # noqa: E402

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


def _make_analysis_state() -> AnalysisState:
    """Return a fresh AnalysisState instance bypassing the Reflex runtime guard."""
    return AnalysisState(_reflex_internal_init=True)


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
        ("INaL", np.float64),
        ("IKL", np.float64),
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
        "INaL",
        "IKL",
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
    s._current_sweeps = [_make_sweep()]
    assert s.is_multi_sweep is False


def test_is_multi_sweep_true_for_multiple_sweeps() -> None:
    """is_multi_sweep is True when current_sweeps has more than one entry."""
    s = _make_state()
    s._current_sweeps = [_make_sweep(), _make_sweep()]
    assert s.is_multi_sweep is True


# ---------------------------------------------------------------------------
# load_protocol_preset
# ---------------------------------------------------------------------------


async def test_load_protocol_preset_action_potential_sets_stimulus_duration() -> None:
    """load_protocol_preset('Action Potential') sets stimulus_duration to 30.0."""
    ps = _make_protocol_state()
    with patch.object(ProtocolState, "get_state", new=_make_get_state_fn({})):
        [_ async for _ in ps.load_protocol_preset(ACTION_POTENTIAL)]
    assert ps.stimulus_duration == pytest.approx(30.0)


async def test_load_protocol_preset_action_potential_sets_clamp_mode() -> None:
    """load_protocol_preset('Action Potential') sets clamp_mode to 'Current Clamp'."""
    ps = _make_protocol_state()
    with patch.object(ProtocolState, "get_state", new=_make_get_state_fn({})):
        [_ async for _ in ps.load_protocol_preset(ACTION_POTENTIAL)]
    assert ps.clamp_mode == "Current Clamp"


async def test_load_protocol_preset_repetitive_firing_sets_stimulus_duration() -> None:
    """load_protocol_preset('Repetitive Firing') sets stimulus_duration to 180.0."""
    ps = _make_protocol_state()
    with patch.object(ProtocolState, "get_state", new=_make_get_state_fn({})):
        [_ async for _ in ps.load_protocol_preset(REPETITIVE_FIRING)]
    assert ps.stimulus_duration == pytest.approx(180.0)


async def test_load_protocol_preset_iv_curve_sets_voltage_clamp_mode() -> None:
    """load_protocol_preset('I-V Curve') sets clamp_mode to 'Voltage Clamp'."""
    ps = _make_protocol_state()
    with patch.object(ProtocolState, "get_state", new=_make_get_state_fn({})):
        [_ async for _ in ps.load_protocol_preset("I-V Curve")]
    assert ps.clamp_mode == "Voltage Clamp"


async def test_set_clamp_mode_clears_current_sweeps() -> None:
    """set_clamp_mode resets current_sweeps on the SimulationState."""
    ps = _make_protocol_state()
    sim_st = _make_state()
    sim_st._current_sweeps = [_make_sweep()]
    with patch.object(
        ProtocolState, "get_state", new=_make_get_state_fn({SimulationState: sim_st})
    ):
        [_ async for _ in ps.set_clamp_mode("Voltage Clamp")]
    assert len(sim_st._current_sweeps) == 0


async def test_set_clamp_mode_clears_stored_traces() -> None:
    """set_clamp_mode resets stored_traces on the SimulationState."""
    ps = _make_protocol_state()
    sim_st = _make_state()
    sim_st.stored_traces = [_make_sweep()]
    with patch.object(
        ProtocolState, "get_state", new=_make_get_state_fn({SimulationState: sim_st})
    ):
        [_ async for _ in ps.set_clamp_mode("Voltage Clamp")]
    assert len(sim_st.stored_traces) == 0


async def test_set_clamp_mode_resets_cont_has_state() -> None:
    """set_clamp_mode resets _cont_has_state on the SimulationState."""
    ps = _make_protocol_state()
    sim_st = _make_state()
    sim_st._cont_has_state = True
    with patch.object(
        ProtocolState, "get_state", new=_make_get_state_fn({SimulationState: sim_st})
    ):
        [_ async for _ in ps.set_clamp_mode("Voltage Clamp")]
    assert sim_st._cont_has_state is False


async def test_load_protocol_preset_unknown_name_is_ignored() -> None:
    """load_protocol_preset silently ignores an unknown preset name."""
    ps = _make_protocol_state()
    original = ps.stimulus_duration
    [_ async for _ in ps.load_protocol_preset("NonExistentPreset")]
    assert ps.stimulus_duration == pytest.approx(original)


# ---------------------------------------------------------------------------
# load_neuron_preset
# ---------------------------------------------------------------------------


async def test_load_neuron_preset_fast_spiking_interneuron() -> None:
    """load_neuron_preset enables IKv31 and leaves IKa disabled for Fast-Spiking."""
    ns = _make_neuron_state()
    with patch.object(NeuronState, "get_state", new=_make_get_state_fn({})):
        [_ async for _ in ns.load_neuron_preset(FAST_SPIKING_INTERNEURON)]
    assert ns.ikv31_enabled is True
    assert ns.ika_enabled is False


async def test_load_neuron_preset_sets_active_neuron_type() -> None:
    """load_neuron_preset records the selected neuron type on NeuronState."""
    ns = _make_neuron_state()
    with patch.object(NeuronState, "get_state", new=_make_get_state_fn({})):
        [_ async for _ in ns.load_neuron_preset(CORTICAL_PYRAMIDAL)]
    assert ns.active_neuron_type == CORTICAL_PYRAMIDAL


async def test_load_neuron_preset_resets_previously_enabled_channels() -> None:
    """Loading a second neuron preset disables channels from the first."""
    ns = _make_neuron_state()
    with patch.object(NeuronState, "get_state", new=_make_get_state_fn({})):
        [_ async for _ in ns.load_neuron_preset(FAST_SPIKING_INTERNEURON)]
        assert ns.ikv31_enabled is True
        [_ async for _ in ns.load_neuron_preset(CORTICAL_PYRAMIDAL)]
    assert ns.ikv31_enabled is False


async def test_load_neuron_preset_pyramidal_neuron() -> None:
    """load_neuron_preset enables Ih, INaP, and IM channels for Cortical Pyramidal."""
    ns = _make_neuron_state()
    with patch.object(NeuronState, "get_state", new=_make_get_state_fn({})):
        [_ async for _ in ns.load_neuron_preset(CORTICAL_PYRAMIDAL)]
    assert ns.ih_enabled is True
    assert ns.inap_enabled is True
    assert ns.im_enabled is True


async def test_load_neuron_preset_purkinje_cell() -> None:
    """load_neuron_preset enables ICaL, ICaT, and IKCa channels for Purkinje Cell."""
    ns = _make_neuron_state()
    with patch.object(NeuronState, "get_state", new=_make_get_state_fn({})):
        [_ async for _ in ns.load_neuron_preset(PURKINJE)]
    assert ns.ical_enabled is True
    assert ns.icat_enabled is True
    assert ns.ikca_enabled is True


async def test_load_neuron_preset_dopaminergic_neuron() -> None:
    """load_neuron_preset enables Ih and IM channels for Dopaminergic Neuron."""
    ns = _make_neuron_state()
    with patch.object(NeuronState, "get_state", new=_make_get_state_fn({})):
        [_ async for _ in ns.load_neuron_preset(DOPAMINERGIC)]
    assert ns.ih_enabled is True
    assert ns.im_enabled is True


async def test_load_neuron_preset_thalamic_relay() -> None:
    """load_neuron_preset enables ICaT and Ih channels for Thalamic Relay."""
    ns = _make_neuron_state()
    with patch.object(NeuronState, "get_state", new=_make_get_state_fn({})):
        [_ async for _ in ns.load_neuron_preset(THALAMIC_RELAY)]
    assert ns.icat_enabled is True
    assert ns.ih_enabled is True


# ---------------------------------------------------------------------------
# _build_neuron forwards preset.calcium_dynamics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset_name",
    [TRN, THALAMIC_RELAY, STN, PURKINJE, CA1_PYRAMIDAL],
)
async def test_build_neuron_forwards_preset_calcium_dynamics(preset_name: str) -> None:
    """_build_neuron must propagate the preset's CalciumDynamics to the Neuron.

    Without this, every preset would silently use NeuronConfig's default
    auto-instantiated CalciumDynamics (alpha_ca=1e-4, tau_ca=200 ms),
    masking each preset's tuned values.  For TRN this collapses the HP92
    rebound burst (preset wants alpha_ca=1.2e-5 / tau_ca=20 ms — 8.3× and
    10× different from defaults).  This test asserts the bug class is
    fixed at the build-neuron level rather than only verifying the
    downstream simulation behaviour.

    Args:
        preset_name: Name of a preset that overrides calcium_dynamics.
    """
    ns = _make_neuron_state()
    with patch.object(NeuronState, "get_state", new=_make_get_state_fn({})):
        [_ async for _ in ns.load_neuron_preset(preset_name)]
    neuron = ns._build_neuron()

    expected = NEURON_PRESETS[preset_name].calcium_dynamics
    assert expected is not None, (
        f"Test fixture mismatch: {preset_name} preset must have "
        f"calcium_dynamics set for this regression test to apply"
    )
    actual = neuron.calcium_dynamics
    assert actual is not None, (
        f"{preset_name}: built neuron has no calcium_dynamics — "
        f"_build_neuron is dropping the preset's tuned values."
    )
    assert actual.alpha_ca == pytest.approx(expected.alpha_ca), (
        f"{preset_name}: built neuron alpha_ca={actual.alpha_ca} but "
        f"preset alpha_ca={expected.alpha_ca}"
    )
    assert actual.tau_ca == pytest.approx(expected.tau_ca), (
        f"{preset_name}: built neuron tau_ca={actual.tau_ca} but "
        f"preset tau_ca={expected.tau_ca}"
    )


async def test_trn_hyperpolarization_burst_via_ui_build_neuron() -> None:
    """End-to-end TRN HP92 rebound burst exercised via the UI's _build_neuron path.

    Loads the TRN preset into a NeuronState, builds a neuron via the UI's
    own ``_build_neuron`` method (the same code the live Reflex app uses
    when the user clicks Run), then runs the HYPERPOLARIZATION_STEPS
    protocol via :func:`simulate_batch` and asserts each deeper sweep
    (-3 to -5 µA/cm²) produces a 5+ spike rebound burst at 200–600 Hz.

    This covers the regression discovered when the user reported only ~2
    visible spikes per sweep in the UI: the previous version of
    ``_build_neuron`` silently dropped the preset's calcium_dynamics and
    used the auto-instantiated default (alpha_ca=1e-4, tau_ca=200 ms),
    which collapsed the burst after the first spike.  An assertion-only
    fix in the simulation domain is insufficient — the UI build path must
    be exercised end-to-end.
    """
    import patch_sim
    from patch_sim.clamp_simulations import simulate_batch, simulate_current_clamp
    from patch_sim.presets import build_protocol_from_preset

    ns = _make_neuron_state()
    with patch.object(NeuronState, "get_state", new=_make_get_state_fn({})):
        [_ async for _ in ns.load_neuron_preset(TRN)]
    neuron = ns._build_neuron()

    # Sanity: assert the preset's tuned calcium dynamics survived the
    # build-neuron step (this is what the parametrised regression test
    # above exercises in isolation).
    assert neuron.calcium_dynamics is not None
    assert neuron.calcium_dynamics.alpha_ca == pytest.approx(1.2e-5)
    assert neuron.calcium_dynamics.tau_ca == pytest.approx(20.0)

    # Run the multi-sweep protocol via simulate_batch (the same call site
    # the UI uses for multi-sweep current-clamp runs).
    protocol = build_protocol_from_preset(HYPERPOLARIZATION_STEPS, neuron_preset=TRN)
    results = list(
        simulate_batch(neuron, [sweep for sweep in protocol], simulate_current_clamp)
    )

    # The deeper sweeps (indices 0, 1, 2 — currents -5, -4, -3 µA/cm²)
    # must each produce a 5–15 spike, 200–600 Hz rebound burst.
    for sweep_idx in (0, 1, 2):
        result = results[sweep_idx]
        time_arr = np.asarray(result["time"])
        v_arr = np.asarray(result["voltage"])
        ap_result = patch_sim.analyze_aps(time_arr, v_arr)
        analysis = patch_sim.analyze_bursts(
            ap_result, total_duration_ms=float(time_arr[-1] - time_arr[0])
        )

        assert analysis.burst_count >= 1, (
            f"UI-build-path sweep {sweep_idx}: expected ≥1 LTS rebound "
            f"burst, got burst_count={analysis.burst_count}, "
            f"total APs={ap_result.spike_count}.  This regression points "
            f"at _build_neuron — most likely a preset attribute "
            f"(calcium_dynamics, channel factory, etc.) is not being "
            f"propagated to the built Neuron."
        )
        burst = analysis.bursts[0]
        assert 5 <= burst.spike_count <= 15, (
            f"UI-build-path sweep {sweep_idx}: expected 5–15 spikes "
            f"(Huguenard & Prince 1992), got {burst.spike_count}"
        )
        assert burst.intra_burst_frequency is not None
        assert 200.0 <= burst.intra_burst_frequency <= 600.0, (
            f"UI-build-path sweep {sweep_idx}: expected 200–600 Hz "
            f"intra-burst, got {burst.intra_burst_frequency:.1f} Hz"
        )


async def test_load_neuron_preset_unknown_name_is_ignored() -> None:
    """load_neuron_preset silently ignores an unknown preset name."""
    ns = _make_neuron_state()
    before = ns.active_neuron_type
    [_ async for _ in ns.load_neuron_preset("NonExistentNeuron")]
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
        [_ async for _ in ns.load_neuron_preset(CORTICAL_PYRAMIDAL)]
    assert len(sim_st.stored_traces) == 1


async def test_load_neuron_preset_keeps_sweeps_when_stored() -> None:
    """load_neuron_preset preserves current_sweeps when stored traces exist.

    When the user stores a trace and then changes neuron type, the previous
    simulation should remain visible in the figure alongside the stored trace.
    """
    sim_st = _make_state()
    sim_st.stored_traces = [_make_sweep()]
    sim_st._current_sweeps = [_make_sweep()]
    ns = _make_neuron_state()
    with patch.object(
        NeuronState,
        "get_state",
        new=_make_get_state_fn({SimulationState: sim_st}),
    ):
        [_ async for _ in ns.load_neuron_preset(CORTICAL_PYRAMIDAL)]
    assert len(sim_st._current_sweeps) == 1


async def test_load_neuron_preset_clears_sweeps_without_stored() -> None:
    """load_neuron_preset clears current_sweeps when no stored traces exist.

    Without stored traces the previous simulation is stale and should be
    removed so the figure does not show results from the wrong neuron type.
    """
    sim_st = _make_state()
    sim_st.stored_traces = []
    sim_st._current_sweeps = [_make_sweep()]
    ns = _make_neuron_state()
    with patch.object(
        NeuronState,
        "get_state",
        new=_make_get_state_fn({SimulationState: sim_st}),
    ):
        [_ async for _ in ns.load_neuron_preset(CORTICAL_PYRAMIDAL)]
    assert len(sim_st._current_sweeps) == 0


async def test_store_trace_label_includes_neuron_type() -> None:
    """store_trace includes the active neuron type in the stored trace label."""
    s = _make_state()
    s._label_neuron_type = CORTICAL_PYRAMIDAL
    s._current_sweeps = [_make_sweep()]
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
        [_ async for _ in ps.load_protocol_preset(REPETITIVE_FIRING)]
    assert ps.min_stimulus > 0.0


async def test_protocol_preset_without_active_neuron_type_uses_base_params() -> None:
    """Repetitive Firing with no active neuron type uses the base stimulus."""
    ps = _make_protocol_state()
    mock_neuron = MagicMock()
    mock_neuron.active_neuron_type = ""
    with patch.object(
        ProtocolState, "get_state", new=_make_get_state_fn({NeuronState: mock_neuron})
    ):
        [_ async for _ in ps.load_protocol_preset(REPETITIVE_FIRING)]
    assert ps.min_stimulus == pytest.approx(15.0)


async def test_protocol_preset_with_no_adjustment_entry_uses_base_params() -> None:
    """Repetitive Firing with Squid Giant Axon active falls through to base duration.

    SGA has no Repetitive Firing adjustment, so the base preset value (180 ms)
    should be used unchanged.
    """
    ps = _make_protocol_state()
    mock_neuron = MagicMock()
    mock_neuron.active_neuron_type = SQUID_GIANT_AXON
    with patch.object(
        ProtocolState, "get_state", new=_make_get_state_fn({NeuronState: mock_neuron})
    ):
        [_ async for _ in ps.load_protocol_preset(REPETITIVE_FIRING)]
    assert ps.stimulus_duration == pytest.approx(180.0)


async def test_protocol_preset_dopaminergic_repetitive_firing_long_duration() -> None:
    """Dopaminergic Neuron + Repetitive Firing sets stimulus_duration to 3000 ms."""
    ps = _make_protocol_state()
    mock_neuron = MagicMock()
    mock_neuron.active_neuron_type = DOPAMINERGIC
    with patch.object(
        ProtocolState, "get_state", new=_make_get_state_fn({NeuronState: mock_neuron})
    ):
        [_ async for _ in ps.load_protocol_preset(REPETITIVE_FIRING)]
    assert ps.stimulus_duration == pytest.approx(3000.0)


# ---------------------------------------------------------------------------
# store_trace / clear_stored_traces
# ---------------------------------------------------------------------------


async def test_store_trace_appends_to_stored_traces() -> None:
    """store_trace promotes the current sweep to stored_traces."""
    s = _make_state()
    s._current_sweeps = [_make_sweep()]
    assert len(s.stored_traces) == 0
    with patch.object(SimulationState, "get_state", new=_make_get_state_fn({})):
        await s.store_trace()
    assert len(s.stored_traces) == 1


async def test_store_trace_sets_stored_label() -> None:
    """store_trace labels the stored entry with index and neuron type."""
    s = _make_state()
    s._current_sweeps = [_make_sweep()]
    with patch.object(SimulationState, "get_state", new=_make_get_state_fn({})):
        await s.store_trace()
    assert s.stored_traces[0].label == f"Stored 1 ({s._label_neuron_type})"


async def test_store_trace_twice_increments_label() -> None:
    """Calling store_trace twice creates sequentially numbered labels."""
    s = _make_state()
    s._current_sweeps = [_make_sweep()]
    with patch.object(SimulationState, "get_state", new=_make_get_state_fn({})):
        await s.store_trace()
        await s.store_trace()
    assert len(s.stored_traces) == 2
    assert s.stored_traces[1].label == f"Stored 2 ({s._label_neuron_type})"


async def test_store_trace_does_nothing_when_no_result() -> None:
    """store_trace is a no-op when current_sweeps is empty."""
    s = _make_state()
    assert len(s._current_sweeps) == 0
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
    s._current_sweeps = [_make_sweep()]
    with patch.object(SimulationState, "get_state", new=_make_get_state_fn({})):
        await s.store_trace()
    assert s.has_stored_traces is True


async def test_set_clamp_mode_to_current_resets_protocol_type() -> None:
    """set_clamp_mode('Current Clamp') resets protocol_type to the first CC option."""
    ps = _make_protocol_state()
    ps.clamp_mode = "Voltage Clamp"
    ps.protocol_type = "Ramp"
    with patch.object(ProtocolState, "get_state", new=_make_get_state_fn({})):
        [_ async for _ in ps.set_clamp_mode("Current Clamp")]
    assert ps.protocol_type == constants.CURRENT_PROTOCOLS[0]


async def test_set_clamp_mode_to_voltage_resets_protocol_type() -> None:
    """set_clamp_mode('Voltage Clamp') resets protocol_type to the first VC option."""
    ps = _make_protocol_state()
    with patch.object(ProtocolState, "get_state", new=_make_get_state_fn({})):
        [_ async for _ in ps.set_clamp_mode("Voltage Clamp")]
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


# ---------------------------------------------------------------------------
# active_protocol_preset tracking
# ---------------------------------------------------------------------------


def test_active_protocol_preset_empty_by_default() -> None:
    """active_protocol_preset is an empty string on a fresh ProtocolState."""
    ps = _make_protocol_state()
    assert ps.active_protocol_preset == ""


async def test_active_protocol_preset_set_on_load() -> None:
    """load_protocol_preset records the preset name in active_protocol_preset."""
    ps = _make_protocol_state()
    with patch.object(ProtocolState, "get_state", new=_make_get_state_fn({})):
        [_ async for _ in ps.load_protocol_preset(REPETITIVE_FIRING)]
    assert ps.active_protocol_preset == REPETITIVE_FIRING


async def test_active_protocol_preset_updated_on_second_load() -> None:
    """active_protocol_preset is overwritten when a different preset is loaded."""
    ps = _make_protocol_state()
    with patch.object(ProtocolState, "get_state", new=_make_get_state_fn({})):
        [_ async for _ in ps.load_protocol_preset(ACTION_POTENTIAL)]
        [_ async for _ in ps.load_protocol_preset(REPETITIVE_FIRING)]
    assert ps.active_protocol_preset == REPETITIVE_FIRING


async def test_set_clamp_mode_clears_active_protocol_preset() -> None:
    """Switching clamp mode clears active_protocol_preset to avoid stale re-application.

    If a Current Clamp preset is active and the user manually switches to
    Voltage Clamp, the preset must be cleared so that a subsequent neuron
    change does not re-apply the old Current Clamp preset and flip the mode back.
    """
    ps = _make_protocol_state()
    ps.active_protocol_preset = REPETITIVE_FIRING
    with patch.object(ProtocolState, "get_state", new=_make_get_state_fn({})):
        [_ async for _ in ps.set_clamp_mode("Voltage Clamp")]
    assert ps.active_protocol_preset == ""


# ---------------------------------------------------------------------------
# load_neuron_preset — protocol override re-application
# ---------------------------------------------------------------------------


async def test_load_neuron_preset_reapplies_active_protocol_overrides() -> None:
    """Switching neuron type re-applies the active protocol preset with new overrides.

    Load 'Repetitive Firing' (Squid defaults: duration=180 ms), then switch to
    SNc Dopaminergic which has a longer 3000 ms duration override and lower
    0.3 µA/cm² stimulus (post-#318 review the cell is a Cav1.3+SK tonic
    pacemaker; ≳1 µA/cm² drives depolarisation block, mid-range drive can
    produce abortive doublets).  The protocol params should update
    automatically.
    """
    ns = _make_neuron_state()
    ps = _make_protocol_state()
    ps.active_protocol_preset = REPETITIVE_FIRING
    ps._apply_protocol_preset(REPETITIVE_FIRING)
    sim_st = _make_state()
    with patch.object(
        NeuronState,
        "get_state",
        new=_make_get_state_fn({ProtocolState: ps, SimulationState: sim_st}),
    ):
        [_ async for _ in ns.load_neuron_preset(DOPAMINERGIC)]
    assert ps.stimulus_duration == pytest.approx(3000.0)
    assert ps.min_stimulus == pytest.approx(0.3)


async def test_load_neuron_preset_no_active_protocol_skips_override() -> None:
    """Switching neuron type with no active preset leaves protocol params unchanged."""
    ns = _make_neuron_state()
    ps = _make_protocol_state()
    # active_protocol_preset is empty — no override should be applied
    original_duration = ps.stimulus_duration
    sim_st = _make_state()
    with patch.object(
        NeuronState,
        "get_state",
        new=_make_get_state_fn({ProtocolState: ps, SimulationState: sim_st}),
    ):
        [_ async for _ in ns.load_neuron_preset(DOPAMINERGIC)]
    assert ps.stimulus_duration == pytest.approx(original_duration)


async def test_load_neuron_preset_squid_uses_base_protocol_params() -> None:
    """Switching to Squid Giant Axon applies the base preset params (no adjustments)."""
    ns = _make_neuron_state()
    ps = _make_protocol_state()
    ps.active_protocol_preset = REPETITIVE_FIRING
    ps._apply_protocol_preset(REPETITIVE_FIRING)
    sim_st = _make_state()
    with patch.object(
        NeuronState,
        "get_state",
        new=_make_get_state_fn({ProtocolState: ps, SimulationState: sim_st}),
    ):
        [_ async for _ in ns.load_neuron_preset(SQUID_GIANT_AXON)]
    # Squid Giant Axon has no Repetitive Firing entry in NEURON_PROTOCOL_ADJUSTMENTS
    assert ps.stimulus_duration == pytest.approx(180.0)
    assert ps.min_stimulus == pytest.approx(15.0)


async def test_load_neuron_preset_reapplies_protocol_neuron_overrides() -> None:
    """Switching neuron type re-applies PROTOCOL_NEURON_OVERRIDES on top of preset.

    The 'Na+ Channel Activation' preset sets g_K=0.  Loading a new neuron
    preset would normally restore g_K to its default value, but the override
    must be re-applied on top so that g_K stays at 0.
    """
    ns = _make_neuron_state()
    ps = _make_protocol_state()
    ps.active_protocol_preset = "Na+ Channel Activation"
    ps._apply_protocol_preset("Na+ Channel Activation")
    sim_st = _make_state()
    with patch.object(
        NeuronState,
        "get_state",
        new=_make_get_state_fn({ProtocolState: ps, SimulationState: sim_st}),
    ):
        [_ async for _ in ns.load_neuron_preset(CORTICAL_PYRAMIDAL)]
    assert ns.g_K == pytest.approx(0.0)


async def test_load_neuron_preset_syncs_figure_clamp_mode() -> None:
    """load_neuron_preset syncs _figure_clamp_mode on SimulationState."""
    ns = _make_neuron_state()
    ps = _make_protocol_state()
    ps.active_protocol_preset = REPETITIVE_FIRING
    ps._apply_protocol_preset(REPETITIVE_FIRING)
    sim_st = _make_state()
    sim_st._figure_clamp_mode = "Voltage Clamp"

    async def _get_state(_self, cls):
        """Return the appropriate state instance for the given class."""
        if cls is ProtocolState:
            return ps
        if cls is SimulationState:
            return sim_st
        return MagicMock()

    with patch.object(NeuronState, "get_state", new=_get_state):
        [_ async for _ in ns.load_neuron_preset(DOPAMINERGIC)]
    assert sim_st._figure_clamp_mode == ps.clamp_mode


# ---------------------------------------------------------------------------
# NeuronState↔NeuronConfig sync regression tests (#232)
# ---------------------------------------------------------------------------


def test_neuron_state_mirrors_all_neuron_config_scalar_fields() -> None:
    """Every NeuronConfig scalar field appears on NeuronState with the correct default.

    This test fails if a new scalar field is added to NeuronConfig without
    the corresponding entry flowing through to NeuronState.
    """
    ns = _make_neuron_state()
    for name in NEURON_CONFIG_SCALAR_FIELDS:
        assert hasattr(ns, name), f"NeuronState missing field: {name}"
        assert getattr(ns, name) == pytest.approx(
            NEURON_CONFIG_SCALAR_DEFAULTS[name]
        ), f"NeuronState.{name} default does not match NeuronConfig"


def test_build_neuron_forwards_all_neuron_config_scalar_fields() -> None:
    """_build_neuron() forwards every scalar field value to the constructed Neuron.

    Mutates each NeuronState field to a widely-spaced sentinel, calls
    _build_neuron(), then checks the resulting Neuron.  Wide spacing (100 *
    index) makes field-swap bugs produce large diffs rather than near-misses.
    Fails if any field is silently dropped in the kwargs construction.
    """
    ns = _make_neuron_state()
    expected: dict[str, float] = {}
    for i, name in enumerate(NEURON_CONFIG_SCALAR_FIELDS):
        sentinel = NEURON_CONFIG_SCALAR_DEFAULTS[name] + 100.0 * (i + 1)
        setattr(ns, name, sentinel)
        expected[name] = sentinel

    neuron = ns._build_neuron()
    for name, value in expected.items():
        assert getattr(neuron, name) == pytest.approx(value), (
            f"_build_neuron dropped field: {name}"
        )


@pytest.mark.parametrize("preset_name", list(NEURON_PRESETS))
def test_neuron_config_to_ui_state_covers_all_scalar_fields(
    preset_name: str,
) -> None:
    """neuron_config_to_ui_state() returns a key for every scalar field.

    Parametrized over all presets so that value-forwarding bugs on specific
    presets are caught, not just field presence.  Fails if a new NeuronConfig
    scalar field is missing from the returned dict or has a wrong value.

    Args:
        preset_name: Name of the preset to test.
    """
    config = NEURON_PRESETS[preset_name]
    state = neuron_config_to_ui_state(config)
    for name in NEURON_CONFIG_SCALAR_FIELDS:
        assert name in state, f"neuron_config_to_ui_state missing key: {name}"
        assert state[name] == pytest.approx(getattr(config, name)), (
            f"neuron_config_to_ui_state value mismatch for: {name}"
        )


# ---------------------------------------------------------------------------
# initialize_defaults
# ---------------------------------------------------------------------------


async def test_initialize_defaults_applies_sga_q10() -> None:
    """initialize_defaults sets Q10 to the SGA preset value (1.0).

    Core regression for issue #235: the SGA preset overrides Q10 to 1.0 but
    NeuronConfig.Q10 defaults to 3.0, so without initialize_defaults the app
    starts with the wrong Q10.
    """
    ns = _make_neuron_state()
    ps = _make_protocol_state()
    sim_st = _make_state()
    with patch.object(
        SimulationState,
        "get_state",
        new=_make_get_state_fn({NeuronState: ns, ProtocolState: ps}),
    ):
        [_ async for _ in sim_st.initialize_defaults()]
    assert ns.Q10 == pytest.approx(1.0)


async def test_initialize_defaults_applies_action_potential_with_sga_adjustment() -> (
    None
):
    """initialize_defaults applies the SGA-specific stimulus_duration=10.0 override.

    The base Action Potential preset uses stimulus_duration=30.0, but the SGA
    neuron-specific adjustment reduces it to 10.0 to elicit exactly one spike.
    Without this override the membrane re-fires within the 30 ms window.
    """
    ns = _make_neuron_state()
    ps = _make_protocol_state()
    sim_st = _make_state()
    with patch.object(
        SimulationState,
        "get_state",
        new=_make_get_state_fn({NeuronState: ns, ProtocolState: ps}),
    ):
        [_ async for _ in sim_st.initialize_defaults()]
    assert ps.stimulus_duration == pytest.approx(10.0)
    assert ps.active_protocol_preset == DEFAULT_PROTOCOL_PRESET
    assert ps.clamp_mode == "Current Clamp"


async def test_initialize_defaults_syncs_simulation_shadows() -> None:
    """initialize_defaults syncs _label_neuron_type and _figure_clamp_mode."""
    ns = _make_neuron_state()
    ps = _make_protocol_state()
    sim_st = _make_state()
    with patch.object(
        SimulationState,
        "get_state",
        new=_make_get_state_fn({NeuronState: ns, ProtocolState: ps}),
    ):
        [_ async for _ in sim_st.initialize_defaults()]
    assert sim_st._label_neuron_type == DEFAULT_NEURON_PRESET
    assert sim_st._figure_clamp_mode == "Current Clamp"


async def test_initialize_defaults_is_idempotent() -> None:
    """Calling initialize_defaults twice leaves state unchanged on the second call."""
    ns = _make_neuron_state()
    ps = _make_protocol_state()
    sim_st = _make_state()
    with patch.object(
        SimulationState,
        "get_state",
        new=_make_get_state_fn({NeuronState: ns, ProtocolState: ps}),
    ):
        [_ async for _ in sim_st.initialize_defaults()]
        q10_after_first = ns.Q10
        duration_after_first = ps.stimulus_duration
        [_ async for _ in sim_st.initialize_defaults()]
    assert ns.Q10 == pytest.approx(q10_after_first)
    assert ps.stimulus_duration == pytest.approx(duration_after_first)


async def test_reset_to_defaults_preserves_current_selection() -> None:
    """reset_to_defaults re-applies the current selection, not the startup defaults.

    The Reset button should restore preset defaults for whichever neuron and
    protocol the user has selected, rather than switching back to the startup
    defaults.  This test sets a non-default neuron and protocol, calls
    reset_to_defaults, and verifies the selection is preserved.
    reset() is patched to a no-op because the Reflex parent-state chain is not
    available in isolated unit tests.
    """
    ns = _make_neuron_state()
    ns.active_neuron_type = CORTICAL_PYRAMIDAL
    ps = _make_protocol_state()
    ps.active_protocol_preset = REPETITIVE_FIRING
    sim_st = _make_state()
    with (
        patch.object(SimulationState, "reset", return_value=None),
        patch.object(NeuronState, "reset", return_value=None),
        patch.object(ProtocolState, "reset", return_value=None),
        patch.object(
            SimulationState,
            "get_state",
            new=_make_get_state_fn({NeuronState: ns, ProtocolState: ps}),
        ),
    ):
        yielded = [e async for e in sim_st.reset_to_defaults()]

    assert ns.active_neuron_type == CORTICAL_PYRAMIDAL
    assert ps.active_protocol_preset == REPETITIVE_FIRING
    assert SimulationState.run_membrane_test in yielded
    assert SimulationState.initialize_defaults not in yielded


# ---------------------------------------------------------------------------
# Analysis results cleared when traces are cleared (issue #271)
# ---------------------------------------------------------------------------


def _prime_analysis(an_st: AnalysisState) -> None:
    """Populate every analysis field on an AnalysisState with non-empty data.

    Args:
        an_st: The AnalysisState instance to populate.
    """
    an_st.ap_metrics = [{"peak_v": 40.0}]
    an_st.ap_summary = {"mean_peak_v": 40.0}
    an_st.ap_is_multi_sweep = True
    an_st.ca_transient_metrics = [{"peak_concentration": "1.0"}]
    an_st.ca_transient_summary = {"transient_count": "1"}
    an_st.burst_metrics = [{"start_time": "10.0"}]
    an_st.burst_summary = {"burst_count": "1"}
    an_st.fi_data = {"current_steps": [0.1]}
    an_st.iv_data = {"voltage_steps": [-70.0]}
    an_st.gv_data = {"conductance": [0.5]}
    an_st.sfa_data = {"isi": [10.0]}
    an_st.hyperpolarization_data = {"sag_ratio": 0.1}
    an_st.phase_plane_data = {"v": [-65.0]}


def _assert_analysis_cleared(an_st: AnalysisState) -> None:
    """Assert that all per-simulation analysis fields are reset to empty defaults.

    Args:
        an_st: The AnalysisState instance to inspect.
    """
    assert an_st.ap_metrics == []
    assert an_st.ap_summary == {}
    assert an_st.ap_is_multi_sweep is False
    assert an_st.ca_transient_metrics == []
    assert an_st.ca_transient_summary == {}
    assert an_st.burst_metrics == []
    assert an_st.burst_summary == {}
    assert an_st.fi_data == {}
    assert an_st.iv_data == {}
    assert an_st.gv_data == {}
    assert an_st.sfa_data == {}
    assert an_st.hyperpolarization_data == {}
    assert an_st.phase_plane_data == {}


async def test_set_clamp_mode_clears_analysis_results() -> None:
    """set_clamp_mode clears all per-simulation analysis fields on AnalysisState."""
    ps = _make_protocol_state()
    sim_st = _make_state()
    an_st = _make_analysis_state()
    _prime_analysis(an_st)
    with patch.object(
        ProtocolState,
        "get_state",
        new=_make_get_state_fn({SimulationState: sim_st, AnalysisState: an_st}),
    ):
        [_ async for _ in ps.set_clamp_mode("Voltage Clamp")]
    _assert_analysis_cleared(an_st)


async def test_set_clamp_mode_preserves_membrane_test_cache() -> None:
    """set_clamp_mode does not clear membrane test cache on AnalysisState.

    The mt_* fields are fingerprint-based and should survive protocol switches.
    """
    ps = _make_protocol_state()
    sim_st = _make_state()
    an_st = _make_analysis_state()
    an_st.mt_input_resistance = "100 kΩ·cm²"
    an_st.mt_neuron_fingerprint = "abc123"
    with patch.object(
        ProtocolState,
        "get_state",
        new=_make_get_state_fn({SimulationState: sim_st, AnalysisState: an_st}),
    ):
        [_ async for _ in ps.set_clamp_mode("Voltage Clamp")]
    assert an_st.mt_input_resistance == "100 kΩ·cm²"
    assert an_st.mt_neuron_fingerprint == "abc123"


async def test_load_protocol_preset_clears_analysis_results() -> None:
    """load_protocol_preset clears all per-simulation analysis fields."""
    ps = _make_protocol_state()
    sim_st = _make_state()
    an_st = _make_analysis_state()
    _prime_analysis(an_st)
    with patch.object(
        ProtocolState,
        "get_state",
        new=_make_get_state_fn({SimulationState: sim_st, AnalysisState: an_st}),
    ):
        [_ async for _ in ps.load_protocol_preset(ACTION_POTENTIAL)]
    _assert_analysis_cleared(an_st)


async def test_load_neuron_preset_clears_analysis_when_no_stored_traces() -> None:
    """load_neuron_preset clears analysis results when no stored traces exist.

    Without stored traces the simulation sweeps are cleared, so analysis
    results from a previous run should not persist for the new neuron type.
    """
    ns = _make_neuron_state()
    sim_st = _make_state()
    an_st = _make_analysis_state()
    sim_st.stored_traces = []
    _prime_analysis(an_st)
    with patch.object(
        NeuronState,
        "get_state",
        new=_make_get_state_fn({SimulationState: sim_st, AnalysisState: an_st}),
    ):
        [_ async for _ in ns.load_neuron_preset(CORTICAL_PYRAMIDAL)]
    _assert_analysis_cleared(an_st)


async def test_load_neuron_preset_preserves_analysis_with_stored_traces() -> None:
    """load_neuron_preset keeps analysis results when stored traces exist.

    When the user has stored a trace and changes neuron type, the previous
    simulation (and its analysis) remain visible alongside the stored trace.
    """
    ns = _make_neuron_state()
    sim_st = _make_state()
    an_st = _make_analysis_state()
    sim_st.stored_traces = [_make_sweep()]
    _prime_analysis(an_st)
    with patch.object(
        NeuronState,
        "get_state",
        new=_make_get_state_fn({SimulationState: sim_st, AnalysisState: an_st}),
    ):
        [_ async for _ in ns.load_neuron_preset(CORTICAL_PYRAMIDAL)]
    assert an_st.ap_metrics == [{"peak_v": 40.0}]
    assert an_st.fi_data == {"current_steps": [0.1]}


async def test_clear_stored_traces_preserves_analysis_results() -> None:
    """clear_stored_traces does not affect analysis results on AnalysisState.

    Analysis results correspond to the current simulation sweeps, not the
    stored reference traces, so removing stored traces must not clear analysis.
    """
    s = _make_state()
    s.stored_traces = [_make_sweep()]
    an_st = _make_analysis_state()
    _prime_analysis(an_st)
    with patch.object(SimulationState, "get_state", new=_make_get_state_fn({})):
        await s.clear_stored_traces()
    assert an_st.ap_metrics == [{"peak_v": 40.0}]
    assert an_st.fi_data == {"current_steps": [0.1]}


def test_analysis_state_clear_results_resets_simulation_fields() -> None:
    """clear_results resets per-simulation fields and leaves mt_* fields untouched."""
    an_st = _make_analysis_state()
    _prime_analysis(an_st)
    an_st.mt_input_resistance = "50 kΩ·cm²"
    an_st.mt_neuron_fingerprint = "xyz"
    an_st.clear_results()
    _assert_analysis_cleared(an_st)
    assert an_st.mt_input_resistance == "50 kΩ·cm²"
    assert an_st.mt_neuron_fingerprint == "xyz"


# ---------------------------------------------------------------------------
# Cell-area metadata flowing through NeuronState (#222)
# ---------------------------------------------------------------------------


def test_build_neuron_carries_area_for_cortical_pyramidal() -> None:
    """A built Neuron carries the cortical pyramidal preset's area_cm2."""
    ns = _make_neuron_state()
    ns.active_neuron_type = CORTICAL_PYRAMIDAL
    neuron = ns._build_neuron()
    assert neuron.area_cm2 == pytest.approx(20e-6)


def test_build_neuron_has_no_area_for_squid() -> None:
    """The squid preset has no area_cm2 — built Neuron exposes ``None``."""
    ns = _make_neuron_state()
    ns.active_neuron_type = SQUID_GIANT_AXON
    neuron = ns._build_neuron()
    assert neuron.area_cm2 is None
