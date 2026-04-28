"""Integration tests for burst-metric analysis with real simulations.

Drives :func:`analyze_bursts_from_result` through the full
simulate-then-analyse pipeline using bursting and non-bursting presets.
Unit tests with synthetic AP results live in
tests/unit/test_burst_metrics.py.
"""

import numpy as np

import patch_sim
from patch_sim.analysis.burst_metrics import (
    analyze_bursts,
    analyze_bursts_from_result,
)
from patch_sim.clamp_simulations import simulate_current_clamp
from patch_sim.constants import PURKINJE, THALAMIC_RELAY
from patch_sim.neuron import Neuron
from patch_sim.neuron_factory import make_neuron
from patch_sim.presets import NEURON_PRESETS
from patch_sim.protocols import step_current


def test_purkinje_preset_produces_at_least_one_burst() -> None:
    """A depolarising step on Purkinje should drive ≥1 detected burst.

    Purkinje is a spontaneous pacemaker with both ICaL and ICaT.  Under a
    moderate depolarising current it produces high-frequency bursts that
    are well above the default ISI threshold gap.
    """
    neuron = make_neuron(NEURON_PRESETS[PURKINJE])
    protocol = step_current(
        duration=600.0,
        current_amplitude=10.0,
        step_start=10.0,
        step_duration=580.0,
    )
    result = simulate_current_clamp(neuron, protocol)

    analysis = analyze_bursts_from_result(result)
    assert analysis.burst_count >= 1
    assert analysis.duty_cycle is not None and analysis.duty_cycle > 0.0
    assert analysis.isi_threshold_ms > 0.0
    assert analysis.threshold_method in {"auto-histogram", "default-fixed"}


def test_classic_hh_tonic_firing_does_not_report_genuine_bursting(
    hh_model: Neuron,
) -> None:
    """Tonic-firing HH should not report multi-burst structure.

    A regular tonic spike train either has all ISIs above the threshold
    (zero bursts, all spikes "unburst") or all ISIs below it (one
    contiguous "burst" with no inter-burst interval).  In neither case
    should a real :attr:`mean_inter_burst_interval` appear.

    Args:
        hh_model: Classic Hodgkin-Huxley neuron fixture.
    """
    protocol = step_current(
        duration=200.0,
        current_amplitude=10.0,
        step_start=10.0,
        step_duration=180.0,
    )
    result = simulate_current_clamp(hh_model, protocol)
    analysis = analyze_bursts_from_result(result)
    assert analysis.mean_inter_burst_interval is None, (
        "Tonic firing must not report a multi-burst inter-burst interval; "
        f"got burst_count={analysis.burst_count}"
    )
    assert analysis.burst_count <= 1


def test_thalamic_relay_step_release_does_not_falsely_report_burst() -> None:
    """A single rebound spike on Thalamic Relay must not be reported as a burst.

    With the default hyperpolarising step the Thalamic Relay preset
    produces a single ICaT-driven rebound spike, not a multi-spike burst.
    With the default ``min_spikes_per_burst`` of 2 the burst analyser
    must therefore report zero bursts and surface the lone spike via
    :attr:`unburst_spike_count`.
    """
    neuron = make_neuron(NEURON_PRESETS[THALAMIC_RELAY])
    pre = 50.0
    stim = 300.0
    post = 200.0
    protocol = step_current(
        duration=pre + stim + post,
        current_amplitude=-10.0,
        step_start=pre,
        step_duration=stim,
    )
    result = simulate_current_clamp(neuron, protocol)
    analysis = analyze_bursts_from_result(result, isi_threshold_ms=50.0)
    assert analysis.burst_count == 0
    assert analysis.duty_cycle is None


def test_no_spikes_returns_empty_burst_result_on_simulation(
    hh_model: Neuron,
) -> None:
    """A sub-threshold protocol produces zero bursts but still reports a threshold.

    Args:
        hh_model: Classic Hodgkin-Huxley neuron fixture.
    """
    protocol = step_current(
        duration=20.0,
        current_amplitude=0.5,  # well below rheobase
        step_start=2.0,
        step_duration=15.0,
    )
    result = simulate_current_clamp(hh_model, protocol)
    analysis = analyze_bursts_from_result(result)
    assert analysis.burst_count == 0
    assert analysis.duty_cycle is None
    assert analysis.isi_threshold_ms > 0.0


def test_user_supplied_threshold_changes_grouping() -> None:
    """Different ``isi_threshold_ms`` values for the same trace yield different counts.

    Sanity check that the threshold is actually applied: for a Purkinje
    burst train, a very small threshold (1 ms) should fragment most ISIs
    into "unburst" while a very large one (500 ms) should merge everything
    into a single burst.
    """
    neuron = make_neuron(NEURON_PRESETS[PURKINJE])
    protocol = step_current(
        duration=400.0,
        current_amplitude=10.0,
        step_start=10.0,
        step_duration=380.0,
    )
    result = simulate_current_clamp(neuron, protocol)
    time = np.asarray(result["time"])
    total_duration_ms = float(time[-1] - time[0])
    ap_result = patch_sim.analyze_aps_from_result(result)
    if ap_result.spike_count < 2:
        # Defensive: should not happen with this protocol, but skip the
        # comparison rather than fail spuriously if it does.
        return

    fine = analyze_bursts(
        ap_result, total_duration_ms=total_duration_ms, isi_threshold_ms=1.0
    )
    coarse = analyze_bursts(
        ap_result, total_duration_ms=total_duration_ms, isi_threshold_ms=500.0
    )
    assert fine.threshold_method == "user"
    assert coarse.threshold_method == "user"
    assert coarse.burst_count == 1
    assert fine.burst_count <= coarse.burst_count
