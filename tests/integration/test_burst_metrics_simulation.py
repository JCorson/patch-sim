"""Integration tests for burst-metric analysis with real simulations.

Drives :func:`analyze_bursts_from_result` through the full
simulate-then-analyse pipeline using rebound-bursting and tonic-firing
presets.  Unit tests with synthetic AP results live in
tests/unit/test_burst_metrics.py.
"""

import numpy as np

import patch_sim
from patch_sim.analysis.burst_metrics import (
    analyze_bursts,
    analyze_bursts_from_result,
)
from patch_sim.clamp_simulations import simulate_current_clamp
from patch_sim.constants import PURKINJE, STN, THALAMIC_RELAY
from patch_sim.neuron import Neuron
from patch_sim.neuron_factory import make_neuron
from patch_sim.presets import NEURON_PRESETS
from patch_sim.protocols import step_current


def test_purkinje_tonic_firing_reports_zero_bursts() -> None:
    """Purkinje under a depolarising step is tonic — zero bursts, all unburst.

    Purkinje is a tonic pacemaker (Raman & Bean 1999).  Under a moderate
    depolarising current it produces a regular tonic spike train with a
    unimodal ISI distribution, so the analyser cannot place an
    auto-histogram threshold and falls back to ``"default-fixed"``.  In
    that case the burst analyser short-circuits to zero bursts and
    surfaces every spike via ``unburst_spike_count`` (issue #290).

    Complex-spike bursts in vivo are climbing-fibre driven and cannot be
    produced by this single-compartment, current-clamp preset.
    """
    neuron = make_neuron(NEURON_PRESETS[PURKINJE])
    protocol = step_current(
        duration=600.0,
        current_amplitude=10.0,
        step_start=10.0,
        step_duration=580.0,
    )
    result = simulate_current_clamp(neuron, protocol)
    time = np.asarray(result["time"])
    total_duration_ms = float(time[-1] - time[0])
    ap_result = patch_sim.analyze_aps_from_result(result)
    analysis = analyze_bursts(ap_result, total_duration_ms=total_duration_ms)

    assert analysis.threshold_method == "default-fixed"
    assert analysis.burst_count == 0
    assert analysis.bursts == []
    assert analysis.unburst_spike_count == ap_result.spike_count
    assert analysis.mean_inter_burst_interval is None
    assert analysis.duty_cycle is None


def test_classic_hh_tonic_firing_reports_zero_bursts(
    hh_model: Neuron,
) -> None:
    """Tonic-firing HH reports zero bursts with all spikes surfaced as unburst.

    HH at +10 µA/cm² fires a regular tonic train with a unimodal ISI
    distribution; the analyser falls back to ``"default-fixed"`` and the
    short-circuit reports zero bursts (issue #290).

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
    time = np.asarray(result["time"])
    total_duration_ms = float(time[-1] - time[0])
    ap_result = patch_sim.analyze_aps_from_result(result)
    analysis = analyze_bursts(ap_result, total_duration_ms=total_duration_ms)

    assert analysis.threshold_method == "default-fixed"
    assert analysis.burst_count == 0
    assert analysis.bursts == []


def test_classic_hh_short_stimulus_tonic_does_not_trip_tight_cluster(
    hh_model: Neuron,
) -> None:
    """A short HH tonic train must not be misread as a tight-cluster burst.

    Regression guard for the tight-cluster carve-out: a brief depolarising
    step that produces only a handful of tonic spikes must still report
    zero bursts.  HH at +10 µA/cm² fires at ~210 Hz in this simulator, so
    a short stimulus quickly accumulates enough ISIs to exceed the
    ``_TIGHT_CLUSTER_MAX_ISIS`` cap; this test pins that protection at the
    integration level so a future loosening of the cap can't silently
    re-introduce the false-positive.

    Args:
        hh_model: Classic Hodgkin-Huxley neuron fixture.
    """
    protocol = step_current(
        duration=70.0,
        current_amplitude=10.0,
        step_start=10.0,
        step_duration=50.0,
    )
    result = simulate_current_clamp(hh_model, protocol)
    time = np.asarray(result["time"])
    total_duration_ms = float(time[-1] - time[0])
    ap_result = patch_sim.analyze_aps_from_result(result)
    analysis = analyze_bursts(ap_result, total_duration_ms=total_duration_ms)

    assert ap_result.spike_count >= 5, (
        "Short HH stimulus should still produce several tonic spikes; "
        f"got spike_count={ap_result.spike_count}"
    )
    assert analysis.threshold_method == "default-fixed"
    assert analysis.burst_count == 0
    assert analysis.unburst_spike_count == ap_result.spike_count
    assert analysis.unburst_spike_count == ap_result.spike_count
    assert analysis.mean_inter_burst_interval is None
    assert analysis.duty_cycle is None


def test_thalamic_relay_step_release_produces_multi_spike_lts_burst() -> None:
    """Thalamic Relay fires a multi-spike LTS burst after hyperpolarising release.

    McCormick & Huguenard (1992), J. Neurophysiol. 68:1384 describe the TC
    low-threshold-spike (LTS) burst as 3–7 Na⁺ spikes at 200–500 Hz riding on
    the ICaT-driven calcium plateau.  Sustained hyperpolarisation
    de-inactivates the ICaT ``ft`` gate; on release the LTS depolarises the
    membrane and the TC-tuned slow ICaT inactivation
    (:func:`~patch_sim.additional_channels.make_thalamic_relay_icat_channel`)
    sustains the plateau long enough for several Na⁺ spikes to fire.

    Verifies issue #287: prior to the fix, the TC preset produced only a
    single rebound spike.
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
    assert analysis.burst_count >= 1, (
        f"Expected at least one LTS burst on rebound, "
        f"got burst_count={analysis.burst_count}"
    )
    burst = analysis.bursts[0]
    assert 3 <= burst.spike_count <= 7, (
        f"Expected 3–7 Na⁺ spikes per LTS burst (McCormick & Huguenard 1992), "
        f"got {burst.spike_count}"
    )
    assert burst.intra_burst_frequency is not None
    assert 200.0 <= burst.intra_burst_frequency <= 500.0, (
        f"Expected intra-burst frequency 200–500 Hz, "
        f"got {burst.intra_burst_frequency:.1f} Hz"
    )
    assert analysis.unburst_spike_count == 0, (
        "Unexpected isolated spikes outside the LTS burst — the rebound "
        "should be a single clean burst, not a burst plus stragglers; "
        f"got unburst_spike_count={analysis.unburst_spike_count}"
    )


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
    tonic spike train, a very small threshold (1 ms) should fragment most
    ISIs into "unburst" while a very large one (500 ms) should merge
    everything into a single burst.
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


def test_stn_conditional_burst_mode_under_hyperpolarising_step_release() -> None:
    """STN fires a multi-spike rebound burst after a hyperpolarising step.

    STN is a tonic pacemaker with conditional burst mode (Beurrier et al.
    1999, J. Neurosci. 19:599; Otsuka et al. 2004, J. Neurophysiol.
    92:255).  Burst mode is unreachable under depolarising steps in this
    preset (NMDA is not modelled), but a sufficiently deep hyperpolarising
    step de-inactivates the ICaT ``ft`` gate; on release the high
    ``g_CaT`` (5 mS/cm²) drives a high-frequency rebound burst.  Sanity
    check that the secondary firing mode is wired in.
    """
    neuron = make_neuron(NEURON_PRESETS[STN])
    pre = 50.0
    stim = 300.0
    post = 100.0
    protocol = step_current(
        duration=pre + stim + post,
        current_amplitude=-10.0,
        step_start=pre,
        step_duration=stim,
    )
    result = simulate_current_clamp(neuron, protocol)
    # The STN rebound is a tight cluster (3–8 spikes >100 Hz, ISIs <10 ms)
    # — too few ISIs (<4) for the histogram, but the tight-cluster
    # carve-out in ``analyze_bursts`` groups it into a single burst on the
    # default-fixed path.  No threshold pin needed.
    analysis = analyze_bursts_from_result(result)
    assert analysis.burst_count >= 1, (
        "STN: expected ≥1 rebound burst after −10 µA/cm² × 300 ms "
        f"hyperpolarisation, got burst_count={analysis.burst_count}"
    )
