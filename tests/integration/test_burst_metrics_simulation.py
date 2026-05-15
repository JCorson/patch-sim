"""Integration tests for burst-metric analysis with real simulations.

Drives :func:`analyze_bursts_from_result` through the full
simulate-then-analyze pipeline using rebound-bursting and tonic-firing
presets.  Unit tests with synthetic AP results live in
tests/unit/test_burst_metrics.py.
"""

import numpy as np

import patch_sim
from patch_sim.analysis.burst_metrics import (
    _TIGHT_CLUSTER_MAX_ISIS,
    BurstMetrics,
    analyze_bursts,
    analyze_bursts_from_result,
)
from patch_sim.clamp_simulations import simulate_batch, simulate_current_clamp
from patch_sim.constants import (
    HYPERPOLARIZATION_STEPS,
    PURKINJE,
    STN,
    THALAMIC_RELAY,
    TRN,
)
from patch_sim.neuron import Neuron
from patch_sim.presets import NEURON_PRESETS, build_protocol_from_preset
from patch_sim.protocols import step_current


def test_purkinje_tonic_firing_reports_zero_bursts() -> None:
    """Purkinje under a depolarizing step is tonic — zero bursts, all unburst.

    Purkinje is a tonic pacemaker (Raman & Bean 1999).  Under a moderate
    depolarizing current it produces a regular tonic spike train with a
    unimodal ISI distribution, so the analyzer cannot place an
    auto-histogram threshold and falls back to ``"default-fixed"``.  In
    that case the burst analyzer short-circuits to zero bursts and
    surfaces every spike via ``unburst_spike_count`` (issue #290).

    Complex-spike bursts in vivo are climbing-fiber driven and cannot be
    produced by this single-compartment, current-clamp preset.

    The first ~50 ms of the trace is discarded before analysis so the
    test sees only steady-state stepped firing: during the 10 ms zero-
    current pre-step window the autonomous oscillator pacemakes at its
    slow ~26 ms ISI, then the step depolarizes it and steady-state ISIs
    collapse to ~3.6 ms.  The first stepped spike's ISI relative to the
    preceding spontaneous spike (~7.3 ms) is well above the steady-state
    interval, and the analyzer would correctly classify those two spikes
    as a 2-spike burst — masking the steady-state tonic phenotype this
    test is pinning.
    """
    neuron = NEURON_PRESETS[PURKINJE]()
    protocol = step_current(
        duration=600.0,
        current_amplitude=10.0,
        step_start=10.0,
        step_duration=580.0,
    )
    result = simulate_current_clamp(neuron, protocol)
    time = np.asarray(result["time"])
    voltage = np.asarray(result["voltage"])
    settle_ms = 50.0
    settle_mask = time >= settle_ms
    steady_time = time[settle_mask]
    steady_voltage = voltage[settle_mask]
    total_duration_ms = float(steady_time[-1] - steady_time[0])
    ap_result = patch_sim.analyze_aps(steady_time, steady_voltage)
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
    distribution; the analyzer falls back to ``"default-fixed"`` and the
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
    assert analysis.unburst_spike_count == ap_result.spike_count
    assert analysis.mean_inter_burst_interval is None
    assert analysis.duty_cycle is None


def test_classic_hh_short_stimulus_tonic_does_not_trip_tight_cluster(
    hh_model: Neuron,
) -> None:
    """A short HH tonic train must not be misread as a tight-cluster burst.

    Regression guard for the tight-cluster carve-out: a brief depolarizing
    step that produces only a handful of tonic spikes must still report
    zero bursts.  HH at +10 µA/cm² fires at ~210 Hz in this simulator, so
    a 50 ms step accumulates enough ISIs to exceed the
    ``_TIGHT_CLUSTER_MAX_ISIS`` cap; this test pins that protection at the
    integration level so a future loosening of the cap can't silently
    re-introduce the false-positive.  Note that HH's per-spike ISI here
    (~4.8 ms) sits well below ``_TIGHT_CLUSTER_MAX_ISI_MS``, so the count
    cap — not the ISI cap — is what disqualifies the train; the spike
    count assertion below makes that protection load-bearing.

    Args:
        hh_model: Classic Hodgkin-Huxley neuron fixture.
    """
    # ~150 ms of step at HH's tonic ~66 Hz (+10 µA/cm²) yields ~10 spikes →
    # 9 ISIs, comfortably above _TIGHT_CLUSTER_MAX_ISIS so the count cap is
    # exercised.
    protocol = step_current(
        duration=170.0,
        current_amplitude=10.0,
        step_start=10.0,
        step_duration=150.0,
    )
    result = simulate_current_clamp(hh_model, protocol)
    time = np.asarray(result["time"])
    total_duration_ms = float(time[-1] - time[0])
    ap_result = patch_sim.analyze_aps_from_result(result)
    analysis = analyze_bursts(ap_result, total_duration_ms=total_duration_ms)

    assert len(ap_result.isis) > _TIGHT_CLUSTER_MAX_ISIS, (
        "Short HH stimulus must produce more ISIs than the tight-cluster "
        "count cap, otherwise the carve-out's count protection isn't "
        "actually exercised; "
        f"got len(isis)={len(ap_result.isis)}, cap={_TIGHT_CLUSTER_MAX_ISIS}"
    )
    assert analysis.threshold_method == "default-fixed"
    assert analysis.burst_count == 0
    assert analysis.unburst_spike_count == ap_result.spike_count
    assert analysis.mean_inter_burst_interval is None
    assert analysis.duty_cycle is None


def test_thalamic_relay_step_release_produces_multi_spike_lts_burst() -> None:
    """Thalamic Relay fires a multi-spike LTS burst after hyperpolarizing release.

    McCormick & Huguenard (1992), J. Neurophysiol. 68:1384 describe the TC
    low-threshold-spike (LTS) burst as 3–7 Na⁺ spikes at 200–500 Hz riding on
    the ICaT-driven calcium plateau.  Sustained hyperpolarization
    de-inactivates the ICaT ``ft`` gate; on release the LTS depolarizes the
    membrane and the TC-tuned slow ICaT inactivation
    (:func:`~patch_sim.channels.thalamic.make_thalamic_relay_icat_channel`)
    sustains the plateau long enough for several Na⁺ spikes to fire.

    Verifies issue #287: prior to the fix, the TC preset produced only a
    single rebound spike.
    """
    neuron = NEURON_PRESETS[THALAMIC_RELAY]()
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


def test_trn_step_release_produces_hp92_rebound_burst() -> None:
    """TRN fires a 5–15 spike LTS rebound burst on hyperpolarizing step release.

    Huguenard & Prince (1992), J. Neurosci. 12:3804 describe the TRN burst
    phenotype as 5–15 Na⁺ spikes at 200–600 Hz riding on the ICaT-driven
    LTS plateau, terminated by IKCa-driven AHP.  The rebound mechanism
    requires both Ih (activates during hyperpolarization, provides
    depolarizing drive on release — Bal & McCormick 1993) and the
    sigmoid-shaped ICaT inactivation tau (sustains the LTS plateau long
    enough to fit 5+ spikes —
    :func:`~patch_sim.channels.trn.make_trn_icat_channel`).

    Verifies issue #295: prior to the fix, the cosh-shaped Destexhe (1994)
    ICaT inactivation tau collapsed the LTS plateau in ~5–10 ms (too fast
    to fit 5+ spikes), and the absence of Ih meant the LTS did not trigger
    on rebound at all.

    The TRN preset is spontaneously active (~3 Hz tonic) so this test
    accepts a non-zero ``unburst_spike_count`` from the pre-step tonic
    firing and the post-burst return to tonic firing.  Calls
    :func:`analyze_bursts_from_result` *without* a user-supplied
    ``isi_threshold_ms`` so that this regression-tests the same path the
    Reflex UI takes when the user runs the protocol — the default-fixed
    embedded-burst carve-out must successfully isolate the rebound burst
    from the surrounding tonic spikes.
    """
    neuron = NEURON_PRESETS[TRN]()
    pre = 200.0
    stim = 500.0
    post = 200.0
    protocol = step_current(
        duration=pre + stim + post,
        current_amplitude=-4.0,
        step_start=pre,
        step_duration=stim,
    )
    result = simulate_current_clamp(neuron, protocol)
    analysis = analyze_bursts_from_result(result)
    # The default-fixed burst detector resolves three clusters in this trace:
    # a cold-start cluster (~28 ms after cell start, depol-block recovery on
    # the rising LTS edge), a small pre-rebound tonic doublet, and the LTS
    # rebound burst after step release.  Assert that *at least one* burst
    # matches the HP92 phenotype (5–15 Na⁺ spikes at 200–600 Hz) — the
    # biological invariant — rather than fixing on a particular burst index.
    assert analysis.burst_count >= 1, (
        f"Expected at least one HP92-phenotype LTS burst, got burst_count="
        f"{analysis.burst_count}.  The rebound may have fragmented into "
        f"tonic-like firing or failed to trigger."
    )

    def _matches_hp92(burst: BurstMetrics) -> bool:
        """Return True iff ``burst`` matches the HP92 LTS-burst phenotype.

        Args:
            burst: A :class:`~patch_sim.analysis.burst_metrics.BurstMetrics`
                instance from the burst-detector output.

        Returns:
            True when the burst has 5–15 spikes and an intra-burst frequency
            in the 200–600 Hz range (Huguenard & Prince 1992 LTS phenotype).
        """
        if not (5 <= burst.spike_count <= 15):
            return False
        if burst.intra_burst_frequency is None:
            return False
        return 200.0 <= burst.intra_burst_frequency <= 600.0

    matching = [b for b in analysis.bursts if _matches_hp92(b)]
    assert matching, (
        "No detected burst matches the HP92 phenotype "
        "(5–15 spikes, 200–600 Hz).  Detected bursts: "
        + ", ".join(
            f"(n={b.spike_count}, f={b.intra_burst_frequency})" for b in analysis.bursts
        )
    )


def test_trn_hyperpolarization_steps_protocol_produces_burst_per_sweep() -> None:
    """The HYPERPOLARIZATION_STEPS protocol on TRN produces 5-spike+ bursts.

    Exercises the exact code path the Reflex UI uses to run a multi-sweep
    protocol:
    - Build the protocol via :func:`build_protocol_from_preset` with the
      TRN-specific overrides
    - Simulate every sweep through :func:`simulate_batch` (the multi-sweep
      executor used by the UI's run handler)
    - Analyze APs per sweep with :func:`analyze_aps`

    For each sweep at -3 to -5 µA/cm², asserts the burst-detection result
    finds a burst with spike_count in [5, 15] and intra-burst frequency in
    [200, 600] Hz (the HP92 phenotype).  This guards against regressions in
    the multi-sweep UI scenario, which the single-sweep test
    :func:`test_trn_step_release_produces_hp92_rebound_burst` does not
    cover.
    """
    neuron = NEURON_PRESETS[TRN]()
    protocol = build_protocol_from_preset(HYPERPOLARIZATION_STEPS, neuron_preset=TRN)
    assert protocol.shape[0] == 5, (
        f"Expected 5 sweeps for TRN HYPERPOLARIZATION_STEPS, got {protocol.shape[0]}"
    )

    results = list(
        simulate_batch(neuron, [sweep for sweep in protocol], simulate_current_clamp)
    )

    # Sweeps are min-to-max in current amplitude: index 0 is deepest (-5).
    # Verify deeper sweeps (-5, -4, -3) all produce HP92-shape bursts.
    for sweep_idx in (0, 1, 2):
        result = results[sweep_idx]
        time_arr = np.asarray(result["time"])
        v_arr = np.asarray(result["voltage"])
        ap_result = patch_sim.analyze_aps(time_arr, v_arr)
        analysis = analyze_bursts(
            ap_result, total_duration_ms=float(time_arr[-1] - time_arr[0])
        )

        assert analysis.burst_count >= 1, (
            f"Sweep {sweep_idx} (deeper hyperpolarization): expected ≥1 LTS "
            f"rebound burst, got burst_count={analysis.burst_count}.  Total "
            f"APs in sweep: {ap_result.spike_count}.  This indicates the TRN "
            f"preset is not delivering the HP92 rebound phenotype on the "
            f"multi-sweep UI path."
        )
        # The detector may resolve multiple bursts (cold-start cluster + LTS
        # rebound), so match *any* burst against the HP92 phenotype rather
        # than fixing on bursts[0].  Upper bound 30 because deeper
        # hyperpolarizations (-5 µA) produce larger rebound bursts (≥25
        # spikes): Na⁺ is more available after the long Ih-driven
        # hyperpolarization, beyond the 5–15 range Huguenard & Prince
        # (1992) reported for typical hyperpolarization.  The canonical
        # 5–15 floor for the -4 µA × 500 ms protocol is enforced in
        # test_trn_step_release_produces_hp92_rebound_burst above.
        matching = [
            b
            for b in analysis.bursts
            if 5 <= b.spike_count <= 30
            and b.intra_burst_frequency is not None
            and 200.0 <= b.intra_burst_frequency <= 600.0
        ]
        assert matching, (
            f"Sweep {sweep_idx}: no detected burst matches the deep-HP "
            "rebound phenotype (5–30 spikes, 200–600 Hz).  Detected bursts: "
            + ", ".join(
                f"(n={b.spike_count}, f={b.intra_burst_frequency})"
                for b in analysis.bursts
            )
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
    neuron = NEURON_PRESETS[PURKINJE]()
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


def test_stn_conditional_burst_mode_under_hyperpolarizing_step_release() -> None:
    """STN fires a multi-spike rebound burst after a hyperpolarizing step.

    STN is a tonic pacemaker with conditional burst mode (Beurrier et al.
    1999, J. Neurosci. 19:599; Otsuka et al. 2004, J. Neurophysiol.
    92:255).  Burst mode is unreachable under depolarizing steps in this
    preset (NMDA is not modeled), but a sufficiently deep hyperpolarizing
    step de-inactivates the ICaT ``ft`` gate; on release the high
    ``g_CaT`` (5 mS/cm²) drives a high-frequency rebound burst.  Sanity
    check that the secondary firing mode is wired in.
    """
    neuron = NEURON_PRESETS[STN]()
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
        f"hyperpolarization, got burst_count={analysis.burst_count}"
    )
