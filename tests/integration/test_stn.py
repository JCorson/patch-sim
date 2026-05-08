"""Behavioral tests for the Subthalamic Nucleus (STN) neuron preset.

Pins the STN tonic-pacemaker phenotype: 5–50 Hz autonomous firing and AP
shape consistent with Bevan & Wilson (1999); the conditional burst mode
is covered by ``test_stn_conditional_burst_mode_under_…`` in
``test_burst_metrics_simulation.py``.  Bands cite Beurrier et al. (1999)
and Bevan & Wilson (1999).
"""

import numpy as np
import pytest

from patch_sim.analysis.ap_metrics import analyze_aps_from_result
from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ, simulate_current_clamp
from patch_sim.neuron import Neuron
from patch_sim.presets import make_stn
from tests.integration._ap_shape import assert_ap_shape

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# 1 s of zero-current simulation captures the STN's autonomous tonic
# pacemaker phenotype directly: at the 5–50 Hz Bevan & Wilson (1999) rate
# this yields 5–50 spikes — sufficient for both rate and AP-shape stats.
_STN_SIMULATION_MS = 1000.0
_STN_REFERENCE = "Beurrier et al. 1999 / Bevan & Wilson 1999"


@pytest.fixture
def stn_neuron() -> Neuron:
    """STN neuron instance for all tests in this module."""
    return make_stn()


@pytest.fixture
def stn_ap_shape_result(stn_neuron: Neuron):
    """AP analysis from a zero-current spontaneous-firing trace.

    All AP-shape assertions in this module run against the autonomous
    trace (the cell's intrinsic operating point per Bevan & Wilson 1999).
    The driven 2 µA/cm² operating point is exercised by
    ``test_repetitive_firing_preset`` in ``test_preset_protocols.py``.
    """
    zero_current = np.zeros(_ms_to_samples(_STN_SIMULATION_MS) + 1)
    result = simulate_current_clamp(stn_neuron, current_external=zero_current)
    return analyze_aps_from_result(result)


def _ms_to_samples(ms: float) -> int:
    """Convert ms to simulation samples at SIM_SAMPLING_FREQ."""
    return int(ms * SIM_SAMPLING_FREQ / 1000.0)


# ---------------------------------------------------------------------------
# Spontaneous tonic pacemaking — Bevan & Wilson (1999), 5–50 Hz in vitro.
# ---------------------------------------------------------------------------


def test_stn_spontaneous_pacemaking(stn_ap_shape_result) -> None:
    """STN fires autonomously at 5–50 Hz without injected current.

    Bevan & Wilson (1999) characterise the STN as a tonic pacemaker
    sustained intrinsically by INaP and Ih.  The driven 2 µA/cm²
    operating point is exercised by ``test_repetitive_firing_preset``
    in ``test_preset_protocols.py``.
    """
    assert_ap_shape(
        stn_ap_shape_result,
        reference=_STN_REFERENCE,
        firing_rate_hz=(5.0, 50.0),
        min_spike_count=5,
    )


def test_stn_tonic_firing_is_regular_single_spikes(stn_ap_shape_result) -> None:
    """STN tonic firing is regular single spikes, not doublets/spikelets.

    Bevan & Wilson (1999) describe the STN as a regular tonic pacemaker;
    Hallworth, Wilson & Bevan (2003), J. Neurosci. 23:7525, report ISI
    CV ≈ 0.05–0.15 in tonic mode.  Doublets/triplets only appear in
    burst mode (Beurrier et al. 1999, NMDA-induced plateau or rebound
    bursts), not at zero current.

    This test catches two failure modes that the AP-shape tests average
    away (issue #326):

      • Bimodal ISI (full spike followed by an abortive spikelet a few
        ms later) → high CV(ISI).
      • Spike peaks below 0 mV (abortive overshoot when fast-Na ``h``
        has not fully de-inactivated) — Bevan & Wilson 1999 put the
        full STN AP peak in the +5 to +25 mV band.
    """
    isis = np.asarray(stn_ap_shape_result.isis)
    assert isis.size >= 4, (
        f"need ≥5 spikes for a stable CV(ISI) estimate; got "
        f"{stn_ap_shape_result.spike_count} spikes"
    )
    cv_isi = float(isis.std(ddof=0) / isis.mean())
    assert cv_isi < 0.2, (
        f"[Hallworth, Wilson & Bevan 2003] STN tonic ISI is regular "
        f"(CV 0.05–0.15); got CV(ISI)={cv_isi:.3f} over ISIs={isis.tolist()}"
    )

    bad_peaks = [
        (s.index, s.peak_voltage)
        for s in stn_ap_shape_result.spikes
        if s.peak_voltage <= 0.0
    ]
    assert not bad_peaks, (
        f"[Bevan & Wilson 1999] every STN tonic spike should overshoot "
        f"0 mV; got abortive peaks (index, peak_mV)={bad_peaks}"
    )


# ---------------------------------------------------------------------------
# AP shape — STN tonic phenotype, Bevan & Wilson (1999).
# ---------------------------------------------------------------------------


def test_stn_ap_threshold_in_stn_range(stn_ap_shape_result) -> None:
    """Mean AP threshold falls in the STN range (−60 to −40 mV)."""
    assert_ap_shape(
        stn_ap_shape_result,
        reference=_STN_REFERENCE,
        threshold_mv=(-60.0, -40.0),
    )


def test_stn_ap_peak_voltage_in_stn_range(stn_ap_shape_result) -> None:
    """Mean AP peak voltage falls within the STN range (0 to +30 mV).

    STN APs have a comparatively low overshoot relative to other neurons —
    Bevan & Wilson (1999) report typical AP amplitudes of 60–80 mV from
    a threshold near −55 mV, putting the peak in the +5 to +25 mV range.
    """
    assert_ap_shape(
        stn_ap_shape_result,
        reference=_STN_REFERENCE,
        peak_mv=(0.0, 30.0),
    )


def test_stn_ap_half_width_in_stn_range(stn_ap_shape_result) -> None:
    """Mean AP half-width matches the STN tonic phenotype (0.4–1.2 ms)."""
    assert_ap_shape(
        stn_ap_shape_result,
        reference=_STN_REFERENCE,
        half_width_ms=(0.4, 1.2),
    )


def test_stn_ap_ahp_depth_in_stn_range(stn_ap_shape_result) -> None:
    """Mean AHP depth falls within the STN range (−85 to −55 mV)."""
    assert_ap_shape(
        stn_ap_shape_result,
        reference=_STN_REFERENCE,
        ahp_mv=(-85.0, -55.0),
    )


# ---------------------------------------------------------------------------
# Slow INaP inactivation engages during sustained drive — issue #324.
# ---------------------------------------------------------------------------


def test_stn_inap_slow_inactivation_engages_during_drive(
    stn_neuron: Neuron,
) -> None:
    """The sNaP gate closes during +5 µA/cm² × 200 ms drive.

    Direct mechanism check for #324: the Magistretti & Alonso 1999 slow
    inactivation gate baked into ``make_inap_channel`` must be doing real
    work during sustained suprathreshold drive.  By the end of the step,
    sNaP should have lost at least half its rest availability so the
    persistent Na⁺ contribution to the depol-block plateau is suppressed.

    The 0.5 threshold is loose because in STN sNaP co-acts with the
    fast-Na sNa gate and K_ATP (all part of the #324 fix) to rescue the
    cell from the plateau; once voltage starts recovering during the
    step, sNaP starts recovering too, so it never reaches the deep
    ≈ 0.05 plateau value it would attain if the cell hung at −15 mV
    indefinitely.
    """
    n_pre = _ms_to_samples(100.0)
    n_step = _ms_to_samples(200.0)
    n_post = _ms_to_samples(50.0)
    current = np.concatenate(
        [
            np.zeros(n_pre),
            np.full(n_step, 5.0),
            np.zeros(n_post + 1),
        ]
    )
    result = simulate_current_clamp(stn_neuron, current_external=current)
    sNaP_at_rest = float(result["sNaP"][n_pre - 1])
    sNaP_at_step_end = float(result["sNaP"][n_pre + n_step - 1])
    fraction_inactivated = 1.0 - sNaP_at_step_end / sNaP_at_rest
    assert fraction_inactivated > 0.5, (
        f"sNaP did not meaningfully inactivate: rest={sNaP_at_rest:.3f}, "
        f"step end={sNaP_at_step_end:.3f}, "
        f"fraction inactivated={fraction_inactivated:.2f} (expected > 0.5)"
    )


def test_stn_fast_na_slow_inactivation_engages_during_drive(
    stn_neuron: Neuron,
) -> None:
    """The sNa gate (fast Na slow inactivation) closes during +5 µA/cm² × 200 ms.

    Direct mechanism check for #324 fix part 2: the Fleidervish & Gutnick
    1996 / Mickus et al. 1999 / Do & Bean 2003 slow voltage-dependent
    inactivation gate baked into ``make_stn_na_channel`` must lose at
    least half its rest availability by the end of a sustained
    suprathreshold step, abolishing the residual h-tail that pinned the
    cell at −15 mV.
    """
    n_pre = _ms_to_samples(100.0)
    n_step = _ms_to_samples(200.0)
    n_post = _ms_to_samples(50.0)
    current = np.concatenate(
        [
            np.zeros(n_pre),
            np.full(n_step, 5.0),
            np.zeros(n_post + 1),
        ]
    )
    result = simulate_current_clamp(stn_neuron, current_external=current)
    sNa_at_rest = float(result["sNa"][n_pre - 1])
    sNa_at_step_end = float(result["sNa"][n_pre + n_step - 1])
    fraction_inactivated = 1.0 - sNa_at_step_end / sNa_at_rest
    assert fraction_inactivated > 0.5, (
        f"sNa did not meaningfully inactivate: rest={sNa_at_rest:.3f}, "
        f"step end={sNa_at_step_end:.3f}, "
        f"fraction inactivated={fraction_inactivated:.2f} (expected > 0.5)"
    )


def test_stn_katp_engages_during_drive(stn_neuron: Neuron) -> None:
    """K_ATP opens during +5 µA/cm² × 200 ms drive and stays closed at rest.

    Direct mechanism check for #324 fix part 3: the K_ATP gate
    (Stanford & Lacey 1996; Bevan & Wilson 1999; Hahn & McIntyre 2010)
    must engage strongly during sustained suprathreshold drive so the
    outward K⁺ current it provides can dominate the residual fast-Na
    drive at the depolarised plateau.  Subthreshold availability must
    stay near zero or autonomous tonic firing would silence.
    """
    n_pre = _ms_to_samples(100.0)
    n_step = _ms_to_samples(200.0)
    n_post = _ms_to_samples(50.0)
    current = np.concatenate(
        [
            np.zeros(n_pre),
            np.full(n_step, 5.0),
            np.zeros(n_post + 1),
        ]
    )
    result = simulate_current_clamp(stn_neuron, current_external=current)
    kATP_at_rest = float(result["kATP"][n_pre - 1])
    kATP_at_step_end = float(result["kATP"][n_pre + n_step - 1])
    # The "rest" sample is one slice of an autonomously oscillating
    # trajectory; with τ_kATP ≈ 400 ms and ~50 ms AP cycle, kATP
    # time-averages to a few percent during normal pacemaking rather than
    # being strictly zero, but stays low enough not to silence firing.
    assert kATP_at_rest < 0.05, (
        f"kATP not closed at autonomous rest: rest={kATP_at_rest:.3f} (expected < 0.05)"
    )
    # During sustained drive kATP doesn't reach steady-state because sNaP
    # and sNa repolarise the cell before τ_kATP (≈400 ms at V½) has time
    # to fully activate; engagement is still several-fold above rest.
    # The 0.08 floor reflects the post-#326 tuning, where a halved INaP
    # makes the depolarisation-block plateau shallower and shorter-lived
    # — kATP engages a few-fold above rest rather than reaching the
    # ~0.19 it attained when INaP was holding the cell at the plateau
    # for longer.  The third assertion below (>4× rest) is the primary
    # mechanism check; this absolute floor just guards against silence.
    assert kATP_at_step_end > 0.08, (
        f"kATP did not meaningfully open during drive: "
        f"step end={kATP_at_step_end:.3f} (expected > 0.08)"
    )
    assert kATP_at_step_end > 4.0 * kATP_at_rest, (
        f"kATP step-end {kATP_at_step_end:.3f} not meaningfully above "
        f"autonomous-firing rest {kATP_at_rest:.3f} (expected > 4× rest)"
    )


# ---------------------------------------------------------------------------
# Depolarisation-block recovery — issue #324 regression.
# ---------------------------------------------------------------------------


def test_stn_recovers_from_sustained_suprathreshold_drive(
    stn_neuron: Neuron,
) -> None:
    """STN repolarises after +5 µA/cm² × 200 ms (full #324 regression).

    Regression test for #324: the three opt-in mechanisms (INaP sNaP,
    fast-Na sNa, K_ATP) cooperate so the membrane escapes the
    depolarisation-block plateau seeded by sustained suprathreshold
    drive at the F-I sweep upper bound (+5 µA/cm²).  The cell should
    settle below −50 mV during the last 200 ms of a 700 ms post-stimulus
    epoch, confirming a return to the autonomous-oscillation regime
    (cycling around −60 mV) rather than hanging at −15 mV.
    """
    n_pre = _ms_to_samples(100.0)
    n_step = _ms_to_samples(200.0)
    n_post = _ms_to_samples(700.0)
    current = np.concatenate(
        [
            np.zeros(n_pre),
            np.full(n_step, 5.0),
            np.zeros(n_post + 1),
        ]
    )
    result = simulate_current_clamp(stn_neuron, current_external=current)
    tail = np.asarray(result["voltage"][-_ms_to_samples(200.0) :])
    mean_v = float(tail.mean())
    assert mean_v < -50.0, (
        f"STN failed to recover from +5 µA/cm² × 200 ms; "
        f"mean V in last 200 ms = {mean_v:.2f} mV"
    )
