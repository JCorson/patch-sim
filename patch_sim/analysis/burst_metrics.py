"""Burst detection and burst-metric extraction.

Provides :func:`analyze_bursts` for grouping detected action potentials into
bursts and computing standard burst metrics (burst count, spikes per burst,
intra-burst frequency, inter-burst interval, duty cycle), and
:func:`analyze_bursts_from_result` as a convenience wrapper for
:class:`~patch_sim.clamp_simulations.SimulationResult` structured arrays.

Burst detection consumes the spike list produced by
:func:`~patch_sim.analysis.ap_metrics.analyze_aps`.  Spikes are grouped
into bursts whose intra-burst inter-spike intervals are at or below a
threshold; bursts are separated by gaps where the inter-spike interval
strictly exceeds the threshold.
The threshold can be supplied by the caller, or auto-detected from a
log-spaced histogram of the inter-spike intervals.  When too few intervals
exist or the distribution is unimodal, the default-fixed path falls back
to the tight-cluster intra-burst cap (~12 ms): contiguous runs of ISIs at
or below this group into bursts; longer ISIs end them.  A size cap on the
default-fixed path (≤8 spikes per burst) prevents sustained high-frequency
tonic firing from being fabricated as one giant burst (issue #290).

For non-bursting neurons (sub-threshold protocols, tonic firing) the
analysis degrades gracefully: ``burst_count`` is 0 and aggregate metrics
are ``None``.

Data classes:
    BurstMetrics: Per-burst measurement record.
    BurstAnalysisResult: Aggregated burst analysis.
"""

# Needed so that the SimulationResult annotation in the *_from_result wrapper
# is not evaluated eagerly at import time — SimulationResult is only imported
# under TYPE_CHECKING to avoid a circular dependency.
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import numpy as np

from patch_sim.analysis.ap_metrics import (
    APAnalysisResult,
    analyze_aps_from_result,
)

if TYPE_CHECKING:
    from patch_sim.clamp_simulations import SimulationResult

logger = logging.getLogger(__name__)

#: Minimum number of ISIs required to attempt a histogram-based threshold.
_MIN_ISIS_FOR_HISTOGRAM: int = 4

#: Number of bins used in the log-spaced ISI histogram.
_HISTOGRAM_BINS: int = 20

#: Minimum bin separation between the two modes of a putative bimodal ISI
#: distribution.  A smaller separation is treated as unimodal.
_MIN_PEAK_SEPARATION_BINS: int = 2

#: Default minimum number of spikes required to call a group of consecutive
#: spikes a "burst".  Single spikes between long gaps are reported via
#: :attr:`BurstAnalysisResult.unburst_spike_count`.
_DEFAULT_MIN_SPIKES_PER_BURST: int = 2

#: Lower bound on the auto-detected threshold (ms).  Guards against degenerate
#: histogram cases where the minimum lands below the smallest observed ISI.
_AUTO_THRESHOLD_FLOOR_MS: float = 1e-3

#: Minimum bin count for a histogram bin to qualify as a mode.  Prevents
#: single-observation noise from being treated as a peak in narrow
#: unimodal distributions.
_MIN_PEAK_COUNT: int = 2

#: A valid bimodal distribution must have a valley with bin count below
#: this fraction of the smaller of its two peaks.  Rejects shallow dips
#: that are typical of noisy unimodal data.
_VALLEY_DEPTH_FRACTION: float = 0.5

#: Minimum ratio (in log10 ISI space) between the centres of the two
#: candidate peaks.  Genuine burst/pause distributions are typically
#: separated by an order of magnitude or more (e.g. 5 ms intra-burst vs
#: 200 ms inter-burst); narrow noisy unimodal data sees its peak centres
#: only fractions of a log unit apart.  log10(3.0) ≈ 0.48 corresponds to
#: a 3× ratio in linear ISI space, which comfortably accepts real bursts
#: while rejecting noise.
_MIN_PEAK_LOG_RATIO: float = 0.48

#: Maximum ISI (ms) for a default-fixed cluster to be treated as a single
#: tight burst rather than a non-burst tonic train.  12 ms ≈ 83 Hz sits
#: below the textbook tonic-firing range (Purkinje ~50 Hz, HH at 10 µA/cm²
#: ~60 Hz, ISIs 17–20 ms; Raman & Bean 1999) and well above burst-mode
#: ISIs (TC LTS 200–500 Hz / 2–5 ms; STN rebound >100 Hz / <10 ms;
#: McCormick & Huguenard 1992 J. Neurophysiol. 68:1384; Beurrier et al.
#: 1999 J. Neurosci. 19:599).  In this simulator the protected tonic
#: presets actually fire faster than the textbook rates (HH and Purkinje
#: both ~210 Hz at 10 µA/cm²), so the count cap below provides the
#: primary protection against fast sustained trains; the ISI cap guards
#: against slower textbook-rate tonic firing that future presets or
#: parameter sweeps may produce.
_TIGHT_CLUSTER_MAX_ISI_MS: float = 12.0

#: Maximum ISI count for the tight-cluster carve-out.  Real LTS-style
#: bursts top out at 7–8 spikes (TC LTS 3–7 spikes; STN rebound 3–8
#: spikes), i.e. ≤7 ISIs; any longer cluster is a sustained
#: high-frequency tonic train, not a single burst.
_TIGHT_CLUSTER_MAX_ISIS: int = 7


@dataclasses.dataclass
class BurstMetrics:
    """Metrics for a single detected burst.

    Attributes:
        index: Zero-based burst number within the trace.
        start_time: Peak time of the first spike in the burst (ms).
        end_time: Peak time of the last spike in the burst (ms).
        duration: ``end_time - start_time`` (ms); 0 for single-spike bursts.
        spike_count: Number of spikes contained in the burst.
        intra_burst_frequency: Mean firing rate within the burst (Hz),
            computed as ``(spike_count - 1) * 1000 / duration``.  ``None``
            for single-spike bursts (which have zero duration).
        mean_intra_burst_isi: Mean of the inter-spike intervals inside the
            burst (ms).  ``None`` for single-spike bursts.
    """

    index: int
    start_time: float
    end_time: float
    duration: float
    spike_count: int
    intra_burst_frequency: float | None
    mean_intra_burst_isi: float | None


@dataclasses.dataclass
class BurstAnalysisResult:
    """Complete burst analysis of a spike train.

    Attributes:
        burst_count: Number of detected bursts.
        bursts: Per-burst metrics.
        unburst_spike_count: Number of spikes that were not assigned to any
            burst because they were isolated by long inter-spike intervals.
        mean_spikes_per_burst: Mean ``spike_count`` across detected bursts,
            or ``None`` when no bursts were detected.
        mean_intra_burst_frequency: Mean ``intra_burst_frequency`` across
            multi-spike bursts, or ``None`` when no multi-spike burst exists.
        mean_inter_burst_interval: Mean gap between the end of one burst and
            the start of the next (ms), or ``None`` when fewer than two
            bursts were detected.
        duty_cycle: Fraction of the recording window spent inside a burst
            (``sum(burst.duration) / total_duration``).  ``None`` when no
            bursts were detected.
        isi_threshold_ms: The inter-spike interval threshold (ms) used to
            separate intra-burst ISIs from inter-burst gaps.  Always set,
            even when no bursts were detected, so downstream consumers can
            display the threshold that was applied.
        threshold_method: How :attr:`isi_threshold_ms` was determined.  One
            of ``"auto-histogram"``, ``"default-fixed"``, or ``"user"``.
            ``"default-fixed"`` carries the tight-cluster intra-burst cap
            (12 ms): the grouper still finds bursts on this path whenever
            a contiguous run of ISIs falls inside that cap.  See
            :func:`analyze_bursts` for the size-cap rule that prevents
            sustained tonic trains from being fabricated as bursts.
    """

    burst_count: int
    bursts: list[BurstMetrics]
    unburst_spike_count: int
    mean_spikes_per_burst: float | None
    mean_intra_burst_frequency: float | None
    mean_inter_burst_interval: float | None
    duty_cycle: float | None
    isi_threshold_ms: float
    threshold_method: str


def _estimate_isi_threshold(isis: list[float]) -> tuple[float, str]:
    """Estimate the intra/inter-burst ISI threshold from a list of ISIs.

    The estimator builds a histogram of ``log10(isis)`` and looks for a
    bimodal distribution.  When two non-adjacent peaks separated by a
    strictly lower minimum are present, the threshold is placed at the bin
    minimum (in linear ms).  Otherwise the function falls back to
    :data:`_TIGHT_CLUSTER_MAX_ISI_MS` and reports method
    ``"default-fixed"``: this is the same threshold the grouper applies
    when no histogram-derived threshold is available, so a contiguous run
    of intra-burst-territory ISIs (each ≤ 12 ms) groups into a burst while
    longer ISIs end it.

    Args:
        isis: Inter-spike intervals (ms).  Values must be positive.

    Returns:
        Tuple ``(threshold_ms, method)`` where ``method`` is one of
        ``"auto-histogram"`` or ``"default-fixed"``.
    """
    if len(isis) < _MIN_ISIS_FOR_HISTOGRAM:
        return _TIGHT_CLUSTER_MAX_ISI_MS, "default-fixed"

    isi_arr = np.asarray(isis, dtype=float)
    if np.any(isi_arr <= 0.0) or not np.all(np.isfinite(isi_arr)):
        return _TIGHT_CLUSTER_MAX_ISI_MS, "default-fixed"

    log_isi = np.log10(isi_arr)
    if float(np.ptp(log_isi)) < 1e-6:
        # All ISIs are essentially identical → no meaningful split possible.
        return _TIGHT_CLUSTER_MAX_ISI_MS, "default-fixed"

    hist, edges = np.histogram(log_isi, bins=_HISTOGRAM_BINS)

    # Find local maxima: bins strictly greater than the left neighbour and
    # at least as great as the right, populated above _MIN_PEAK_COUNT.  The
    # asymmetric ``> left`` / ``>= right`` tie-breaker is intentional: on a
    # plateau (e.g. ``[2, 3, 3, 2]``) it registers a single peak at the
    # plateau's left edge instead of two adjacent peaks, which would later
    # be filtered out by ``_MIN_PEAK_SEPARATION_BINS`` anyway.  The
    # _MIN_PEAK_COUNT floor rejects single-observation bumps that arise in
    # narrow unimodal distributions.
    n_bins = len(hist)
    peaks: list[int] = []
    for i in range(n_bins):
        if hist[i] < _MIN_PEAK_COUNT:
            continue
        left = hist[i - 1] if i > 0 else -1
        right = hist[i + 1] if i + 1 < n_bins else -1
        if hist[i] > left and hist[i] >= right:
            peaks.append(i)

    if len(peaks) < 2:
        return _TIGHT_CLUSTER_MAX_ISI_MS, "default-fixed"

    # Pick the two largest peaks that are both well-separated in bin index
    # and in log10 ISI space.  The latter ensures genuine order-of-magnitude
    # separation between bursts and gaps; narrow unimodal data with a ~1×
    # peak-to-peak ratio is rejected.
    peaks_sorted = sorted(peaks, key=lambda b: hist[b], reverse=True)
    left_peak = right_peak = -1
    bin_centres = 0.5 * (edges[:-1] + edges[1:])
    for i in range(len(peaks_sorted)):
        for j in range(i + 1, len(peaks_sorted)):
            a, b = sorted((peaks_sorted[i], peaks_sorted[j]))
            log_gap = float(bin_centres[b] - bin_centres[a])
            if b - a >= _MIN_PEAK_SEPARATION_BINS and log_gap >= _MIN_PEAK_LOG_RATIO:
                left_peak, right_peak = a, b
                break
        if left_peak != -1:
            break

    if left_peak == -1:
        return _TIGHT_CLUSTER_MAX_ISI_MS, "default-fixed"

    # Locate the minimum-count bin between the two peaks.  Require the
    # valley to be substantially lower than the smaller peak — shallow dips
    # are a hallmark of noisy unimodal data, not true bimodality.
    between = hist[left_peak + 1 : right_peak]
    if len(between) == 0:
        return _TIGHT_CLUSTER_MAX_ISI_MS, "default-fixed"
    min_offset = int(np.argmin(between))
    min_bin = left_peak + 1 + min_offset
    smaller_peak_count = min(hist[left_peak], hist[right_peak])
    if hist[min_bin] >= smaller_peak_count * _VALLEY_DEPTH_FRACTION:
        return _TIGHT_CLUSTER_MAX_ISI_MS, "default-fixed"

    # Threshold = midpoint of the chosen bin in log space, converted back to ms.
    log_threshold = 0.5 * (edges[min_bin] + edges[min_bin + 1])
    threshold_ms = float(10.0**log_threshold)
    threshold_ms = max(
        _AUTO_THRESHOLD_FLOOR_MS, min(threshold_ms, float(isi_arr.max()))
    )
    return threshold_ms, "auto-histogram"


def _group_spikes_into_bursts(
    peak_times: list[float],
    isis: list[float],
    threshold_ms: float,
    min_spikes_per_burst: int,
) -> tuple[list[BurstMetrics], int]:
    """Walk the spike list and partition spikes into bursts.

    A burst extends through every spike whose preceding ISI is at or below
    ``threshold_ms``.  An ISI strictly above the threshold ends the current
    burst; the next spike starts a new candidate burst.  Candidate bursts
    with fewer than ``min_spikes_per_burst`` spikes are discarded; their
    spikes are added to the unburst count.

    Args:
        peak_times: Spike peak times (ms), in order.
        isis: Inter-spike intervals (ms).  ``len(isis)`` must equal
            ``len(peak_times) - 1``.
        threshold_ms: Maximum intra-burst ISI (ms).
        min_spikes_per_burst: Minimum spikes for a group to count as a burst.

    Returns:
        Tuple ``(bursts, unburst_spike_count)``.  ``bursts`` is the list of
        per-burst metrics in chronological order; ``unburst_spike_count`` is
        the number of spikes that were dropped from short candidate groups.
    """
    bursts: list[BurstMetrics] = []
    unburst = 0

    n_spikes = len(peak_times)
    if n_spikes == 0:
        return bursts, 0

    # Walk through ISIs, breaking groups whenever an ISI exceeds threshold.
    # Groups are closed both when an inter-burst gap is found and at the
    # end of the spike list.  A 0-ISI single-spike trace falls through to
    # the "close the final group" line and is handled there.
    group_start = 0  # index into peak_times of the first spike in the group
    group_isis: list[float] = []

    def _close_group(end_idx: int) -> None:
        """Finalise the current candidate group ending at ``end_idx``.

        Promotes the group to a :class:`BurstMetrics` when its spike count
        meets ``min_spikes_per_burst``; otherwise its spikes are added to
        the running unburst count.

        Args:
            end_idx: Index (in ``peak_times``) of the last spike in the group.
        """
        nonlocal unburst
        spike_count = end_idx - group_start + 1
        if spike_count >= min_spikes_per_burst:
            start_t = float(peak_times[group_start])
            end_t = float(peak_times[end_idx])
            duration = end_t - start_t
            if spike_count > 1 and duration > 0:
                freq: float | None = (spike_count - 1) * 1000.0 / duration
                mean_isi: float | None = (
                    float(np.mean(group_isis)) if group_isis else None
                )
            else:
                freq = None
                mean_isi = None
            bursts.append(
                BurstMetrics(
                    index=len(bursts),
                    start_time=start_t,
                    end_time=end_t,
                    duration=duration,
                    spike_count=spike_count,
                    intra_burst_frequency=freq,
                    mean_intra_burst_isi=mean_isi,
                )
            )
        else:
            unburst += spike_count

    for i, isi in enumerate(isis):
        if isi <= threshold_ms:
            group_isis.append(float(isi))
        else:
            _close_group(i)
            group_start = i + 1
            group_isis = []

    # Close the final group at the last spike.
    _close_group(n_spikes - 1)

    return bursts, unburst


def _empty_result(
    isi_threshold_ms: float,
    threshold_method: str,
    unburst_spike_count: int = 0,
) -> BurstAnalysisResult:
    """Return a result representing a trace with no detected bursts.

    Args:
        isi_threshold_ms: Threshold that was applied (ms).
        threshold_method: How the threshold was chosen.
        unburst_spike_count: Spikes to surface as unburst.  Defaults to 0
            for the no-spikes call site; the default-fixed tonic
            short-circuit in :func:`analyze_bursts` (when the
            tight-cluster carve-out does not apply) passes the full spike
            count through.

    Returns:
        A :class:`BurstAnalysisResult` with ``burst_count`` of 0 and all
        aggregate metrics set to ``None``.
    """
    return BurstAnalysisResult(
        burst_count=0,
        bursts=[],
        unburst_spike_count=unburst_spike_count,
        mean_spikes_per_burst=None,
        mean_intra_burst_frequency=None,
        mean_inter_burst_interval=None,
        duty_cycle=None,
        isi_threshold_ms=isi_threshold_ms,
        threshold_method=threshold_method,
    )


def analyze_bursts(
    ap_result: APAnalysisResult,
    *,
    total_duration_ms: float,
    isi_threshold_ms: float | None = None,
    min_spikes_per_burst: int = _DEFAULT_MIN_SPIKES_PER_BURST,
) -> BurstAnalysisResult:
    """Group detected spikes into bursts and compute burst metrics.

    Args:
        ap_result: AP analysis from :func:`~patch_sim.analysis.analyze_aps`.
            Provides the spike list and ISIs that drive grouping.
        total_duration_ms: Length of the recording window (ms).  Used as the
            denominator of :attr:`BurstAnalysisResult.duty_cycle`.  Typically
            ``time[-1] - time[0]``.
        isi_threshold_ms: Optional user-supplied threshold (ms).  When
            ``None``, the function attempts a histogram-based auto-detect.
            If auto-detect cannot find a bimodal split it returns method
            ``"default-fixed"`` with threshold
            :data:`_TIGHT_CLUSTER_MAX_ISI_MS` (12 ms): contiguous runs of
            ISIs at or below this group into bursts, longer ISIs end
            them.  In the default-fixed path a size cap of
            :data:`_TIGHT_CLUSTER_MAX_ISIS` + 1 spikes per burst applies,
            so sustained high-frequency tonic firing is not fabricated
            into one giant burst (issue #290).  Caller-supplied
            (``"user"``) and histogram-derived (``"auto-histogram"``)
            thresholds bypass the size cap.
        min_spikes_per_burst: Minimum spike count for a group to count as a
            burst.  Defaults to 2; isolated spikes are tracked via
            :attr:`BurstAnalysisResult.unburst_spike_count`.

    Returns:
        A :class:`BurstAnalysisResult` with per-burst and aggregate metrics.
    """
    if isi_threshold_ms is not None:
        threshold = float(isi_threshold_ms)
        method = "user"
    else:
        threshold, method = _estimate_isi_threshold(list(ap_result.isis))

    if ap_result.spike_count == 0:
        return _empty_result(threshold, method)

    # 1-spike traces fall through to the grouper so they are correctly
    # routed to ``unburst_spike_count`` under the default
    # ``min_spikes_per_burst=2`` (or to a single 1-spike burst when the
    # caller has lowered the minimum).
    peak_times = [s.peak_time for s in ap_result.spikes]
    bursts, unburst = _group_spikes_into_bursts(
        peak_times,
        list(ap_result.isis),
        threshold,
        min_spikes_per_burst,
    )

    # In the default-fixed path, apply a size cap so sustained
    # high-frequency tonic firing (issue #290) is not fabricated into one
    # giant burst.  Real LTS-style bursts top out at 7–8 spikes (TC LTS
    # 3–7; STN rebound 3–8); anything longer with all-short ISIs is
    # tonic.  Caller-supplied (``"user"``) and histogram-derived
    # (``"auto-histogram"``) thresholds bypass the cap — those callers
    # have explicit knowledge of the trace shape.
    if method == "default-fixed" and isi_threshold_ms is None and bursts:
        survivors = [b for b in bursts if b.spike_count <= _TIGHT_CLUSTER_MAX_ISIS + 1]
        rejected_spike_count = sum(
            b.spike_count for b in bursts if b.spike_count > _TIGHT_CLUSTER_MAX_ISIS + 1
        )
        unburst += rejected_spike_count
        bursts = [
            BurstMetrics(
                index=i,
                start_time=b.start_time,
                end_time=b.end_time,
                duration=b.duration,
                spike_count=b.spike_count,
                intra_burst_frequency=b.intra_burst_frequency,
                mean_intra_burst_isi=b.mean_intra_burst_isi,
            )
            for i, b in enumerate(survivors)
        ]

    if not bursts:
        return BurstAnalysisResult(
            burst_count=0,
            bursts=[],
            unburst_spike_count=unburst,
            mean_spikes_per_burst=None,
            mean_intra_burst_frequency=None,
            mean_inter_burst_interval=None,
            duty_cycle=None,
            isi_threshold_ms=threshold,
            threshold_method=method,
        )

    spike_counts = [b.spike_count for b in bursts]
    mean_spikes = float(np.mean(spike_counts))

    freq_values = [
        b.intra_burst_frequency for b in bursts if b.intra_burst_frequency is not None
    ]
    mean_freq: float | None = float(np.mean(freq_values)) if freq_values else None

    if len(bursts) >= 2:
        ibis = [
            bursts[k + 1].start_time - bursts[k].end_time
            for k in range(len(bursts) - 1)
        ]
        mean_ibi: float | None = float(np.mean(ibis))
    else:
        mean_ibi = None

    if total_duration_ms > 0:
        duty_cycle: float | None = float(
            sum(b.duration for b in bursts) / total_duration_ms
        )
    else:
        duty_cycle = None

    return BurstAnalysisResult(
        burst_count=len(bursts),
        bursts=bursts,
        unburst_spike_count=unburst,
        mean_spikes_per_burst=mean_spikes,
        mean_intra_burst_frequency=mean_freq,
        mean_inter_burst_interval=mean_ibi,
        duty_cycle=duty_cycle,
        isi_threshold_ms=threshold,
        threshold_method=method,
    )


def analyze_bursts_from_result(
    result: SimulationResult,
    *,
    isi_threshold_ms: float | None = None,
    min_spikes_per_burst: int = _DEFAULT_MIN_SPIKES_PER_BURST,
    dvdt_threshold: float = 20.0,
    min_spike_height: float = -20.0,
) -> BurstAnalysisResult:
    """Extract burst metrics from a :class:`SimulationResult`.

    Convenience wrapper that runs :func:`analyze_aps_from_result` to obtain
    the spike list and then forwards to :func:`analyze_bursts`.  Callers
    that have already run :func:`~patch_sim.analyze_aps` (or
    :func:`~patch_sim.analyze_aps_from_result`) on the same trace should
    prefer :func:`analyze_bursts` directly to avoid re-detecting spikes.

    Args:
        result: A structured NumPy array returned by
            :func:`~patch_sim.simulate_current_clamp` or its voltage-clamp
            equivalents.  Must contain ``"time"`` and ``"voltage"`` fields.
        isi_threshold_ms: Optional user-supplied threshold (ms).
        min_spikes_per_burst: Minimum spike count for a burst.
        dvdt_threshold: dV/dt threshold passed to spike detection (mV/ms).
        min_spike_height: Minimum peak voltage for spike detection (mV).

    Returns:
        A :class:`BurstAnalysisResult`.
    """
    ap_result = analyze_aps_from_result(
        result,
        dvdt_threshold=dvdt_threshold,
        min_spike_height=min_spike_height,
    )
    time = result["time"]
    total_duration_ms = float(time[-1] - time[0]) if len(time) > 1 else 0.0
    return analyze_bursts(
        ap_result,
        total_duration_ms=total_duration_ms,
        isi_threshold_ms=isi_threshold_ms,
        min_spikes_per_burst=min_spikes_per_burst,
    )
