"""Shared helper: pin a post-release LTS rebound burst from the spike train.

The default-fixed burst detector's tight-cluster size cap (issue #290)
reclassifies the TRN LTS rebound — which runs to ~15 Na⁺ spikes once the
depolarized h-gate shift accelerates Na⁺ recovery from inactivation — as
sustained tonic firing.  As a result :func:`analyze_bursts_from_result`
surfaces only the short cold-start cluster and never the rebound itself, so
matching against detector output cannot distinguish a real rebound from a
phenotype-matching cold-start cluster.  Tests pin the rebound directly from
the voltage trace instead, via this helper.
"""

import numpy as np


def post_release_rebound(
    time_ms: np.ndarray,
    voltage_mv: np.ndarray,
    release_ms: float,
    isi_max_ms: float = 8.0,
) -> tuple[int, float | None]:
    """Return the longest post-release high-frequency spike run.

    Detects AP peaks (local maxima above 0 mV), keeps those at or after
    ``release_ms``, and returns the longest run of consecutive peaks joined
    by ISIs ≤ ``isi_max_ms`` together with its mean intra-burst frequency.

    Args:
        time_ms: Simulation time axis in ms.
        voltage_mv: Membrane voltage in mV.
        release_ms: Time the hyperpolarizing step releases, in ms.
        isi_max_ms: Maximum ISI (ms) for two consecutive spikes to count as
            part of the same rebound burst.

    Returns:
        Tuple ``(spike_count, intra_burst_frequency_hz)`` for the longest
        post-release run.  ``spike_count`` is 0 (frequency ``None``) when
        there are no post-release spikes; frequency is ``None`` for a
        degenerate single-spike run.
    """
    t = np.asarray(time_ms)
    v = np.asarray(voltage_mv)
    is_peak = np.zeros(v.shape, dtype=bool)
    is_peak[1:-1] = (v[1:-1] > v[:-2]) & (v[1:-1] > v[2:]) & (v[1:-1] > 0.0)
    peak_times = [float(pt) for pt in t[is_peak].tolist() if float(pt) >= release_ms]
    if not peak_times:
        return 0, None
    runs: list[list[float]] = [[peak_times[0]]]
    for prev, nxt in zip(peak_times[:-1], peak_times[1:]):
        if nxt - prev <= isi_max_ms:
            runs[-1].append(nxt)
        else:
            runs.append([nxt])
    run: list[float] = []
    for candidate in runs:
        if len(candidate) > len(run):
            run = candidate
    if len(run) < 2:
        return len(run), None
    span_ms = run[-1] - run[0]
    freq = (len(run) - 1) / span_ms * 1000.0 if span_ms > 0 else None
    return len(run), freq
