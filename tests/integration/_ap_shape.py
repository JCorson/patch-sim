"""Shared AP-shape assertion helper for integration tests.

The :func:`assert_ap_shape` helper standardises how preset tests assert that a
neuron's mean AP-shape metrics fall within literature-cited tolerance bands.
Each band kwarg is a ``(low, high)`` tuple; passing ``None`` skips that metric.
The mandatory ``reference`` string is reproduced in every failure message so
that future readers can audit the chosen bounds against the source paper.

Lives in ``_ap_shape.py`` (leading underscore = test-internal) rather than
``conftest.py`` because pytest reserves ``conftest.py`` for fixtures and
plugins; importing plain functions from it is an anti-pattern.
"""

from patch_sim.analysis.ap_metrics import APAnalysisResult

_Band = tuple[float, float]


def assert_ap_shape(
    ap_result: APAnalysisResult,
    *,
    reference: str,
    half_width_ms: _Band | None = None,
    peak_mv: _Band | None = None,
    ahp_mv: _Band | None = None,
    threshold_mv: _Band | None = None,
    firing_rate_hz: _Band | None = None,
    min_spike_count: int | None = None,
) -> None:
    """Assert mean AP-shape metrics fall within literature-cited ranges.

    Each band is a closed interval ``(low, high)``.  ``None`` skips that
    check.  The helper accumulates every failing metric and raises a single
    :class:`AssertionError` listing all offenders, which is more useful than
    failing on the first violation when calibrating tolerances against a
    new paper.

    Args:
        ap_result: Output of :func:`~patch_sim.analysis.ap_metrics.analyze_aps`
            or :func:`~patch_sim.analysis.ap_metrics.analyze_aps_from_result`.
        reference: Literature citation (e.g. ``"McCormick et al. 1985"``);
            included in every failure message.
        half_width_ms: Acceptable mean half-width band in ms.
        peak_mv: Acceptable mean peak-voltage band in mV.
        ahp_mv: Acceptable mean AHP-depth band in mV (depth is the post-spike
            voltage minimum; values are negative).
        threshold_mv: Acceptable mean threshold-voltage band in mV.
        firing_rate_hz: Acceptable mean firing rate in Hz (1000 / mean ISI).
        min_spike_count: Minimum spike count required.  Defaults to 1 if any
            shape band is supplied (you cannot measure shape without spikes).

    Raises:
        AssertionError: When any supplied band is violated, when fewer than
            ``min_spike_count`` spikes are detected, or when a metric is
            ``None`` (no spikes / insufficient ISIs) but a band was supplied.
    """
    needs_spikes = any(
        band is not None
        for band in (half_width_ms, peak_mv, ahp_mv, threshold_mv, firing_rate_hz)
    )
    effective_min = (
        min_spike_count if min_spike_count is not None else (1 if needs_spikes else 0)
    )
    if ap_result.spike_count < effective_min:
        raise AssertionError(
            f"[{reference}] Expected ≥{effective_min} spike(s), got "
            f"{ap_result.spike_count}.  Cannot evaluate shape metrics."
        )

    failures: list[str] = []

    def _check(metric_name: str, value: float | None, band: _Band | None) -> None:
        """Append a failure message if value is missing or outside band."""
        if band is None:
            return
        low, high = band
        if value is None:
            failures.append(
                f"{metric_name}=None (no spikes or fewer than 2 ISIs); "
                f"expected in [{low}, {high}]"
            )
            return
        if not (low <= value <= high):
            failures.append(f"{metric_name}={value:.3f} outside [{low}, {high}]")

    _check("mean_half_width_ms", ap_result.mean_half_width, half_width_ms)
    _check("mean_peak_voltage_mv", ap_result.mean_peak_voltage, peak_mv)
    _check("mean_ahp_depth_mv", ap_result.mean_ahp_depth, ahp_mv)
    _check(
        "mean_threshold_voltage_mv",
        ap_result.mean_threshold_voltage,
        threshold_mv,
    )
    _check("firing_rate_hz", ap_result.firing_rate, firing_rate_hz)

    if failures:
        joined = "\n  - ".join(failures)
        raise AssertionError(f"[{reference}] AP-shape assertion failed:\n  - {joined}")
