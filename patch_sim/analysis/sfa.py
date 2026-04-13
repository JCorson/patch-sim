"""Spike-frequency adaptation (SFA) analysis.

Computes instantaneous firing frequency and an adaptation index from a
pre-computed :class:`~patch_sim.analysis.ap_metrics.APAnalysisResult`.

Data classes:
    SFACurve: Instantaneous-frequency curve for a single sweep.
    SFAAnalysisResult: Collection of curves, one per eligible sweep.
"""

from __future__ import annotations

import dataclasses

from patch_sim.analysis.ap_metrics import APAnalysisResult


@dataclasses.dataclass
class SFACurve:
    """Instantaneous-frequency curve for a single current clamp sweep.

    Attributes:
        spike_indices: Zero-based index for each ISI pair (0, 1, 2, …).
            Length equals ``len(instantaneous_frequencies)``.
        instantaneous_frequencies: Instantaneous firing frequency in Hz
            (``1000 / ISI_ms``) for each consecutive spike pair.
        adaptation_index: ``(f_first - f_last) / f_first``, dimensionless.
            Ranges from 0 (no adaptation) to ~1 (strong adaptation).
            Can be negative when the neuron accelerates.
        label: Optional display label for multi-sweep overlay (e.g.
            ``"10.0 µA/cm²"``).  Empty string for single-sweep use.
    """

    spike_indices: list[int]
    instantaneous_frequencies: list[float]
    adaptation_index: float
    label: str


@dataclasses.dataclass
class SFAAnalysisResult:
    """Spike-frequency adaptation analysis, possibly across multiple sweeps.

    Attributes:
        curves: One :class:`SFACurve` per sweep that contained at least two
            spikes (one ISI).  Sweeps with fewer spikes are excluded.
    """

    curves: list[SFACurve]


def compute_sfa(ap_result: APAnalysisResult, label: str = "") -> SFACurve | None:
    """Compute an instantaneous-frequency curve from a pre-computed AP analysis.

    Reuses ``ap_result.isis`` directly; no additional spike detection is run.
    Returns ``None`` when fewer than two spikes were detected (zero ISIs).

    Args:
        ap_result: AP analysis result whose ``isis`` field is populated.
        label: Display label for multi-sweep overlay (e.g. current step).
            Defaults to an empty string.

    Returns:
        An :class:`SFACurve`, or ``None`` when ``ap_result.isis`` is empty.
    """
    if not ap_result.isis:
        return None

    inst_freqs = [1000.0 / isi for isi in ap_result.isis]
    spike_indices = list(range(len(inst_freqs)))
    f_first = inst_freqs[0]
    f_last = inst_freqs[-1]
    adaptation_index = (f_first - f_last) / f_first if f_first != 0.0 else 0.0

    return SFACurve(
        spike_indices=spike_indices,
        instantaneous_frequencies=inst_freqs,
        adaptation_index=adaptation_index,
        label=label,
    )


def analyze_sfa(
    ap_results: list[APAnalysisResult],
    labels: list[str] | None = None,
) -> SFAAnalysisResult:
    """Compute spike-frequency adaptation curves for multiple sweeps.

    Calls :func:`compute_sfa` for each sweep and collects the non-``None``
    results.  Sweeps with fewer than two spikes are silently excluded.

    Args:
        ap_results: List of AP analysis results, one per sweep.
        labels: Optional parallel list of display labels, aligned with
            ``ap_results``.  Defaults to empty strings for all sweeps.

    Returns:
        An :class:`SFAAnalysisResult` whose ``curves`` list contains one
        entry per sweep that had at least two spikes.
    """
    if labels is None:
        labels = [""] * len(ap_results)

    curves: list[SFACurve] = []
    for ap_result, label in zip(ap_results, labels):
        curve = compute_sfa(ap_result, label=label)
        if curve is not None:
            curves.append(curve)

    return SFAAnalysisResult(curves=curves)
