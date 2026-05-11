"""Pure analysis-formatting helpers for SimulationState.

Each helper takes ``patch_sim`` analysis dataclasses or :class:`Sweep` lists
and returns JSON-serialisable display dicts ready for AnalysisState.  None
of these functions touches Reflex state; they are extracted here so the
sweep executor and ``SimulationState`` itself can remain focused on flow
control rather than display formatting.
"""

import logging
from typing import Any

import numpy as np

import patch_sim
from patch_sim.analysis.fi_curve import _fi_point_from_ap_result
from patch_sim.analysis.hyperpolarization import _sag_point_from_ap_result
from patch_sim.constants import CURRENT_CLAMP
from patch_sim_ui.sweep import Sweep

logger = logging.getLogger(__name__)

#: Number of voltage points used for the pre-computed Boltzmann fit curve.
_GV_FIT_POINTS = 200


def _fmt_optional(value: "float | None", fmt: str) -> str:
    """Format a float value, or return an em-dash when the value is None.

    Args:
        value: The float to format, or None.
        fmt: Python format spec string (e.g. ``".1f"``).

    Returns:
        A formatted string or ``"—"`` when value is None.
    """
    return f"{value:{fmt}}" if value is not None else "—"


def _format_spike_dict(index: int, spike: "patch_sim.SpikeMetrics") -> dict[str, Any]:
    """Serialise a single :class:`~patch_sim.SpikeMetrics` to a display dict.

    Args:
        index: Display index to assign (may differ from spike.index in
            multi-sweep mode where spikes are renumbered globally).
        spike: The spike to serialise.

    Returns:
        A dict with pre-formatted string values for each metric column
        (``index``, ``threshold_voltage``, ``peak_voltage``, ``rise_time``,
        ``half_width``, ``ahp_depth``).
    """
    return {
        "index": index,
        "threshold_voltage": f"{spike.threshold_voltage:.1f}",
        "peak_voltage": f"{spike.peak_voltage:.1f}",
        "rise_time": f"{spike.rise_time:.2f}",
        "half_width": f"{spike.half_width:.2f}",
        "ahp_depth": _fmt_optional(spike.ahp_depth, ".1f"),
    }


def _format_ca_transient_dict(
    index: int, transient: "patch_sim.CalciumTransient"
) -> dict[str, Any]:
    """Serialise a single :class:`~patch_sim.CalciumTransient` to a display dict.

    The ``decay_tau`` column carries a trailing ``*`` when the τ value came
    from the 1/e fallback rather than a converged ``curve_fit``, so the user
    can tell when the decay-fit convergence failed.

    Args:
        index: Display index to assign.
        transient: The transient to serialise.

    Returns:
        A dict with pre-formatted string values for each metric column
        (``index``, ``sweep_index``, ``peak_time``, ``peak_concentration``,
        ``time_to_peak``, ``decay_tau``, ``amplitude``).  All concentrations
        are in µM and all times in ms.  ``sweep_index`` is 0 for
        single-sweep results; multi-sweep callers overwrite it with a
        1-based sweep number.
    """
    if transient.decay_tau is None:
        decay_str = "—"
    elif transient.decay_fit_converged:
        decay_str = f"{transient.decay_tau:.1f}"
    else:
        decay_str = f"{transient.decay_tau:.1f}*"

    return {
        "index": index,
        "sweep_index": 0,
        "peak_time": f"{transient.peak_time:.1f}",
        "peak_concentration": f"{transient.peak_concentration:.3f}",
        "time_to_peak": f"{transient.time_to_peak:.2f}",
        "decay_tau": decay_str,
        "amplitude": f"{transient.amplitude:.3f}",
    }


def _serialise_ca_transient_summary(
    result: "patch_sim.CalciumTransientAnalysisResult",
) -> dict[str, Any]:
    """Serialise the aggregate part of a calcium-transient analysis.

    Args:
        result: The analysis result to summarise.

    Returns:
        A dict with pre-formatted string values for the metrics-panel
        aggregate row.  All concentrations are in µM and all times in ms.
    """
    return {
        "transient_count": str(result.transient_count),
        "baseline_concentration": _fmt_optional(result.baseline_concentration, ".3f"),
        "mean_peak_concentration": _fmt_optional(result.mean_peak_concentration, ".3f"),
        "mean_time_to_peak": _fmt_optional(result.mean_time_to_peak, ".2f"),
        "mean_decay_tau": _fmt_optional(result.mean_decay_tau, ".1f"),
        "mean_amplitude": _fmt_optional(result.mean_amplitude, ".3f"),
    }


def _compute_ca_transient_data(
    result: "patch_sim.SimulationResult",
) -> "tuple[list[dict[str, Any]], dict[str, Any]]":
    """Compute serialised calcium-transient analysis from a single-sweep result.

    Returns empty ``([], {})`` when the simulation result has no ``ca_i``
    field (i.e. calcium dynamics were not active) or when no transients
    were detected.

    Args:
        result: A single-sweep :class:`~patch_sim.SimulationResult`.

    Returns:
        A 2-tuple ``(metrics, summary)`` where ``metrics`` is a list of
        per-transient display dicts and ``summary`` is the aggregate summary
        dict.  Both are empty when no transients were detected.
    """
    analysis = patch_sim.analyze_calcium_transients_from_result(result)
    if analysis is None or analysis.transient_count == 0:
        return [], {}

    metrics = [
        _format_ca_transient_dict(i, t) for i, t in enumerate(analysis.transients)
    ]
    summary = _serialise_ca_transient_summary(analysis)
    return metrics, summary


def _compute_multi_sweep_ca_transient_data(
    sweeps: "list[Sweep]",
) -> "tuple[list[dict[str, Any]], dict[str, Any]]":
    """Pool calcium-transient analysis across a multi-sweep simulation result.

    Used for both multi-sweep current clamp and multi-sweep voltage clamp:
    each sweep that exposes a ``ca_i`` trace is analyzed in isolation, and
    each per-transient display dict carries its 1-based ``sweep_index`` (in
    protocol order) so the UI can identify which sweep produced which
    transient.  Aggregate summary statistics are pooled across all sweeps.

    Args:
        sweeps: Ordered list of :class:`Sweep` objects.

    Returns:
        A 2-tuple ``(metrics, summary)`` of pre-formatted display dicts.  Both
        are empty when no sweep has a ``ca_i`` field or when no transients
        were detected across the pool.
    """
    pooled: list[tuple[int, patch_sim.CalciumTransient]] = []
    baselines: list[float] = []
    for sweep_idx, sweep in enumerate(sweeps):
        ca_trace = sweep.channel_gating.get("ca_i")
        if not ca_trace:
            continue
        time_arr = np.array(sweep.time)
        ca_arr = np.array(ca_trace)
        analysis = patch_sim.analyze_calcium_transients(time_arr, ca_arr)
        if analysis.baseline_concentration is not None:
            baselines.append(analysis.baseline_concentration)
        pooled.extend((sweep_idx, t) for t in analysis.transients)

    if not pooled:
        return [], {}

    metrics: list[dict[str, Any]] = []
    for i, (sweep_idx, transient) in enumerate(pooled):
        entry = _format_ca_transient_dict(i, transient)
        # 1-based sweep number for direct display in the UI table.
        entry["sweep_index"] = sweep_idx + 1
        metrics.append(entry)

    peak_vals = [t.peak_concentration for _, t in pooled]
    ttp_vals = [t.time_to_peak for _, t in pooled]
    amp_vals = [t.amplitude for _, t in pooled]
    tau_vals = [t.decay_tau for _, t in pooled if t.decay_tau is not None]

    summary: dict[str, Any] = {
        "transient_count": str(len(pooled)),
        "baseline_concentration": _fmt_optional(
            float(np.mean(baselines)) if baselines else None, ".3f"
        ),
        "mean_peak_concentration": f"{float(np.mean(peak_vals)):.3f}",
        "mean_time_to_peak": f"{float(np.mean(ttp_vals)):.2f}",
        "mean_decay_tau": _fmt_optional(
            float(np.mean(tau_vals)) if tau_vals else None, ".1f"
        ),
        "mean_amplitude": f"{float(np.mean(amp_vals)):.3f}",
    }
    return metrics, summary


def _format_burst_dict(index: int, burst: "patch_sim.BurstMetrics") -> dict[str, Any]:
    """Serialise a single :class:`~patch_sim.BurstMetrics` to a display dict.

    Args:
        index: Display index to assign.  May differ from ``burst.index`` when
            multi-sweep results are pooled.
        burst: The burst to serialise.

    Returns:
        A dict with pre-formatted string values for each metric column
        (``index``, ``sweep_index``, ``start_time``, ``end_time``,
        ``duration``, ``spike_count``, ``intra_burst_frequency``,
        ``mean_intra_burst_isi``).  ``sweep_index`` is 0 for single-sweep
        results; multi-sweep callers overwrite it with a 1-based sweep
        number.
    """
    return {
        "index": index,
        "sweep_index": 0,
        "start_time": f"{burst.start_time:.1f}",
        "end_time": f"{burst.end_time:.1f}",
        "duration": f"{burst.duration:.1f}",
        "spike_count": str(burst.spike_count),
        "intra_burst_frequency": _fmt_optional(burst.intra_burst_frequency, ".1f"),
        "mean_intra_burst_isi": _fmt_optional(burst.mean_intra_burst_isi, ".2f"),
    }


def _serialise_burst_summary(
    result: "patch_sim.BurstAnalysisResult",
) -> dict[str, Any]:
    """Serialise the aggregate part of a burst-analysis result.

    Args:
        result: The analysis result to summarise.

    Returns:
        A dict with pre-formatted string values for the metrics-panel
        aggregate row.  Frequencies are in Hz, durations and intervals in ms,
        and ``duty_cycle`` is reported as a fraction with 3 decimals.
    """
    return {
        "burst_count": str(result.burst_count),
        "unburst_spike_count": str(result.unburst_spike_count),
        "mean_spikes_per_burst": _fmt_optional(result.mean_spikes_per_burst, ".2f"),
        "mean_intra_burst_frequency": _fmt_optional(
            result.mean_intra_burst_frequency, ".1f"
        ),
        "mean_inter_burst_interval": _fmt_optional(
            result.mean_inter_burst_interval, ".1f"
        ),
        "duty_cycle": _fmt_optional(result.duty_cycle, ".3f"),
        "isi_threshold_ms": f"{result.isi_threshold_ms:.1f}",
        "threshold_method": result.threshold_method,
    }


def _compute_burst_data(
    ap_result: "patch_sim.APAnalysisResult",
    time_arr: np.ndarray,
) -> "tuple[list[dict[str, Any]], dict[str, Any]]":
    """Compute serialised burst-analysis from a single-sweep AP result.

    Returns ``([], {})`` when the spike train is too short to support any
    burst analysis (fewer than 2 spikes); otherwise always emits the
    summary so the UI can surface the threshold and method even when zero
    bursts were detected.

    Args:
        ap_result: AP analysis from :func:`patch_sim.analyze_aps_from_result`.
        time_arr: The recording's time array (ms); used to compute the
            window length for the duty cycle.

    Returns:
        A 2-tuple ``(metrics, summary)`` where ``metrics`` is a list of
        per-burst display dicts (empty when no bursts were detected) and
        ``summary`` is the aggregate summary dict.
    """
    if ap_result.spike_count < 2:
        return [], {}

    total_duration_ms = float(time_arr[-1] - time_arr[0]) if len(time_arr) > 1 else 0.0
    analysis = patch_sim.analyze_bursts(ap_result, total_duration_ms=total_duration_ms)
    metrics = [_format_burst_dict(i, b) for i, b in enumerate(analysis.bursts)]
    summary = _serialise_burst_summary(analysis)
    return metrics, summary


def _compute_multi_sweep_burst_data(
    sweeps: "list[Sweep]",
) -> "tuple[list[dict[str, Any]], dict[str, Any]]":
    """Pool burst analysis across the current-clamp sweeps of a multi-sweep run.

    Each current-clamp sweep is analyzed in isolation; the resulting bursts
    are pooled with a 1-based ``sweep_index`` so the UI table can identify
    the source sweep.  Aggregate metrics are computed by pooling across all
    bursts (intra-burst frequency, spikes/burst) or all per-sweep IBIs
    (inter-burst interval), so each metric is a true mean over its
    underlying observations rather than a mean of per-sweep means.
    ``duty_cycle`` is total burst time divided by total CC-sweep duration.

    The ``isi_threshold_ms`` shown in the summary is the median across
    analyzed sweeps; the ``threshold_method`` is a single label when every
    sweep agreed (``"auto-histogram"`` / ``"default-fixed"``) or
    ``"mixed: <a> N/M, <b> K/M"`` when sweeps used different methods.

    Sweeps are assumed to be in chronological order in the input list; the
    pooled per-burst table preserves that order so the UI shows bursts in
    the same sequence the user produced them.

    Args:
        sweeps: Chronologically ordered list of :class:`Sweep` objects.
            Voltage-clamp sweeps are ignored — burst analysis is meaningful
            only for current-clamp recordings.

    Returns:
        A 2-tuple ``(metrics, summary)`` of pre-formatted display dicts.
        Both are empty when no current-clamp sweep produced ≥2 spikes.
    """
    pooled: list[tuple[int, patch_sim.BurstMetrics]] = []
    per_sweep_results: list[tuple[int, patch_sim.BurstAnalysisResult]] = []
    total_window_ms = 0.0

    for sweep_idx, sweep in enumerate(sweeps):
        if sweep.clamp_mode != CURRENT_CLAMP:
            continue
        time_arr = np.array(sweep.time)
        sweep_duration_ms = (
            float(time_arr[-1] - time_arr[0]) if len(time_arr) > 1 else 0.0
        )
        # Every CC sweep contributes to the duty-cycle denominator, not just
        # sweeps that fire enough spikes to run the burst analyzer.  Otherwise
        # a 10-sweep run where 2 fire would report a duty cycle as a fraction
        # of those 2 sweeps' duration, overstating the active-burst time.
        total_window_ms += sweep_duration_ms

        voltage_arr = np.array(sweep.voltage)
        ap_result = patch_sim.analyze_aps(time_arr, voltage_arr)
        if ap_result.spike_count < 2:
            continue
        analysis = patch_sim.analyze_bursts(
            ap_result, total_duration_ms=sweep_duration_ms
        )
        per_sweep_results.append((sweep_idx, analysis))
        pooled.extend((sweep_idx, b) for b in analysis.bursts)

    if not per_sweep_results:
        return [], {}

    metrics: list[dict[str, Any]] = []
    for i, (sweep_idx, burst) in enumerate(pooled):
        entry = _format_burst_dict(i, burst)
        entry["sweep_index"] = sweep_idx + 1
        metrics.append(entry)

    burst_count = sum(r.burst_count for _, r in per_sweep_results)
    unburst = sum(r.unburst_spike_count for _, r in per_sweep_results)
    spike_counts = [b.spike_count for _, b in pooled]
    mean_spikes = float(np.mean(spike_counts)) if spike_counts else None
    freq_values = [
        b.intra_burst_frequency
        for _, b in pooled
        if b.intra_burst_frequency is not None
    ]
    mean_freq = float(np.mean(freq_values)) if freq_values else None
    # Pool individual IBIs across all sweeps rather than averaging per-sweep
    # means.  IBIs aren't defined across sweep boundaries (the gap between
    # sweep N's last burst and sweep N+1's first isn't physiological), so
    # they're computed within each sweep and then concatenated.
    pooled_ibis: list[float] = []
    for _, r in per_sweep_results:
        for k in range(len(r.bursts) - 1):
            pooled_ibis.append(r.bursts[k + 1].start_time - r.bursts[k].end_time)
    mean_ibi = float(np.mean(pooled_ibis)) if pooled_ibis else None
    total_burst_ms = sum(b.duration for _, b in pooled)
    duty_cycle = (
        float(total_burst_ms / total_window_ms) if total_window_ms > 0 else None
    )

    # Median threshold across analyzed sweeps; method is a single label when
    # every sweep agreed, otherwise a "mixed: ..." breakdown.  This avoids
    # silently masking auto/fixed disagreement that would happen if we just
    # picked one sweep's threshold as representative.
    thresholds = [r.isi_threshold_ms for _, r in per_sweep_results]
    methods = [r.threshold_method for _, r in per_sweep_results]
    median_threshold = float(np.median(thresholds))
    unique_methods = sorted(set(methods))
    if len(unique_methods) == 1:
        method_label = unique_methods[0]
    else:
        method_label = "mixed: " + ", ".join(
            f"{m} {methods.count(m)}/{len(methods)}" for m in unique_methods
        )

    summary: dict[str, Any] = {
        "burst_count": str(burst_count),
        "unburst_spike_count": str(unburst),
        "mean_spikes_per_burst": _fmt_optional(mean_spikes, ".2f"),
        "mean_intra_burst_frequency": _fmt_optional(mean_freq, ".1f"),
        "mean_inter_burst_interval": _fmt_optional(mean_ibi, ".1f"),
        "duty_cycle": _fmt_optional(duty_cycle, ".3f"),
        "isi_threshold_ms": f"{median_threshold:.1f}",
        "threshold_method": method_label,
    }
    return metrics, summary


def _serialise_sfa_curve(curve: "patch_sim.SFACurve") -> dict[str, Any]:
    """Serialise a single :class:`~patch_sim.SFACurve` to a plain dict.

    Args:
        curve: The SFA curve to serialise.

    Returns:
        A dict with ``spike_indices``, ``instantaneous_frequencies``,
        ``adaptation_index``, and ``label`` keys suitable for UI state transfer.
    """
    return {
        "spike_indices": curve.spike_indices,
        "instantaneous_frequencies": curve.instantaneous_frequencies,
        "adaptation_index": curve.adaptation_index,
        "label": curve.label,
    }


def _build_phase_plane_data(sweeps: "list[Sweep]") -> dict[str, Any]:
    """Serialise current-clamp sweeps into phase-plane data for AnalysisState.

    Only sweeps that have a non-empty ``dvdt`` field are included.  Voltage
    clamp sweeps are silently skipped because their voltage is prescribed and
    dV/dt carries no physiological information.

    Args:
        sweeps: The current sweep list from SimulationState.

    Returns:
        A dict with a ``"sweeps"`` key whose value is a list of dicts, each
        containing ``"voltage"``, ``"dvdt"``, ``"label"``, and ``"color"``.
        Returns an empty dict when no eligible sweeps exist.
    """
    eligible = [
        {
            "voltage": s.voltage,
            "dvdt": s.dvdt,
            "label": s.label,
            "color": s.color,
        }
        for s in sweeps
        if s.clamp_mode == CURRENT_CLAMP and s.dvdt
    ]
    return {"sweeps": eligible} if eligible else {}


def _compute_impedance_data(
    sweep: "Sweep",
    pre_stimulus_duration: float,
    stimulus_duration: float,
    start_frequency: float,
    end_frequency: float,
    area_cm2: "float | None",
) -> dict[str, Any]:
    """Serialise a chirp current-clamp sweep into impedance data for AnalysisState.

    Computes the FFT-based membrane impedance over the chirp's swept band via
    :func:`patch_sim.analyze_impedance`, restricting the analysis to the chirp
    window ``[pre_stimulus_duration, pre_stimulus_duration + stimulus_duration]``
    (or to the whole recording when ``stimulus_duration`` is zero, matching the
    chirp builder).  Absolute MΩ values are used when the neuron declares a
    surface area, otherwise the per-area kΩ·cm² spectrum is reported.

    Args:
        sweep: The single current-clamp sweep produced by a chirp protocol;
            its ``stimulus`` field holds the injected current (µA/cm²).
        pre_stimulus_duration: Pre-stimulus padding before the chirp (ms).
        stimulus_duration: Duration of the chirp stimulus (ms); zero means the
            chirp fills the recording from ``pre_stimulus_duration`` onward.
        start_frequency: Chirp sweep start frequency in Hz.
        end_frequency: Chirp sweep end frequency in Hz.
        area_cm2: Membrane surface area in cm², or ``None`` when undeclared.

    Returns:
        A dict with keys ``frequencies``, ``magnitude``, ``phase`` (lists),
        ``resonance_frequency``, ``quality_factor``, ``peak_impedance``
        (pre-formatted strings), and ``units``.  Returns an empty dict when the
        impedance analysis is not applicable (see :func:`patch_sim.analyze_impedance`).
    """
    time_arr = np.asarray(sweep.time, dtype=float)
    if len(time_arr) < 2:
        return {}
    voltage_arr = np.asarray(sweep.voltage, dtype=float)
    current_arr = np.asarray(sweep.stimulus, dtype=float)
    stim_start_ms = float(pre_stimulus_duration)
    stim_end_ms = (
        float(pre_stimulus_duration + stimulus_duration)
        if stimulus_duration > 0.0
        else float(time_arr[-1])
    )
    profile = patch_sim.analyze_impedance(
        time_arr,
        voltage_arr,
        current_arr,
        stim_start_ms,
        stim_end_ms,
        start_frequency,
        end_frequency,
        area_cm2=area_cm2,
    )
    if profile is None:
        return {}
    if profile.magnitude_mohm is not None:
        magnitude = profile.magnitude_mohm
        peak = profile.peak_impedance_mohm
        units = "MΩ"
    else:
        magnitude = profile.magnitude
        peak = profile.peak_impedance
        units = "kΩ·cm²"
    return {
        "frequencies": profile.frequencies,
        "magnitude": magnitude,
        "phase": profile.phase,
        "resonance_frequency": _fmt_optional(profile.resonance_frequency, ".2f"),
        "quality_factor": _fmt_optional(profile.quality_factor, ".2f"),
        "peak_impedance": _fmt_optional(peak, ".3g"),
        "units": units,
    }


def _compute_cc_multi_sweep_analysis(
    sweeps: "list[Sweep]",
    min_stimulus: float,
    max_stimulus: float,
    stimulus_step: float,
    pre_stimulus_duration: float,
    stimulus_duration: float,
) -> "tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]":  # noqa: E501
    """Compute AP metrics, F-I, SFA, and hyperpolarization data from multi-sweep CC.

    Runs spike detection once per sweep and derives all outputs from those
    results, avoiding redundant :func:`analyze_aps` calls.

    When all current steps are negative (``max_stimulus <= 0``),
    hyperpolarization sag and rebound analysis is returned in place of F-I data
    (which is not meaningful for negative currents).

    Args:
        sweeps: Ordered list of :class:`Sweep` objects from the simulation.
        min_stimulus: Minimum injected current step (µA/cm²).
        max_stimulus: Maximum injected current step (µA/cm²).
        stimulus_step: Step size between current commands (µA/cm²).
        pre_stimulus_duration: Duration before the step begins (ms).
        stimulus_duration: Duration of the current step (ms).

    Returns:
        A 5-tuple ``(ap_metrics, ap_summary, fi_data, sfa_data, hyp_data)``
        where ``ap_metrics`` is a list of per-spike dicts (pooled and
        renumbered across all sweeps), ``ap_summary`` is an aggregate dict,
        ``fi_data`` is a serialised :class:`~patch_sim.FIAnalysisResult` dict
        (empty when all steps are negative), ``sfa_data`` is a serialised SFA
        dict with one curve per sweep that had at least two spikes, and
        ``hyp_data`` is a serialised
        :class:`~patch_sim.HyperpolarizationAnalysisResult` dict (empty when
        any step is positive).  ``ap_metrics`` and ``ap_summary`` are empty
        when no spikes are detected.
    """
    n_steps = round((max_stimulus - min_stimulus) / stimulus_step) + 1
    current_steps = list(np.linspace(min_stimulus, max_stimulus, n_steps))

    time_arr = np.array(sweeps[0].time)
    stim_start = pre_stimulus_duration
    stim_end = pre_stimulus_duration + stimulus_duration

    # Run analyze_aps once per sweep; reuse results for both AP metrics and F-I.
    per_sweep_ap = [
        patch_sim.analyze_aps(time_arr, np.array(s.voltage)) for s in sweeps
    ]

    # --- AP metrics (pooled across all sweeps) ---
    all_spikes = [spike for ap in per_sweep_ap for spike in ap.spikes]

    if all_spikes:
        ap_metrics: list[dict[str, Any]] = [
            _format_spike_dict(i, s) for i, s in enumerate(all_spikes)
        ]
        thresh_vals = [s.threshold_voltage for s in all_spikes]
        peak_vals = [s.peak_voltage for s in all_spikes]
        rise_vals = [s.rise_time for s in all_spikes]
        hw_vals = [s.half_width for s in all_spikes]
        ahp_vals = [s.ahp_depth for s in all_spikes if s.ahp_depth is not None]
        ap_summary: dict[str, Any] = {
            "spike_count": str(len(all_spikes)),
            "mean_threshold_voltage": f"{float(np.mean(thresh_vals)):.1f}",
            "mean_peak_voltage": f"{float(np.mean(peak_vals)):.1f}",
            "mean_rise_time": f"{float(np.mean(rise_vals)):.2f}",
            "mean_half_width": f"{float(np.mean(hw_vals)):.2f}",
            "mean_ahp_depth": (f"{float(np.mean(ahp_vals)):.1f}" if ahp_vals else "—"),
            # mean_isi and firing_rate omitted: shown per-sweep in the F-I curve.
        }
    else:
        ap_metrics = []
        ap_summary = {}

    # --- SFA data (one curve per sweep with >= 2 spikes) ---
    sfa_curves = [
        patch_sim.compute_sfa(ap_result, label=f"{i_step:.1f} µA/cm²")
        for ap_result, i_step in zip(per_sweep_ap, current_steps)
    ]
    sfa_data: dict[str, Any] = {
        "curves": [_serialise_sfa_curve(c) for c in sfa_curves if c is not None]
    }
    if not sfa_data["curves"]:
        sfa_data = {}

    # Detect whether all steps are hyperpolarizing (negative).  When true,
    # F-I analysis is skipped in favor of sag/rebound analysis.
    is_hyperpolarizing = max_stimulus < 0.0

    # --- F-I / hyperpolarization data ---
    if len(sweeps) != len(current_steps):
        logger.warning(
            "Multi-sweep analysis skipped: %d sweeps but %d current steps derived "
            "from protocol (min=%.3g, max=%.3g, step=%.3g)",
            len(sweeps),
            len(current_steps),
            min_stimulus,
            max_stimulus,
            stimulus_step,
        )
        if ap_summary:
            ap_summary["rheobase"] = "—"
        return ap_metrics, ap_summary, {}, sfa_data, {}

    if is_hyperpolarizing:
        time_arr = np.array(sweeps[0].time)
        sag_points: list[patch_sim.SagPoint] = [
            _sag_point_from_ap_result(
                time_arr,
                np.array(s.voltage),
                ap_result,
                i_step,
                stim_start,
                stim_end,
            )
            for s, ap_result, i_step in zip(sweeps, per_sweep_ap, current_steps)
        ]
        sag_points.sort(key=lambda p: p.current_step)
        hyp_result = patch_sim.HyperpolarizationAnalysisResult(points=sag_points)
        hyp_data: dict[str, Any] = {
            "current_steps": hyp_result.current_steps,
            "sag_amplitudes": hyp_result.sag_amplitudes,
            "rebound_spike_counts": hyp_result.rebound_spike_counts,
        }
        return ap_metrics, ap_summary, {}, sfa_data, hyp_data

    fi_points: list[patch_sim.FIPoint] = [
        _fi_point_from_ap_result(ap_result, i_step, stim_start, stim_end)
        for ap_result, i_step in zip(per_sweep_ap, current_steps)
    ]

    fi_points.sort(key=lambda p: p.current_step)
    fi_result = patch_sim.FIAnalysisResult(points=fi_points)
    rheobase = patch_sim.estimate_rheobase(fi_result)
    fi_data: dict[str, Any] = {
        "current_steps": fi_result.current_steps,
        "mean_firing_rates": fi_result.mean_firing_rates,
        "initial_firing_rates": fi_result.initial_firing_rates,
        "steady_state_firing_rates": fi_result.steady_state_firing_rates,
    }
    if ap_summary:
        ap_summary["rheobase"] = f"{rheobase:.2f}" if rheobase is not None else "—"
    return ap_metrics, ap_summary, fi_data, sfa_data, {}


def _compute_iv_data(
    sweeps: "list[Sweep]",
    min_stimulus: float,
    max_stimulus: float,
    stimulus_step: float,
    pre_stimulus_duration: float,
    stimulus_duration: float,
) -> "tuple[dict[str, Any], patch_sim.IVAnalysisResult | None]":
    """Compute I-V analysis data from multi-sweep voltage clamp results.

    Derives voltage step values from the protocol parameters, extracts total
    current arrays from each sweep, and calls :func:`patch_sim.analyze_iv`.
    The serialised dict is suitable for use as a Reflex state variable; the
    raw :class:`~patch_sim.IVAnalysisResult` is also returned so callers can
    derive further analyses (e.g. the g-V curve) without re-running the
    simulation.

    Args:
        sweeps: Ordered list of :class:`Sweep` objects from the simulation.
        min_stimulus: Minimum voltage step command (mV).
        max_stimulus: Maximum voltage step command (mV).
        stimulus_step: Step size between voltage commands (mV).
        pre_stimulus_duration: Duration before the step begins (ms).
        stimulus_duration: Duration of the voltage step (ms).

    Returns:
        A 2-tuple ``(iv_data, iv_result)`` where *iv_data* is a dict with keys
        ``voltages``, ``peak_inward_currents``, ``peak_outward_currents``, and
        ``steady_state_currents`` (each a list of floats sorted by voltage) and
        *iv_result* is the underlying :class:`~patch_sim.IVAnalysisResult`.
        Both are empty / ``None`` when fewer than two sweeps are provided or
        when the sweep count does not match the number of voltage steps.
    """
    if len(sweeps) < 2:
        return {}, None

    n_steps = round((max_stimulus - min_stimulus) / stimulus_step) + 1
    voltage_steps = list(np.linspace(min_stimulus, max_stimulus, n_steps))

    if len(sweeps) != len(voltage_steps):
        return {}, None

    time_arr = np.array(sweeps[0].time)
    currents = [np.array(s.total_current) for s in sweeps]

    stim_start = pre_stimulus_duration
    stim_end = pre_stimulus_duration + stimulus_duration

    iv_result = patch_sim.analyze_iv(
        time_arr, currents, voltage_steps, stim_start, stim_end
    )
    iv_data: dict[str, Any] = {
        "voltages": iv_result.voltage_steps,
        "peak_inward_currents": iv_result.peak_inward_currents,
        "peak_outward_currents": iv_result.peak_outward_currents,
        "steady_state_currents": iv_result.steady_state_currents,
    }
    return iv_data, iv_result


def _compute_tau_v_data(
    sweeps: "list[Sweep]",
    min_stimulus: float,
    max_stimulus: float,
    stimulus_step: float,
    pre_stimulus_duration: float,
    stimulus_duration: float,
) -> "dict[str, Any]":
    """Compute serialised τ-V analysis data from multi-sweep voltage clamp results.

    Derives voltage step values from the protocol parameters, extracts total
    current arrays from each sweep, and calls :func:`patch_sim.analyze_tau_v`.
    The serialised dict is suitable for use as a Reflex state variable.

    Args:
        sweeps: Ordered list of :class:`Sweep` objects from the simulation.
        min_stimulus: Minimum voltage step command (mV).
        max_stimulus: Maximum voltage step command (mV).
        stimulus_step: Step size between voltage commands (mV).
        pre_stimulus_duration: Duration before the step begins (ms).
        stimulus_duration: Duration of the voltage step (ms).

    Returns:
        A dict with keys ``voltages``, ``tau_activation``,
        ``tau_inactivation``, ``tau_inactivation_slow``, and
        ``has_double_exp`` (each a list of floats / nullable floats / bools
        sorted by voltage), or an empty dict when fewer than two sweeps are
        available or the sweep count does not match the voltage steps.
    """
    if len(sweeps) < 2:
        return {}

    n_steps = round((max_stimulus - min_stimulus) / stimulus_step) + 1
    voltage_steps = list(np.linspace(min_stimulus, max_stimulus, n_steps))

    if len(sweeps) != len(voltage_steps):
        return {}

    time_arr = np.array(sweeps[0].time)
    currents = [np.array(s.total_current) for s in sweeps]

    stim_start = pre_stimulus_duration
    stim_end = pre_stimulus_duration + stimulus_duration

    tau_v_result = patch_sim.analyze_tau_v(
        time_arr, currents, voltage_steps, stim_start, stim_end
    )

    if not tau_v_result.points:
        return {}

    return {
        "voltages": tau_v_result.voltage_steps,
        "tau_activation": tau_v_result.tau_activation_values,
        "tau_inactivation": tau_v_result.tau_inactivation_values,
        "tau_inactivation_slow": tau_v_result.tau_inactivation_slow_values,
        "has_double_exp": [p.inactivation_is_double for p in tau_v_result.points],
    }


def _compute_gv_data(
    iv_result: "patch_sim.IVAnalysisResult",
    reversal_potential: float,
) -> "dict[str, Any]":
    """Compute g-V analysis data from an I-V result and a reversal potential.

    Calls :func:`patch_sim.compute_gv` to derive normalized conductance and fit
    a Boltzmann sigmoid.  A dense voltage array (:data:`_GV_FIT_POINTS` points)
    spanning the range of included steps is pre-computed so the plotting
    function can draw a smooth fit curve without importing scipy.

    Args:
        iv_result: Pre-computed I-V analysis result.
        reversal_potential: Reversal potential for the dominant inward current
            carrier (mV), used to compute driving force at each step.

    Returns:
        A dict with keys ``voltages``, ``g_normalized``,
        ``reversal_potential``, ``boltzmann_converged``, ``v_half``, ``k``,
        ``fit_voltages``, and ``fit_g_normalized``.  Returns an empty dict when
        no valid conductance points can be extracted.
    """
    gv_result = patch_sim.compute_gv(iv_result, reversal_potential)
    if not gv_result.points:
        return {}

    fit = gv_result.boltzmann
    fit_voltages: list[float] = []
    fit_gn: list[float] = []
    if fit.converged and len(gv_result.voltage_steps) >= 2:
        v_min = min(gv_result.voltage_steps)
        v_max = max(gv_result.voltage_steps)
        v_arr = np.linspace(v_min, v_max, _GV_FIT_POINTS)
        fit_voltages = v_arr.tolist()
        fit_gn = [float(patch_sim.boltzmann(v, fit.v_half, fit.k)) for v in v_arr]

    return {
        "voltages": gv_result.voltage_steps,
        "g_normalized": gv_result.g_normalized_values,
        "reversal_potential": gv_result.reversal_potential,
        "boltzmann_converged": fit.converged,
        "v_half": fit.v_half,
        "k": fit.k,
        "fit_voltages": fit_voltages,
        "fit_g_normalized": fit_gn,
    }
