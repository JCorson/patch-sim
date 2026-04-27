"""Simulation state for the patch_sim web UI.

SimulationState owns simulation results, sweep collections, continuous mode,
and the figure computed var.  Cross-cutting state lives in the sibling
substates (NeuronState, ProtocolState, VisibilityState, AnalysisState, LogState).
"""

import asyncio
import dataclasses
import json
import logging
import pathlib
import time
import uuid
from typing import Any, AsyncGenerator

import numpy as np
import reflex as rx
from reflex.config import get_config as _get_config

import patch_sim
import patch_sim.clamp_simulations
from patch_sim.analysis.fi_curve import _fi_point_from_ap_result
from patch_sim.analysis.hyperpolarization import _sag_point_from_ap_result
from patch_sim.constants import (
    CURRENT_CLAMP,
    VOLTAGE_CLAMP,
)
from patch_sim_ui import constants, presets
from patch_sim_ui.api import traces
from patch_sim_ui.channels import (
    ADDITIONAL_CURRENT_FIELD_MAP,
    ADDITIONAL_GATING_FIELD_MAP,
)
from patch_sim_ui.plotting import (
    TraceVisibility,
    build_figure,
    compute_trace_visibility_map,
)
from patch_sim_ui.state._common import (
    _LOG_SCROLL_JS,
    _PLOTLY_GD_JS,
)
from patch_sim_ui.state.analysis import AnalysisState
from patch_sim_ui.state.log import LogState
from patch_sim_ui.state.neuron import NeuronState
from patch_sim_ui.state.protocol import ProtocolState
from patch_sim_ui.state.visibility import VisibilityState
from patch_sim_ui.sweep import Sweep

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Module-level helpers used only by SimulationState                  #
# ------------------------------------------------------------------ #

# Client-side sweep highlight / selection module.  Injected via
# rx.call_script() after every figure render in multi-sweep mode.
# Loaded from assets/sweep_highlight.js at import time; placeholder tokens
# (/*DIM_OPACITY*/, /*HOVER_WIDTH*/, /*DIM_WIDTH*/, /*SELECTED_SWEEP*/) are
# substituted at call time in _sweep_highlight_js().
_SWEEP_HIGHLIGHT_JS: str = (
    pathlib.Path(__file__).parents[2] / "assets" / "sweep_highlight.js"
).read_text()

# Client-side fetch-and-swap for side-channel figure delivery.  Loaded once
# at import; placeholder tokens (/*API_URL*/, /*TOKEN*/, /*DARK_AXIS_STYLE*/,
# /*POST_JS*/) are substituted per call in _build_fetch_figure_js().
_FETCH_FIGURE_JS: str = (
    pathlib.Path(__file__).parents[2] / "assets" / "fetch_figure.js"
).read_text()


#: Debounce window (seconds) for slider-driven membrane test requests.
_MT_DEBOUNCE_S: float = 0.3

#: Number of voltage points used for the pre-computed Boltzmann fit curve.
_GV_FIT_POINTS = 200


# ------------------------------------------------------------------ #
# Analysis computation helpers                                       #
# ------------------------------------------------------------------ #


def _fmt_optional(value: "float | None", fmt: str) -> str:
    """Format a float value, or return an em-dash when the value is None.

    Args:
        value: The float to format, or None.
        fmt: Python format spec string (e.g. ``".1f"``).

    Returns:
        A formatted string or ``"—"`` when value is None.
    """
    return f"{value:{fmt}}" if value is not None else "\u2014"


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
    each sweep that exposes a ``ca_i`` trace is analysed in isolation, and
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
        ca_trace = sweep.additional_gating.get("ca_i")
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
            "mean_ahp_depth": (
                f"{float(np.mean(ahp_vals)):.1f}" if ahp_vals else "\u2014"
            ),
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
    # F-I analysis is skipped in favour of sag/rebound analysis.
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
            ap_summary["rheobase"] = "\u2014"
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
        ap_summary["rheobase"] = f"{rheobase:.2f}" if rheobase is not None else "\u2014"
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


def _compute_gv_data(
    iv_result: "patch_sim.IVAnalysisResult",
    reversal_potential: float,
) -> "dict[str, Any]":
    """Compute g-V analysis data from an I-V result and a reversal potential.

    Calls :func:`patch_sim.compute_gv` to derive normalised conductance and fit
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


# ------------------------------------------------------------------ #
# Simulation computation helpers                                      #
# ------------------------------------------------------------------ #


@dataclasses.dataclass(frozen=True)
class _SimResult:
    """Output of a complete simulation run, ready to apply to state.

    Produced by :func:`_compute_simulation` and consumed by
    :meth:`SimulationState._do_apply_simulation`.  All fields have
    empty defaults so callers only set what is relevant for the current
    clamp mode.
    """

    sweeps: list[Sweep]
    sim_token: str
    iv_data: dict[str, Any] = dataclasses.field(default_factory=dict)
    gv_data: dict[str, Any] = dataclasses.field(default_factory=dict)
    ap_metrics: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    ap_summary: dict[str, Any] = dataclasses.field(default_factory=dict)
    ap_is_multi_sweep: bool = False
    ca_transient_metrics: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    ca_transient_summary: dict[str, Any] = dataclasses.field(default_factory=dict)
    fi_data: dict[str, Any] = dataclasses.field(default_factory=dict)
    sfa_data: dict[str, Any] = dataclasses.field(default_factory=dict)
    hyperpolarization_data: dict[str, Any] = dataclasses.field(default_factory=dict)
    phase_plane_data: dict[str, Any] = dataclasses.field(default_factory=dict)


def _compute_simulation(
    neuron: "patch_sim.Neuron",
    protocols: "list[tuple[np.ndarray, str]]",
    mode: str,
    stored_traces: "list[Sweep]",
    show_hover: bool,
    min_stimulus: float,
    max_stimulus: float,
    stimulus_step: float,
    pre_stimulus_duration: float,
    stimulus_duration: float,
) -> _SimResult:
    """Run the simulation synchronously and compute all analysis.

    Designed to be called via ``run_in_executor`` in production (no state
    mutation) and directly in tests.  Raises :exc:`ValueError` on invalid
    protocol parameters so the caller can propagate the error to
    :attr:`~SimulationState.error_message`.

    Args:
        neuron: Built neuron model.
        protocols: List of ``(stimulus_array, label)`` tuples from
            :meth:`~patch_sim_ui.state.protocol.ProtocolState._build_protocols`.
        mode: ``"Current Clamp"`` or ``"Voltage Clamp"``.
        stored_traces: Snapshot of current stored traces for figure building.
        show_hover: Whether hover tooltips are enabled.
        min_stimulus: Minimum stimulus value for analysis range.
        max_stimulus: Maximum stimulus value for analysis range.
        stimulus_step: Stimulus step size for analysis range.
        pre_stimulus_duration: Pre-stimulus duration (ms) for analysis windows.
        stimulus_duration: Stimulus duration (ms) for analysis windows.

    Returns:
        A :class:`_SimResult` containing sweeps, figure token, and all
        analysis data ready for :meth:`~SimulationState._do_apply_simulation`.
    """
    sim_fn = (
        patch_sim.simulate_current_clamp
        if mode == CURRENT_CLAMP
        else patch_sim.simulate_voltage_clamp
    )
    is_multi = len(protocols) > 1

    if is_multi:
        new_sweeps: list[Sweep] = []
        for sweep_result, (protocol, label) in zip(
            patch_sim.simulate_batch(neuron, [p for p, _ in protocols], sim_fn),
            protocols,
        ):
            color_index = len(new_sweeps) % len(constants.SWEEP_COLORS)
            new_sweeps.append(
                Sweep.from_result(
                    sweep_result,
                    protocol,
                    label,
                    constants.SWEEP_COLORS[color_index],
                    mode,
                )
            )

        fig = build_figure(
            current_sweeps=new_sweeps,
            visibility=TraceVisibility(),
            clamp_mode=mode,
            stored_traces=stored_traces,
            show_hover=show_hover,
        )
        sim_token = uuid.uuid4().hex
        traces.put(sim_token, fig)

        if mode == VOLTAGE_CLAMP:
            iv_data, iv_result = _compute_iv_data(
                new_sweeps,
                min_stimulus,
                max_stimulus,
                stimulus_step,
                pre_stimulus_duration,
                stimulus_duration,
            )
            if iv_result is not None:
                na_channel = next(
                    (
                        ch
                        for ch in neuron.core_channels
                        if isinstance(ch.reversal_spec, patch_sim.NernstSpec)
                        and ch.reversal_spec.species is patch_sim.IonSpecies.SODIUM
                    ),
                    None,
                )
                gv_data = (
                    _compute_gv_data(iv_result, na_channel.reversal_potential(neuron))
                    if na_channel is not None
                    else {}
                )
            else:
                gv_data = {}
            ms_ca_metrics, ms_ca_summary = _compute_multi_sweep_ca_transient_data(
                new_sweeps
            )
            return _SimResult(
                sweeps=new_sweeps,
                sim_token=sim_token,
                iv_data=iv_data,
                gv_data=gv_data,
                ca_transient_metrics=ms_ca_metrics,
                ca_transient_summary=ms_ca_summary,
            )

        ms_metrics, ms_summary, ms_fi, ms_sfa, ms_hyp = (
            _compute_cc_multi_sweep_analysis(
                new_sweeps,
                min_stimulus,
                max_stimulus,
                stimulus_step,
                pre_stimulus_duration,
                stimulus_duration,
            )
        )
        ms_ca_metrics, ms_ca_summary = _compute_multi_sweep_ca_transient_data(
            new_sweeps
        )
        return _SimResult(
            sweeps=new_sweeps,
            sim_token=sim_token,
            ap_metrics=ms_metrics,
            ap_summary=ms_summary,
            ap_is_multi_sweep=True,
            ca_transient_metrics=ms_ca_metrics,
            ca_transient_summary=ms_ca_summary,
            fi_data=ms_fi,
            sfa_data=ms_sfa,
            hyperpolarization_data=ms_hyp,
            phase_plane_data=_build_phase_plane_data(new_sweeps),
        )

    # Single sweep
    stimulus, _ = protocols[0]
    result = sim_fn(neuron, stimulus)
    sweep = Sweep.from_result(result, stimulus, "", "", mode)

    fig = build_figure(
        current_sweeps=[sweep],
        visibility=TraceVisibility(),
        clamp_mode=mode,
        stored_traces=stored_traces,
        show_hover=show_hover,
    )
    sim_token = uuid.uuid4().hex
    traces.put(sim_token, fig)

    if mode == CURRENT_CLAMP:
        ap_result = patch_sim.analyze_aps_from_result(result)
        sfa_curve = patch_sim.compute_sfa(ap_result)
        ca_metrics, ca_summary = _compute_ca_transient_data(result)
        return _SimResult(
            sweeps=[sweep],
            sim_token=sim_token,
            ap_metrics=[_format_spike_dict(s.index, s) for s in ap_result.spikes],
            ap_summary={
                "spike_count": str(ap_result.spike_count),
                "mean_threshold_voltage": _fmt_optional(
                    ap_result.mean_threshold_voltage, ".1f"
                ),
                "mean_peak_voltage": _fmt_optional(ap_result.mean_peak_voltage, ".1f"),
                "mean_rise_time": _fmt_optional(ap_result.mean_rise_time, ".2f"),
                "mean_half_width": _fmt_optional(ap_result.mean_half_width, ".2f"),
                "mean_ahp_depth": _fmt_optional(ap_result.mean_ahp_depth, ".1f"),
                "mean_isi": _fmt_optional(ap_result.mean_isi, ".1f"),
                "firing_rate": _fmt_optional(ap_result.firing_rate, ".1f"),
                "adaptation_index": (
                    f"{sfa_curve.adaptation_index:.2f}"
                    if sfa_curve is not None
                    else "—"
                ),
                "rheobase": (
                    f"≤ {min_stimulus:.2f}" if ap_result.spike_count >= 1 else "—"
                ),
            },
            ca_transient_metrics=ca_metrics,
            ca_transient_summary=ca_summary,
            sfa_data=(
                {"curves": [_serialise_sfa_curve(sfa_curve)]}
                if sfa_curve is not None
                else {}
            ),
            phase_plane_data=_build_phase_plane_data([sweep]),
        )

    # Single VC sweep — IV/GV require multi-sweep, but calcium transients can
    # still appear (e.g. a depolarising VC step opens Ca channels).
    ca_metrics, ca_summary = _compute_ca_transient_data(result)
    return _SimResult(
        sweeps=[sweep],
        sim_token=sim_token,
        ca_transient_metrics=ca_metrics,
        ca_transient_summary=ca_summary,
    )


class SimulationState(rx.State):
    """State for simulation results, sweep collections, and figure rendering."""

    # Synced copy of NeuronState.active_neuron_type used by store_trace for
    # labelling (avoids making those handlers async).
    _label_neuron_type: str = "Squid Giant Axon (Classic HH)"
    # Synced copy of ProtocolState.clamp_mode used by _rebuild_figure_and_fetch_js
    # and _apply_visibility_js (both are synchronous, cannot call get_state).
    _figure_clamp_mode: str = CURRENT_CLAMP

    # ------------------------------------------------------------------ #
    # Simulation results                                                  #
    # ------------------------------------------------------------------ #
    # Backend-only: excluded from the Reflex state delta so that millions of
    # floats are not JSON-serialised on every state flush.  Served to the
    # browser via /api/figure/{sim_token} (see patch_sim_ui/api/traces.py).
    _current_sweeps: list[Sweep] = []  # Latest simulation result
    stored_traces: list[Sweep] = []  # Oscilloscope-style stored reference traces
    # Short token identifying the latest figure in the side-channel store.
    # Pushed to the client so JS can fetch /api/figure/{sim_token}.
    sim_token: str = ""

    # ------------------------------------------------------------------ #
    # Continuous simulation mode                                        #
    # ------------------------------------------------------------------ #
    continuous_mode: bool = False  # user has toggled continuous on
    continuous_loop_running: bool = False  # background task is executing

    # Terminal neuron state carried between continuous loop iterations.
    # Prefixed with _ by convention; still full Reflex state vars.
    _cont_V: float = 0.0
    _cont_gating: dict[str, float] = {}
    _cont_ca_i: float = 0.0
    _cont_has_state: bool = False  # True once at least one iteration has run
    _mt_request_id: int = 0  # incremented per slider event; debounce uses this

    # ------------------------------------------------------------------ #
    # UI state                                                           #
    # ------------------------------------------------------------------ #
    is_running: bool = False
    error_message: str = ""
    show_hover: bool = True  # Whether plot hover tooltips are visible
    # Index of the last click-selected sweep seeded into the JS on figure
    # rebuild (-1 = none selected).  Selection state is managed entirely
    # client-side by ``window._psSweep``; this field is only written by
    # Python (reset on run/clamp-mode change) so that the correct seed value
    # is injected when the JS module is re-initialised after a figure rebuild.
    #
    # Known limitation: this field is never updated from the client side, so
    # any figure rebuild triggered by a Python state change while the user has
    # a client-side selection active will re-seed JS with -1, silently
    # clearing the selection.
    selected_sweep: int = -1

    # ------------------------------------------------------------------ #
    # Analysis sidebar state                                             #
    # ------------------------------------------------------------------ #
    analysis_panel_open: bool = True

    # ------------------------------------------------------------------ #
    # Computed properties                                               #
    # ------------------------------------------------------------------ #
    @rx.var
    def has_result(self) -> bool:
        """Whether a simulation result is available."""
        return len(self._current_sweeps) > 0

    @rx.var
    def has_stored_traces(self) -> bool:
        """Whether any oscilloscope-stored reference traces exist."""
        return len(self.stored_traces) > 0

    @rx.var
    def is_multi_sweep(self) -> bool:
        """Whether the current simulation has multiple sweeps (e.g. I-V protocol)."""
        return len(self._current_sweeps) > 1

    @rx.var
    def continuous_active(self) -> bool:
        """True when continuous mode is enabled and the loop is running."""
        return self.continuous_mode and self.continuous_loop_running

    # ------------------------------------------------------------------ #
    # Event handlers                                                     #
    # ------------------------------------------------------------------ #
    def _clear_for_new_protocol(self) -> None:
        """Clear all simulation results and reset continuous state.

        Called by ProtocolState when the clamp mode or a protocol preset
        changes, ensuring stale results are not shown for the new protocol.
        """
        self._current_sweeps = []
        self.stored_traces = []
        self.sim_token = ""
        self._cont_has_state = False
        self.selected_sweep = -1

    async def initialize_defaults(self) -> AsyncGenerator[Any, None]:
        """Apply the default neuron and protocol presets on app startup or reset.

        Reflex class-level defaults set ``active_neuron_type`` and the scalar
        neuron fields independently, so the displayed preset can be out of sync
        with the underlying parameters (e.g. ``Q10`` defaults to ``3.0`` from
        :class:`~patch_sim.NeuronConfig` but the SGA preset requires ``1.0``).
        This method bootstraps a consistent state by calling
        :meth:`NeuronState._apply_neuron_preset` and
        :meth:`ProtocolState._apply_protocol_preset` for the configured
        defaults, then yields :meth:`run_membrane_test` to populate the passive
        properties panel.

        Idempotent: safe to call repeatedly.
        """
        neuron_st = await self.get_state(NeuronState)
        neuron_st._apply_neuron_preset(presets.DEFAULT_NEURON_PRESET)

        proto_st = await self.get_state(ProtocolState)
        proto_st._apply_protocol_preset(
            presets.DEFAULT_PROTOCOL_PRESET, presets.DEFAULT_NEURON_PRESET
        )

        for key, value in presets.PROTOCOL_NEURON_OVERRIDES.get(
            presets.DEFAULT_PROTOCOL_PRESET, {}
        ).items():
            setattr(neuron_st, key, value)

        self._label_neuron_type = presets.DEFAULT_NEURON_PRESET
        self._figure_clamp_mode = proto_st.clamp_mode

        yield SimulationState.run_membrane_test

    async def reset_to_defaults(self) -> AsyncGenerator[Any, None]:
        """Reset state to preset defaults for the current neuron and protocol.

        Captures the active neuron type and protocol preset before resetting,
        then re-applies those same selections so the user's neuron and protocol
        choice is preserved while all parameter values are restored to their
        preset defaults.
        """
        neuron_st = await self.get_state(NeuronState)
        neuron_type = neuron_st.active_neuron_type
        proto_st = await self.get_state(ProtocolState)
        protocol_preset = proto_st.active_protocol_preset

        self.reset()
        neuron_st.reset()
        proto_st.reset()
        vis_st = await self.get_state(VisibilityState)
        vis_st.reset()
        analysis_st = await self.get_state(AnalysisState)
        analysis_st.reset()
        log_st = await self.get_state(LogState)
        log_st.reset()

        neuron_st._apply_neuron_preset(neuron_type)
        if protocol_preset:
            proto_st._apply_protocol_preset(protocol_preset, neuron_type)
            for key, value in presets.PROTOCOL_NEURON_OVERRIDES.get(
                protocol_preset, {}
            ).items():
                setattr(neuron_st, key, value)

        self._label_neuron_type = neuron_type
        self._figure_clamp_mode = proto_st.clamp_mode

        yield rx.call_script(
            "var gd=document.getElementById('ps-trace-plot');"
            "if(gd&&gd.data){Plotly.purge(gd);}"
        )
        yield SimulationState.run_membrane_test

    def toggle_analysis_panel(self) -> None:
        """Toggle the right-hand analysis sidebar open or closed."""
        self.analysis_panel_open = not self.analysis_panel_open

    def toggle_hover(self):
        """Toggle plot hover tooltips on or off.

        Flips ``show_hover`` and issues a client-side ``Plotly.relayout`` call
        to update the figure's ``hovermode`` without triggering a full rebuild.
        When hover is disabled, ``hovermode`` is set to ``false``.  When
        re-enabled it is restored to ``"x unified"`` for single-sweep traces or
        ``"x"`` for multi-sweep (I-V Curve) results.

        Returns:
            A ``rx.call_script`` event that applies the relayout in-browser.
        """
        self.show_hover = not self.show_hover
        if self.show_hover:
            hovermode = "x" if len(self._current_sweeps) > 1 else "x unified"
            hovermode_js = f'"{hovermode}"'
        else:
            hovermode_js = "false"
        js = (
            f"{_PLOTLY_GD_JS}"
            f"if(gd&&gd.layout)Plotly.relayout(gd,{{hovermode:{hovermode_js}}})"
        )
        return rx.call_script(js)

    # ------------------------------------------------------------------ #
    # Trace management                                                   #
    # ------------------------------------------------------------------ #
    def _apply_visibility_js(self, vis_st: "VisibilityState") -> str | None:
        """Build a JS snippet to re-apply trace visibility, hover, and sweep highlight.

        Called after any operation that triggers a full figure rebuild (run
        simulation, store trace, clear stored traces) so that traces the user
        has toggled off are correctly hidden again, the hover mode matches the
        current ``show_hover`` flag, and sweep highlight listeners are
        (re-)attached in multi-sweep mode.

        Args:
            vis_st: Current VisibilityState instance providing show_* values.
                Passed in by the caller rather than fetched via ``self.get_state``
                because this method is synchronous and ``get_state`` is async-only.

        Returns:
            A JS string that re-applies trace visibility, hover mode, and
            sweep highlight, or ``None`` when nothing needs to be applied.
        """
        trace_map = compute_trace_visibility_map(
            current_sweeps=self._current_sweeps,
            clamp_mode=self._figure_clamp_mode,
            additional_current_field_map=ADDITIONAL_CURRENT_FIELD_MAP,
            additional_gating_field_map=ADDITIONAL_GATING_FIELD_MAP,
            stored_traces=self.stored_traces,
        )
        hidden: list[int] = []
        for field_name, indices in trace_map.items():
            if not getattr(vis_st, field_name, True):
                hidden.extend(indices)

        parts: list[str] = []
        if hidden:
            parts.append(
                f"if(gd&&gd.data)Plotly.restyle(gd,"
                f"{{visible:false}},{json.dumps(hidden)});"
            )
        if not self.show_hover:
            parts.append("if(gd&&gd.layout)Plotly.relayout(gd,{hovermode:false});")

        # Inject sweep highlight listeners in multi-sweep mode.
        is_multi = len(self._current_sweeps) > 1
        if is_multi:
            parts.append(self._sweep_highlight_js())

        if not parts:
            return None
        body = "".join(parts)
        return (
            f"setTimeout(function(){{"
            f"var gd=document.getElementById('ps-trace-plot');"
            f"{body}"
            f"}},0)"
        )

    def _build_fetch_figure_js(self, token: str, vis_st: "VisibilityState") -> str:
        """Build a JS snippet that fetches the stored figure and swaps it in.

        Fetches ``/api/figure/{token}`` (JSON Plotly figure), calls
        ``Plotly.react`` to replace the current traces, then immediately
        re-applies the current visibility state (hidden traces, hover mode,
        sweep-highlight listeners).

        The entire snippet is wrapped in ``setTimeout(async fn, 0)`` so it
        runs after the Reflex state flush has finished updating the DOM.

        Args:
            token: UUID hex token identifying the figure in the side-channel
                store.
            vis_st: Current ``VisibilityState`` instance providing show_* values.

        Returns:
            A self-contained JS string suitable for ``rx.call_script``.
        """
        trace_map = compute_trace_visibility_map(
            current_sweeps=self._current_sweeps,
            clamp_mode=self._figure_clamp_mode,
            additional_current_field_map=ADDITIONAL_CURRENT_FIELD_MAP,
            additional_gating_field_map=ADDITIONAL_GATING_FIELD_MAP,
            stored_traces=self.stored_traces,
        )
        hidden: list[int] = []
        for field_name, indices in trace_map.items():
            if not getattr(vis_st, field_name, True):
                hidden.extend(indices)

        post_parts: list[str] = []
        if hidden:
            post_parts.append(
                f"Plotly.restyle(gd,{{visible:false}},{json.dumps(hidden)});"
            )
        if not self.show_hover:
            post_parts.append("Plotly.relayout(gd,{hovermode:false});")
        if len(self._current_sweeps) > 1:
            post_parts.append(self._sweep_highlight_js())
        post_js = "".join(post_parts)

        api_url = _get_config().api_url.rstrip("/")
        return (
            _FETCH_FIGURE_JS.replace("/*API_URL*/", api_url)
            .replace("/*TOKEN*/", token)
            .replace("/*DARK_AXIS_STYLE*/", json.dumps(constants.DARK_AXIS_STYLE))
            .replace("/*POST_JS*/", post_js)
        )

    def _sweep_highlight_js(self) -> str:
        """Return the sweep highlight JS with styling constants substituted.

        Returns:
            A self-executing JS function string.
        """
        return (
            _SWEEP_HIGHLIGHT_JS.replace(
                "/*DIM_OPACITY*/", str(constants.HIGHLIGHT_DIM_OPACITY)
            )
            .replace("/*HOVER_WIDTH*/", str(constants.HIGHLIGHT_HOVER_WIDTH))
            .replace("/*DIM_WIDTH*/", str(constants.HIGHLIGHT_DIM_WIDTH))
            .replace("/*SELECTED_SWEEP*/", str(self.selected_sweep))
        )

    def _rebuild_figure_and_fetch_js(
        self, stored_traces: list[Sweep], vis_st: "VisibilityState"
    ) -> str:
        """Build and store a fresh figure, return JS to fetch-and-swap it.

        Builds a figure from ``_current_sweeps`` and *stored_traces*, serialises
        it into the side-channel store under a fresh token, updates
        ``sim_token``, and returns the fetch-and-swap JS snippet ready for
        ``rx.call_script``.

        Args:
            stored_traces: Reference traces to include alongside current sweeps.
            vis_st: Current VisibilityState for post-swap visibility application.

        Returns:
            JS string for ``rx.call_script``.
        """
        fig = build_figure(
            current_sweeps=self._current_sweeps,
            visibility=TraceVisibility(),
            clamp_mode=self._figure_clamp_mode,
            stored_traces=stored_traces,
            show_hover=self.show_hover,
        )
        new_token = uuid.uuid4().hex
        traces.put(new_token, fig)
        self.sim_token = new_token
        return self._build_fetch_figure_js(new_token, vis_st)

    # ------------------------------------------------------------------ #
    # Stored trace management                                            #
    # ------------------------------------------------------------------ #
    def _do_store_trace(self) -> None:
        """Snapshot the current sweep into stored traces without applying visibility JS.

        Core logic extracted so tests can call it synchronously.
        """
        if not self.has_result:
            return
        idx = len(self.stored_traces)
        color = constants.STORED_TRACE_COLORS[idx % len(constants.STORED_TRACE_COLORS)]
        label = f"Stored {idx + 1} ({self._label_neuron_type})"
        self.stored_traces.append(
            self._current_sweeps[0].model_copy(update={"color": color, "label": label})
        )

    async def store_trace(self) -> None:
        """Snapshot the current sweep into the oscilloscope stored traces."""
        self._do_store_trace()
        vis_st = await self.get_state(VisibilityState)
        if self._current_sweeps:
            return rx.call_script(
                self._rebuild_figure_and_fetch_js(self.stored_traces, vis_st)
            )
        js = self._apply_visibility_js(vis_st)
        if js:
            return rx.call_script(js)

    def _do_clear_stored_traces(self) -> None:
        """Remove all stored traces without applying visibility JS.

        Core logic extracted so tests can call it synchronously.
        """
        self.stored_traces = []

    async def clear_stored_traces(self) -> None:
        """Remove all oscilloscope stored traces."""
        self._do_clear_stored_traces()
        vis_st = await self.get_state(VisibilityState)
        if self._current_sweeps:
            return rx.call_script(self._rebuild_figure_and_fetch_js([], vis_st))
        js = self._apply_visibility_js(vis_st)
        if js:
            return rx.call_script(js)

    def _do_apply_simulation(
        self,
        result: _SimResult,
        analysis_st: "AnalysisState",
    ) -> None:
        """Apply a :class:`_SimResult` to self and *analysis_st*.

        Core logic extracted for testability.  In production this is called
        inside ``async with self:`` (under the Reflex state lock) immediately
        after :func:`_compute_simulation` returns from the thread executor.
        Tests may call it directly on bare state instances.

        Args:
            result: Computation result from :func:`_compute_simulation`.
            analysis_st: The :class:`AnalysisState` sibling to update.
        """
        self._current_sweeps = result.sweeps
        self.sim_token = result.sim_token
        analysis_st.clear_results()
        analysis_st.iv_data = result.iv_data
        analysis_st.gv_data = result.gv_data
        analysis_st.ap_metrics = result.ap_metrics
        analysis_st.ap_summary = result.ap_summary
        analysis_st.ap_is_multi_sweep = result.ap_is_multi_sweep
        analysis_st.ca_transient_metrics = result.ca_transient_metrics
        analysis_st.ca_transient_summary = result.ca_transient_summary
        analysis_st.fi_data = result.fi_data
        analysis_st.sfa_data = result.sfa_data
        analysis_st.hyperpolarization_data = result.hyperpolarization_data
        analysis_st.phase_plane_data = result.phase_plane_data

    # ------------------------------------------------------------------ #
    # Continuous simulation mode                                        #
    # ------------------------------------------------------------------ #
    def toggle_continuous_mode(self) -> None:
        """Enable or disable continuous simulation mode."""
        if self.continuous_loop_running:
            # Signal the running loop to stop
            self.continuous_mode = False
        else:
            self.continuous_mode = True
            # Returning an event from a sync handler chains it: Reflex will
            # dispatch run_continuous immediately after toggle_continuous_mode
            # completes, without requiring this handler to be async.
            return SimulationState.run_continuous  # type: ignore[return-value]

    @rx.event(background=True)
    async def run_continuous(self) -> AsyncGenerator[Any, None]:
        """Run simulations in a continuous loop until continuous_mode is False.

        Each iteration picks up the terminal neuron state (voltage, gating
        variables, Ca²⁺) from the previous iteration, giving true continuity.
        Parameter changes made while the loop runs take effect at the start
        of the next iteration.
        """
        async with self:
            self.continuous_loop_running = True
            self.error_message = ""

        async with self:
            proto_st = await self.get_state(ProtocolState)
        logger.info(
            "Continuous simulation started: mode=%s, protocol=%s",
            proto_st.clamp_mode,
            proto_st.protocol_type,
        )
        _iteration = 0
        try:
            while True:
                _iter_start = time.monotonic()
                # Snapshot all state needed for this iteration.
                async with self:
                    if not self.continuous_mode:
                        break

                    proto_st = await self.get_state(ProtocolState)
                    mode = proto_st.clamp_mode
                    ptype = proto_st.protocol_type
                    neuron_st = await self.get_state(NeuronState)
                    neuron = neuron_st._build_neuron()
                    stimulus = proto_st._build_protocols()[0][0]
                    prior_V = self._cont_V
                    prior_ca_i = self._cont_ca_i
                    stored_traces_snap = list(self.stored_traces)
                    show_hover_snap = self.show_hover
                    # Validate that the stored gating state is compatible with the
                    # current neuron before choosing the from-state path.  A neuron
                    # switch during the previous simulation iteration can leave
                    # _cont_gating with a different set of variable names (e.g.
                    # 'r' from an Ih channel the new neuron does not have), which
                    # would raise KeyError inside _simulate_*_core.
                    _cur_gv = frozenset(gv.name for gv in neuron.all_gating_variables)
                    _stored_gv = frozenset(self._cont_gating.keys())
                    use_prior_state = self._cont_has_state and _cur_gv == _stored_gv
                    if self._cont_has_state and not use_prior_state:
                        logger.warning(
                            "Continuous mode: gating variable mismatch after "
                            "neuron change (current=%s stored=%s) — "
                            "restarting from steady state.",
                            sorted(_cur_gv),
                            sorted(_stored_gv),
                        )
                    prior_gating = dict(self._cont_gating) if use_prior_state else {}

                    _iteration += 1
                    logger.debug(
                        "Continuous iteration %d: mode=%s protocol=%s "
                        "use_prior_state=%s",
                        _iteration,
                        mode,
                        ptype,
                        use_prior_state,
                    )

                # Run simulation outside the state lock in a thread executor.
                loop = asyncio.get_running_loop()
                try:
                    if mode == CURRENT_CLAMP:
                        if use_prior_state:
                            result = await loop.run_in_executor(
                                None,
                                patch_sim.simulate_current_clamp_from_state,
                                neuron,
                                stimulus,
                                prior_V,
                                prior_gating,
                                prior_ca_i,
                            )
                        else:
                            result = await loop.run_in_executor(
                                None,
                                patch_sim.simulate_current_clamp,
                                neuron,
                                stimulus,
                            )
                    else:
                        if use_prior_state:
                            result = await loop.run_in_executor(
                                None,
                                patch_sim.simulate_voltage_clamp_from_state,
                                neuron,
                                stimulus,
                                prior_gating,
                                prior_ca_i,
                            )
                        else:
                            result = await loop.run_in_executor(
                                None,
                                patch_sim.simulate_voltage_clamp,
                                neuron,
                                stimulus,
                            )
                except ValueError as exc:
                    logger.exception("Continuous simulation error: %s", exc)
                    async with self:
                        self.error_message = str(exc)
                        self.continuous_mode = False
                    break

                # Extract terminal state for next iteration.
                last_V = float(result["voltage"][-1])
                gating_vars = {gv.name for gv in neuron.all_gating_variables}
                gating_cols = [col for col in result.dtype.names if col in gating_vars]
                last_gating = {col: float(result[col][-1]) for col in gating_cols}
                last_ca_i = (
                    float(result["ca_i"][-1]) if "ca_i" in result.dtype.names else 0.0
                )

                sweep = Sweep.from_result(result, stimulus, "", "", mode)

                # Build and serialise the figure outside the state lock so that
                # fig.to_json() does not inflate the flush time.
                _cont_fig = build_figure(
                    current_sweeps=[sweep],
                    visibility=TraceVisibility(),
                    clamp_mode=mode,
                    stored_traces=stored_traces_snap,
                    show_hover=show_hover_snap,
                )
                _cont_token = uuid.uuid4().hex
                traces.put(_cont_token, _cont_fig)

                async with self:
                    if not self.continuous_mode:
                        break
                    self._current_sweeps = [sweep]
                    self.sim_token = _cont_token
                    self._cont_V = last_V
                    self._cont_gating = last_gating
                    self._cont_ca_i = last_ca_i
                    self._cont_has_state = True
                    vis_st = await self.get_state(VisibilityState)

                yield rx.call_script(self._build_fetch_figure_js(_cont_token, vis_st))

                _iter_elapsed_ms = (time.monotonic() - _iter_start) * 1000
                if _iter_elapsed_ms > 500:
                    logger.warning(
                        "Continuous iteration %d took %.0f ms (>500 ms threshold)",
                        _iteration,
                        _iter_elapsed_ms,
                    )

                # Yield to allow UI events (parameter changes) to be processed.
                await asyncio.sleep(0)

        finally:
            async with self:
                self.continuous_mode = False
                self.continuous_loop_running = False
                logger.info(
                    "Continuous simulation stopped after %d iteration(s)", _iteration
                )
                log_st = await self.get_state(LogState)
                log_st._refresh_logs()
                vis_st = await self.get_state(VisibilityState)
            js = self._apply_visibility_js(vis_st)
            if js:
                yield rx.call_script(js)
            yield rx.call_script(_LOG_SCROLL_JS)

    @rx.event(background=True)
    async def run_simulation(self) -> AsyncGenerator[Any, None]:
        """Build protocol and run the simulation asynchronously.

        Uses background=True so the UI stays responsive during long runs.
        All inputs are snapshotted inside the state lock before blocking work
        begins; :func:`_compute_simulation` runs via ``run_in_executor`` so the
        event loop can flush ``is_running=True`` to the client first; results
        are applied under the lock via :meth:`_do_apply_simulation`.
        """
        async with self:
            self.is_running = True
            self.error_message = ""
            self.selected_sweep = -1
            neuron_st = await self.get_state(NeuronState)
            neuron = neuron_st._build_neuron()
            proto_st = await self.get_state(ProtocolState)
            mode = proto_st.clamp_mode
            ptype = proto_st.protocol_type
            try:
                protocols = proto_st._build_protocols()
            except ValueError as exc:
                logger.exception("Simulation error: %s", exc)
                self.error_message = str(exc)
                self.is_running = False
                return
            # Snapshot scalar protocol params so the executor call is thread-safe.
            stored_traces_snap = list(self.stored_traces)
            show_hover_snap = self.show_hover
            min_stimulus = proto_st.min_stimulus
            max_stimulus = proto_st.max_stimulus
            stimulus_step = proto_st.stimulus_step
            pre_stimulus_duration = proto_st.pre_stimulus_duration
            stimulus_duration = proto_st.stimulus_duration

        logger.info("Simulation started: mode=%s, protocol=%s", mode, ptype)

        _start_ms = time.monotonic() * 1000
        _sim_result: _SimResult | None = None
        try:
            loop = asyncio.get_running_loop()
            _sim_result = await loop.run_in_executor(
                None,
                _compute_simulation,
                neuron,
                protocols,
                mode,
                stored_traces_snap,
                show_hover_snap,
                min_stimulus,
                max_stimulus,
                stimulus_step,
                pre_stimulus_duration,
                stimulus_duration,
            )
        except ValueError as exc:
            logger.exception("Simulation error: %s", exc)
            async with self:
                self.error_message = str(exc)
        else:
            elapsed = time.monotonic() * 1000 - _start_ms
            logger.info("Simulation complete: %.0f ms", elapsed)
            async with self:
                analysis_st = await self.get_state(AnalysisState)
                self._do_apply_simulation(_sim_result, analysis_st)
            yield SimulationState.run_membrane_test
        finally:
            async with self:
                self.is_running = False
                log_st = await self.get_state(LogState)
                log_st._refresh_logs()
                vis_st = await self.get_state(VisibilityState)
            if _sim_result is not None:
                # Side-channel path: fetch the figure from /api/figure/{token}
                # and swap it in via Plotly.react (includes visibility re-apply).
                yield rx.call_script(
                    self._build_fetch_figure_js(_sim_result.sim_token, vis_st)
                )
            else:
                # Error path: no figure was produced; re-apply existing visibility.
                js = self._apply_visibility_js(vis_st)
                if js:
                    yield rx.call_script(js)
            yield rx.call_script(_LOG_SCROLL_JS)

    @rx.event(background=True)
    async def run_membrane_test_debounced(self) -> AsyncGenerator[Any, None]:
        """Debounced entry point for slider-driven membrane test requests.

        Increments ``_mt_request_id``, sleeps :data:`_MT_DEBOUNCE_S` seconds,
        then bails out if a newer request arrived.  Otherwise yields
        :meth:`run_membrane_test`.  Page-load and post-simulation triggers
        call :meth:`run_membrane_test` directly and are unaffected.
        """
        async with self:
            self._mt_request_id += 1
            request_id = self._mt_request_id

        await asyncio.sleep(_MT_DEBOUNCE_S)

        async with self:
            if self._mt_request_id != request_id:
                return

        yield SimulationState.run_membrane_test

    @rx.event(background=True)
    async def run_membrane_test(self) -> None:
        """Run the dedicated membrane test protocol and cache passive properties.

        Builds the neuron from current NeuronState parameters, checks whether
        the neuron fingerprint has changed since the last run (cache hit → skip),
        and if stale runs a fixed small hyperpolarising current step via
        :func:`patch_sim.run_membrane_test`.  Results are stored in
        ``AnalysisState.mt_*`` fields and persist across protocol and simulation
        changes until the neuron parameters change.
        """
        async with self:
            neuron_st = await self.get_state(NeuronState)
            # Use _compute_fingerprint() rather than the @rx.var: ComputedVar
            # values are cached by Reflex and may not yet reflect the latest
            # state when accessed from a background task.
            fingerprint = neuron_st._compute_fingerprint()
            area_cm2 = (
                neuron_st.area_cm2
                if (neuron_st.has_area_cm2 and neuron_st.area_cm2 > 0)
                else None
            )
            analysis_st = await self.get_state(AnalysisState)
            if (
                analysis_st.mt_neuron_fingerprint == fingerprint
                and analysis_st.mt_props_available
            ):
                # Sim cache hit: re-derive display from cached raw values
                # against the current area state without re-running the
                # membrane-test simulation.
                self._apply_membrane_test_display(analysis_st, area_cm2)
                logger.debug(
                    "run_membrane_test: cache hit, refreshed display only"
                )
                return
            neuron = neuron_st._build_neuron()

        loop = asyncio.get_running_loop()
        props = await loop.run_in_executor(None, patch_sim.run_membrane_test, neuron)

        async with self:
            # Re-read fingerprint: if neuron changed while we were computing,
            # discard results to avoid caching stale passive properties.
            neuron_st = await self.get_state(NeuronState)
            current_fp = neuron_st._compute_fingerprint()
            if current_fp != fingerprint:
                logger.info(
                    "run_membrane_test: neuron changed during computation, discarding"
                )
                return
            analysis_st = await self.get_state(AnalysisState)
            if props is None:
                analysis_st.mt_props_available = False
                analysis_st.mt_raw_r_in_density = 0.0
                analysis_st.mt_raw_c_m_density = 0.0
                analysis_st.mt_raw_has_c_m = False
                analysis_st.mt_raw_tau_m = 0.0
                analysis_st.mt_fit_converged = False
            else:
                analysis_st.mt_props_available = True
                analysis_st.mt_raw_r_in_density = float(props.input_resistance)
                analysis_st.mt_raw_tau_m = float(props.time_constant)
                if props.membrane_capacitance is not None:
                    analysis_st.mt_raw_c_m_density = float(props.membrane_capacitance)
                    analysis_st.mt_raw_has_c_m = True
                else:
                    analysis_st.mt_raw_c_m_density = 0.0
                    analysis_st.mt_raw_has_c_m = False
                analysis_st.mt_fit_converged = props.fit_converged
            self._apply_membrane_test_display(analysis_st, area_cm2)
            analysis_st.mt_neuron_fingerprint = fingerprint
            logger.info(
                "run_membrane_test: R_in=%s %s, τ_m=%s ms, C_m=%s %s",
                analysis_st.mt_input_resistance,
                analysis_st.mt_r_units,
                analysis_st.mt_time_constant,
                analysis_st.mt_membrane_capacitance,
                analysis_st.mt_c_units,
            )

    @staticmethod
    def _apply_membrane_test_display(
        analysis_st: "AnalysisState", area_cm2: float | None
    ) -> None:
        """Populate AnalysisState display strings from cached raw props.

        Reads ``mt_raw_*`` fields and ``area_cm2`` and writes the formatted
        ``mt_input_resistance`` / ``mt_membrane_capacitance`` /
        ``mt_time_constant`` strings plus ``mt_r_units`` / ``mt_c_units`` /
        ``mt_units_mode``.  When the cached props were unavailable, all
        displays are em-dashes.

        Args:
            analysis_st: AnalysisState instance to update in place.
            area_cm2: Membrane area in cm², or ``None`` for density mode.
        """
        em_dash = "—"
        density_r_units = "kΩ·cm²"
        density_c_units = "µF/cm²"
        if not analysis_st.mt_props_available:
            analysis_st.mt_input_resistance = em_dash
            analysis_st.mt_time_constant = em_dash
            analysis_st.mt_membrane_capacitance = em_dash
            analysis_st.mt_r_units = density_r_units
            analysis_st.mt_c_units = density_c_units
            analysis_st.mt_units_mode = "density"
            return

        analysis_st.mt_time_constant = f"{analysis_st.mt_raw_tau_m:.2f}"

        if area_cm2 is not None and area_cm2 > 0:
            r_mohm = patch_sim.density_to_absolute_r_in(
                analysis_st.mt_raw_r_in_density, area_cm2
            )
            c_pf = (
                patch_sim.density_to_absolute_c_m(
                    analysis_st.mt_raw_c_m_density, area_cm2
                )
                if analysis_st.mt_raw_has_c_m
                else None
            )
            analysis_st.mt_input_resistance = (
                f"{r_mohm:.2f}" if r_mohm is not None else em_dash
            )
            analysis_st.mt_membrane_capacitance = (
                f"{c_pf:.2f}" if c_pf is not None else em_dash
            )
            analysis_st.mt_r_units = "MΩ"
            analysis_st.mt_c_units = "pF"
            analysis_st.mt_units_mode = "absolute"
        else:
            analysis_st.mt_input_resistance = (
                f"{analysis_st.mt_raw_r_in_density:.2f}"
            )
            analysis_st.mt_membrane_capacitance = (
                f"{analysis_st.mt_raw_c_m_density:.2f}"
                if analysis_st.mt_raw_has_c_m
                else em_dash
            )
            analysis_st.mt_r_units = density_r_units
            analysis_st.mt_c_units = density_c_units
            analysis_st.mt_units_mode = "density"
