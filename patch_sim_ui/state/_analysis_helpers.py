"""Pure analysis computation helpers used by SimulationState.

All functions in this module are stateless and have no Reflex dependency.
They convert raw simulation results into serialised dicts suitable for use
as Reflex state variables (i.e. JSON-serialisable plain Python objects).

Keeping these separate from simulation.py makes each piece independently
testable and keeps SimulationState focused on Reflex lifecycle concerns.
"""

import logging
from typing import Any

import numpy as np

import patch_sim
from patch_sim.analysis.fi_curve import _fi_point_from_ap_result
from patch_sim.constants import CURRENT_CLAMP
from patch_sim_ui.plotting import Sweep

logger = logging.getLogger(__name__)

#: Number of voltage points used for the pre-computed Boltzmann fit curve.
_GV_FIT_POINTS = 200


# ------------------------------------------------------------------ #
# Spike / AP formatting                                              #
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


# ------------------------------------------------------------------ #
# Phase-plane                                                        #
# ------------------------------------------------------------------ #


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


# ------------------------------------------------------------------ #
# Multi-sweep analysis                                               #
# ------------------------------------------------------------------ #


def _compute_cc_multi_sweep_analysis(
    sweeps: "list[Sweep]",
    min_stimulus: float,
    max_stimulus: float,
    stimulus_step: float,
    pre_stimulus_duration: float,
    stimulus_duration: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compute AP metrics, F-I data, and SFA data from multi-sweep CC results.

    Runs spike detection once per sweep and derives all three outputs from those
    results, avoiding the redundant ``analyze_aps`` call that would occur if
    the AP, F-I, and SFA helpers were called separately.

    Args:
        sweeps: Ordered list of :class:`Sweep` objects from the simulation.
        min_stimulus: Minimum injected current step (µA/cm²).
        max_stimulus: Maximum injected current step (µA/cm²).
        stimulus_step: Step size between current commands (µA/cm²).
        pre_stimulus_duration: Duration before the step begins (ms).
        stimulus_duration: Duration of the current step (ms).

    Returns:
        A 4-tuple ``(ap_metrics, ap_summary, fi_data, sfa_data)`` where
        ``ap_metrics`` is a list of per-spike dicts (pooled and renumbered
        across all sweeps), ``ap_summary`` is an aggregate dict without
        ``mean_isi`` / ``firing_rate`` (those are shown in the F-I curve),
        ``fi_data`` is a serialised :class:`~patch_sim.FIAnalysisResult` dict,
        and ``sfa_data`` is a serialised SFA dict with one curve per sweep that
        had at least two spikes.  ``ap_metrics`` and ``ap_summary`` are empty
        when no spikes are detected.  ``fi_data`` is empty when the sweep count
        does not match the derived step count.
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

    # --- F-I data (derived from the same per-sweep AP results) ---
    if len(sweeps) != len(current_steps):
        logger.warning(
            "F-I analysis skipped: %d sweeps but %d current steps derived "
            "from protocol (min=%.3g, max=%.3g, step=%.3g)",
            len(sweeps),
            len(current_steps),
            min_stimulus,
            max_stimulus,
            stimulus_step,
        )
        if ap_summary:
            ap_summary["rheobase"] = "\u2014"
        return ap_metrics, ap_summary, {}, sfa_data

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
    return ap_metrics, ap_summary, fi_data, sfa_data


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
