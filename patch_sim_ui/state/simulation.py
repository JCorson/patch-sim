"""Simulation state for the patch_sim web UI.

SimulationState owns simulation results, sweep collections, continuous mode,
and the figure computed var.  Cross-cutting state lives in the sibling
substates (NeuronState, ProtocolState, VisibilityState, AnalysisState, LogState).
"""

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator

import numpy as np
import plotly.graph_objects as go
import reflex as rx

import patch_sim
import patch_sim.clamp_simulations
from patch_sim.analysis.fi_curve import _fi_point_from_ap_result
from patch_sim.constants import (
    CURRENT_CLAMP,
    VOLTAGE_CLAMP,
)
from patch_sim_ui import constants
from patch_sim_ui.plotting import (
    Sweep,
    TraceVisibility,
    build_figure,
    compute_trace_visibility_map,
)
from patch_sim_ui.state._common import (
    _ADDITIONAL_CURRENT_FIELD_MAP,
    _ADDITIONAL_GATING_FIELD_MAP,
    _LOG_SCROLL_JS,
    _PLOTLY_GD_JS,
)
from patch_sim_ui.state.analysis import AnalysisState
from patch_sim_ui.state.log import LogState
from patch_sim_ui.state.neuron import NeuronState
from patch_sim_ui.state.protocol import ProtocolState
from patch_sim_ui.state.visibility import VisibilityState

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Module-level helpers used only by SimulationState                  #
# ------------------------------------------------------------------ #

# Client-side sweep highlight / selection module.  Injected via
# rx.call_script() after every figure render in multi-sweep mode.
_SWEEP_HIGHLIGHT_JS = """
(function() {
  // State object persists across re-inits and retries.
  if (!window._psSweep) window._psSweep = {};
  var S = window._psSweep;

  // Cancel any pending retry so we don't get duplicate inits.
  if (S._initTimer) { clearTimeout(S._initTimer); S._initTimer = null; }

  function setup(retries) {
    var gd = document.querySelector('.js-plotly-plot');
    // Plotly may not have rendered yet; retry up to 10 times (1 s total).
    if (!gd || !gd.data || !gd.data.length) {
      if (retries > 0) {
        S._initTimer = setTimeout(function() { setup(retries - 1); }, 100);
      }
      return;
    }

    // Build sweep-to-trace index map from meta.sweep.
    var sweepMap = {};   // sweepIdx -> [traceIdx, ...]
    var maxSweep = -1;
    var origOpacity = [];
    var origWidth = [];
    var origShowlegend = [];  // original showlegend for each trace
    for (var i = 0; i < gd.data.length; i++) {
      var m = gd.data[i].meta;
      var si = (m && typeof m.sweep === 'number') ? m.sweep : -1;
      origOpacity[i] = (gd.data[i].opacity != null) ? gd.data[i].opacity : 1;
      origWidth[i] = (gd.data[i].line && gd.data[i].line.width != null)
                     ? gd.data[i].line.width : 2;
      origShowlegend[i] = gd.data[i].showlegend === true;
      if (si >= 0) {
        if (!sweepMap[si]) sweepMap[si] = [];
        sweepMap[si].push(i);
        if (si > maxSweep) maxSweep = si;
      }
    }
    S.sweepMap = sweepMap;
    S.nSweeps = maxSweep + 1;
    S.origOpacity = origOpacity;
    S.origWidth = origWidth;
    S.origShowlegend = origShowlegend;
    // legendPositions[p] = true means position p within a sweep's trace list
    // originally had showlegend=true (based on sweep 0).
    var legendPositions = {};
    if (sweepMap[0]) {
      for (var p = 0; p < sweepMap[0].length; p++) {
        if (origShowlegend[sweepMap[0][p]]) legendPositions[p] = true;
      }
    }
    S.legendPositions = legendPositions;
    S.selectedSweep = /*SELECTED_SWEEP*/;

    if (S.nSweeps <= 1) {
      // Single-sweep mode — remove listeners and bail.
      if (S._cleanup) { S._cleanup(); S._cleanup = null; }
      return;
    }

    function _applySweepStyle(activeSweep, dimOpacity) {
      if (S.nSweeps <= 1) return;
      var opacities = new Array(gd.data.length);
      var widths = new Array(gd.data.length);
      var showlegends = new Array(gd.data.length);
      for (var i = 0; i < gd.data.length; i++) {
        var m = gd.data[i].meta;
        var si = (m && typeof m.sweep === 'number') ? m.sweep : -1;
        if (si < 0 || gd.data[i].visible === false) {
          opacities[i] = S.origOpacity[i];
          widths[i] = S.origWidth[i];
          showlegends[i] = S.origShowlegend[i];
        } else if (si === activeSweep) {
          opacities[i] = 1;
          widths[i] = S.origWidth[i];
          // Show legend entry for this trace if it occupies a legend position.
          var pos = S.sweepMap[si] ? S.sweepMap[si].indexOf(i) : -1;
          showlegends[i] = pos >= 0 && S.legendPositions[pos] === true;
        } else {
          opacities[i] = dimOpacity;
          widths[i] = /*DIM_WIDTH*/;
          showlegends[i] = false;
        }
      }
      var indices = [];
      for (var i = 0; i < gd.data.length; i++) indices.push(i);
      var styleUpdate = {
        'opacity': opacities, 'line.width': widths, 'showlegend': showlegends
      };
      Plotly.restyle(gd, styleUpdate, indices);
    }

    function _clearStyle() {
      var opacities = [];
      var widths = [];
      var showlegends = [];
      for (var i = 0; i < gd.data.length; i++) {
        opacities.push(S.origOpacity[i]);
        widths.push(S.origWidth[i]);
        showlegends.push(S.origShowlegend[i]);
      }
      var indices = [];
      for (var i = 0; i < gd.data.length; i++) indices.push(i);
      var styleUpdate = {
        'opacity': opacities, 'line.width': widths, 'showlegend': showlegends
      };
      Plotly.restyle(gd, styleUpdate, indices);
    }

    function _applyHoverHighlight(activeSweep) {
      if (S.nSweeps <= 1) return;
      var widths = new Array(gd.data.length);
      for (var i = 0; i < gd.data.length; i++) {
        var m = gd.data[i].meta;
        var si = (m && typeof m.sweep === 'number') ? m.sweep : -1;
        if (si === activeSweep) {
          widths[i] = /*HOVER_WIDTH*/;
        } else {
          widths[i] = S.origWidth[i];
        }
      }
      var indices = [];
      for (var i = 0; i < gd.data.length; i++) indices.push(i);
      Plotly.restyle(gd, {'line.width': widths}, indices);
    }

    function _selectSweep(idx) {
      S.selectedSweep = idx;
      _applySweepStyle(idx, /*DIM_OPACITY*/);
    }

    function _deselect() {
      S.selectedSweep = -1;
      _clearStyle();
    }

    // Resolve which sweep is nearest to the mouse cursor.
    // Converts the raw MouseEvent pixel position to data coordinates using
    // Plotly's internal axis layout, then finds the sweep whose trace value
    // at that x-position is closest to the cursor's y-position.
    //
    // ⚠ Plotly private API (tested against Plotly.js 2.x / 3.x):
    //   gd._fullLayout  — computed layout with pixel geometry (_size, _offset, _length)
    //   gd._fullData[i] — fully-resolved trace data with typed arrays decoded
    // If Plotly restructures these internals in a future release this function
    // will silently return -1 (no sweep resolved) without breaking anything else.
    //
    // Note: gd.data[i].x holds a Plotly v3 binary descriptor {dtype,bdata};
    // decoded typed arrays live in gd._fullData[i].x.
    function _resolveSweepFromMouse(evt) {
      if (!evt || !evt.event || !gd._fullLayout) return -1;
      var fl = gd._fullLayout;
      if (!fl._size) return -1;
      var rect = gd.getBoundingClientRect();
      // ya._offset is measured from the figure div top, so py must be too.
      // px is relative to plot area (after left margin) because xa._offset=0.
      var px = evt.event.clientX - rect.left - fl._size.l;
      var py = evt.event.clientY - rect.top;

      // Identify which subplot row contains the cursor.
      var matchedYa = null;
      var matchedYaKey = null;
      var yAxisKeys = ['yaxis', 'yaxis2', 'yaxis3'];
      for (var k = 0; k < yAxisKeys.length; k++) {
        var ya = fl[yAxisKeys[k]];
        if (!ya || ya._length == null) continue;
        if (py >= ya._offset && py <= ya._offset + ya._length) {
          matchedYa = ya;
          matchedYaKey = (yAxisKeys[k] === 'yaxis')
            ? 'y' : yAxisKeys[k].replace('yaxis', 'y');
          break;
        }
      }
      if (!matchedYa) return -1;

      // Convert pixel X to data X using the shared primary xaxis.
      var xa = fl.xaxis;
      if (!xa || !xa._length) return -1;
      var dataX = xa.range[0]
        + ((px - (xa._offset || 0)) / xa._length)
        * (xa.range[1] - xa.range[0]);

      // Convert pixel Y to data Y (screen top = data max).
      var pyInAxis = py - matchedYa._offset;
      var dataY = matchedYa.range[1]
        - (pyInAxis / matchedYa._length)
        * (matchedYa.range[1] - matchedYa.range[0]);

      // Find the sweep whose trace at dataX is closest to dataY.
      // Search only in the matched subplot axis — cross-axis comparison is
      // invalid because each subplot has different units and scale.
      var bestSweep = -1;
      var bestDist = Infinity;
      for (var _i = 0; _i < gd.data.length; _i++) {
        var _td = gd.data[_i];
        var _tm = _td.meta;
        var _tsi = (_tm && typeof _tm.sweep === 'number') ? _tm.sweep : -1;
        if (_tsi < 0) continue;
        var _tya = _td.yaxis || 'y';
        if (_tya !== matchedYaKey) continue;
        // Include hidden traces — we resolve by data proximity, not visibility.
        var _fd = gd._fullData[_i];
        var xArr = _fd && _fd.x;
        var yArr = _fd && _fd.y;
        if (!xArr || !yArr || !xArr.length) continue;
        // Binary search for the nearest x index.
        // xArr is the simulation time axis — always monotonically increasing.
        var lo = 0, hi = xArr.length - 1;
        while (lo < hi) {
          var mid = (lo + hi) >> 1;
          if (xArr[mid] < dataX) lo = mid + 1; else hi = mid;
        }
        var traceY = yArr[lo];
        if (traceY == null || traceY !== traceY) continue;  // null or NaN
        var dist = Math.abs(traceY - dataY);
        if (dist < bestDist) { bestDist = dist; bestSweep = _tsi; }
      }
      return bestSweep;
    }

    // Re-apply if a sweep was already selected (e.g. after figure rebuild).
    if (S.selectedSweep >= 0 && S.selectedSweep < S.nSweeps) {
      _applySweepStyle(S.selectedSweep, /*DIM_OPACITY*/);
    }

    // Remove old listeners before attaching new ones.
    if (S._cleanup) { S._cleanup(); S._cleanup = null; }

    function onNativeClick(nativeEvt) {
      var si = _resolveSweepFromMouse({event: nativeEvt});
      if (si >= 0) {
        if (S.selectedSweep === si) { _deselect(); }
        else { _selectSweep(si); }
      } else {
        _deselect();
      }
    }

    var _moveTimer = null;
    function onNativeMousemove(nativeEvt) {
      if (S.selectedSweep >= 0) return;
      // Debounce: skip if a frame is already scheduled (~16 ms / one frame).
      if (_moveTimer) return;
      _moveTimer = setTimeout(function() { _moveTimer = null; }, 16);
      var si = _resolveSweepFromMouse({event: nativeEvt});
      if (si >= 0) { _applyHoverHighlight(si); }
      else { _clearStyle(); }
    }

    function onNativeMouseleave() {
      if (S.selectedSweep >= 0) return;
      _clearStyle();
    }

    function onKeydown(evt) {
      if (S.nSweeps <= 1) return;
      if (!document.querySelector('.js-plotly-plot')) return;
      if (evt.key === 'Escape') {
        _deselect();
      } else if (evt.key === 'ArrowUp') {
        evt.preventDefault();
        var next = (S.selectedSweep < 0) ? 0 : (S.selectedSweep + 1) % S.nSweeps;
        _selectSweep(next);
      } else if (evt.key === 'ArrowDown') {
        evt.preventDefault();
        var prev = (S.selectedSweep < 0)
          ? S.nSweeps - 1
          : (S.selectedSweep - 1 + S.nSweeps) % S.nSweeps;
        _selectSweep(prev);
      }
    }

    gd.addEventListener('click', onNativeClick);
    gd.addEventListener('mousemove', onNativeMousemove);
    gd.addEventListener('mouseleave', onNativeMouseleave);
    document.addEventListener('keydown', onKeydown);

    S._cleanup = function() {
      gd.removeEventListener('click', onNativeClick);
      gd.removeEventListener('mousemove', onNativeMousemove);
      gd.removeEventListener('mouseleave', onNativeMouseleave);
      document.removeEventListener('keydown', onKeydown);
    };
  }

  setup(10);
})();
"""


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

    # --- Passive properties (first subthreshold, non-zero-current sweep) ---
    passive = None
    for ap_result, sweep, i_step in zip(per_sweep_ap, sweeps, current_steps):
        if ap_result.spike_count == 0 and i_step != 0.0:
            passive = patch_sim.analyze_passive_properties(
                time_arr,
                np.array(sweep.voltage),
                current_amplitude=i_step,
                stim_start_ms=stim_start,
                stim_end_ms=stim_end,
            )
            break

    def _fmt_passive() -> dict[str, Any]:
        """Serialise passive property results to display strings.

        Returns:
            Dict with ``input_resistance``, ``time_constant``, and
            ``membrane_capacitance`` as pre-formatted strings.
        """
        if passive is None:
            return {
                "input_resistance": "\u2014",
                "time_constant": "\u2014",
                "membrane_capacitance": "\u2014",
            }
        return {
            "input_resistance": f"{passive.input_resistance:.2f}",
            "time_constant": f"{passive.time_constant:.2f}",
            "membrane_capacitance": (
                f"{passive.membrane_capacitance:.2f}"
                if passive.membrane_capacitance is not None
                else "\u2014"
            ),
        }

    # --- AP metrics (pooled across all sweeps) ---
    all_spikes = [spike for ap in per_sweep_ap for spike in ap.spikes]

    if all_spikes:
        ap_metrics: list[dict[str, Any]] = [
            {
                "index": i,
                "threshold_voltage": f"{s.threshold_voltage:.1f}",
                "peak_voltage": f"{s.peak_voltage:.1f}",
                "rise_time": f"{s.rise_time:.2f}",
                "half_width": f"{s.half_width:.2f}",
                "ahp_depth": (
                    f"{s.ahp_depth:.1f}" if s.ahp_depth is not None else "\u2014"
                ),
            }
            for i, s in enumerate(all_spikes)
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
            **_fmt_passive(),
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


_GV_FIT_POINTS = 200  # number of voltage points for the pre-computed Boltzmann curve


def _compute_gv_data(
    iv_result: "patch_sim.IVAnalysisResult",
    reversal_potential: float,
) -> "dict[str, Any]":
    """Compute g-V analysis data from an I-V result and a reversal potential.

    Calls :func:`patch_sim.compute_gv` to derive normalised conductance and fit
    a Boltzmann sigmoid.  A dense voltage array (200 points) spanning the range
    of included steps is pre-computed so the plotting function can draw a smooth
    fit curve without importing scipy.

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


class SimulationState(rx.State):
    """State for simulation results, sweep collections, and figure rendering."""

    # Synced copy of NeuronState.active_neuron_type used by store_trace for
    # labelling (avoids making those handlers async).
    _label_neuron_type: str = "Squid Giant Axon (Classic HH)"
    # Synced copy of ProtocolState.clamp_mode used by figure_data and
    # _apply_visibility_js (both are synchronous, cannot call get_state).
    _figure_clamp_mode: str = CURRENT_CLAMP

    # ------------------------------------------------------------------ #
    # Simulation results                                                  #
    # ------------------------------------------------------------------ #
    current_sweeps: list[Sweep] = []  # Latest simulation result
    stored_traces: list[Sweep] = []  # Oscilloscope-style stored reference traces

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
        return len(self.current_sweeps) > 0

    @rx.var
    def has_stored_traces(self) -> bool:
        """Whether any oscilloscope-stored reference traces exist."""
        return len(self.stored_traces) > 0

    @rx.var
    def is_multi_sweep(self) -> bool:
        """Whether the current simulation has multiple sweeps (e.g. I-V protocol)."""
        return len(self.current_sweeps) > 1

    @rx.var
    def continuous_active(self) -> bool:
        """True when continuous mode is enabled and the loop is running."""
        return self.continuous_mode and self.continuous_loop_running

    @rx.var
    def figure_data(self) -> go.Figure:
        """Plotly figure rebuilt when sweeps, clamp mode, or hover state change.

        All traces are built with full visibility; toggling show_* flags is
        handled client-side via ``Plotly.restyle`` so that figure rebuilds are
        not triggered by visibility changes.  The ``show_hover`` flag is
        respected here so that hovermode is baked into the figure data and takes
        effect immediately, even without a client-side relayout.

        Dark/light theming is applied client-side by the ``rx.plotly``
        component via its ``layout`` and ``template`` props, so no server-side
        colour mode state is needed.
        """
        return build_figure(
            current_sweeps=self.current_sweeps,
            visibility=TraceVisibility(),  # all visible; toggling handled client-side
            clamp_mode=self._figure_clamp_mode,
            stored_traces=self.stored_traces,
            show_hover=self.show_hover,
        )

    # ------------------------------------------------------------------ #
    # Event handlers                                                     #
    # ------------------------------------------------------------------ #
    def _clear_for_new_protocol(self) -> None:
        """Clear all simulation results and reset continuous state.

        Called by ProtocolState when the clamp mode or a protocol preset
        changes, ensuring stale results are not shown for the new protocol.
        """
        self.current_sweeps = []
        self.stored_traces = []
        self._cont_has_state = False
        self.selected_sweep = -1

    async def reset_to_defaults(self) -> None:
        """Reset all state vars across all substates to their class-level defaults."""
        self.reset()
        neuron_st = await self.get_state(NeuronState)
        neuron_st.reset()
        proto_st = await self.get_state(ProtocolState)
        proto_st.reset()
        vis_st = await self.get_state(VisibilityState)
        vis_st.reset()
        analysis_st = await self.get_state(AnalysisState)
        analysis_st.reset()
        log_st = await self.get_state(LogState)
        log_st.reset()

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
            hovermode = "x" if len(self.current_sweeps) > 1 else "x unified"
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
            current_sweeps=self.current_sweeps,
            clamp_mode=self._figure_clamp_mode,
            additional_current_field_map=_ADDITIONAL_CURRENT_FIELD_MAP,
            additional_gating_field_map=_ADDITIONAL_GATING_FIELD_MAP,
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
        is_multi = len(self.current_sweeps) > 1
        if is_multi:
            parts.append(self._sweep_highlight_js())

        if not parts:
            return None
        body = "".join(parts)
        return (
            f"setTimeout(function(){{"
            f"var gd=document.querySelector('.js-plotly-plot');"
            f"{body}"
            f"}},0)"
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
            self.current_sweeps[0].model_copy(update={"color": color, "label": label})
        )

    async def store_trace(self) -> None:
        """Snapshot the current sweep into the oscilloscope stored traces."""
        self._do_store_trace()
        vis_st = await self.get_state(VisibilityState)
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
        js = self._apply_visibility_js(vis_st)
        if js:
            return rx.call_script(js)

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
                    use_prior_state = self._cont_has_state
                    prior_V = self._cont_V
                    prior_gating = dict(self._cont_gating)
                    prior_ca_i = self._cont_ca_i

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

                async with self:
                    if not self.continuous_mode:
                        break
                    self.current_sweeps = [sweep]
                    self._cont_V = last_V
                    self._cont_gating = last_gating
                    self._cont_ca_i = last_ca_i
                    self._cont_has_state = True

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
        State is snapshotted inside the lock before any blocking work begins,
        and all blocking simulation calls are offloaded via run_in_executor so
        the event loop can flush the is_running=True update to the client
        before the computation starts.
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

        _start_ms = time.monotonic() * 1000
        logger.info(
            "Simulation started: mode=%s, protocol=%s",
            mode,
            ptype,
        )
        loop = asyncio.get_running_loop()
        sim_fn = (
            patch_sim.simulate_current_clamp
            if mode == CURRENT_CLAMP
            else patch_sim.simulate_voltage_clamp
        )
        is_multi = len(protocols) > 1
        try:
            if is_multi:
                # Run each sweep independently so gating variables are reset
                # between steps — matching real patch-clamp I-V experiments.
                def _run_batch() -> list[Sweep]:
                    """Run all sweeps via simulate_batch and assemble Sweep list."""
                    new_sweeps: list[Sweep] = []
                    for sweep_result, (protocol, label) in zip(
                        patch_sim.simulate_batch(
                            neuron, [p for p, _ in protocols], sim_fn
                        ),
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
                    return new_sweeps

                new_sweeps = await loop.run_in_executor(None, _run_batch)
                async with self:
                    self.current_sweeps = new_sweeps
                    analysis_st = await self.get_state(AnalysisState)
                    if mode == VOLTAGE_CLAMP:
                        analysis_st.ap_metrics = []
                        analysis_st.ap_summary = {}
                        analysis_st.ap_is_multi_sweep = False
                        analysis_st.fi_data = {}
                        analysis_st.sfa_data = {}
                        iv_data, iv_result = _compute_iv_data(
                            new_sweeps,
                            proto_st.min_stimulus,
                            proto_st.max_stimulus,
                            proto_st.stimulus_step,
                            proto_st.pre_stimulus_duration,
                            proto_st.stimulus_duration,
                        )
                        analysis_st.iv_data = iv_data
                        if iv_result is not None:
                            na_channel = next(
                                (
                                    ch
                                    for ch in neuron.core_channels
                                    if isinstance(
                                        ch.reversal_spec, patch_sim.NernstSpec
                                    )
                                    and ch.reversal_spec.species
                                    is patch_sim.IonSpecies.SODIUM
                                ),
                                None,
                            )
                            if na_channel is not None:
                                e_rev = na_channel.reversal_potential(neuron)
                                analysis_st.gv_data = _compute_gv_data(iv_result, e_rev)
                            else:
                                analysis_st.gv_data = {}
                        else:
                            analysis_st.gv_data = {}
                    else:
                        ms_metrics, ms_summary, ms_fi, ms_sfa = (
                            _compute_cc_multi_sweep_analysis(
                                new_sweeps,
                                proto_st.min_stimulus,
                                proto_st.max_stimulus,
                                proto_st.stimulus_step,
                                proto_st.pre_stimulus_duration,
                                proto_st.stimulus_duration,
                            )
                        )
                        analysis_st.ap_metrics = ms_metrics
                        analysis_st.ap_summary = ms_summary
                        analysis_st.ap_is_multi_sweep = True
                        analysis_st.iv_data = {}
                        analysis_st.gv_data = {}
                        analysis_st.fi_data = ms_fi
                        analysis_st.sfa_data = ms_sfa

            else:
                stimulus, _ = protocols[0]
                result = await loop.run_in_executor(None, sim_fn, neuron, stimulus)
                async with self:
                    self.current_sweeps = [
                        Sweep.from_result(result, stimulus, "", "", mode)
                    ]
                    analysis_st = await self.get_state(AnalysisState)
                    if mode == CURRENT_CLAMP:
                        ap_result = patch_sim.analyze_aps_from_result(result)
                        sfa_curve = patch_sim.compute_sfa(ap_result)
                        passive = patch_sim.analyze_passive_from_result(
                            result,
                            current_amplitude=proto_st.min_stimulus,
                            stim_start_ms=proto_st.pre_stimulus_duration,
                            stim_end_ms=(
                                proto_st.pre_stimulus_duration
                                + proto_st.stimulus_duration
                            ),
                        )
                        analysis_st.ap_metrics = [
                            {
                                "index": s.index,
                                "threshold_voltage": f"{s.threshold_voltage:.1f}",
                                "peak_voltage": f"{s.peak_voltage:.1f}",
                                "rise_time": f"{s.rise_time:.2f}",
                                "half_width": f"{s.half_width:.2f}",
                                "ahp_depth": (
                                    f"{s.ahp_depth:.1f}"
                                    if s.ahp_depth is not None
                                    else "\u2014"
                                ),
                            }
                            for s in ap_result.spikes
                        ]
                        analysis_st.ap_summary = {
                            "spike_count": str(ap_result.spike_count),
                            "mean_threshold_voltage": (
                                f"{ap_result.mean_threshold_voltage:.1f}"
                                if ap_result.mean_threshold_voltage is not None
                                else "\u2014"
                            ),
                            "mean_peak_voltage": (
                                f"{ap_result.mean_peak_voltage:.1f}"
                                if ap_result.mean_peak_voltage is not None
                                else "\u2014"
                            ),
                            "mean_rise_time": (
                                f"{ap_result.mean_rise_time:.2f}"
                                if ap_result.mean_rise_time is not None
                                else "\u2014"
                            ),
                            "mean_half_width": (
                                f"{ap_result.mean_half_width:.2f}"
                                if ap_result.mean_half_width is not None
                                else "\u2014"
                            ),
                            "mean_ahp_depth": (
                                f"{ap_result.mean_ahp_depth:.1f}"
                                if ap_result.mean_ahp_depth is not None
                                else "\u2014"
                            ),
                            "mean_isi": (
                                f"{ap_result.mean_isi:.1f}"
                                if ap_result.mean_isi is not None
                                else "\u2014"
                            ),
                            "firing_rate": (
                                f"{ap_result.firing_rate:.1f}"
                                if ap_result.firing_rate is not None
                                else "\u2014"
                            ),
                            "adaptation_index": (
                                f"{sfa_curve.adaptation_index:.2f}"
                                if sfa_curve is not None
                                else "\u2014"
                            ),
                            "rheobase": (
                                f"\u2264\u202f{proto_st.min_stimulus:.2f}"
                                if ap_result.spike_count >= 1
                                else "\u2014"
                            ),
                            "input_resistance": (
                                f"{passive.input_resistance:.2f}"
                                if passive is not None
                                else "\u2014"
                            ),
                            "time_constant": (
                                f"{passive.time_constant:.2f}"
                                if passive is not None
                                else "\u2014"
                            ),
                            "membrane_capacitance": (
                                f"{passive.membrane_capacitance:.2f}"
                                if passive is not None
                                and passive.membrane_capacitance is not None
                                else "\u2014"
                            ),
                        }
                        analysis_st.ap_is_multi_sweep = False
                        analysis_st.iv_data = {}
                        analysis_st.gv_data = {}
                        analysis_st.fi_data = {}
                        analysis_st.sfa_data = (
                            {"curves": [_serialise_sfa_curve(sfa_curve)]}
                            if sfa_curve is not None
                            else {}
                        )
                    else:
                        analysis_st.ap_metrics = []
                        analysis_st.ap_summary = {}
                        analysis_st.ap_is_multi_sweep = False
                        analysis_st.iv_data = {}
                        analysis_st.gv_data = {}
                        analysis_st.fi_data = {}
                        analysis_st.sfa_data = {}

        except ValueError as exc:
            logger.exception("Simulation error: %s", exc)
            async with self:
                self.error_message = str(exc)
        else:
            elapsed = time.monotonic() * 1000 - _start_ms
            logger.info("Simulation complete: %.0f ms", elapsed)
        finally:
            async with self:
                self.is_running = False
                log_st = await self.get_state(LogState)
                log_st._refresh_logs()
                vis_st = await self.get_state(VisibilityState)
            js = self._apply_visibility_js(vis_st)
            if js:
                yield rx.call_script(js)
            yield rx.call_script(_LOG_SCROLL_JS)
