"""Shared constants, field lists, factory functions, and free helpers.

All pre-class module-level code that is reused across multiple substate
modules lives here so that each substate file can import what it needs
without re-defining it.
"""

import json
from typing import Any

import numpy as np
import reflex as rx

import patch_sim
from patch_sim_ui.plotting import Sweep, compute_trace_visibility_map

# ------------------------------------------------------------------ #
# Additional-channel visibility field maps                           #
# ------------------------------------------------------------------ #
# Maps from additional-channel sweep keys to show_* field names.     #
# Used by compute_trace_visibility_map and build_visibility_js.       #

_ADDITIONAL_CURRENT_FIELD_MAP: dict[str, str] = {
    "Ih": "show_ih_current",
    "IKa": "show_ika_current",
    "IKv31": "show_ikv31_current",
    "INaP": "show_inap_current",
    "INaR": "show_inar_current",
    "IM": "show_im_current",
    "IKir": "show_ikir_current",
    "IKCa": "show_ikca_current",
    "ICaL": "show_ical_current",
    "ICaT": "show_icat_current",
    "ICaN": "show_ican_current",
}

_ADDITIONAL_GATING_FIELD_MAP: dict[str, str] = {
    "r": "show_ih_gating",
    "a": "show_ika_gating",
    "b": "show_ika_gating",
    "nk": "show_ikv31_gating",
    "p": "show_inap_gating",
    "s": "show_inar_gating",
    "hr": "show_inar_gating",
    "w": "show_im_gating",
    "kir": "show_ikir_gating",
    "q": "show_ikca_gating",
    "d": "show_ical_gating",
    "f": "show_ical_gating",
    "dt": "show_icat_gating",
    "ft": "show_icat_gating",
    "dn": "show_ican_gating",
    "fn": "show_ican_gating",
}

# ------------------------------------------------------------------ #
# Float setter field lists                                           #
# ------------------------------------------------------------------ #
# Split by substate so each substate only iterates its own fields.   #
# The combined _FLOAT_FIELDS is kept for backward compatibility.     #

_NEURON_FLOAT_FIELDS: list[str] = [
    "g_Na",
    "g_K",
    "g_L",
    "C_m",
    "v_rest",
    "Na_out",
    "Na_in",
    "K_out",
    "K_in",
    "Cl_out",
    "Cl_in",
    "Ca_out",
    "Ca_in",
    "T",
]

_CHANNEL_FLOAT_FIELDS: list[str] = [
    "ih_g_max",
    "ika_g_max",
    "ikv31_g_max",
    "inap_g_max",
    "inar_g_max",
    "im_g_max",
    "ikir_g_max",
    "ikca_g_max",
    "ical_g_max",
    "icat_g_max",
    "ican_g_max",
]

_PROTOCOL_FLOAT_FIELDS: list[str] = [
    # Shared timing
    "pre_stimulus_duration",
    "stimulus_duration",
    "post_stimulus_duration",
    # Current clamp protocol params
    "start_current",
    "end_current",
    "pulse_amplitude",
    "pulse_width",
    "pulse_interval",
    "dc_offset",
    "amplitude",
    "frequency",
    "start_frequency",
    "end_frequency",
    "mean_current",
    "std_current",
    # Voltage clamp protocol params
    "holding_voltage",
    "vc_start_voltage",
    "vc_end_voltage",
    "vc_pulse_amplitude",
    "vc_pulse_width",
    "vc_pulse_interval",
]

_FLOAT_FIELDS: list[str] = (
    _NEURON_FLOAT_FIELDS + _PROTOCOL_FLOAT_FIELDS + _CHANNEL_FLOAT_FIELDS
)

# ------------------------------------------------------------------ #
# Bool setter field lists                                            #
# ------------------------------------------------------------------ #

# Visibility fields: show_* — toggled client-side via Plotly.restyle.
_VISIBILITY_FIELDS: list[str] = [
    "show_voltage",
    "show_total_current",
    "show_sodium_current",
    "show_potassium_current",
    "show_leak_current",
    "show_potassium_activation",
    "show_sodium_activation",
    "show_sodium_inactivation",
    # Additional channel visibility
    "show_ih_current",
    "show_ih_gating",
    "show_ika_current",
    "show_ika_gating",
    "show_ikv31_current",
    "show_ikv31_gating",
    "show_inap_current",
    "show_inap_gating",
    "show_inar_current",
    "show_inar_gating",
    "show_im_current",
    "show_im_gating",
    "show_ikir_current",
    "show_ikir_gating",
    "show_ikca_current",
    "show_ikca_gating",
    "show_ical_current",
    "show_ical_gating",
    "show_icat_current",
    "show_icat_gating",
    "show_ican_current",
    "show_ican_gating",
]

# Non-visibility bool fields: channel enable/disable toggles.
_NON_VISIBILITY_BOOL_FIELDS: list[str] = [
    "ih_enabled",
    "ika_enabled",
    "ikv31_enabled",
    "inap_enabled",
    "inar_enabled",
    "im_enabled",
    "ikir_enabled",
    "ikca_enabled",
    "ical_enabled",
    "icat_enabled",
    "ican_enabled",
]

# Kept for backward compatibility; callers outside this module may reference it.
_BOOL_FIELDS: list[str] = _VISIBILITY_FIELDS + _NON_VISIBILITY_BOOL_FIELDS

# ------------------------------------------------------------------ #
# Shared JS snippets                                                 #
# ------------------------------------------------------------------ #

# Shared JS snippet for targeting the Plotly graph element.
_PLOTLY_GD_JS = "var gd=document.querySelector('.js-plotly-plot');"

# Scroll the log panel viewport to the top so the newest entry (displayed
# first in newest-first order) is always visible after a refresh.
_LOG_SCROLL_JS = (
    "var vp=document.querySelector("
    "'#log-scroll-area [data-radix-scroll-area-viewport]');"
    "if(vp)vp.scrollTop=0;"
)

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


# ------------------------------------------------------------------ #
# Event handler factories                                            #
# ------------------------------------------------------------------ #


def _make_bool_setter(field_name: str, class_name: str = "AppState"):
    """Factory returning a bool event handler for ``field_name``.

    Args:
        field_name: Name of the state attribute to update.
        class_name: Owning state class name used in ``__qualname__``.

    Returns:
        An event handler method that sets the bool field.
    """

    def setter(self, value: bool) -> None:
        """Set the field from a checkbox event."""
        setattr(self, field_name, value)

    setter.__name__ = f"set_{field_name}"
    setter.__qualname__ = f"{class_name}.set_{field_name}"
    setter.__doc__ = f"Set {field_name} from a checkbox event."
    return setter


def _make_visibility_setter(field_name: str, class_name: str = "AppState"):
    """Factory returning a visibility event handler for ``field_name``.

    The generated handler updates the server-side state var and issues a
    ``Plotly.restyle`` call to toggle the corresponding trace(s) client-side,
    avoiding a full figure rebuild for what is otherwise a trivial DOM change.

    Args:
        field_name: Name of the show_* state attribute to update.
        class_name: Owning state class name used in ``__qualname__``.

    Returns:
        An event handler method that sets the bool field and yields a
        ``rx.call_script`` event to apply the visibility change in-browser.
    """

    def setter(self, value: bool):
        """Set the visibility flag and apply a client-side Plotly restyle."""
        setattr(self, field_name, value)
        trace_map = compute_trace_visibility_map(
            current_sweeps=self.current_sweeps,
            saved_sweeps=self.saved_sweeps,
            clamp_mode=self._figure_clamp_mode,
            additional_current_field_map=_ADDITIONAL_CURRENT_FIELD_MAP,
            additional_gating_field_map=_ADDITIONAL_GATING_FIELD_MAP,
            stored_traces=self.stored_traces,
        )
        indices = trace_map.get(field_name, [])
        if indices:
            visible_js = "true" if value else "false"
            js = (
                f"{_PLOTLY_GD_JS}"
                f"if(gd&&gd.data)Plotly.restyle(gd,"
                f"{{visible:{visible_js}}},{json.dumps(indices)})"
            )
            return rx.call_script(js)

    setter.__name__ = f"set_{field_name}"
    setter.__qualname__ = f"{class_name}.set_{field_name}"
    setter.__doc__ = f"Set {field_name} and apply client-side visibility restyle."
    return setter


def _make_float_setter(field_name: str, class_name: str = "AppState"):
    """Factory returning a float-coercing event handler for ``field_name``.

    Args:
        field_name: Name of the state attribute to update.
        class_name: Owning state class name used in ``__qualname__``.

    Returns:
        An event handler method that accepts ``str | list[float] | float``
        and delegates to ``_set_float``.
    """

    def setter(self, value: "str | list[float] | float") -> None:
        """Set the field from an input or slider event."""
        self._set_float(field_name, value)

    setter.__name__ = f"set_{field_name}"
    setter.__qualname__ = f"{class_name}.set_{field_name}"
    setter.__doc__ = f"Set {field_name} from an input or slider event."
    return setter


# ------------------------------------------------------------------ #
# Free helper functions                                              #
# ------------------------------------------------------------------ #


def _compute_iv_data(
    sweeps: list[Sweep],
    min_stimulus: float,
    max_stimulus: float,
    stimulus_step: float,
    pre_stimulus_duration: float,
    stimulus_duration: float,
) -> dict[str, Any]:
    """Compute I-V analysis data from multi-sweep voltage clamp results.

    Derives voltage step values from the protocol parameters, extracts total
    current arrays from each sweep, and calls :func:`patch_sim.analyze_iv`.
    The result is serialised into a plain dict suitable for use as a Reflex
    state variable.

    Args:
        sweeps: Ordered list of :class:`Sweep` objects from the simulation.
        min_stimulus: Minimum voltage step command (mV).
        max_stimulus: Maximum voltage step command (mV).
        stimulus_step: Step size between voltage commands (mV).
        pre_stimulus_duration: Duration before the step begins (ms).
        stimulus_duration: Duration of the voltage step (ms).

    Returns:
        A dict with keys ``voltages``, ``peak_inward_currents``,
        ``peak_outward_currents``, and ``steady_state_currents``, each a list
        of floats sorted by voltage.  Returns an empty dict when fewer than
        two sweeps are provided or when the number of sweeps does not match
        the number of voltage steps derived from the protocol parameters.
    """
    if len(sweeps) < 2:
        return {}

    n_steps = round((max_stimulus - min_stimulus) / stimulus_step) + 1
    voltage_steps = list(np.linspace(min_stimulus, max_stimulus, n_steps))

    if len(sweeps) != len(voltage_steps):
        return {}

    time = np.array(sweeps[0].time)
    currents = [np.array(s.total_current) for s in sweeps]

    stim_start = pre_stimulus_duration
    stim_end = pre_stimulus_duration + stimulus_duration

    iv_result = patch_sim.analyze_iv(
        time, currents, voltage_steps, stim_start, stim_end
    )
    return {
        "voltages": iv_result.voltage_steps,
        "peak_inward_currents": iv_result.peak_inward_currents,
        "peak_outward_currents": iv_result.peak_outward_currents,
        "steady_state_currents": iv_result.steady_state_currents,
    }
