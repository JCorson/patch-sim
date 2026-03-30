"""Application state for the patch_sim web UI.

All reactive variables and event handlers live here. The state drives
the Reflex component tree via computed properties.
"""

import asyncio
import json
import logging
import time

import numpy as np
import plotly.graph_objects as go
import reflex as rx

import patch_sim
import patch_sim.clamp_simulations
from patch_sim_ui.log_handler import MAX_LOG_ENTRIES, StateLogHandler, UILogRecord
from patch_sim.constants import (
    DEFAULT_C_M,
    DEFAULT_CA_IN,
    DEFAULT_CA_OUT,
    DEFAULT_CL_IN,
    DEFAULT_CL_OUT,
    DEFAULT_G_ICAN,
    DEFAULT_G_ICAL,
    DEFAULT_G_ICAT,
    DEFAULT_G_IH,
    DEFAULT_G_IKA,
    DEFAULT_G_IKCA,
    DEFAULT_G_IKIR,
    DEFAULT_G_IM,
    DEFAULT_G_K,
    DEFAULT_G_L,
    DEFAULT_G_NA,
    DEFAULT_G_NAP,
    DEFAULT_G_NAR,
    DEFAULT_K_IN,
    DEFAULT_K_OUT,
    DEFAULT_NA_IN,
    DEFAULT_NA_OUT,
    DEFAULT_T,
    DEFAULT_V_REST,
)
from patch_sim.additional_channels import (
    make_ican_channel,
    make_ical_channel,
    make_icat_channel,
    make_ih_channel,
    make_ika_channel,
    make_ikca_channel,
    make_ikir_channel,
    make_im_channel,
    make_inap_channel,
    make_inar_channel,
)
from patch_sim_ui import constants, presets
from patch_sim_ui.constants import CURRENT_CLAMP
from patch_sim_ui.plotting import (
    Sweep,
    TraceVisibility,
    build_figure,
    compute_trace_visibility_map,
)
from patch_sim_ui.protocol_builders import (
    build_current_protocol,
    build_voltage_protocol,
)

logger = logging.getLogger("patch_sim_ui.state")

# ------------------------------------------------------------------ #
# Channel registry                                                   #
# ------------------------------------------------------------------ #
# Each entry is (enabled_attr, g_max_attr, factory, extra_kwargs).   #
# extra_kwargs maps kwarg name → state attribute name.               #

_CHANNEL_REGISTRY: list[tuple[str, str, object, dict[str, str]]] = [
    ("ih_enabled", "ih_g_max", make_ih_channel, {}),
    ("ika_enabled", "ika_g_max", make_ika_channel, {}),
    ("inap_enabled", "inap_g_max", make_inap_channel, {}),
    ("inar_enabled", "inar_g_max", make_inar_channel, {}),
    ("im_enabled", "im_g_max", make_im_channel, {}),
    ("ikir_enabled", "ikir_g_max", make_ikir_channel, {}),
    ("ikca_enabled", "ikca_g_max", make_ikca_channel, {}),
    ("ical_enabled", "ical_g_max", make_ical_channel, {}),
    ("icat_enabled", "icat_g_max", make_icat_channel, {}),
    ("ican_enabled", "ican_g_max", make_ican_channel, {}),
]

# ------------------------------------------------------------------ #
# Additional-channel visibility field maps                          #
# ------------------------------------------------------------------ #
# Maps from additional-channel sweep keys to show_* field names.    #
# Used by compute_trace_visibility_map and _apply_visibility_js.    #

_ADDITIONAL_CURRENT_FIELD_MAP: dict[str, str] = {
    "Ih": "show_ih_current",
    "IKa": "show_ika_current",
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
# Float setter generation                                            #
# ------------------------------------------------------------------ #
# Reflex's auto-generated set_X handlers are typed `value: float`,   #
# but rx.input.on_change passes str and rx.slider.on_change passes   #
# list[float].  The generated setters below coerce both via          #
# AppState._set_float.                                               #

_FLOAT_FIELDS: list[str] = [
    # Neuron parameters
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
    # Protocol params — shared
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
    "vc_holding_voltage",
    "vc_start_voltage",
    "vc_end_voltage",
    "vc_pulse_amplitude",
    "vc_pulse_width",
    "vc_pulse_interval",
    # Additional channel params
    "ih_g_max",
    "ika_g_max",
    "inap_g_max",
    "inar_g_max",
    "im_g_max",
    "ikir_g_max",
    "ikca_g_max",
    "ical_g_max",
    "icat_g_max",
    "ican_g_max",
]


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


def _make_bool_setter(field_name: str):
    """Factory returning a bool event handler for ``field_name``.

    Args:
        field_name: Name of the AppState attribute to update.

    Returns:
        An event handler method that sets the bool field.
    """

    def setter(self, value: bool) -> None:
        """Set the field from a checkbox event."""
        setattr(self, field_name, value)

    setter.__name__ = f"set_{field_name}"
    setter.__qualname__ = f"AppState.set_{field_name}"
    setter.__doc__ = f"Set {field_name} from a checkbox event."
    return setter


def _make_visibility_setter(field_name: str):
    """Factory returning a visibility event handler for ``field_name``.

    The generated handler updates the server-side state var and issues a
    ``Plotly.restyle`` call to toggle the corresponding trace(s) client-side,
    avoiding a full figure rebuild for what is otherwise a trivial DOM change.

    Args:
        field_name: Name of the AppState show_* attribute to update.

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
            clamp_mode=self.clamp_mode,
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
    setter.__qualname__ = f"AppState.set_{field_name}"
    setter.__doc__ = f"Set {field_name} and apply client-side visibility restyle."
    return setter


def _make_float_setter(field_name: str):
    """Factory returning a float-coercing event handler for ``field_name``.

    Args:
        field_name: Name of the AppState attribute to update.

    Returns:
        An event handler method that accepts ``str | list[float] | float``
        and delegates to ``AppState._set_float``.
    """

    def setter(self, value: "str | list[float] | float") -> None:
        """Set the field from an input or slider event."""
        self._set_float(field_name, value)

    setter.__name__ = f"set_{field_name}"
    setter.__qualname__ = f"AppState.set_{field_name}"
    setter.__doc__ = f"Set {field_name} from an input or slider event."
    return setter


class AppState(rx.State):
    """Top-level application state."""

    # ------------------------------------------------------------------ #
    # Neuron parameters                                                   #
    # ------------------------------------------------------------------ #
    g_Na: float = DEFAULT_G_NA
    g_K: float = DEFAULT_G_K
    g_L: float = DEFAULT_G_L
    C_m: float = DEFAULT_C_M
    v_rest: float = DEFAULT_V_REST
    Na_out: float = DEFAULT_NA_OUT
    Na_in: float = DEFAULT_NA_IN
    K_out: float = DEFAULT_K_OUT
    K_in: float = DEFAULT_K_IN
    Cl_out: float = DEFAULT_CL_OUT
    Cl_in: float = DEFAULT_CL_IN
    Ca_out: float = DEFAULT_CA_OUT
    Ca_in: float = DEFAULT_CA_IN
    T: float = DEFAULT_T

    # ------------------------------------------------------------------ #
    # Additional channels                                                 #
    # ------------------------------------------------------------------ #
    ih_enabled: bool = False
    ih_g_max: float = DEFAULT_G_IH
    ika_enabled: bool = False
    ika_g_max: float = DEFAULT_G_IKA
    inap_enabled: bool = False
    inap_g_max: float = DEFAULT_G_NAP
    inar_enabled: bool = False
    inar_g_max: float = DEFAULT_G_NAR
    im_enabled: bool = False
    im_g_max: float = DEFAULT_G_IM
    ikir_enabled: bool = False
    ikir_g_max: float = DEFAULT_G_IKIR
    ikca_enabled: bool = False
    ikca_g_max: float = DEFAULT_G_IKCA
    ical_enabled: bool = False
    ical_g_max: float = DEFAULT_G_ICAL
    icat_enabled: bool = False
    icat_g_max: float = DEFAULT_G_ICAT
    ican_enabled: bool = False
    ican_g_max: float = DEFAULT_G_ICAN

    # ------------------------------------------------------------------ #
    # Experiment mode                                                     #
    # ------------------------------------------------------------------ #
    active_neuron_type: str = ""  # name of the currently selected neuron preset, or ""
    clamp_mode: str = CURRENT_CLAMP  # CURRENT_CLAMP | VOLTAGE_CLAMP

    # ------------------------------------------------------------------ #
    # Protocol parameters                                                #
    # ------------------------------------------------------------------ #
    protocol_type: str = "Step"
    pre_stimulus_duration: float = 10.0
    stimulus_duration: float = 30.0
    post_stimulus_duration: float = 10.0

    # Stimulus amplitude params — shared (units depend on clamp_mode)
    min_stimulus: float = 10.0
    max_stimulus: float = 10.0
    stimulus_step: float = 0.0

    # Current clamp protocol params
    start_current: float = 0.0
    end_current: float = 15.0
    pulse_amplitude: float = 10.0
    pulse_width: float = 2.0
    pulse_interval: float = 10.0
    dc_offset: float = 8.0
    amplitude: float = 4.0
    frequency: float = 50.0
    start_frequency: float = 1.0
    end_frequency: float = 100.0
    mean_current: float = 8.0
    std_current: float = 2.0

    # Voltage clamp protocol params
    vc_holding_voltage: float = -70.0
    vc_start_voltage: float = -70.0
    vc_end_voltage: float = 40.0
    vc_pulse_amplitude: float = 20.0
    vc_pulse_width: float = 2.0
    vc_pulse_interval: float = 10.0
    # ------------------------------------------------------------------ #
    # Simulation results                                                  #
    # ------------------------------------------------------------------ #
    current_sweeps: list[Sweep] = []  # Latest simulation result
    saved_sweeps: list[Sweep] = []  # User-saved sweeps for comparison overlay
    stored_traces: list[Sweep] = []  # Oscilloscope-style stored reference traces

    # ------------------------------------------------------------------ #
    # Trace visibility checkboxes                                        #
    # ------------------------------------------------------------------ #
    show_voltage: bool = True
    show_total_current: bool = True
    show_sodium_current: bool = True
    show_potassium_current: bool = True
    show_leak_current: bool = False
    show_potassium_activation: bool = True
    show_sodium_activation: bool = True
    show_sodium_inactivation: bool = True
    show_ih_current: bool = True
    show_ih_gating: bool = True
    show_ika_current: bool = True
    show_ika_gating: bool = True
    show_inap_current: bool = True
    show_inap_gating: bool = True
    show_inar_current: bool = True
    show_inar_gating: bool = True
    show_im_current: bool = True
    show_im_gating: bool = True
    show_ikir_current: bool = True
    show_ikir_gating: bool = True
    show_ikca_current: bool = True
    show_ikca_gating: bool = True
    show_ical_current: bool = True
    show_ical_gating: bool = True
    show_icat_current: bool = True
    show_icat_gating: bool = True
    show_ican_current: bool = True
    show_ican_gating: bool = True

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
    # any figure rebuild triggered by a Python state change (e.g. add_sweep)
    # while the user has a client-side selection active will re-seed JS with
    # -1, silently clearing the selection.
    selected_sweep: int = -1

    # ------------------------------------------------------------------ #
    # Log panel state                                                    #
    # ------------------------------------------------------------------ #
    log_panel_open: bool = False
    log_entries: list[UILogRecord] = []
    log_level_filter: str = "DEBUG"

    # ------------------------------------------------------------------ #
    # Derived reversal potentials (shown as read-only in neuron panel)  #
    # ------------------------------------------------------------------ #
    @rx.var
    def E_Na(self) -> float:
        """Sodium reversal potential in mV."""
        return float(patch_sim.nernst_potential(1, self.T, self.Na_out, self.Na_in))

    @rx.var
    def E_K(self) -> float:
        """Potassium reversal potential in mV."""
        return float(patch_sim.nernst_potential(1, self.T, self.K_out, self.K_in))

    @rx.var
    def E_L(self) -> float:
        """Leak reversal potential in mV."""
        return float(patch_sim.nernst_potential(-1, self.T, self.Cl_out, self.Cl_in))

    @rx.var
    def E_Ca(self) -> float:
        """Calcium reversal potential in mV (z=+2)."""
        return float(patch_sim.nernst_potential(2, self.T, self.Ca_out, self.Ca_in))

    @rx.var
    def protocol_options(self) -> list[str]:
        """Protocol type options filtered by clamp mode."""
        if self.clamp_mode == CURRENT_CLAMP:
            return constants.CURRENT_PROTOCOLS
        return constants.VOLTAGE_PROTOCOLS

    @rx.var
    def has_result(self) -> bool:
        """Whether a simulation result is available."""
        return len(self.current_sweeps) > 0

    @rx.var
    def has_stored_traces(self) -> bool:
        """Whether any oscilloscope-stored reference traces exist."""
        return len(self.stored_traces) > 0

    @rx.var
    def continuous_active(self) -> bool:
        """True when continuous mode is enabled and the loop is running."""
        return self.continuous_mode and self.continuous_loop_running

    @rx.var
    def is_step_single_sweep(self) -> bool:
        """True when the Step protocol is configured as a single sweep.

        A single sweep is produced whenever min_stimulus == max_stimulus,
        regardless of the stimulus_step value.

        Returns:
            True if min_stimulus equals max_stimulus, False otherwise.
        """
        return self.min_stimulus == self.max_stimulus

    @rx.var
    def can_run_continuous(self) -> bool:
        """True when the active protocol is compatible with continuous mode.

        Multi-sweep Step configurations (min_stimulus != max_stimulus) are
        excluded; all other protocols run as a single sweep and are compatible.
        """
        if self.protocol_type != "Step":
            return True
        return self.is_step_single_sweep

    @rx.var
    def filtered_log_entries(self) -> list[UILogRecord]:
        """Log entries filtered to the selected minimum level, newest first.

        Returns:
            Entries whose numeric level is >= the selected filter level,
            in reverse chronological order so the most recent entry is
            always visible at the top of the panel without scrolling.
        """
        min_level = logging.getLevelName(self.log_level_filter)
        if not isinstance(min_level, int):
            min_level = logging.DEBUG
        return [
            e
            for e in reversed(self.log_entries)
            if logging.getLevelName(e.level) >= min_level
        ]

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
            saved_sweeps=self.saved_sweeps,
            visibility=TraceVisibility(),  # all visible; toggling handled client-side
            clamp_mode=self.clamp_mode,
            stored_traces=self.stored_traces,
            show_hover=self.show_hover,
        )

    # ------------------------------------------------------------------ #
    # Log panel event handlers                                          #
    # ------------------------------------------------------------------ #
    def toggle_log_panel(self):
        """Toggle the log panel open/closed, refreshing logs on open."""
        self.log_panel_open = not self.log_panel_open
        if self.log_panel_open:
            self._refresh_logs()
            return rx.call_script(_LOG_SCROLL_JS)

    def refresh_logs(self):
        """Public event handler: drain buffered records into state."""
        self._refresh_logs()
        return rx.call_script(_LOG_SCROLL_JS)

    def _refresh_logs(self) -> None:
        """Drain buffered log records into state, capping at MAX_LOG_ENTRIES."""
        new_records = StateLogHandler.drain()
        combined = list(self.log_entries) + new_records
        self.log_entries = combined[-MAX_LOG_ENTRIES:]

    def clear_logs(self) -> None:
        """Clear all displayed log entries."""
        self.log_entries = []

    def set_log_level_filter(self, value: str) -> None:
        """Set the minimum display level for log entries.

        Args:
            value: Level name string (e.g. ``"DEBUG"``, ``"INFO"``).
        """
        self.log_level_filter = value

    # ------------------------------------------------------------------ #
    # Event handlers                                                     #
    # ------------------------------------------------------------------ #
    def set_protocol_type(self, value: str) -> None:
        """Set protocol_type from a select event."""
        self.protocol_type = value

    def set_clamp_mode(self, mode: str) -> None:
        """Switch between Current Clamp and Voltage Clamp modes."""
        self.clamp_mode = mode
        # Reset protocol type to first option for the new mode
        if mode == CURRENT_CLAMP:
            self.protocol_type = constants.CURRENT_PROTOCOLS[0]
        else:
            self.protocol_type = constants.VOLTAGE_PROTOCOLS[0]
        self.current_sweeps = []
        self.saved_sweeps = []
        self.stored_traces = []
        self._cont_has_state = False
        self.selected_sweep = -1

    def reset_to_defaults(self) -> None:
        """Reset all parameters and sweeps to their class-level defaults."""
        self.reset()

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

    def load_neuron_preset(self, name: str) -> None:
        """Load a neuron-type preset, updating only neuron parameters.

        Sets conductances and auxiliary channel configuration for the selected
        neuron type without touching any protocol settings.  Records the active
        neuron type so that subsequent protocol preset loads can apply
        neuron-specific adjustments via NEURON_PROTOCOL_ADJUSTMENTS.

        Args:
            name: Key into presets.NEURON_PRESETS.  Ignored if not found.
        """
        if name not in presets.NEURON_PRESETS:
            logger.debug("load_neuron_preset: unknown preset %r ignored", name)
            return
        logger.info("Loaded neuron preset: %s", name)
        config = presets.NEURON_PRESETS[name]
        for key, value in config.items():
            setattr(self, key, value)
        self.active_neuron_type = name
        self.current_sweeps = []
        self.saved_sweeps = []
        self.stored_traces = []
        self._cont_has_state = False

    def load_protocol_preset(self, name: str) -> None:
        """Load a protocol preset, applying neuron-type adjustments if active.

        Applies the base protocol preset, then overlays any entries from
        presets.NEURON_PROTOCOL_ADJUSTMENTS for the currently active neuron
        type so that the stimulus parameters suit the selected cell type.

        Args:
            name: Key into presets.PROTOCOL_PRESETS.  Ignored if not found.
        """
        if name not in presets.PROTOCOL_PRESETS:
            logger.debug("load_protocol_preset: unknown preset %r ignored", name)
            return
        logger.info("Loaded protocol preset: %s", name)
        config = dict(presets.PROTOCOL_PRESETS[name])
        adjustments = presets.NEURON_PROTOCOL_ADJUSTMENTS.get(
            self.active_neuron_type, {}
        ).get(name, {})
        config.update(adjustments)
        for key, value in config.items():
            setattr(self, key, value)
        self.current_sweeps = []
        self.saved_sweeps = []
        self.stored_traces = []
        self._cont_has_state = False

    # ------------------------------------------------------------------ #
    # Numeric field setters                                              #
    # ------------------------------------------------------------------ #
    def _set_float(self, field: str, value: "str | list[float] | float") -> None:
        """Coerce value to float and set the named field.

        Args:
            field: Name of the AppState attribute to update.
            value: Raw value from an input or slider event.
        """
        v = value[0] if isinstance(value, list) else value
        try:
            setattr(self, field, float(v))
        except (ValueError, TypeError):
            pass

    # One setter per float field — generated at class-definition time so
    # Reflex's metaclass sees them as regular event handlers.
    for _f in _FLOAT_FIELDS:
        vars()[f"set_{_f}"] = _make_float_setter(_f)

    # Stimulus range setters — custom logic to keep stimulus_step valid.
    def set_min_stimulus(self, value: str | float) -> None:
        """Set min_stimulus, auto-setting stimulus_step when a range is opened.

        If the new min_stimulus differs from max_stimulus and stimulus_step is
        currently 0, stimulus_step is set to 1.0 so the Step protocol remains
        in a valid multi-sweep state without requiring a separate user action.

        Args:
            value: Raw input value from the UI field.
        """
        self._set_float("min_stimulus", value)
        if self.min_stimulus != self.max_stimulus and self.stimulus_step == 0.0:
            self.stimulus_step = 1.0

    def set_max_stimulus(self, value: str | float) -> None:
        """Set max_stimulus, auto-setting stimulus_step when a range is opened.

        If the new max_stimulus differs from min_stimulus and stimulus_step is
        currently 0, stimulus_step is set to 1.0 so the Step protocol remains
        in a valid multi-sweep state without requiring a separate user action.

        Args:
            value: Raw input value from the UI field.
        """
        self._set_float("max_stimulus", value)
        if self.min_stimulus != self.max_stimulus and self.stimulus_step == 0.0:
            self.stimulus_step = 1.0

    def set_stimulus_step(self, value: str | float) -> None:
        """Set stimulus_step, rejecting non-positive values when min != max.

        When min_stimulus differs from max_stimulus a step of 0 (or negative)
        would produce an invalid multi-sweep configuration.  Such values are
        ignored, leaving the previous step unchanged.  When
        min_stimulus == max_stimulus (single-sweep mode) any value is accepted.

        Args:
            value: Raw input value from the UI text field.
        """
        try:
            parsed = float(value)
        except (ValueError, TypeError):
            logger.debug("set_stimulus_step: could not parse %r as float", value)
            return
        if self.min_stimulus != self.max_stimulus and parsed <= 0.0:
            logger.debug(
                "set_stimulus_step: rejected value %s"
                " (non-positive step in multi-sweep mode)",
                parsed,
            )
            self.stimulus_step = 1.0
            return
        self.stimulus_step = parsed

    # Non-visibility bool setters (channel enable/disable).
    for _f in _NON_VISIBILITY_BOOL_FIELDS:
        vars()[f"set_{_f}"] = _make_bool_setter(_f)

    # Visibility setters: update state + issue client-side Plotly.restyle.
    for _f in _VISIBILITY_FIELDS:
        vars()[f"set_{_f}"] = _make_visibility_setter(_f)

    # ------------------------------------------------------------------ #
    # Sweep management                                                   #
    # ------------------------------------------------------------------ #
    def _apply_visibility_js(self) -> str | None:
        """Build a JS snippet to re-apply trace visibility, hover, and sweep highlight.

        Called after any operation that triggers a full figure rebuild (run
        simulation, add sweep, clear sweeps) so that traces the user has
        toggled off are correctly hidden again, the hover mode matches the
        current ``show_hover`` flag, and sweep highlight listeners are
        (re-)attached in multi-sweep mode.

        Returns:
            A JS string that re-applies trace visibility, hover mode, and
            sweep highlight, or ``None`` when nothing needs to be applied.
        """
        trace_map = compute_trace_visibility_map(
            current_sweeps=self.current_sweeps,
            saved_sweeps=self.saved_sweeps,
            clamp_mode=self.clamp_mode,
            additional_current_field_map=_ADDITIONAL_CURRENT_FIELD_MAP,
            additional_gating_field_map=_ADDITIONAL_GATING_FIELD_MAP,
            stored_traces=self.stored_traces,
        )
        hidden: list[int] = []
        for field_name, indices in trace_map.items():
            if not getattr(self, field_name):
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

    def add_sweep(self):
        """Promote current simulation result to the saved sweep overlay."""
        if not self.has_result:
            return
        logger.debug("Adding %d sweep(s) to overlay", len(self.current_sweeps))
        for sweep in self.current_sweeps:
            idx = len(self.saved_sweeps)
            color = constants.SWEEP_COLORS[idx % len(constants.SWEEP_COLORS)]
            self.saved_sweeps.append(
                sweep.model_copy(update={"color": color, "label": f"Sweep {idx + 1}"})
            )
        js = self._apply_visibility_js()
        if js:
            return rx.call_script(js)

    def clear_sweeps(self):
        """Remove all saved sweeps."""
        logger.debug("Cleared %d saved sweep(s)", len(self.saved_sweeps))
        self.saved_sweeps = []
        js = self._apply_visibility_js()
        if js:
            return rx.call_script(js)

    # ------------------------------------------------------------------ #
    # Stored trace management                                            #
    # ------------------------------------------------------------------ #
    def store_trace(self) -> None:
        """Snapshot the current sweep into the oscilloscope stored traces."""
        if not self.has_result:
            return
        idx = len(self.stored_traces)
        color = constants.STORED_TRACE_COLORS[idx % len(constants.STORED_TRACE_COLORS)]
        label = f"Stored {idx + 1}"
        self.stored_traces.append(
            self.current_sweeps[0].model_copy(update={"color": color, "label": label})
        )
        js = self._apply_visibility_js()
        if js:
            return rx.call_script(js)

    def clear_stored_traces(self) -> None:
        """Remove all oscilloscope stored traces."""
        self.stored_traces = []
        js = self._apply_visibility_js()
        if js:
            return rx.call_script(js)

    # ------------------------------------------------------------------ #
    # Simulation helpers (called while holding the state lock)          #
    # ------------------------------------------------------------------ #
    def _build_neuron(self) -> "patch_sim.HodgkinHuxley":
        """Construct a HodgkinHuxley neuron from current state parameters."""
        additional_channels = []
        for enabled_attr, g_max_attr, factory, extra_kwargs in _CHANNEL_REGISTRY:
            if getattr(self, enabled_attr):
                kwargs = {k: getattr(self, v) for k, v in extra_kwargs.items()}
                additional_channels.append(
                    factory(g_max=getattr(self, g_max_attr), **kwargs)  # type: ignore[operator]
                )
        needs_calcium = (
            self.ikca_enabled
            or self.ical_enabled
            or self.icat_enabled
            or self.ican_enabled
        )
        calcium_dynamics = patch_sim.CalciumDynamics() if needs_calcium else None
        return patch_sim.HodgkinHuxley(
            g_Na=self.g_Na,
            g_K=self.g_K,
            g_L=self.g_L,
            C_m=self.C_m,
            v_rest=self.v_rest,
            Na_out=self.Na_out,
            Na_in=self.Na_in,
            K_out=self.K_out,
            K_in=self.K_in,
            Cl_out=self.Cl_out,
            Cl_in=self.Cl_in,
            Ca_out=self.Ca_out,
            Ca_in=self.Ca_in,
            T=self.T,
            additional_channels=tuple(additional_channels),
            calcium_dynamics=calcium_dynamics,
        )

    def _build_protocols(self) -> "list[tuple[np.ndarray, str]]":
        """Build stimulus arrays from current protocol state.

        Returns:
            List of (stimulus_array, sweep_label) pairs. Single-sweep protocols
            return a one-element list with an empty label; multi-sweep protocols
            (e.g. I-V Curve) return one entry per sweep with a descriptive label.
        """
        fs = patch_sim.clamp_simulations.SIM_SAMPLING_FREQ
        if self.clamp_mode == "Current Clamp":
            return build_current_protocol(
                protocol_type=self.protocol_type,
                sampling_frequency=fs,
                pre_stimulus_duration=self.pre_stimulus_duration,
                stimulus_duration=self.stimulus_duration,
                post_stimulus_duration=self.post_stimulus_duration,
                current_min=self.min_stimulus,
                current_max=self.max_stimulus,
                current_step=self.stimulus_step,
                start_current=self.start_current,
                end_current=self.end_current,
                pulse_amplitude=self.pulse_amplitude,
                pulse_width=self.pulse_width,
                pulse_interval=self.pulse_interval,
                dc_offset=self.dc_offset,
                amplitude=self.amplitude,
                frequency=self.frequency,
                start_frequency=self.start_frequency,
                end_frequency=self.end_frequency,
                mean_current=self.mean_current,
                std_current=self.std_current,
            )
        else:
            return build_voltage_protocol(
                protocol_type=self.protocol_type,
                sampling_frequency=fs,
                pre_stimulus_duration=self.pre_stimulus_duration,
                stimulus_duration=self.stimulus_duration,
                post_stimulus_duration=self.post_stimulus_duration,
                holding_voltage=self.vc_holding_voltage,
                start_voltage=self.vc_start_voltage,
                end_voltage=self.vc_end_voltage,
                pulse_amplitude=self.vc_pulse_amplitude,
                pulse_width=self.vc_pulse_width,
                pulse_interval=self.vc_pulse_interval,
                voltage_min=self.min_stimulus,
                voltage_max=self.max_stimulus,
                voltage_step=self.stimulus_step,
            )

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
            return AppState.run_continuous  # type: ignore[return-value]

    @rx.event(background=True)
    async def run_continuous(self) -> None:
        """Run simulations in a continuous loop until continuous_mode is False.

        Each iteration picks up the terminal neuron state (voltage, gating
        variables, Ca²⁺) from the previous iteration, giving true continuity.
        Parameter changes made while the loop runs take effect at the start
        of the next iteration.
        """
        async with self:
            self.continuous_loop_running = True
            self.error_message = ""

        logger.info(
            "Continuous simulation started: mode=%s, protocol=%s",
            self.clamp_mode,
            self.protocol_type,
        )
        _iteration = 0
        try:
            while True:
                _iter_start = time.monotonic()
                # Snapshot all state needed for this iteration.
                async with self:
                    if not self.continuous_mode:
                        break

                    mode = self.clamp_mode
                    ptype = self.protocol_type
                    neuron = self._build_neuron()
                    stimulus = self._build_protocols()[0][0]
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
                            df = await loop.run_in_executor(
                                None,
                                patch_sim.simulate_current_clamp_from_state,
                                neuron,
                                stimulus,
                                prior_V,
                                prior_gating,
                                prior_ca_i,
                            )
                        else:
                            df = await loop.run_in_executor(
                                None,
                                patch_sim.simulate_current_clamp,
                                neuron,
                                stimulus,
                            )
                    else:
                        if use_prior_state:
                            df = await loop.run_in_executor(
                                None,
                                patch_sim.simulate_voltage_clamp_from_state,
                                neuron,
                                stimulus,
                                prior_gating,
                                prior_ca_i,
                            )
                        else:
                            df = await loop.run_in_executor(
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
                last_V = float(df["voltage"].iloc[-1])
                gating_vars = {gv.name for gv in neuron.all_gating_variables}
                gating_cols = [col for col in df.columns if col in gating_vars]
                last_gating = {col: float(df[col].iloc[-1]) for col in gating_cols}
                last_ca_i = float(df["ca_i"].iloc[-1]) if "ca_i" in df.columns else 0.0

                sweep = Sweep.from_dataframe(df, stimulus, "", "", mode)

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
                self._refresh_logs()
            js = self._apply_visibility_js()
            if js:
                yield rx.call_script(js)
            yield rx.call_script(_LOG_SCROLL_JS)

    @rx.event(background=True)
    async def run_simulation(self) -> None:
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
            neuron = self._build_neuron()
            mode = self.clamp_mode
            ptype = self.protocol_type
            try:
                protocols = self._build_protocols()
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
                    for sweep_df, (protocol, label) in zip(
                        patch_sim.simulate_batch(
                            neuron, [p for p, _ in protocols], sim_fn
                        ),
                        protocols,
                    ):
                        color_index = len(new_sweeps) % len(constants.SWEEP_COLORS)
                        new_sweeps.append(
                            Sweep.from_dataframe(
                                sweep_df,
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

            else:
                stimulus, _ = protocols[0]
                df = await loop.run_in_executor(None, sim_fn, neuron, stimulus)
                async with self:
                    self.current_sweeps = [
                        Sweep.from_dataframe(df, stimulus, "", "", mode)
                    ]

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
                self._refresh_logs()
            js = self._apply_visibility_js()
            if js:
                yield rx.call_script(js)
            yield rx.call_script(_LOG_SCROLL_JS)
