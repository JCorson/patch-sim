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
