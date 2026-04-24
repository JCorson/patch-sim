// Fetch a stored Plotly figure from the side-channel HTTP route and swap it
// into the #ps-trace-plot div.
//
// Placeholder tokens (each of the form slash-star NAME star-slash) are
// substituted server-side by SimulationState._build_fetch_figure_js:
//   API_URL          -> base URL from Reflex config (no trailing slash)
//   TOKEN            -> uuid4 hex identifying the figure in the store
//   DARK_AXIS_STYLE  -> JSON object of axis colour overrides
//   POST_JS          -> additional JS to run after Plotly.react (visibility
//                       reapply, hover, sweep highlight)
//
// Wrapped in setTimeout(..., 0) so it runs after Reflex's state flush has
// updated the DOM (display:none -> display:block on the mount div).
setTimeout(async function () {
  try {
    var resp = await fetch("/*API_URL*/" + "/api/figure/" + "/*TOKEN*/");
    if (!resp.ok) {
      console.error("[patch_sim] fetch figure HTTP", resp.status);
      return;
    }
    var fig = await resp.json();

    // Wait for two preconditions before rendering:
    //
    // 1. The mount div has display:block and non-zero offsetHeight.  It is
    //    always in the DOM but stays display:none until has_result flips
    //    true, and React may still be flushing that flip when we arrive.
    //    Calling Plotly.react on a zero-height div produces an invisible plot.
    //
    // 2. window.Plotly is defined.  react-plotly.js is lazy-loaded
    //    (ClientSide dynamic import) when the first <Plot> component mounts,
    //    which happens concurrently with this fetch on the first simulation
    //    run.  If we call Plotly.react before the module finishes loading,
    //    it throws ReferenceError and the plot stays blank until the next run.
    //
    // TODO: replace this poll with a proper signal once we have one — e.g.
    // a ResizeObserver on the mount div plus a Plotly-ready event.  2 s at
    // 50 ms gives the browser generous room for a slow frame and a slow
    // module import without blocking forever.
    var gd = null;
    for (var _i = 0; _i < 40; _i++) {
      gd = document.getElementById("ps-trace-plot");
      if (gd && gd.offsetHeight > 0 && typeof window.Plotly !== "undefined") {
        break;
      }
      gd = null;
      await new Promise(function (r) {
        setTimeout(r, 50);
      });
    }
    if (!gd) {
      console.error("[patch_sim] plot div / Plotly not ready after 2 s");
      return;
    }

    // Dark-mode layout overrides.  The server builds the figure with
    // plotly_white; on a dark surface we override transparent backgrounds,
    // text/axis colours, and gridlines (plotly_white's near-white grid is
    // very prominent against a dark paper).
    var _dark = document.documentElement.classList.contains("dark");
    var _layout = Object.assign({}, fig.layout);
    if (_dark) {
      _layout.paper_bgcolor = "rgba(0,0,0,0)";
      _layout.plot_bgcolor = "rgba(0,0,0,0)";
      _layout.font = Object.assign({}, fig.layout.font, { color: "#e8e8e8" });
      _layout.legend = Object.assign({}, fig.layout.legend, {
        bgcolor: "rgba(40,40,40,0.9)",
        font: { color: "#e8e8e8" },
      });
      _layout.legend2 = Object.assign({}, fig.layout.legend2, {
        bgcolor: "rgba(40,40,40,0.9)",
        font: { color: "#e8e8e8" },
      });
      var _ax = /*DARK_AXIS_STYLE*/;
      Object.keys(fig.layout).forEach(function (k) {
        if (/^[xy]axis/.test(k)) {
          _layout[k] = Object.assign({}, fig.layout[k], _ax);
        }
      });
    }

    // The mount is a plain HTML div (not a react-plotly.js component), so
    // Plotly.react's output is not wiped by a subsequent React re-render.
    await Plotly.react(gd, fig.data, _layout, { responsive: true });
    /*POST_JS*/
  } catch (err) {
    console.error("[patch_sim] fetch figure error:", err);
  }
}, 0);
