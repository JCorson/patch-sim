// Download the trace figure as a PNG or SVG image.
//
// Placeholder tokens (each of the form slash-star NAME star-slash) are
// substituted server-side by _render_export_figure_js:
//   API_URL   -> base URL from Reflex config (no trailing slash)
//   TOKEN     -> uuid4 hex identifying the figure in the side-channel store
//   FORMAT    -> JSON string, "png" or "svg"
//   FILENAME  -> JSON string, download filename without extension
//
// The canonical (light-theme) figure is re-fetched from the server so the
// export is independent of the on-screen dark-mode restyling.  Per-trace
// visibility is copied from the on-screen plot, and the invisible hover
// carrier traces are dropped.  The image is rendered at the on-screen size.
(async function () {
  try {
    var gd = document.getElementById("ps-trace-plot");
    if (!gd || !gd.data || typeof window.Plotly === "undefined") {
      console.error("[patch_sim] export figure: plot is not ready");
      return;
    }
    var resp = await fetch("/*API_URL*/" + "/api/figure/" + "/*TOKEN*/");
    if (!resp.ok) {
      console.error("[patch_sim] export figure HTTP", resp.status);
      return;
    }
    var fig = await resp.json();
    var format = /*FORMAT*/;
    var filename = /*FILENAME*/;

    var data = [];
    fig.data.forEach(function (trace, i) {
      if (trace.meta && trace.meta.carrier) return;
      var copy = Object.assign({}, trace);
      var live = gd.data[i];
      if (live && live.visible !== undefined) copy.visible = live.visible;
      // WebGL traces do not render to vector output.
      if (format === "svg" && copy.type === "scattergl") copy.type = "scatter";
      data.push(copy);
    });

    await Plotly.downloadImage(
      { data: data, layout: fig.layout },
      {
        format: format,
        filename: filename,
        width: gd.offsetWidth,
        height: gd.offsetHeight,
        scale: format === "png" ? 3 : 1,
      }
    );
  } catch (err) {
    console.error("[patch_sim] export figure error:", err);
  }
})();
