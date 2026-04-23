"""Plotly figure construction for simulation results.

Pure functions — no Reflex dependency.
Follows the same separation as protocol_builders.py.
"""

from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from patch_sim_ui.constants import (
    CC_VOLTAGE_COLOR,
    CHANNEL_COLORS,
    CURRENT_CLAMP,
    GATING_VAR_COLORS,
    STIMULUS_COLOR,
    STORED_TRACE_COLORS,
    SWEEP_COLORS,
    VOLTAGE_CLAMP,
)
from patch_sim_ui.sweep import Sweep

# Maximum number of hover carrier points; keeps tooltip HTML build time short.
_MAX_HOVER_POINTS = 2000

# Hover tooltip column width in characters.
_HOVER_COL_WIDTH = 8

# Inline CSS applied to hover tooltip ``<span>`` elements.
_HOVER_MONO_STYLE = "font-family: monospace; font-size: 11px;"

# Relative row heights for the 3-row subplot layout (response, gating, stimulus).
_SUBPLOT_ROW_HEIGHTS = [0.5, 0.25, 0.25]

# Vertical gap between subplot rows (fraction of total figure height).
_SUBPLOT_VERT_SPACING = 0.08

# Line width used for the total-current trace in Voltage Clamp mode.
_TOTAL_CURRENT_LINE_WIDTH = 4

# Plot margin in pixels: left, right, top, bottom.
_PLOT_MARGIN = {"l": 60, "r": 20, "t": 30, "b": 40}

# Shared layout kwargs applied to every analysis sub-figure (F-I, I-V, g-V,
# SFA, phase-plane).  Kept separate from the main trace plot (_PLOT_MARGIN)
# because the analysis figures are smaller and have tighter margins.
_ANALYSIS_FIGURE_LAYOUT: dict = {
    "template": "plotly_white",
    "margin": {"l": 50, "r": 10, "t": 10, "b": 40},
    "height": 260,
}


@dataclass
class TraceVisibility:
    """Visibility flags for every plotted trace.

    All classic flags default to ``True`` so callers only need to supply
    the flags they want to override.  Additional-channel dicts are empty
    by default, which ``build_figure`` and ``_build_hover_tables`` treat as
    "show all".

    Attributes:
        voltage: Show the membrane voltage trace (Current Clamp).
        total_current: Show the summed ion current trace (Voltage Clamp).
        sodium_current: Show I_Na.
        potassium_current: Show I_K.
        na_leak_current: Show I_NaL (Na⁺ leak).
        k_leak_current: Show I_KL (K⁺ leak).
        potassium_activation: Show gating variable n.
        sodium_activation: Show gating variable m.
        sodium_inactivation: Show gating variable h.
        additional_currents: Visibility flags for extra channel currents,
            keyed by channel name.
        additional_gating: Visibility flags for extra gating variables,
            keyed by variable name.
    """

    voltage: bool = True
    total_current: bool = True
    sodium_current: bool = True
    potassium_current: bool = True
    na_leak_current: bool = True
    k_leak_current: bool = True
    potassium_activation: bool = True
    sodium_activation: bool = True
    sodium_inactivation: bool = True
    additional_currents: dict[str, bool] = field(default_factory=dict)
    additional_gating: dict[str, bool] = field(default_factory=dict)


def _build_hover_tables(
    current_sweeps: list[Sweep],
    visibility: TraceVisibility,
    add_current_keys: list[str],
    add_gating_keys: list[str],
    is_vc: bool,
    stride: int,
) -> tuple[list[str], list[str], list[str]]:
    """Build per-time-point HTML table strings for multi-sweep hover tooltips.

    For each downsampled time point, produces a monospace HTML table showing
    all sweeps as rows and visible quantities as columns — one table string
    per subplot (response, gating, stimulus).

    Args:
        current_sweeps: The N sweeps from the multi-sweep run.
        visibility: Consolidated trace visibility flags.
        add_current_keys: Ordered list of additional current channel names.
        add_gating_keys: Ordered list of additional gating variable names.
        is_vc: True when in Voltage Clamp mode.
        stride: Downsampling stride — every Nth time point is included.

    Returns:
        A 3-tuple of ``(resp_html, gating_html, stim_html)``, each a list of
        HTML strings, one per downsampled time point.
    """
    time_vals = current_sweeps[0].time
    indices = np.arange(0, len(time_vals), stride)
    n_pts = len(indices)
    col_w = _HOVER_COL_WIDTH
    label_w = max((len(s.label) for s in current_sweeps), default=6)
    mono_style = _HOVER_MONO_STYLE
    n_sweeps = len(current_sweeps)
    # Pre-compute padded labels once — reused across all subplots.
    labels = [f"{s.label:<{label_w}}" for s in current_sweeps]

    def _fmt_table(header: str, row_groups: list[list[str]]) -> list[str]:
        """Format a list of HTML table strings from per-time-point row groups.

        Args:
            header: The fixed header row string shared across all time points.
            row_groups: One list of row strings per time point.

        Returns:
            List of HTML strings, one per time point.
        """
        return [
            f'<span style="{mono_style}">{header}<br>{"<br>".join(rows)}</span>'
            for rows in row_groups
        ]

    def _build_rows_html(
        cols: list[tuple[str, str, str]],
        additional_attr: str,
        fmt_spec: str,
    ) -> list[str]:
        """Build per-time-point HTML for one subplot given column specs.

        Pre-extracts all column data into a numpy array and downsamples once
        with fancy indexing, eliminating per-element ``getattr`` and
        ``dict.get`` calls from the inner loop.

        Closes over ``indices``, ``current_sweeps``, ``col_w``, ``label_w``,
        ``n_pts``, ``n_sweeps``, ``labels``, and ``_fmt_table``.

        Args:
            cols: Column spec triples ``(header_label, source, data_key)``.
                ``source`` is either ``"classic"`` (attribute on ``Sweep``) or
                ``"additional"`` (looked up via ``additional_attr``).
            additional_attr: Name of the ``Sweep`` dict attribute used for
                ``"additional"`` columns (e.g. ``"additional_currents"`` or
                ``"additional_gating"``).
            fmt_spec: Python format spec for numeric values (e.g. ``".2f"``).

        Returns:
            List of HTML strings, one per time point, or a list of empty
            strings when ``cols`` is empty.
        """
        if not cols:
            return [""] * n_pts
        header = " " * label_w + "".join(f"{c[0]:>{col_w}}" for c in cols)
        # Pre-extract every column into a (n_sweeps, n_hover_pts) array.
        # Downsampling via fancy indexing is done once per column rather than
        # per time-point, eliminating getattr/dict.get from the inner loop.
        col_arrays: list[np.ndarray] = []
        for _, src, key in cols:
            if src == "classic":
                raw = np.array([getattr(s, key) for s in current_sweeps], dtype=float)
            else:
                raw = np.array(
                    [getattr(s, additional_attr).get(key, []) for s in current_sweeps],
                    dtype=float,
                )
            col_arrays.append(raw[:, indices])  # (n_sweeps, n_hover_pts)
        # Stack and transpose to (n_hover_pts, n_sweeps, n_cols) for time-first
        # iteration — each slice data_t[t] is (n_sweeps, n_cols).
        # Convert to a nested Python list via .tolist() so that inner
        # iteration works on Python floats rather than numpy scalars,
        # avoiding per-element boxing overhead.
        data_py: list[list[list[float]]] = (
            np.stack(col_arrays).transpose(2, 1, 0).tolist()
        )
        row_groups = [
            [
                labels[s] + "".join(f"{v:>{col_w}{fmt_spec}}" for v in data_py[t][s])
                for s in range(n_sweeps)
            ]
            for t in range(n_pts)
        ]
        return _fmt_table(header, row_groups)

    # --- Response subplot (row 1) ---
    # Col spec: (header_label, source, data_key)
    if is_vc:
        resp_cols: list[tuple[str, str, str]] = []
        if visibility.total_current:
            resp_cols.append(("I_total", "classic", "total_current"))
        if visibility.sodium_current:
            resp_cols.append(("I_Na", "classic", "sodium_current"))
        if visibility.potassium_current:
            resp_cols.append(("I_K", "classic", "potassium_current"))
        if visibility.na_leak_current:
            resp_cols.append(("I_NaL", "classic", "na_leak_current"))
        if visibility.k_leak_current:
            resp_cols.append(("I_KL", "classic", "k_leak_current"))
        for ch_name in add_current_keys:
            if visibility.additional_currents.get(ch_name, True):
                resp_cols.append((f"I_{ch_name}", "additional", ch_name))
    else:
        resp_cols = [("V (mV)", "classic", "voltage")]

    resp_html = _build_rows_html(resp_cols, "additional_currents", ".2f")

    # --- Gating subplot (row 2) ---
    gating_cols: list[tuple[str, str, str]] = []
    if visibility.potassium_activation:
        gating_cols.append(("n", "classic", "potassium_activation"))
    if visibility.sodium_activation:
        gating_cols.append(("m", "classic", "sodium_activation"))
    if visibility.sodium_inactivation:
        gating_cols.append(("h", "classic", "sodium_inactivation"))
    for gv_name in add_gating_keys:
        if visibility.additional_gating.get(gv_name, True):
            gating_cols.append((gv_name, "additional", gv_name))

    gating_html = _build_rows_html(gating_cols, "additional_gating", ".3f")

    # --- Stimulus subplot (row 3) ---
    stim_col_label = "Cmd (mV)" if is_vc else "Stim"
    stim_header = " " * label_w + f"{stim_col_label:>{col_w}}"
    # Pre-extract stimulus arrays and downsample once: (n_sweeps, n_hover_pts).
    # .tolist() converts to Python floats before the formatting loop.
    stim_raw = np.array([s.stimulus for s in current_sweeps], dtype=float)
    stim_py: list[list[float]] = stim_raw[:, indices].T.tolist()
    stim_row_groups = [
        [labels[s] + f"{stim_py[t][s]:>{col_w}.2f}" for s in range(n_sweeps)]
        for t in range(n_pts)
    ]
    stim_html = _fmt_table(stim_header, stim_row_groups)

    return resp_html, gating_html, stim_html


def compute_trace_visibility_map(
    current_sweeps: list[Sweep],
    clamp_mode: str,
    additional_current_field_map: dict[str, str] | None = None,
    additional_gating_field_map: dict[str, str] | None = None,
    stored_traces: list[Sweep] | None = None,
) -> dict[str, list[int]]:
    """Return a mapping from show_* field names to Plotly trace indices.

    Mirrors the trace-insertion order used by ``build_figure`` so callers can
    toggle individual traces via ``Plotly.restyle`` without rebuilding the
    figure.  Only current-sweep traces controlled by a ``show_*`` field are
    mapped; stored traces and carrier traces advance the counter but are not
    included (they are always visible).

    Args:
        current_sweeps: Latest simulation result sweeps.
        clamp_mode: Active UI clamp mode (CURRENT_CLAMP or VOLTAGE_CLAMP).
        additional_current_field_map: Optional mapping from additional-current
            sweep keys (e.g. ``"Ih"``) to ``show_*`` field names
            (e.g. ``"show_ih_current"``).  Keys absent from the map advance
            the index counter but are not recorded.
        additional_gating_field_map: Optional mapping from additional-gating
            sweep keys (e.g. ``"r"``) to ``show_*`` field names
            (e.g. ``"show_ih_gating"``).  Keys absent from the map advance
            the index counter but are not recorded.
        stored_traces: Oscilloscope-style stored reference traces.  Always
            visible; their indices are not recorded in the result map.

    Returns:
        Dict mapping each ``show_*`` field name to the list of Plotly trace
        indices it controls.  When the same field controls one trace per sweep
        (e.g. multi-sweep Step) the list contains one index per sweep.
    """
    result: dict[str, list[int]] = {}
    idx = 0
    add_curr = additional_current_field_map or {}
    add_gating = additional_gating_field_map or {}
    is_multi_sweep = len(current_sweeps) > 1

    def _map(field: str) -> None:
        """Record the current index under field and advance the counter."""
        nonlocal idx
        result.setdefault(field, []).append(idx)
        idx += 1

    def _skip() -> None:
        """Advance the trace index counter without recording a mapping."""
        nonlocal idx
        idx += 1

    for sweep in current_sweeps:
        if sweep.clamp_mode == CURRENT_CLAMP:
            _map("show_voltage")
        else:
            # Voltage Clamp: matches _add_vc_currents insertion order.
            _map("show_total_current")
            _map("show_sodium_current")
            _map("show_potassium_current")
            _map("show_na_leak_current")
            _map("show_k_leak_current")
            for ch_name in sweep.additional_currents:
                field = add_curr.get(ch_name)
                if field:
                    _map(field)
                else:
                    _skip()

        # Gating row is always present in both modes.
        _map("show_potassium_activation")
        _map("show_sodium_activation")
        _map("show_sodium_inactivation")
        for gv_name in sweep.additional_gating:
            field = add_gating.get(gv_name)
            if field:
                _map(field)
            else:
                _skip()

        _skip()  # stimulus — always visible, no show_* field

    for sweep in stored_traces or []:
        # Stored traces are always visible; just advance the counter.
        if sweep.clamp_mode == CURRENT_CLAMP:
            _skip()  # voltage
        else:
            _skip()  # total_current
        _skip()  # stimulus

    # Invisible hover-carrier traces in multi-sweep mode — always visible.
    if is_multi_sweep and current_sweeps:
        for _ in range(3):
            _skip()

    return result


def build_figure(
    current_sweeps: list[Sweep],
    visibility: TraceVisibility,
    clamp_mode: str,
    stored_traces: list[Sweep] | None = None,
    show_hover: bool = True,
) -> go.Figure:
    """Build a Plotly figure from current sweeps and stored reference traces.

    Current sweeps are rendered with visibility-flag filtering applied via
    Plotly's ``visible`` property so that the subplot layout remains fixed
    regardless of which traces are toggled.  This allows ``react-plotly.js``
    to use the efficient ``Plotly.react()`` path instead of a full rebuild.

    Stored traces are oscilloscope-style background reference snapshots,
    rendered with dashed faded lines; only voltage (CC) or total_current (VC)
    and stimulus are plotted.

    Both **Current Clamp** and **Voltage Clamp** use a fixed 3-row layout.
    In Voltage Clamp mode all ion current channels are overlaid on a single
    subplot (row 1) with each channel identified by a fixed colour from
    ``CHANNEL_COLORS`` and a legend entry.

    When there are multiple current sweeps (multi-sweep mode), hover on all
    real traces is suppressed and replaced with invisible carrier traces that
    display a monospace HTML table — sweeps as rows, quantities as columns —
    via ``hovertemplate``/``customdata``.  Single-sweep modes keep the
    standard ``hovermode="x unified"`` behaviour.

    Args:
        current_sweeps: Latest simulation result (1 sweep for standard runs,
            N sweeps for multi-sweep Step).
        visibility: Consolidated trace visibility flags.
        clamp_mode: Active UI clamp mode, used for layout and axis labels.
        stored_traces: Oscilloscope-style stored reference traces; rendered as
            faded dashed lines behind the live traces.
        show_hover: When ``True`` (default) the figure's ``hovermode`` is set
            to ``"x"`` (multi-sweep) or ``"x unified"`` (single-sweep).  When
            ``False``, ``hovermode`` is set to ``False`` to suppress tooltips.

    Returns:
        A Plotly Figure with response, gating, and stimulus subplots.
    """
    # Collect additional keys present in current sweeps.
    add_gating_keys: list[str] = []
    add_current_keys: list[str] = []
    for sweep in current_sweeps:
        for key in sweep.additional_gating:
            if key not in add_gating_keys:
                add_gating_keys.append(key)
        for key in sweep.additional_currents:
            if key not in add_current_keys:
                add_current_keys.append(key)

    is_vc = clamp_mode == VOLTAGE_CLAMP
    is_multi_sweep = len(current_sweeps) > 1

    # Both modes use a fixed 3-row layout.
    rows = 3
    row_heights = _SUBPLOT_ROW_HEIGHTS
    gating_row = 2
    stimulus_row = 3

    vert_spacing = _SUBPLOT_VERT_SPACING if rows > 1 else 0.0
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=vert_spacing,
    )

    # legend_entries[row] accumulates trace indices for traces shown in the
    # legend.  After all traces are added we suppress the legend for any row
    # that ended up with only one entry (y-axis label is sufficient there).
    legend_entries: dict[int, list[int]] = {1: [], gating_row: []}

    def _scatter(
        x,
        y,
        name,
        row,
        color=None,
        visible=True,
        dash=None,
        width=None,
        hoverinfo=None,
        showlegend=True,
        legend_ref: str = "legend",
        sweep_index: int = -1,
    ):
        """Add a Scattergl trace to the figure.

        Args:
            x: X-axis data values.
            y: Y-axis data values.
            name: Trace name shown in the legend / hover.
            row: Subplot row number (1-based).
            color: Line colour string; ``None`` lets Plotly choose.
            visible: Whether the trace is initially visible.
            dash: Line dash style (e.g. ``"dash"``); ``None`` for solid.
            width: Line width in pixels; ``None`` uses the default.
            hoverinfo: Plotly hoverinfo value (e.g. ``"skip"``); ``None``
                uses the default.
            showlegend: Whether this trace appears in the legend.
            legend_ref: Which Plotly legend object this trace belongs to
                (``"legend"`` for the response row, ``"legend2"`` for the
                gating row).
            sweep_index: Index of the sweep this trace belongs to.  ``-1``
                marks non-sweep traces (saved, stored, carrier).
        """
        line: dict = {"color": color} if color is not None else {}
        if dash is not None:
            line["dash"] = dash
        if width is not None:
            line["width"] = width
        kwargs: dict = {}
        if hoverinfo is not None:
            kwargs["hoverinfo"] = hoverinfo
        trace_idx = len(fig.data)
        if showlegend and row in legend_entries:
            legend_entries[row].append(trace_idx)
        fig.add_trace(
            go.Scattergl(
                x=np.asarray(x),
                y=np.asarray(y),
                name=name,
                mode="lines",
                line=line,
                visible=visible,
                showlegend=showlegend,
                legend=legend_ref,
                meta={"sweep": sweep_index},
                **kwargs,
            ),
            row=row,
            col=1,
        )

    def _add_vc_currents(
        sweep: Sweep,
        pfx: str,
        dash: str | None = None,
        hoverinfo: str | None = None,
        visibility: TraceVisibility | None = None,
        showlegend: bool = True,
        legend_ref: str = "legend",
        sweep_index: int = -1,
    ) -> None:
        """Add Voltage Clamp current traces for one sweep to the figure.

        Iterates over the four classic current channels and any additional
        channels, calling ``_scatter`` for each.

        Args:
            sweep: The sweep whose current data to plot.
            pfx: Label prefix prepended to each trace name.
            dash: Line dash style (e.g. ``"dash"``); ``None`` for solid.
            hoverinfo: Plotly hoverinfo value; ``None`` uses the default.
            visibility: Trace visibility flags; ``None`` treats every trace
                as visible.
            showlegend: Whether these traces appear in the legend.
            legend_ref: Which Plotly legend object these traces belong to.
            sweep_index: Index of the sweep this trace belongs to.  ``-1``
                marks non-sweep traces (saved, stored, carrier).
        """
        classic_defs: list[tuple[str, str, int | None]] = [
            ("total_current", "I_total", _TOTAL_CURRENT_LINE_WIDTH),
            ("sodium_current", "I_Na", None),
            ("potassium_current", "I_K", None),
            ("na_leak_current", "I_NaL", None),
            ("k_leak_current", "I_KL", None),
        ]
        for attr, label, width in classic_defs:
            vis = getattr(visibility, attr) if visibility is not None else True
            _scatter(
                sweep.time,
                getattr(sweep, attr),
                f"{pfx}{label}",
                1,
                CHANNEL_COLORS.get(attr),
                visible=vis,
                width=width,
                dash=dash,
                hoverinfo=hoverinfo,
                showlegend=showlegend,
                legend_ref=legend_ref,
                sweep_index=sweep_index,
            )
        for ch_name, vals in sweep.additional_currents.items():
            vis = (
                visibility.additional_currents.get(ch_name, True)
                if visibility is not None
                else True
            )
            _scatter(
                sweep.time,
                vals,
                f"{pfx}I_{ch_name}",
                1,
                CHANNEL_COLORS.get(ch_name),
                visible=vis,
                dash=dash,
                hoverinfo=hoverinfo,
                showlegend=showlegend,
                legend_ref=legend_ref,
                sweep_index=sweep_index,
            )

    # In multi-sweep mode all real traces suppress hover; carrier traces
    # deliver the table tooltips instead.
    hi = "skip" if is_multi_sweep else None

    for sweep_idx, sweep in enumerate(current_sweeps):
        t = sweep.time
        sweep_mode = sweep.clamp_mode
        pfx = f"{sweep.label} " if sweep.label else ""
        stim_label = (
            "Stimulus (µA/cm²)" if sweep_mode == CURRENT_CLAMP else "Command (mV)"
        )
        # Show legend only for the first sweep; subsequent sweeps share the
        # same colours so their legend entries would be duplicates.
        sl = sweep_idx == 0

        if sweep_mode == CURRENT_CLAMP:
            _scatter(
                t,
                sweep.voltage,
                f"{pfx}Voltage (mV)",
                1,
                CC_VOLTAGE_COLOR,
                visible=visibility.voltage,
                hoverinfo=hi,
                showlegend=sl,
                sweep_index=sweep_idx,
            )
        else:
            # Voltage Clamp: all currents overlaid on row 1 with channel colours.
            # Legend-eligible traces (first sweep) use an empty prefix so the
            # voltage-clamp command level doesn't appear in the legend label
            # (e.g. "I_total" not "100 mV I_total").  Individual hover is
            # suppressed in multi-sweep mode anyway, so the name is legend-only.
            vc_pfx = "" if sl else pfx
            _add_vc_currents(
                sweep,
                vc_pfx,
                hoverinfo=hi,
                visibility=visibility,
                showlegend=sl,
                sweep_index=sweep_idx,
            )

        # Gating row is always present; traces are always added with their
        # visibility flags so the layout never changes on toggle.
        # All gating traces use bare variable names regardless of sweep so that
        # the legend always shows "n", "m", "h" — never a stimulus prefix.
        for gv_attr, gv_label in [
            ("potassium_activation", "n"),
            ("sodium_activation", "m"),
            ("sodium_inactivation", "h"),
        ]:
            _scatter(
                t,
                getattr(sweep, gv_attr),
                gv_label,
                gating_row,
                GATING_VAR_COLORS.get(gv_label),
                visible=getattr(visibility, gv_attr),
                hoverinfo=hi,
                showlegend=sl,
                legend_ref="legend2",
                sweep_index=sweep_idx,
            )
        for gv_name, gv_vals in sweep.additional_gating.items():
            vis = visibility.additional_gating.get(gv_name, True)
            _scatter(
                t,
                gv_vals,
                gv_name,
                gating_row,
                GATING_VAR_COLORS.get(gv_name),
                visible=vis,
                hoverinfo=hi,
                showlegend=sl,
                legend_ref="legend2",
                sweep_index=sweep_idx,
            )

        _scatter(
            t,
            sweep.stimulus,
            stim_label,
            stimulus_row,
            STIMULUS_COLOR,
            hoverinfo=hi,
            showlegend=False,
            sweep_index=sweep_idx,
        )

    for i, sweep in enumerate(stored_traces or []):
        c = sweep.color or STORED_TRACE_COLORS[i % len(STORED_TRACE_COLORS)]
        label = sweep.label or f"Stored {i + 1}"
        if sweep.clamp_mode == CURRENT_CLAMP:
            _scatter(
                sweep.time,
                sweep.voltage,
                label,
                1,
                c,
                dash="dash",
                hoverinfo="skip",
            )
        else:
            _scatter(
                sweep.time,
                sweep.total_current,
                label,
                1,
                c,
                dash="dash",
                hoverinfo="skip",
            )
        _scatter(
            sweep.time,
            sweep.stimulus,
            label,
            stimulus_row,
            c,
            dash="dash",
            hoverinfo="skip",
            showlegend=False,
        )

    # Add invisible hover carrier traces for multi-sweep mode.
    # Each carrier sits on one subplot and delivers a per-time-point HTML
    # table via customdata / hovertemplate.
    if is_multi_sweep and current_sweeps:
        first = current_sweeps[0]
        time_vals = first.time
        n_t = len(time_vals)
        stride = max(1, n_t // _MAX_HOVER_POINTS)
        indices = np.arange(0, n_t, stride)
        carrier_x = np.asarray(time_vals)[indices]

        resp_html, gating_html, stim_html = _build_hover_tables(
            current_sweeps=current_sweeps,
            visibility=visibility,
            add_current_keys=add_current_keys,
            add_gating_keys=add_gating_keys,
            is_vc=is_vc,
            stride=stride,
        )

        # Carrier y values mirror the first sweep so autorange is unaffected.
        if is_vc:
            carrier_y1 = np.asarray(first.total_current)[indices]
        else:
            carrier_y1 = np.asarray(first.voltage)[indices]
        carrier_y_gating = np.asarray(first.potassium_activation)[indices]
        carrier_y_stim = np.asarray(first.stimulus)[indices]

        transparent = dict(color="rgba(0,0,0,0)")
        for carrier_y, html_strings, sub_row in (
            (carrier_y1, resp_html, 1),
            (carrier_y_gating, gating_html, gating_row),
            (carrier_y_stim, stim_html, stimulus_row),
        ):
            fig.add_trace(
                go.Scattergl(
                    x=carrier_x,
                    y=carrier_y,
                    mode="lines",
                    line=transparent,
                    showlegend=False,
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=html_strings,
                    name="",
                    meta={"sweep": -1},
                ),
                row=sub_row,
                col=1,
            )

    # Suppress legend for any subplot row that has only one entry — the
    # y-axis label is sufficient in that case.
    for row_traces in legend_entries.values():
        if len(row_traces) <= 1:
            for idx in row_traces:
                fig.data[idx].showlegend = False

    # Y-axis labels.
    if is_vc:
        fig.update_yaxes(title_text="Current (µA/cm²)", row=1, col=1)
        fig.update_yaxes(title_text="Voltage (mV)", row=stimulus_row, col=1)
    else:
        fig.update_yaxes(title_text="Voltage (mV)", row=1, col=1)
        fig.update_yaxes(title_text="Current (µA/cm²)", row=stimulus_row, col=1)

    # Gating row is always present in both modes.
    fig.update_yaxes(title_text="Gating", row=gating_row, col=1, range=[0, 1])

    fig.update_xaxes(title_text="Time (ms)", row=stimulus_row, col=1)

    # Compute paper-coordinate y of the top of each subplot row so each
    # legend can be anchored inside its own subplot.
    _scale = (1.0 - (rows - 1) * vert_spacing) / sum(_SUBPLOT_ROW_HEIGHTS)
    _row1_top = 1.0
    _row2_top = _row1_top - _SUBPLOT_ROW_HEIGHTS[0] * _scale - vert_spacing

    _legend_common = dict(
        orientation="v",
        x=0.99,
        xanchor="right",
        bgcolor="rgba(255,255,255,0.7)",
        borderwidth=0,
    )

    hovermode: str | bool = (
        ("x" if is_multi_sweep else "x unified") if show_hover else False
    )
    fig.update_layout(
        autosize=True,
        margin=_PLOT_MARGIN,
        hovermode=hovermode,
        showlegend=True,
        legend=dict(**_legend_common, y=_row1_top, yanchor="top"),
        legend2=dict(**_legend_common, y=_row2_top, yanchor="top"),
    )
    return fig


def build_phase_plane_figure(phase_plane_data: dict) -> go.Figure:
    """Build a V vs dV/dt phase-plane figure from serialised sweep data.

    Renders one trace per sweep.  In single-sweep runs the trace uses the
    standard current-clamp colour; in multi-sweep runs each sweep is drawn
    with its stored colour so the trajectories are distinguishable.

    Args:
        phase_plane_data: Dict with key ``"sweeps"``, a list of dicts each
            containing ``"voltage"`` (list[float], mV), ``"dvdt"``
            (list[float], mV/ms), ``"label"`` (str), and ``"color"`` (str).

    Returns:
        A Plotly Figure with a single V vs dV/dt scatter subplot, or an
        empty figure when ``phase_plane_data`` is empty.
    """
    sweeps = phase_plane_data.get("sweeps", [])
    if not sweeps:
        return go.Figure()

    fig = go.Figure()
    for sweep in sweeps:
        voltage = sweep.get("voltage", [])
        dvdt = sweep.get("dvdt", [])
        if not voltage or not dvdt:
            continue
        color = sweep.get("color") or CC_VOLTAGE_COLOR
        label = sweep.get("label", "")
        hover = (
            f"<b>{label}</b><br>"
            "V: %{x:.1f} mV<br>"
            "dV/dt: %{y:.1f} mV/ms"
            "<extra></extra>"
            if label
            else "V: %{x:.1f} mV<br>dV/dt: %{y:.1f} mV/ms<extra></extra>"
        )
        fig.add_trace(
            go.Scattergl(
                x=np.asarray(voltage),
                y=np.asarray(dvdt),
                name=label,
                mode="lines",
                line={"color": color},
                showlegend=False,
                hovertemplate=hover,
            )
        )

    fig.update_layout(
        **_ANALYSIS_FIGURE_LAYOUT,
        hovermode="closest",
        showlegend=False,
        xaxis_title="V (mV)",
        yaxis_title="dV/dt (mV/ms)",
    )
    return fig


def build_fi_figure(fi_data: dict) -> go.Figure:
    """Build a Plotly F-I curve figure from serialised F-I analysis results.

    Renders up to three traces: mean firing rate (green, solid), initial firing
    rate (orange, dashed), and steady-state firing rate (blue, dotted), plotted
    against injected current.  Steps with no spikes (``None`` values) are
    omitted from each trace, so traces may be shorter than the full sweep range.

    Args:
        fi_data: Dict with keys ``current_steps``, ``mean_firing_rates``,
            ``initial_firing_rates``, and ``steady_state_firing_rates``, each
            a list aligned by index.  Rate entries may be ``None`` for silent
            steps.

    Returns:
        A Plotly :class:`go.Figure` ready to be serialised and sent to the UI.
    """
    currents = fi_data["current_steps"]
    mean_rates = fi_data["mean_firing_rates"]
    initial_rates = fi_data["initial_firing_rates"]
    ss_rates = fi_data["steady_state_firing_rates"]

    def _filter(rates: list) -> tuple[list[float], list[float]]:
        """Return (x, y) pairs where rate is not None.

        Args:
            rates: List of rate values (float or None), parallel to
                ``currents``.

        Returns:
            Tuple of (filtered current steps, filtered rate values).
        """
        pairs = [(c, r) for c, r in zip(currents, rates) if r is not None]
        if not pairs:
            return [], []
        xs, ys = zip(*pairs)
        return list(xs), list(ys)

    fig = go.Figure()
    x_mean, y_mean = _filter(mean_rates)
    if x_mean:
        fig.add_trace(
            go.Scatter(
                x=x_mean,
                y=y_mean,
                mode="lines+markers",
                name="Mean",
                line=dict(color="#27ae60"),
                marker=dict(size=5),
            )
        )
    x_init, y_init = _filter(initial_rates)
    if x_init:
        fig.add_trace(
            go.Scatter(
                x=x_init,
                y=y_init,
                mode="lines+markers",
                name="Initial",
                line=dict(color="#e67e22", dash="dash"),
                marker=dict(size=5, symbol="triangle-up"),
            )
        )
    x_ss, y_ss = _filter(ss_rates)
    if x_ss:
        fig.add_trace(
            go.Scatter(
                x=x_ss,
                y=y_ss,
                mode="lines+markers",
                name="Steady-state",
                line=dict(color="#3498db", dash="dot"),
                marker=dict(size=5, symbol="diamond"),
            )
        )
    fig.update_layout(
        **_ANALYSIS_FIGURE_LAYOUT,
        xaxis_title="Current (µA/cm²)",
        yaxis_title="Firing Rate (Hz)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
    )
    return fig


def build_hyperpolarization_figure(hyp_data: dict) -> go.Figure:
    """Build a Plotly sag/rebound figure from serialised hyperpolarization results.

    Renders sag amplitude (mV, left y-axis, teal) and rebound spike count
    (right y-axis, orange) plotted against injected current (µA/cm²).  Steps
    with zero rebound spikes are included as zero in the rebound trace.

    Args:
        hyp_data: Dict with keys ``current_steps``, ``sag_amplitudes``, and
            ``rebound_spike_counts``, each a list aligned by index.

    Returns:
        A Plotly :class:`go.Figure` ready to be serialised and sent to the UI.
    """
    currents = hyp_data["current_steps"]
    sag_amps = hyp_data["sag_amplitudes"]
    rebound_counts = hyp_data["rebound_spike_counts"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=currents,
            y=sag_amps,
            mode="lines+markers",
            name="Sag amplitude",
            line=dict(color="#16a085"),
            marker=dict(size=6),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=currents,
            y=rebound_counts,
            mode="lines+markers",
            name="Rebound spikes",
            line=dict(color="#e67e22", dash="dash"),
            marker=dict(size=6, symbol="triangle-up"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        **_ANALYSIS_FIGURE_LAYOUT,
        xaxis_title="Current (µA/cm²)",
        yaxis=dict(title="Sag amplitude (mV)", color="#16a085"),
        yaxis2=dict(
            title="Rebound spikes",
            overlaying="y",
            side="right",
            color="#e67e22",
            tickformat="d",
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
    )
    return fig


def build_sfa_figure(sfa_data: dict) -> go.Figure:
    """Build a Plotly SFA figure from serialised spike-frequency adaptation data.

    Each curve represents one current clamp sweep and is plotted as
    instantaneous firing frequency (Hz) vs. spike interval index (1-indexed).
    Single-sweep runs display one green curve with an adaptation index
    annotation.  Multi-sweep runs overlay one curve per sweep using distinct
    colours from :data:`~patch_sim_ui.constants.SWEEP_COLORS`, with the
    adaptation index appended to each legend entry.

    Args:
        sfa_data: Dict with a ``"curves"`` key containing a list of curve
            dicts.  Each curve dict must have ``"spike_indices"``
            (list[int]), ``"instantaneous_frequencies"`` (list[float]),
            ``"adaptation_index"`` (float), and ``"label"`` (str).

    Returns:
        A Plotly :class:`go.Figure` ready to be serialised and sent to the UI.
    """
    curves = sfa_data.get("curves", [])
    fig = go.Figure()
    is_multi = len(curves) > 1

    for i, curve in enumerate(curves):
        x = [idx + 1 for idx in curve.get("spike_indices", [])]
        y = curve.get("instantaneous_frequencies", [])
        ai = curve.get("adaptation_index", 0.0)
        label = curve.get("label", "")

        if is_multi:
            color = SWEEP_COLORS[i % len(SWEEP_COLORS)]
            display = label if label else f"Sweep {i + 1}"
            hover_name = f"<b>{display}</b> (AI: {ai:.2f})"
        else:
            color = "#27ae60"
            hover_name = "Inst. frequency"

        hover = (
            f"{hover_name}<br>Interval: %{{x}}<br>Freq: %{{y:.1f}} Hz<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=hover_name,
                line=dict(color=color),
                marker=dict(size=6),
                showlegend=False,
                hovertemplate=hover,
            )
        )

    # For single-sweep, annotate the adaptation index directly on the plot.
    annotations = []
    if not is_multi and curves:
        ai = curves[0].get("adaptation_index", 0.0)
        annotations.append(
            dict(
                text=f"AI = {ai:.2f}",
                xref="paper",
                yref="paper",
                x=0.99,
                y=0.97,
                xanchor="right",
                yanchor="top",
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.7)",
            )
        )

    fig.update_layout(
        **_ANALYSIS_FIGURE_LAYOUT,
        xaxis_title="Spike Interval #",
        yaxis_title="Inst. Frequency (Hz)",
        showlegend=False,
        annotations=annotations,
    )
    return fig


def build_iv_figure(iv_data: dict) -> go.Figure:
    """Build a Plotly I-V curve figure from serialised I-V analysis results.

    Renders three traces: peak inward current (red, solid), peak outward
    current (orange, solid), and steady-state current (blue, dashed) plotted
    against voltage.  The steady-state line uses a dashed style and diamond
    markers so it remains distinguishable when it overlaps with the peak
    outward trace.

    Args:
        iv_data: Dict with keys ``voltages``, ``peak_inward_currents``,
            ``peak_outward_currents``, and ``steady_state_currents``, each a
            list of floats sorted by ascending voltage.

    Returns:
        A Plotly :class:`go.Figure` ready to be serialised and sent to the UI.
    """
    voltages = iv_data["voltages"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=voltages,
            y=iv_data["peak_inward_currents"],
            mode="lines+markers",
            name="Peak inward",
            line=dict(color="#e74c3c"),
            marker=dict(size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=voltages,
            y=iv_data["peak_outward_currents"],
            mode="lines+markers",
            name="Peak outward",
            line=dict(color="#e67e22"),
            marker=dict(size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=voltages,
            y=iv_data["steady_state_currents"],
            mode="lines+markers",
            name="Steady-state",
            line=dict(color="#3498db", dash="dash"),
            marker=dict(size=5, symbol="diamond"),
        )
    )
    fig.update_layout(
        **_ANALYSIS_FIGURE_LAYOUT,
        xaxis_title="Voltage (mV)",
        yaxis_title="Current (µA/cm²)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
    )
    return fig


def build_gv_figure(gv_data: dict) -> go.Figure:
    """Build a Plotly g-V curve figure with optional Boltzmann fit overlay.

    Renders the normalised conductance (G/G_max) data points as a scatter
    trace.  When the Boltzmann fit converged, a smooth fit curve is overlaid
    as a dashed line and the half-activation voltage (V_half) and slope factor
    (k) are shown as an annotation.

    Args:
        gv_data: Dict with keys ``voltages``, ``g_normalized``,
            ``reversal_potential``, ``boltzmann_converged``, ``v_half``, ``k``,
            ``fit_voltages``, and ``fit_g_normalized``.

    Returns:
        A Plotly :class:`go.Figure` ready to be serialised and sent to the UI.
    """
    voltages = gv_data["voltages"]
    g_norm = gv_data["g_normalized"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=voltages,
            y=g_norm,
            mode="markers",
            name="G/G\u2098\u2090\u02e3",
            marker=dict(color="#27ae60", size=7),
        )
    )

    if gv_data.get("boltzmann_converged"):
        v_half = gv_data["v_half"]
        k = gv_data["k"]
        fig.add_trace(
            go.Scatter(
                x=gv_data["fit_voltages"],
                y=gv_data["fit_g_normalized"],
                mode="lines",
                name="Boltzmann fit",
                line=dict(color="#1e8449", dash="dash"),
            )
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            text=f"V\u2081/\u2082 = {v_half:.1f} mV<br>k = {k:.1f} mV",
            showarrow=False,
            font=dict(size=10),
            align="left",
        )

    fig.update_layout(
        **_ANALYSIS_FIGURE_LAYOUT,
        xaxis_title="Voltage (mV)",
        yaxis_title="G / G\u2098\u2090\u02e3",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
    )
    return fig
