"""Simulation result data model and Plotly figure construction.

Pure functions and data containers — no Reflex dependency.
Follows the same separation as protocol_builders.py.
"""

from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel

from patch_sim_ui.constants import (
    CC_VOLTAGE_COLOR,
    CHANNEL_COLORS,
    CURRENT_CLAMP,
    VOLTAGE_CLAMP,
    GATING_VAR_COLORS,
    STIMULUS_COLOR,
    STORED_TRACE_COLORS,
)

# Classic column names that are always present in simulation DataFrames.
_CLASSIC_COLUMNS = frozenset(
    {
        "voltage",
        "total_current",
        "Na_current",
        "K_current",
        "leak_current",
        "potassium_activation",
        "sodium_activation",
        "sodium_inactivation",
    }
)

# Classic current columns (end with _current and are in the classic set).
_CLASSIC_CURRENT_COLUMNS = frozenset(
    {
        "total_current",
        "Na_current",
        "K_current",
        "leak_current",
    }
)

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
        leak_current: Show I_L.
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
    leak_current: bool = True
    potassium_activation: bool = True
    sodium_activation: bool = True
    sodium_inactivation: bool = True
    additional_currents: dict[str, bool] = field(default_factory=dict)
    additional_gating: dict[str, bool] = field(default_factory=dict)


class Sweep(BaseModel):
    """A simulation result snapshot used for overlay display.

    Attributes:
        label: Display name shown in the plot legend.
        color: Hex colour string; empty string means Plotly chooses the colour.
        time: Time axis values in ms.
        voltage: Membrane voltage in mV.
        sodium_current: I_Na in µA/cm².
        potassium_current: I_K in µA/cm².
        leak_current: I_L in µA/cm².
        total_current: Sum of ion currents in µA/cm².
        potassium_activation: Gating variable n (dimensionless, 0–1).
        sodium_activation: Gating variable m (dimensionless, 0–1).
        sodium_inactivation: Gating variable h (dimensionless, 0–1).
        stimulus: Stimulus waveform (current µA/cm² or voltage command mV).
        clamp_mode: CURRENT_CLAMP or VOLTAGE_CLAMP.
        additional_currents: Extra channel currents keyed by channel name.
        additional_gating: Extra gating variable traces keyed by variable name.
    """

    label: str
    color: str
    time: list[float]
    voltage: list[float]
    sodium_current: list[float]
    potassium_current: list[float]
    leak_current: list[float]
    total_current: list[float]
    potassium_activation: list[float]
    sodium_activation: list[float]
    sodium_inactivation: list[float]
    stimulus: list[float]
    clamp_mode: str
    additional_currents: dict[str, list[float]] = {}
    additional_gating: dict[str, list[float]] = {}

    @classmethod
    def from_dataframe(
        cls,
        df,
        stimulus,
        label: str,
        color: str,
        mode: str,
    ) -> "Sweep":
        """Create a Sweep from a simulation result DataFrame.

        Columns not in the classic set are classified as additional: columns
        whose name ends with ``_current`` become ``additional_currents``; all
        other extra columns become ``additional_gating``.

        Args:
            df: Simulation result DataFrame with time as the index.
            stimulus: Stimulus array (current or voltage command).
            label: Display name for this sweep in the legend.
            color: Hex colour string; pass empty string to use Plotly default.
            mode: Clamp mode — CURRENT_CLAMP or VOLTAGE_CLAMP.

        Returns:
            A fully populated Sweep instance.
        """
        columns = df.columns.tolist()

        def _col(name: str) -> list[float]:
            """Return column as list, or empty list if column absent."""
            return df[name].tolist() if name in columns else []

        additional_currents: dict[str, list[float]] = {}
        additional_gating: dict[str, list[float]] = {}
        for col in columns:
            if col in _CLASSIC_COLUMNS:
                continue
            if col.endswith("_current"):
                # Strip trailing _current to get the channel name key
                ch_name = col[: -len("_current")]
                additional_currents[ch_name] = df[col].tolist()
            else:
                additional_gating[col] = df[col].tolist()

        return cls(
            label=label,
            color=color,
            clamp_mode=mode,
            time=df.index.tolist(),
            stimulus=stimulus.tolist(),
            voltage=_col("voltage"),
            sodium_current=_col("Na_current"),
            potassium_current=_col("K_current"),
            leak_current=_col("leak_current"),
            total_current=_col("total_current"),
            potassium_activation=_col("potassium_activation"),
            sodium_activation=_col("sodium_activation"),
            sodium_inactivation=_col("sodium_inactivation"),
            additional_currents=additional_currents,
            additional_gating=additional_gating,
        )


def _build_hover_tables(
    current_sweeps: list[Sweep],
    visibility: TraceVisibility,
    add_current_keys: list[str],
    add_gating_keys: list[str],
    is_vc: bool,
    stride: int,
) -> tuple[list[str], list[str], list[str]]:
    """Build per-time-point HTML table strings for I-V Curve hover tooltips.

    For each downsampled time point, produces a monospace HTML table showing
    all sweeps as rows and visible quantities as columns — one table string
    per subplot (response, gating, stimulus).

    Args:
        current_sweeps: The N sweeps from the I-V Curve run.
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
        if visibility.leak_current:
            resp_cols.append(("I_L", "classic", "leak_current"))
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
    saved_sweeps: list[Sweep],
    clamp_mode: str,
    additional_current_field_map: dict[str, str] | None = None,
    additional_gating_field_map: dict[str, str] | None = None,
    stored_traces: list[Sweep] | None = None,
) -> dict[str, list[int]]:
    """Return a mapping from show_* field names to Plotly trace indices.

    Mirrors the trace-insertion order used by ``build_figure`` so callers can
    toggle individual traces via ``Plotly.restyle`` without rebuilding the
    figure.  Only current-sweep traces controlled by a ``show_*`` field are
    mapped; saved-sweep traces, stored traces, and carrier traces advance the
    counter but are not included (they are always visible).

    Args:
        current_sweeps: Latest simulation result sweeps.
        saved_sweeps: User-saved sweeps displayed as overlays.
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
        (e.g. multi-sweep I-V Curve) the list contains one index per sweep.
    """
    result: dict[str, list[int]] = {}
    idx = 0
    add_curr = additional_current_field_map or {}
    add_gating = additional_gating_field_map or {}
    is_vc = clamp_mode == VOLTAGE_CLAMP
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
            _map("show_leak_current")
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

    for sweep in saved_sweeps:
        # Saved sweeps are always visible; just advance the counter.
        if sweep.clamp_mode == CURRENT_CLAMP:
            _skip()  # voltage
        elif is_vc:
            # _add_vc_currents: 4 classic + additional_currents (no gating).
            for _ in range(4):
                _skip()
            for _ in sweep.additional_currents:
                _skip()
        else:
            _skip()  # total_current only (VC sweep shown in CC context)
        _skip()  # stimulus

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
    saved_sweeps: list[Sweep],
    visibility: TraceVisibility,
    clamp_mode: str,
    stored_traces: list[Sweep] | None = None,
    show_hover: bool = True,
) -> go.Figure:
    """Build a Plotly figure from current and saved sweeps.

    Current sweeps are rendered with visibility-flag filtering applied via
    Plotly's ``visible`` property so that the subplot layout remains fixed
    regardless of which traces are toggled.  This allows ``react-plotly.js``
    to use the efficient ``Plotly.react()`` path instead of a full rebuild.

    Current sweeps are rendered with visibility-flag filtering applied.
    Saved sweeps are rendered as overlays showing the primary trace and
    stimulus only, always visible.
    Stored traces are oscilloscope-style background reference snapshots,
    rendered with dashed faded lines; only voltage (CC) or total_current (VC)
    and stimulus are plotted.

    Both **Current Clamp** and **Voltage Clamp** use a fixed 3-row layout.
    In Voltage Clamp mode all ion current channels are overlaid on a single
    subplot (row 1) with each channel identified by a fixed colour from
    ``CHANNEL_COLORS`` and a legend entry.  Saved VC sweeps are drawn with
    dashed lines in the same channel colours.

    When there are multiple current sweeps (I-V Curve mode), hover on all
    real traces is suppressed and replaced with invisible carrier traces that
    display a monospace HTML table — sweeps as rows, quantities as columns —
    via ``hovertemplate``/``customdata``.  Single-sweep modes keep the
    standard ``hovermode="x unified"`` behaviour.

    Args:
        current_sweeps: Latest simulation result (1 sweep for standard runs,
            N sweeps for I-V Curve).
        saved_sweeps: User-saved sweeps for comparison overlay.
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
        """
        line: dict = {"color": color} if color is not None else {}
        if dash is not None:
            line["dash"] = dash
        if width is not None:
            line["width"] = width
        kwargs: dict = {}
        if hoverinfo is not None:
            kwargs["hoverinfo"] = hoverinfo
        fig.add_trace(
            go.Scattergl(
                x=np.asarray(x),
                y=np.asarray(y),
                name=name,
                mode="lines",
                line=line,
                visible=visible,
                showlegend=showlegend,
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
        """
        classic_defs: list[tuple[str, str, int | None]] = [
            ("total_current", "I_total", _TOTAL_CURRENT_LINE_WIDTH),
            ("sodium_current", "I_Na", None),
            ("potassium_current", "I_K", None),
            ("leak_current", "I_L", None),
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
            )

    # In multi-sweep mode all real traces suppress hover; carrier traces
    # deliver the table tooltips instead.
    hi = "skip" if is_multi_sweep else None

    for sweep_idx, sweep in enumerate(current_sweeps):
        c = sweep.color if sweep.color else None
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
            )
        else:
            # Voltage Clamp: all currents overlaid on row 1 with channel colours.
            _add_vc_currents(
                sweep, pfx, hoverinfo=hi, visibility=visibility, showlegend=sl
            )

        # Gating row is always present; traces are always added with their
        # visibility flags so the layout never changes on toggle.
        for gv_attr, gv_label in [
            ("potassium_activation", "n"),
            ("sodium_activation", "m"),
            ("sodium_inactivation", "h"),
        ]:
            _scatter(
                t,
                getattr(sweep, gv_attr),
                f"{pfx}{gv_label}",
                gating_row,
                GATING_VAR_COLORS.get(gv_label),
                visible=getattr(visibility, gv_attr),
                hoverinfo=hi,
                showlegend=sl,
            )
        for gv_name, gv_vals in sweep.additional_gating.items():
            vis = visibility.additional_gating.get(gv_name, True)
            _scatter(
                t,
                gv_vals,
                f"{pfx}{gv_name}",
                gating_row,
                GATING_VAR_COLORS.get(gv_name),
                visible=vis,
                hoverinfo=hi,
                showlegend=sl,
            )

        _scatter(
            t,
            sweep.stimulus,
            stim_label,
            stimulus_row,
            STIMULUS_COLOR,
            hoverinfo=hi,
            showlegend=False,
        )

    for sweep in saved_sweeps:
        c = sweep.color
        if sweep.clamp_mode == CURRENT_CLAMP:
            _scatter(
                sweep.time,
                sweep.voltage,
                f"{sweep.label} V",
                1,
                c,
                hoverinfo=hi,
            )
        elif is_vc:
            # Saved VC sweeps: all currents overlaid on row 1, dashed lines.
            _add_vc_currents(sweep, f"{sweep.label} ", dash="dash", hoverinfo=hi)
        else:
            _scatter(
                sweep.time,
                sweep.total_current,
                f"{sweep.label} I_total",
                1,
                c,
                hoverinfo=hi,
            )
        _scatter(
            sweep.time,
            sweep.stimulus,
            sweep.label,
            stimulus_row,
            c,
            hoverinfo=hi,
            showlegend=False,
        )

    for i, sweep in enumerate(stored_traces or []):
        c = STORED_TRACE_COLORS[i % len(STORED_TRACE_COLORS)]
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

    # Add invisible hover carrier traces for multi-sweep (I-V Curve) mode.
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
                ),
                row=sub_row,
                col=1,
            )

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

    hovermode: str | bool = (
        ("x" if is_multi_sweep else "x unified") if show_hover else False
    )
    fig.update_layout(
        autosize=True,
        margin=_PLOT_MARGIN,
        template="plotly_white",
        hovermode=hovermode,
        showlegend=True,
        legend=dict(
            orientation="v",
            x=1.0,
            xanchor="left",
            y=1.0,
            yanchor="top",
        ),
    )
    return fig
