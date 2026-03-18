"""Simulation result data model and Plotly figure construction.

Pure functions and data containers — no Reflex dependency.
Follows the same separation as protocol_builders.py.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel

from ap_sim_ui.constants import CHANNEL_COLORS, GATING_VAR_COLORS

# Classic column names that are always present in simulation DataFrames.
_CLASSIC_COLUMNS = frozenset(
    {
        "voltage",
        "total_current",
        "sodium_current",
        "potassium_current",
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
        "sodium_current",
        "potassium_current",
        "leak_current",
    }
)

# Maximum number of hover carrier points; keeps tooltip HTML build time short.
_MAX_HOVER_POINTS = 2000


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
        clamp_mode: "Current Clamp" or "Voltage Clamp".
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
            mode: Clamp mode — "Current Clamp" or "Voltage Clamp".

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
            sodium_current=_col("sodium_current"),
            potassium_current=_col("potassium_current"),
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
    show_total_current: bool,
    show_sodium_current: bool,
    show_potassium_current: bool,
    show_leak_current: bool,
    show_potassium_activation: bool,
    show_sodium_activation: bool,
    show_sodium_inactivation: bool,
    show_additional_currents: dict[str, bool],
    show_additional_gating: dict[str, bool],
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
        show_total_current: Whether the total current trace is visible.
        show_sodium_current: Whether I_Na is visible.
        show_potassium_current: Whether I_K is visible.
        show_leak_current: Whether I_L is visible.
        show_potassium_activation: Whether gating variable n is visible.
        show_sodium_activation: Whether gating variable m is visible.
        show_sodium_inactivation: Whether gating variable h is visible.
        show_additional_currents: Visibility flags for additional current channels.
        show_additional_gating: Visibility flags for additional gating variables.
        add_current_keys: Ordered list of additional current channel names.
        add_gating_keys: Ordered list of additional gating variable names.
        is_vc: True when in Voltage Clamp mode.
        stride: Downsampling stride — every Nth time point is included.

    Returns:
        A 3-tuple of ``(resp_html, gating_html, stim_html)``, each a list of
        HTML strings, one per downsampled time point.
    """
    time_vals = current_sweeps[0].time
    indices = list(range(0, len(time_vals), stride))
    n_pts = len(indices)
    col_w = 8
    label_w = max((len(s.label) for s in current_sweeps), default=6)
    mono_style = "font-family: monospace; font-size: 11px;"

    def _fmt_table(header: str, row_groups: list[list[str]]) -> list[str]:
        """Format a list of HTML table strings from per-time-point row groups.

        Args:
            header: The fixed header row string shared across all time points.
            row_groups: One list of row strings per time point.

        Returns:
            List of HTML strings, one per time point.
        """
        out = []
        for sweep_rows in row_groups:
            body = "<br>".join(sweep_rows)
            out.append(f'<span style="{mono_style}">{header}<br>{body}</span>')
        return out

    # --- Response subplot (row 1) ---
    # Col spec: (header_label, source, data_key)
    if is_vc:
        resp_cols: list[tuple[str, str, str]] = []
        if show_total_current:
            resp_cols.append(("I_total", "classic", "total_current"))
        if show_sodium_current:
            resp_cols.append(("I_Na", "classic", "sodium_current"))
        if show_potassium_current:
            resp_cols.append(("I_K", "classic", "potassium_current"))
        if show_leak_current:
            resp_cols.append(("I_L", "classic", "leak_current"))
        for ch_name in add_current_keys:
            if show_additional_currents.get(ch_name, True):
                resp_cols.append((f"I_{ch_name}", "additional", ch_name))
    else:
        resp_cols = [("V (mV)", "classic", "voltage")]

    if resp_cols:
        resp_header = " " * label_w + "".join(f"{c[0]:>{col_w}}" for c in resp_cols)
        resp_row_groups: list[list[str]] = []
        for idx in indices:
            sweep_rows: list[str] = []
            for sweep in current_sweeps:
                vals: list[str] = []
                for _, src, key in resp_cols:
                    if src == "classic":
                        col_data: list[float] = getattr(sweep, key)
                    else:
                        col_data = sweep.additional_currents.get(key, [])
                    v = col_data[idx] if idx < len(col_data) else float("nan")
                    vals.append(f"{v:>{col_w}.2f}")
                sweep_rows.append(f"{sweep.label:<{label_w}}" + "".join(vals))
            resp_row_groups.append(sweep_rows)
        resp_html = _fmt_table(resp_header, resp_row_groups)
    else:
        resp_html = [""] * n_pts

    # --- Gating subplot (row 2) ---
    gating_cols: list[tuple[str, str, str]] = []
    if show_potassium_activation:
        gating_cols.append(("n", "classic", "potassium_activation"))
    if show_sodium_activation:
        gating_cols.append(("m", "classic", "sodium_activation"))
    if show_sodium_inactivation:
        gating_cols.append(("h", "classic", "sodium_inactivation"))
    for gv_name in add_gating_keys:
        if show_additional_gating.get(gv_name, True):
            gating_cols.append((gv_name, "additional", gv_name))

    if gating_cols:
        gating_header = " " * label_w + "".join(f"{c[0]:>{col_w}}" for c in gating_cols)
        gating_row_groups: list[list[str]] = []
        for idx in indices:
            sweep_rows = []
            for sweep in current_sweeps:
                vals = []
                for _, src, key in gating_cols:
                    if src == "classic":
                        col_data = getattr(sweep, key)
                    else:
                        col_data = sweep.additional_gating.get(key, [])
                    v = col_data[idx] if idx < len(col_data) else float("nan")
                    vals.append(f"{v:>{col_w}.3f}")
                sweep_rows.append(f"{sweep.label:<{label_w}}" + "".join(vals))
            gating_row_groups.append(sweep_rows)
        gating_html = _fmt_table(gating_header, gating_row_groups)
    else:
        gating_html = [""] * n_pts

    # --- Stimulus subplot (row 3) ---
    stim_col_label = "Cmd (mV)" if is_vc else "Stim"
    stim_header = " " * label_w + f"{stim_col_label:>{col_w}}"
    stim_row_groups: list[list[str]] = []
    for idx in indices:
        sweep_rows = []
        for sweep in current_sweeps:
            stim_data = sweep.stimulus
            v = stim_data[idx] if idx < len(stim_data) else float("nan")
            sweep_rows.append(f"{sweep.label:<{label_w}}{v:>{col_w}.2f}")
        stim_row_groups.append(sweep_rows)
    stim_html = _fmt_table(stim_header, stim_row_groups)

    return resp_html, gating_html, stim_html


def build_figure(
    current_sweeps: list[Sweep],
    saved_sweeps: list[Sweep],
    show_voltage: bool,
    show_total_current: bool,
    show_sodium_current: bool,
    show_potassium_current: bool,
    show_leak_current: bool,
    show_potassium_activation: bool,
    show_sodium_activation: bool,
    show_sodium_inactivation: bool,
    clamp_mode: str,
    show_additional_currents: dict[str, bool] | None = None,
    show_additional_gating: dict[str, bool] | None = None,
) -> go.Figure:
    """Build a Plotly figure from current and saved sweeps.

    Current sweeps are rendered with visibility-flag filtering applied via
    Plotly's ``visible`` property so that the subplot layout remains fixed
    regardless of which traces are toggled.  This allows ``react-plotly.js``
    to use the efficient ``Plotly.react()`` path instead of a full rebuild.

    Current sweeps are rendered with visibility-flag filtering applied.
    Saved sweeps are rendered as overlays showing the primary trace and
    stimulus only, always visible.

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
        show_voltage: Whether to render the voltage trace.
        show_total_current: Whether to render total current.
        show_sodium_current: Whether to render I_Na.
        show_potassium_current: Whether to render I_K.
        show_leak_current: Whether to render I_L.
        show_potassium_activation: Whether to render gating variable n.
        show_sodium_activation: Whether to render gating variable m.
        show_sodium_inactivation: Whether to render gating variable h.
        clamp_mode: Active UI clamp mode, used for layout and axis labels.
        show_additional_currents: Mapping from additional channel name to
            visibility flag.  ``None`` means show all.
        show_additional_gating: Mapping from additional gating variable name to
            visibility flag.  ``None`` means show all.

    Returns:
        A Plotly Figure with response, gating, and stimulus subplots.
    """
    if show_additional_currents is None:
        show_additional_currents = {}
    if show_additional_gating is None:
        show_additional_gating = {}

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

    is_vc = clamp_mode == "Voltage Clamp"
    is_multi_sweep = len(current_sweeps) > 1

    # Both modes use a fixed 3-row layout.
    rows = 3
    row_heights = [0.5, 0.25, 0.25]
    gating_row = 2
    stimulus_row = 3

    vert_spacing = 0.08 if rows > 1 else 0.0
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
                x=x, y=y, name=name, mode="lines", line=line, visible=visible, **kwargs
            ),
            row=row,
            col=1,
        )

    # In multi-sweep mode all real traces suppress hover; carrier traces
    # deliver the table tooltips instead.
    hi = "skip" if is_multi_sweep else None

    for sweep in current_sweeps:
        c = sweep.color if sweep.color else None
        t = sweep.time
        sweep_mode = sweep.clamp_mode
        pfx = f"{sweep.label} " if sweep.label else ""
        stim_label = (
            "Stimulus (µA/cm²)" if sweep_mode == "Current Clamp" else "Command (mV)"
        )

        if sweep_mode == "Current Clamp":
            _scatter(
                t,
                sweep.voltage,
                f"{pfx}Voltage (mV)",
                1,
                c,
                visible=show_voltage,
                hoverinfo=hi,
            )
        else:
            # Voltage Clamp: all currents overlaid on row 1 with channel colours.
            _scatter(
                t,
                sweep.total_current,
                f"{pfx}I_total",
                1,
                CHANNEL_COLORS.get("total_current"),
                visible=show_total_current,
                width=4,
                hoverinfo=hi,
            )
            _scatter(
                t,
                sweep.sodium_current,
                f"{pfx}I_Na",
                1,
                CHANNEL_COLORS.get("sodium_current"),
                visible=show_sodium_current,
                hoverinfo=hi,
            )
            _scatter(
                t,
                sweep.potassium_current,
                f"{pfx}I_K",
                1,
                CHANNEL_COLORS.get("potassium_current"),
                visible=show_potassium_current,
                hoverinfo=hi,
            )
            _scatter(
                t,
                sweep.leak_current,
                f"{pfx}I_L",
                1,
                CHANNEL_COLORS.get("leak_current"),
                visible=show_leak_current,
                hoverinfo=hi,
            )
            for ch_name, vals in sweep.additional_currents.items():
                vis = show_additional_currents.get(ch_name, True)
                _scatter(
                    t,
                    vals,
                    f"{pfx}I_{ch_name}",
                    1,
                    CHANNEL_COLORS.get(ch_name),
                    visible=vis,
                    hoverinfo=hi,
                )

        # Gating row is always present; traces are always added with their
        # visibility flags so the layout never changes on toggle.
        _scatter(
            t,
            sweep.potassium_activation,
            f"{pfx}n",
            gating_row,
            GATING_VAR_COLORS.get("n"),
            visible=show_potassium_activation,
            hoverinfo=hi,
        )
        _scatter(
            t,
            sweep.sodium_activation,
            f"{pfx}m",
            gating_row,
            GATING_VAR_COLORS.get("m"),
            visible=show_sodium_activation,
            hoverinfo=hi,
        )
        _scatter(
            t,
            sweep.sodium_inactivation,
            f"{pfx}h",
            gating_row,
            GATING_VAR_COLORS.get("h"),
            visible=show_sodium_inactivation,
            hoverinfo=hi,
        )
        for gv_name, gv_vals in sweep.additional_gating.items():
            vis = show_additional_gating.get(gv_name, True)
            _scatter(
                t,
                gv_vals,
                f"{pfx}{gv_name}",
                gating_row,
                GATING_VAR_COLORS.get(gv_name),
                visible=vis,
                hoverinfo=hi,
            )

        _scatter(t, sweep.stimulus, stim_label, stimulus_row, c, hoverinfo=hi)

    for sweep in saved_sweeps:
        c = sweep.color
        if sweep.clamp_mode == "Current Clamp":
            _scatter(sweep.time, sweep.voltage, f"{sweep.label} V", 1, c, hoverinfo=hi)
        elif is_vc:
            # Saved VC sweeps: all currents overlaid on row 1, dashed lines.
            _scatter(
                sweep.time,
                sweep.total_current,
                f"{sweep.label} I_total",
                1,
                CHANNEL_COLORS.get("total_current"),
                dash="dash",
                width=4,
                hoverinfo=hi,
            )
            _scatter(
                sweep.time,
                sweep.sodium_current,
                f"{sweep.label} I_Na",
                1,
                CHANNEL_COLORS.get("sodium_current"),
                dash="dash",
                hoverinfo=hi,
            )
            _scatter(
                sweep.time,
                sweep.potassium_current,
                f"{sweep.label} I_K",
                1,
                CHANNEL_COLORS.get("potassium_current"),
                dash="dash",
                hoverinfo=hi,
            )
            _scatter(
                sweep.time,
                sweep.leak_current,
                f"{sweep.label} I_L",
                1,
                CHANNEL_COLORS.get("leak_current"),
                dash="dash",
                hoverinfo=hi,
            )
            for ch_name, vals in sweep.additional_currents.items():
                _scatter(
                    sweep.time,
                    vals,
                    f"{sweep.label} I_{ch_name}",
                    1,
                    CHANNEL_COLORS.get(ch_name),
                    dash="dash",
                    hoverinfo=hi,
                )
        else:
            _scatter(
                sweep.time,
                sweep.total_current,
                f"{sweep.label} I_total",
                1,
                c,
                hoverinfo=hi,
            )
        _scatter(sweep.time, sweep.stimulus, sweep.label, stimulus_row, c, hoverinfo=hi)

    # Add invisible hover carrier traces for multi-sweep (I-V Curve) mode.
    # Each carrier sits on one subplot and delivers a per-time-point HTML
    # table via customdata / hovertemplate.
    if is_multi_sweep and current_sweeps:
        first = current_sweeps[0]
        time_vals = first.time
        n_t = len(time_vals)
        stride = max(1, n_t // _MAX_HOVER_POINTS)
        indices = list(range(0, n_t, stride))
        carrier_x = [time_vals[i] for i in indices]

        resp_html, gating_html, stim_html = _build_hover_tables(
            current_sweeps=current_sweeps,
            show_total_current=show_total_current,
            show_sodium_current=show_sodium_current,
            show_potassium_current=show_potassium_current,
            show_leak_current=show_leak_current,
            show_potassium_activation=show_potassium_activation,
            show_sodium_activation=show_sodium_activation,
            show_sodium_inactivation=show_sodium_inactivation,
            show_additional_currents=show_additional_currents,
            show_additional_gating=show_additional_gating,
            add_current_keys=add_current_keys,
            add_gating_keys=add_gating_keys,
            is_vc=is_vc,
            stride=stride,
        )

        # Carrier y values mirror the first sweep so autorange is unaffected.
        if is_vc:
            carrier_y1 = [first.total_current[i] for i in indices]
        else:
            carrier_y1 = [first.voltage[i] for i in indices]
        carrier_y_gating = [first.potassium_activation[i] for i in indices]
        carrier_y_stim = [first.stimulus[i] for i in indices]

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

    hovermode = "x" if is_multi_sweep else "x unified"
    fig.update_layout(
        autosize=True,
        margin={"l": 60, "r": 20, "t": 30, "b": 40},
        template="plotly_white",
        hovermode=hovermode,
        showlegend=False,
    )
    return fig
