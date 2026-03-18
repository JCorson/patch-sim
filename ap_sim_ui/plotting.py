"""Simulation result data model and Plotly figure construction.

Pure functions and data containers — no Reflex dependency.
Follows the same separation as protocol_builders.py.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel

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

    **Current Clamp** always uses a fixed 3-row layout (voltage, gating,
    stimulus).  **Voltage Clamp** always creates rows for all 4 classic
    currents plus any additional currents present in the sweep data, then
    gating and stimulus rows.  The row count only changes when a new
    simulation is run with a different channel configuration.

    In Voltage Clamp mode the y-axis labels identify each panel so the
    legend is hidden.

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

    if is_vc:
        # Fixed layout: all 4 classic currents + all additional from sweep
        # data + gating + stimulus.  Layout depends only on which channels
        # were enabled at simulation time, not on the show_* visibility flags.
        all_vc_currents: list[tuple[str, str]] = [
            ("total_current", "I_total (µA/cm²)"),
            ("sodium_current", "I_Na (µA/cm²)"),
            ("potassium_current", "I_K (µA/cm²)"),
            ("leak_current", "I_L (µA/cm²)"),
        ]
        for ch_name in add_current_keys:
            all_vc_currents.append((f"opt:{ch_name}", f"I_{ch_name} (µA/cm²)"))

        channel_row: dict[str, int] = {
            attr_key: i + 1 for i, (attr_key, _) in enumerate(all_vc_currents)
        }
        n_current_rows = len(all_vc_currents)
        gating_row = n_current_rows + 1
        stimulus_row = n_current_rows + 2
        rows = stimulus_row

        stim_fraction = 0.15
        gating_fraction = 0.15
        current_fraction = (1.0 - stim_fraction - gating_fraction) / n_current_rows
        row_heights = [current_fraction] * n_current_rows + [
            gating_fraction,
            stim_fraction,
        ]
    else:
        # Current Clamp: always 3 rows (voltage, gating, stimulus).
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

    def _scatter(x, y, name, row, color=None, visible=True):
        """Add a Scattergl trace to the figure."""
        line = {"color": color} if color else {}
        fig.add_trace(
            go.Scattergl(x=x, y=y, name=name, mode="lines", line=line, visible=visible),
            row=row,
            col=1,
        )

    for sweep in current_sweeps:
        c = sweep.color if sweep.color else None
        t = sweep.time
        sweep_mode = sweep.clamp_mode
        pfx = f"{sweep.label} " if sweep.label else ""
        stim_label = (
            "Stimulus (µA/cm²)" if sweep_mode == "Current Clamp" else "Command (mV)"
        )

        if sweep_mode == "Current Clamp":
            _scatter(t, sweep.voltage, f"{pfx}Voltage (mV)", 1, c, visible=show_voltage)
        else:
            # Voltage Clamp: one trace per channel row, always added with
            # the appropriate visible flag.
            for attr_key, _ in all_vc_currents:
                row = channel_row[attr_key]
                if attr_key == "total_current":
                    _scatter(
                        t,
                        sweep.total_current,
                        f"{pfx}Total I",
                        row,
                        c,
                        visible=show_total_current,
                    )
                elif attr_key == "sodium_current":
                    _scatter(
                        t,
                        sweep.sodium_current,
                        f"{pfx}I_Na",
                        row,
                        c,
                        visible=show_sodium_current,
                    )
                elif attr_key == "potassium_current":
                    _scatter(
                        t,
                        sweep.potassium_current,
                        f"{pfx}I_K",
                        row,
                        c,
                        visible=show_potassium_current,
                    )
                elif attr_key == "leak_current":
                    _scatter(
                        t,
                        sweep.leak_current,
                        f"{pfx}I_L",
                        row,
                        c,
                        visible=show_leak_current,
                    )
                elif attr_key.startswith("opt:"):
                    ch_name = attr_key[4:]
                    vals = sweep.additional_currents.get(ch_name, [])
                    vis = show_additional_currents.get(ch_name, True)
                    _scatter(t, vals, f"{pfx}I_{ch_name}", row, c, visible=vis)

        # Gating row is always present; traces are always added with their
        # visibility flags so the layout never changes on toggle.
        _scatter(
            t,
            sweep.potassium_activation,
            f"{pfx}n",
            gating_row,
            c,
            visible=show_potassium_activation,
        )
        _scatter(
            t,
            sweep.sodium_activation,
            f"{pfx}m",
            gating_row,
            c,
            visible=show_sodium_activation,
        )
        _scatter(
            t,
            sweep.sodium_inactivation,
            f"{pfx}h",
            gating_row,
            c,
            visible=show_sodium_inactivation,
        )
        for gv_name, gv_vals in sweep.additional_gating.items():
            vis = show_additional_gating.get(gv_name, True)
            _scatter(t, gv_vals, f"{pfx}{gv_name}", gating_row, c, visible=vis)

        _scatter(t, sweep.stimulus, stim_label, stimulus_row, c)

    for sweep in saved_sweeps:
        c = sweep.color
        if sweep.clamp_mode == "Current Clamp":
            _scatter(sweep.time, sweep.voltage, f"{sweep.label} V", 1, c)
        elif is_vc:
            # Show each current channel row for saved VC sweeps.
            for attr_key, _ in all_vc_currents:
                row = channel_row[attr_key]
                if attr_key == "total_current":
                    _scatter(
                        sweep.time,
                        sweep.total_current,
                        f"{sweep.label} I_total",
                        row,
                        c,
                    )
                elif attr_key == "sodium_current":
                    _scatter(
                        sweep.time,
                        sweep.sodium_current,
                        f"{sweep.label} I_Na",
                        row,
                        c,
                    )
                elif attr_key == "potassium_current":
                    _scatter(
                        sweep.time,
                        sweep.potassium_current,
                        f"{sweep.label} I_K",
                        row,
                        c,
                    )
                elif attr_key == "leak_current":
                    _scatter(
                        sweep.time,
                        sweep.leak_current,
                        f"{sweep.label} I_L",
                        row,
                        c,
                    )
                elif attr_key.startswith("opt:"):
                    ch_name = attr_key[4:]
                    vals = sweep.additional_currents.get(ch_name, [])
                    _scatter(sweep.time, vals, f"{sweep.label} I_{ch_name}", row, c)
        else:
            _scatter(sweep.time, sweep.total_current, f"{sweep.label} I_total", 1, c)
        _scatter(sweep.time, sweep.stimulus, sweep.label, stimulus_row, c)

    # Y-axis labels.
    if is_vc:
        for attr_key, y_label in all_vc_currents:
            fig.update_yaxes(title_text=y_label, row=channel_row[attr_key], col=1)
        fig.update_yaxes(title_text="Voltage (mV)", row=stimulus_row, col=1)
    else:
        fig.update_yaxes(title_text="Voltage (mV)", row=1, col=1)
        fig.update_yaxes(title_text="Current (µA/cm²)", row=stimulus_row, col=1)

    # Gating row is always present in both modes.
    fig.update_yaxes(title_text="Gating", row=gating_row, col=1, range=[0, 1])

    fig.update_xaxes(title_text="Time (ms)", row=stimulus_row, col=1)

    if is_vc:
        fig.update_layout(
            autosize=True,
            margin={"l": 60, "r": 20, "t": 30, "b": 40},
            template="plotly_white",
            hovermode="x unified",
            showlegend=False,
        )
    else:
        fig.update_layout(
            autosize=True,
            margin={"l": 60, "r": 20, "t": 30, "b": 40},
            legend={"orientation": "h", "y": 1.08},
            template="plotly_white",
            hovermode="x unified",
        )
    return fig
