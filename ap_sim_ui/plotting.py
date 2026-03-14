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
        optional_currents: Extra channel currents keyed by channel name.
        optional_gating: Extra gating variable traces keyed by variable name.
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
    optional_currents: dict[str, list[float]] = {}
    optional_gating: dict[str, list[float]] = {}

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

        Columns not in the classic set are classified as optional: columns
        whose name ends with ``_current`` become ``optional_currents``; all
        other extra columns become ``optional_gating``.

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

        optional_currents: dict[str, list[float]] = {}
        optional_gating: dict[str, list[float]] = {}
        for col in columns:
            if col in _CLASSIC_COLUMNS:
                continue
            if col.endswith("_current"):
                # Strip trailing _current to get the channel name key
                ch_name = col[: -len("_current")]
                optional_currents[ch_name] = df[col].tolist()
            else:
                optional_gating[col] = df[col].tolist()

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
            optional_currents=optional_currents,
            optional_gating=optional_gating,
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
    show_optional_currents: dict[str, bool] | None = None,
    show_optional_gating: dict[str, bool] | None = None,
) -> go.Figure:
    """Build a Plotly figure from current and saved sweeps.

    Current sweeps are rendered with visibility-flag filtering applied.
    Saved sweeps are rendered as overlays showing the primary trace and
    stimulus only, always visible.

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
        clamp_mode: Active UI clamp mode, used for axis label selection.
        show_optional_currents: Mapping from optional channel name to
            visibility flag.  ``None`` means show all.
        show_optional_gating: Mapping from optional gating variable name to
            visibility flag.  ``None`` means show all.

    Returns:
        A Plotly Figure with response, optional gating, and stimulus subplots.
    """
    if show_optional_currents is None:
        show_optional_currents = {}
    if show_optional_gating is None:
        show_optional_gating = {}

    # Determine whether any optional gating variables should be shown.
    # We check current_sweeps to know which optional gating keys exist.
    opt_gating_keys: list[str] = []
    for sweep in current_sweeps:
        for key in sweep.optional_gating:
            if key not in opt_gating_keys:
                opt_gating_keys.append(key)

    show_any_opt_gating = any(
        show_optional_gating.get(k, True) for k in opt_gating_keys
    )

    show_gating = (
        show_potassium_activation
        or show_sodium_activation
        or show_sodium_inactivation
        or show_any_opt_gating
    )
    rows = 3 if show_gating else 2
    row_heights = [0.5, 0.25, 0.25] if show_gating else [0.6, 0.4]
    stimulus_row = 3 if show_gating else 2
    gating_row = 2 if show_gating else None

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.08,
    )

    def _scatter(x, y, name, row, color=None):
        """Add a Scattergl trace to the figure."""
        line = {"color": color} if color else {}
        fig.add_trace(
            go.Scattergl(x=x, y=y, name=name, mode="lines", line=line),
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
            if show_voltage:
                _scatter(t, sweep.voltage, f"{pfx}Voltage (mV)", 1, c)
            for ch_name, i_vals in sweep.optional_currents.items():
                if show_optional_currents.get(ch_name, True):
                    _scatter(t, i_vals, f"{pfx}I_{ch_name}", 1, c)
        else:
            if show_total_current:
                _scatter(t, sweep.total_current, f"{pfx}Total I", 1, c)
            if show_sodium_current:
                _scatter(t, sweep.sodium_current, f"{pfx}I_Na", 1, c)
            if show_potassium_current:
                _scatter(t, sweep.potassium_current, f"{pfx}I_K", 1, c)
            if show_leak_current:
                _scatter(t, sweep.leak_current, f"{pfx}I_L", 1, c)
            for ch_name, i_vals in sweep.optional_currents.items():
                if show_optional_currents.get(ch_name, True):
                    _scatter(t, i_vals, f"{pfx}I_{ch_name}", 1, c)

        if show_gating and gating_row is not None:
            if show_potassium_activation:
                _scatter(t, sweep.potassium_activation, f"{pfx}n", gating_row, c)
            if show_sodium_activation:
                _scatter(t, sweep.sodium_activation, f"{pfx}m", gating_row, c)
            if show_sodium_inactivation:
                _scatter(t, sweep.sodium_inactivation, f"{pfx}h", gating_row, c)
            for gv_name, gv_vals in sweep.optional_gating.items():
                if show_optional_gating.get(gv_name, True):
                    _scatter(t, gv_vals, f"{pfx}{gv_name}", gating_row, c)

        _scatter(t, sweep.stimulus, stim_label, stimulus_row, c)

    for sweep in saved_sweeps:
        c = sweep.color
        if sweep.clamp_mode == "Current Clamp":
            _scatter(sweep.time, sweep.voltage, f"{sweep.label} V", 1, c)
        else:
            _scatter(sweep.time, sweep.total_current, f"{sweep.label} I_total", 1, c)
        _scatter(sweep.time, sweep.stimulus, sweep.label, stimulus_row, c)

    if clamp_mode == "Current Clamp":
        fig.update_yaxes(title_text="Voltage (mV)", row=1, col=1)
        fig.update_yaxes(title_text="Current (µA/cm²)", row=stimulus_row, col=1)
    else:
        fig.update_yaxes(title_text="Current (µA/cm²)", row=1, col=1)
        fig.update_yaxes(title_text="Voltage (mV)", row=stimulus_row, col=1)

    if show_gating and gating_row is not None:
        fig.update_yaxes(title_text="Gating", row=gating_row, col=1, range=[0, 1])

    fig.update_xaxes(title_text="Time (ms)", row=stimulus_row, col=1)
    fig.update_layout(
        autosize=True,
        margin={"l": 60, "r": 20, "t": 30, "b": 40},
        legend={"orientation": "h", "y": 1.08},
        template="plotly_white",
        hovermode="x unified",
    )
    return fig
