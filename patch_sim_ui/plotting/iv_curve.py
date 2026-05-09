"""I-V curve figure (peak inward, peak outward, steady-state currents)."""

import plotly.graph_objects as go

from patch_sim_ui.plotting._layout import _ANALYSIS_FIGURE_LAYOUT


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
