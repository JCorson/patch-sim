"""Hyperpolarization figure (sag amplitude and rebound spike count)."""

import plotly.graph_objects as go

from patch_sim_ui.plotting._layout import ANALYSIS_FIGURE_LAYOUT


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
        **ANALYSIS_FIGURE_LAYOUT,
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
