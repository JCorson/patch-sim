"""Spike-frequency adaptation (SFA) figure."""

import plotly.graph_objects as go

from patch_sim_ui.constants import SWEEP_COLORS
from patch_sim_ui.plotting._layout import _ANALYSIS_FIGURE_LAYOUT


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
