"""Impedance-profile figure (|Z| magnitude and phase vs frequency)."""

import plotly.graph_objects as go

from patch_sim_ui.plotting._layout import ANALYSIS_FIGURE_LAYOUT


def build_impedance_figure(imp_data: dict) -> go.Figure:
    """Build a Plotly impedance-profile figure from serialised chirp results.

    Renders the impedance magnitude (left y-axis, teal) and phase (right
    y-axis, orange, dashed) plotted against frequency (Hz).  When a resonance
    frequency was detected, a dotted vertical marker is drawn at that
    frequency.

    Args:
        imp_data: Dict with keys ``frequencies``, ``magnitude``, ``phase``
            (lists aligned by index), ``units`` (magnitude unit string), and
            ``resonance_frequency`` (pre-formatted string, ``"—"`` when none).

    Returns:
        A Plotly :class:`go.Figure` ready to be serialised and sent to the UI.
    """
    freqs = imp_data["frequencies"]
    magnitude = imp_data["magnitude"]
    phase = imp_data["phase"]
    units = imp_data.get("units", "kΩ·cm²")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=magnitude,
            mode="lines+markers",
            name="|Z|",
            line=dict(color="#16a085"),
            marker=dict(size=4),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=phase,
            mode="lines",
            name="Phase",
            line=dict(color="#e67e22", dash="dash"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        **ANALYSIS_FIGURE_LAYOUT,
        xaxis_title="Frequency (Hz)",
        yaxis=dict(title=f"|Z| ({units})", color="#16a085"),
        yaxis2=dict(
            title="Phase (°)",
            overlaying="y",
            side="right",
            color="#e67e22",
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

    f_r = imp_data.get("resonance_frequency")
    if f_r is not None:
        try:
            fig.add_vline(x=float(f_r), line_dash="dot", line_color="gray")
        except (TypeError, ValueError):
            pass
    return fig
