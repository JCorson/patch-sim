"""Steady-state inactivation (h∞) curve figure with optional Boltzmann fit."""

import plotly.graph_objects as go

from patch_sim_ui.plotting._layout import ANALYSIS_FIGURE_LAYOUT


def build_inactivation_figure(inactivation_data: dict) -> go.Figure:
    """Build a Plotly h∞ curve figure with optional Boltzmann fit overlay.

    Renders the normalized availability (h∞) data points as a scatter trace.
    When the Boltzmann fit converged, a smooth decreasing fit curve is overlaid
    as a dashed line and the half-inactivation voltage (V_half) and slope
    factor (k) are shown as an annotation.

    Args:
        inactivation_data: Dict with keys ``prepulse_voltages``,
            ``h_normalized``, ``boltzmann_converged``, ``v_half``, ``k``,
            ``fit_voltages``, and ``fit_h_normalized``.

    Returns:
        A Plotly :class:`go.Figure` ready to be serialised and sent to the UI.
    """
    prepulse_voltages = inactivation_data["prepulse_voltages"]
    h_norm = inactivation_data["h_normalized"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=prepulse_voltages,
            y=h_norm,
            mode="markers",
            name="h∞",
            marker=dict(color="#8e44ad", size=7),
        )
    )

    if inactivation_data.get("boltzmann_converged"):
        v_half = inactivation_data["v_half"]
        k = inactivation_data["k"]
        fig.add_trace(
            go.Scatter(
                x=inactivation_data["fit_voltages"],
                y=inactivation_data["fit_h_normalized"],
                mode="lines",
                name="Boltzmann fit",
                line=dict(color="#6c3483", dash="dash"),
            )
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.02,
            xanchor="left",
            yanchor="bottom",
            text=f"V₁/₂ = {v_half:.1f} mV<br>k = {k:.1f} mV",
            showarrow=False,
            font=dict(size=10),
            align="left",
        )

    fig.update_layout(
        **ANALYSIS_FIGURE_LAYOUT,
        xaxis_title="Prepulse voltage (mV)",
        yaxis_title="h∞ (availability)",
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
