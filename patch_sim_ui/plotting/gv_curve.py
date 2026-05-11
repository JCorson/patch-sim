"""g-V (conductance-voltage) curve figure with optional Boltzmann fit."""

import plotly.graph_objects as go

from patch_sim_ui.plotting._layout import ANALYSIS_FIGURE_LAYOUT


def build_gv_figure(gv_data: dict) -> go.Figure:
    """Build a Plotly g-V curve figure with optional Boltzmann fit overlay.

    Renders the normalized conductance (G/G_max) data points as a scatter
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
            name="G/Gₘₐˣ",
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
            text=f"V₁/₂ = {v_half:.1f} mV<br>k = {k:.1f} mV",
            showarrow=False,
            font=dict(size=10),
            align="left",
        )

    fig.update_layout(
        **ANALYSIS_FIGURE_LAYOUT,
        xaxis_title="Voltage (mV)",
        yaxis_title="G / Gₘₐˣ",
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
