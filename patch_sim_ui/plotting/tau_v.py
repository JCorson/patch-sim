"""τ-V figure (activation/inactivation time constants vs voltage)."""

import plotly.graph_objects as go

from patch_sim_ui.plotting._layout import _ANALYSIS_FIGURE_LAYOUT


def build_tau_v_figure(tau_v_data: dict) -> go.Figure:
    """Build a Plotly τ-V figure with activation and inactivation traces.

    Renders up to three traces against command voltage:
        - τ_activation (green, lines + markers).
        - τ_inactivation fast (purple, lines + markers).
        - τ_inactivation slow (light purple, dashed; only when at least one
          step had a double-exponential inactivation fit accepted).

    The y-axis uses a logarithmic scale because τ commonly spans an order
    of magnitude across voltages.  ``connectgaps=False`` is set on each
    trace so that sweeps where the fit failed (``None`` τ) appear as gaps
    rather than being interpolated across, making missing data visible at
    a glance.

    Args:
        tau_v_data: Dict with keys ``voltages``, ``tau_activation``,
            ``tau_inactivation``, ``tau_inactivation_slow``, and
            ``has_double_exp``.  Each list runs parallel to ``voltages``.

    Returns:
        A Plotly :class:`go.Figure` ready to be serialised and sent to the UI.
    """
    voltages = tau_v_data["voltages"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=voltages,
            y=tau_v_data["tau_activation"],
            mode="lines+markers",
            name="τ activation",
            line=dict(color="#16a085"),
            marker=dict(size=6),
            connectgaps=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=voltages,
            y=tau_v_data["tau_inactivation"],
            mode="lines+markers",
            name="τ inactivation",
            line=dict(color="#8e44ad"),
            marker=dict(size=6),
            connectgaps=False,
        )
    )
    has_double = tau_v_data.get("has_double_exp", [])
    if any(has_double):
        fig.add_trace(
            go.Scatter(
                x=voltages,
                y=tau_v_data.get("tau_inactivation_slow", []),
                mode="lines+markers",
                name="τ inactivation (slow)",
                line=dict(color="#bb8fce", dash="dash"),
                marker=dict(size=6, symbol="diamond"),
                connectgaps=False,
            )
        )
    fig.update_layout(
        **_ANALYSIS_FIGURE_LAYOUT,
        xaxis_title="Voltage (mV)",
        yaxis_title="τ (ms)",
        yaxis_type="log",
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
