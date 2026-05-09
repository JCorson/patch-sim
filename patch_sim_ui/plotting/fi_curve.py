"""F-I curve figure (firing rate vs injected current)."""

import plotly.graph_objects as go

from patch_sim_ui.plotting._layout import ANALYSIS_FIGURE_LAYOUT


def build_fi_figure(fi_data: dict) -> go.Figure:
    """Build a Plotly F-I curve figure from serialised F-I analysis results.

    Renders up to three traces: mean firing rate (green, solid), initial firing
    rate (orange, dashed), and steady-state firing rate (blue, dotted), plotted
    against injected current.  Steps with no spikes (``None`` values) are
    omitted from each trace, so traces may be shorter than the full sweep range.

    Args:
        fi_data: Dict with keys ``current_steps``, ``mean_firing_rates``,
            ``initial_firing_rates``, and ``steady_state_firing_rates``, each
            a list aligned by index.  Rate entries may be ``None`` for silent
            steps.

    Returns:
        A Plotly :class:`go.Figure` ready to be serialised and sent to the UI.
    """
    currents = fi_data["current_steps"]
    mean_rates = fi_data["mean_firing_rates"]
    initial_rates = fi_data["initial_firing_rates"]
    ss_rates = fi_data["steady_state_firing_rates"]

    def _filter(rates: list) -> tuple[list[float], list[float]]:
        """Return (x, y) pairs where rate is not None.

        Args:
            rates: List of rate values (float or None), parallel to
                ``currents``.

        Returns:
            Tuple of (filtered current steps, filtered rate values).
        """
        pairs = [(c, r) for c, r in zip(currents, rates) if r is not None]
        if not pairs:
            return [], []
        xs, ys = zip(*pairs)
        return list(xs), list(ys)

    fig = go.Figure()
    x_mean, y_mean = _filter(mean_rates)
    if x_mean:
        fig.add_trace(
            go.Scatter(
                x=x_mean,
                y=y_mean,
                mode="lines+markers",
                name="Mean",
                line=dict(color="#27ae60"),
                marker=dict(size=5),
            )
        )
    x_init, y_init = _filter(initial_rates)
    if x_init:
        fig.add_trace(
            go.Scatter(
                x=x_init,
                y=y_init,
                mode="lines+markers",
                name="Initial",
                line=dict(color="#e67e22", dash="dash"),
                marker=dict(size=5, symbol="triangle-up"),
            )
        )
    x_ss, y_ss = _filter(ss_rates)
    if x_ss:
        fig.add_trace(
            go.Scatter(
                x=x_ss,
                y=y_ss,
                mode="lines+markers",
                name="Steady-state",
                line=dict(color="#3498db", dash="dot"),
                marker=dict(size=5, symbol="diamond"),
            )
        )
    fig.update_layout(
        **ANALYSIS_FIGURE_LAYOUT,
        xaxis_title="Current (µA/cm²)",
        yaxis_title="Firing Rate (Hz)",
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
