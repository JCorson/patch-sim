"""V vs dV/dt phase-plane figure."""

import numpy as np
import plotly.graph_objects as go

from patch_sim_ui.constants import CC_VOLTAGE_COLOR
from patch_sim_ui.plotting._layout import _ANALYSIS_FIGURE_LAYOUT


def build_phase_plane_figure(phase_plane_data: dict) -> go.Figure:
    """Build a V vs dV/dt phase-plane figure from serialised sweep data.

    Renders one trace per sweep.  In single-sweep runs the trace uses the
    standard current-clamp colour; in multi-sweep runs each sweep is drawn
    with its stored colour so the trajectories are distinguishable.

    Args:
        phase_plane_data: Dict with key ``"sweeps"``, a list of dicts each
            containing ``"voltage"`` (list[float], mV), ``"dvdt"``
            (list[float], mV/ms), ``"label"`` (str), and ``"color"`` (str).

    Returns:
        A Plotly Figure with a single V vs dV/dt scatter subplot, or an
        empty figure when ``phase_plane_data`` is empty.
    """
    sweeps = phase_plane_data.get("sweeps", [])
    if not sweeps:
        return go.Figure()

    fig = go.Figure()
    for sweep in sweeps:
        voltage = sweep.get("voltage", [])
        dvdt = sweep.get("dvdt", [])
        if not voltage or not dvdt:
            continue
        color = sweep.get("color") or CC_VOLTAGE_COLOR
        label = sweep.get("label", "")
        hover = (
            f"<b>{label}</b><br>"
            "V: %{x:.1f} mV<br>"
            "dV/dt: %{y:.1f} mV/ms"
            "<extra></extra>"
            if label
            else "V: %{x:.1f} mV<br>dV/dt: %{y:.1f} mV/ms<extra></extra>"
        )
        fig.add_trace(
            go.Scattergl(
                x=np.asarray(voltage),
                y=np.asarray(dvdt),
                name=label,
                mode="lines",
                line={"color": color},
                showlegend=False,
                hovertemplate=hover,
            )
        )

    fig.update_layout(
        **_ANALYSIS_FIGURE_LAYOUT,
        hovermode="closest",
        showlegend=False,
        xaxis_title="V (mV)",
        yaxis_title="dV/dt (mV/ms)",
    )
    return fig
