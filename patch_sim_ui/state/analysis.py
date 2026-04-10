"""Analysis results state for the patch_sim web UI."""

from typing import Any

import plotly.graph_objects as go
import reflex as rx

from patch_sim_ui.plotting import build_iv_figure


class AnalysisState(rx.State):
    """State for AP and I-V analysis results."""

    ap_metrics: list[dict[str, Any]] = []  # Per-spike metrics (serialized)
    ap_summary: dict[str, Any] = {}  # Aggregate summary statistics
    iv_data: dict[str, Any] = {}  # Serialized IVAnalysisResult for the UI

    @rx.var
    def has_ap_metrics(self) -> bool:
        """Return True when AP analysis results are available for display."""
        return len(self.ap_metrics) > 0

    @rx.var
    def has_iv_data(self) -> bool:
        """Return True when I-V analysis results are available for display."""
        return len(self.iv_data) > 0

    @rx.var
    def iv_figure(self) -> go.Figure:
        """Return a Plotly I-V curve figure.

        Returns an empty figure when no I-V data is available.
        """
        if not self.iv_data:
            return go.Figure()
        return build_iv_figure(self.iv_data)
