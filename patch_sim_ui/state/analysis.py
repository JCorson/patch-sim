"""Analysis results state for the patch_sim web UI."""

from typing import Any

import plotly.graph_objects as go
import reflex as rx

from patch_sim_ui.plotting import build_fi_figure, build_iv_figure


class AnalysisState(rx.State):
    """State for AP, F-I, and I-V analysis results."""

    ap_metrics: list[dict[str, Any]] = []  # Per-spike metrics (serialized)
    ap_summary: dict[str, Any] = {}  # Aggregate summary statistics
    ap_is_multi_sweep: bool = False  # True when AP data is pooled from multiple sweeps
    fi_data: dict[str, Any] = {}  # Serialized FIAnalysisResult for the UI
    iv_data: dict[str, Any] = {}  # Serialized IVAnalysisResult for the UI

    @rx.var
    def has_ap_metrics(self) -> bool:
        """Return True when AP analysis results are available for display."""
        return len(self.ap_metrics) > 0

    @rx.var
    def has_ap_or_fi(self) -> bool:
        """Return True when either AP metrics or F-I data are available."""
        return len(self.ap_metrics) > 0 or len(self.fi_data) > 0

    @rx.var
    def has_fi_data(self) -> bool:
        """Return True when F-I analysis results are available for display."""
        return len(self.fi_data) > 0

    @rx.var
    def has_iv_data(self) -> bool:
        """Return True when I-V analysis results are available for display."""
        return len(self.iv_data) > 0

    @rx.var
    def fi_figure(self) -> go.Figure:
        """Return a Plotly F-I curve figure.

        Returns an empty figure when no F-I data is available.
        """
        if not self.fi_data:
            return go.Figure()
        return build_fi_figure(self.fi_data)

    @rx.var
    def iv_figure(self) -> go.Figure:
        """Return a Plotly I-V curve figure.

        Returns an empty figure when no I-V data is available.
        """
        if not self.iv_data:
            return go.Figure()
        return build_iv_figure(self.iv_data)
