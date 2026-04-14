"""Analysis results state for the patch_sim web UI."""

from typing import Any

import plotly.graph_objects as go
import reflex as rx

from patch_sim_ui.plotting import (
    build_fi_figure,
    build_gv_figure,
    build_iv_figure,
    build_sfa_figure,
)


class AnalysisState(rx.State):
    """State for AP, F-I, I-V, g-V, and SFA analysis results."""

    ap_metrics: list[dict[str, Any]] = []  # Per-spike metrics (serialized)
    ap_summary: dict[str, Any] = {}  # Aggregate summary statistics
    ap_is_multi_sweep: bool = False  # True when AP data is pooled from multiple sweeps
    fi_data: dict[str, Any] = {}  # Serialized FIAnalysisResult for the UI
    iv_data: dict[str, Any] = {}  # Serialized IVAnalysisResult for the UI
    gv_data: dict[str, Any] = {}  # Serialized GVAnalysisResult for the UI
    sfa_data: dict[str, Any] = {}  # Serialized SFAAnalysisResult for the UI

    # Membrane test results — persisted across protocol/simulation changes.
    # Only invalidated when neuron parameters change (neuron_fingerprint mismatch).
    mt_input_resistance: str = ""  # R_in formatted as string (kΩ·cm²)
    mt_time_constant: str = ""  # τ_m formatted as string (ms)
    mt_membrane_capacitance: str = ""  # C_m formatted as string (µF/cm²)
    mt_neuron_fingerprint: str = ""  # fingerprint used to detect stale cache
    mt_fit_converged: bool = True  # False when exponential fit did not converge

    @rx.var
    def has_membrane_test(self) -> bool:
        """Return True when the membrane test has been run at least once.

        Uses ``mt_neuron_fingerprint`` as the sentinel: it is set unconditionally
        at the end of ``run_membrane_test`` (even when the fit fails and values
        are em-dashes), making it a more robust indicator than ``mt_input_resistance``.
        """
        return self.mt_neuron_fingerprint != ""

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
    def has_gv_data(self) -> bool:
        """Return True when g-V analysis results are available for display."""
        return len(self.gv_data) > 0

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

    @rx.var
    def gv_figure(self) -> go.Figure:
        """Return a Plotly g-V curve figure with Boltzmann fit overlay.

        Returns an empty figure when no g-V data is available.
        """
        if not self.gv_data:
            return go.Figure()
        return build_gv_figure(self.gv_data)

    @rx.var
    def has_sfa_data(self) -> bool:
        """Return True when SFA analysis results are available for display."""
        return len(self.sfa_data) > 0

    @rx.var
    def sfa_figure(self) -> go.Figure:
        """Return a Plotly SFA (instantaneous frequency) figure.

        Returns an empty figure when no SFA data is available.
        """
        if not self.sfa_data:
            return go.Figure()
        return build_sfa_figure(self.sfa_data)
