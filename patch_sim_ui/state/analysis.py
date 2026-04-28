"""Analysis results state for the patch_sim web UI."""

from typing import Any

import plotly.graph_objects as go
import reflex as rx

from patch_sim_ui.plotting import (
    build_fi_figure,
    build_gv_figure,
    build_hyperpolarization_figure,
    build_iv_figure,
    build_phase_plane_figure,
    build_sfa_figure,
)


class AnalysisState(rx.State):
    """State for AP, F-I, I-V, g-V, SFA, and phase-plane analysis results."""

    ap_metrics: list[dict[str, Any]] = []  # Per-spike metrics (serialized)
    ap_summary: dict[str, Any] = {}  # Aggregate summary statistics
    ap_is_multi_sweep: bool = False  # True when AP data is pooled from multiple sweeps
    # Per-transient calcium-transient metrics (serialized).
    ca_transient_metrics: list[dict[str, Any]] = []
    # Aggregate calcium-transient summary statistics (em-dash for None).
    ca_transient_summary: dict[str, Any] = {}
    fi_data: dict[str, Any] = {}  # Serialized FIAnalysisResult for the UI
    iv_data: dict[str, Any] = {}  # Serialized IVAnalysisResult for the UI
    gv_data: dict[str, Any] = {}  # Serialized GVAnalysisResult for the UI
    sfa_data: dict[str, Any] = {}  # Serialized SFAAnalysisResult for the UI
    hyperpolarization_data: dict[str, Any] = {}  # Serialized sag/rebound analysis
    phase_plane_data: dict[str, Any] = {}  # Serialized V vs dV/dt sweep data

    # Membrane test results — persisted across protocol/simulation changes.
    # Only invalidated when neuron parameters change (neuron_fingerprint mismatch).
    mt_input_resistance: str = ""  # R_in formatted as string (mode-aware)
    mt_time_constant: str = ""  # τ_m formatted as string (ms)
    mt_membrane_capacitance: str = ""  # C_m formatted as string (mode-aware)
    mt_r_units: str = "kΩ·cm²"  # units suffix for mt_input_resistance
    mt_c_units: str = "µF/cm²"  # units suffix for mt_membrane_capacitance
    mt_units_mode: str = "density"  # "density" or "absolute"
    mt_neuron_fingerprint: str = ""  # fingerprint used to detect stale cache
    mt_fit_converged: bool = True  # False when exponential fit did not converge

    def clear_results(self) -> None:
        """Clear all analysis result fields to their empty defaults.

        Called by SimulationState before populating new analysis data so that
        stale results from a previous run do not persist into the new one.
        """
        self.ap_metrics = []
        self.ap_summary = {}
        self.ap_is_multi_sweep = False
        self.ca_transient_metrics = []
        self.ca_transient_summary = {}
        self.fi_data = {}
        self.iv_data = {}
        self.gv_data = {}
        self.sfa_data = {}
        self.hyperpolarization_data = {}
        self.phase_plane_data = {}

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
    def has_ca_transient_metrics(self) -> bool:
        """Return True when calcium-transient analysis results are available."""
        return len(self.ca_transient_metrics) > 0

    @rx.var
    def has_ap_or_fi(self) -> bool:
        """Return True when AP metrics, F-I, or hyperpolarization data are available."""
        return (
            len(self.ap_metrics) > 0
            or len(self.fi_data) > 0
            or len(self.hyperpolarization_data) > 0
        )

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
    def has_hyperpolarization_data(self) -> bool:
        """Return True when hyperpolarization (sag/rebound) results are available."""
        return len(self.hyperpolarization_data) > 0

    @rx.var
    def hyperpolarization_figure(self) -> go.Figure:
        """Return a Plotly sag/rebound figure.

        Returns an empty figure when no hyperpolarization data is available.
        """
        if not self.hyperpolarization_data:
            return go.Figure()
        return build_hyperpolarization_figure(self.hyperpolarization_data)

    @rx.var
    def sfa_figure(self) -> go.Figure:
        """Return a Plotly SFA (instantaneous frequency) figure.

        Returns an empty figure when no SFA data is available.
        """
        if not self.sfa_data:
            return go.Figure()
        return build_sfa_figure(self.sfa_data)

    @rx.var
    def has_phase_plane_data(self) -> bool:
        """Return True when phase-plane data is available for display."""
        return len(self.phase_plane_data) > 0

    @rx.var
    def phase_plane_figure(self) -> go.Figure:
        """Return a Plotly V vs dV/dt phase-plane figure.

        Returns an empty figure when no phase-plane data is available.
        """
        if not self.phase_plane_data:
            return go.Figure()
        return build_phase_plane_figure(self.phase_plane_data)
