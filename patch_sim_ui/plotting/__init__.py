"""Plotly figure construction for simulation results.

Pure functions — no Reflex dependency.  Each builder lives in its own
submodule; this module re-exports the public API for backward compatibility
with callers that import from ``patch_sim_ui.plotting`` directly.
"""

from patch_sim_ui.plotting.fi_curve import build_fi_figure
from patch_sim_ui.plotting.gv_curve import build_gv_figure
from patch_sim_ui.plotting.hyperpolarization import build_hyperpolarization_figure
from patch_sim_ui.plotting.iv_curve import build_iv_figure
from patch_sim_ui.plotting.phase_plane import build_phase_plane_figure
from patch_sim_ui.plotting.sfa import build_sfa_figure
from patch_sim_ui.plotting.tau_v import build_tau_v_figure
from patch_sim_ui.plotting.traces import (
    TraceVisibility,
    build_figure,
    compute_trace_visibility_map,
)

__all__ = [
    "TraceVisibility",
    "build_figure",
    "build_fi_figure",
    "build_gv_figure",
    "build_hyperpolarization_figure",
    "build_iv_figure",
    "build_phase_plane_figure",
    "build_sfa_figure",
    "build_tau_v_figure",
    "compute_trace_visibility_map",
]
