"""UI constants: parameter ranges and color palettes."""

# Re-export domain constants from core so existing UI imports continue to work.
from patch_sim.constants import (  # noqa: F401
    CURRENT_CLAMP,
    CURRENT_PROTOCOLS,
    VOLTAGE_CLAMP,
    VOLTAGE_PROTOCOLS,
)
from patch_sim_ui.channels import CHANNELS

# Slider ranges for neuron parameters
PARAM_RANGES: dict[str, tuple[float, float, float]] = {
    # name: (min, max, step)
    "C_m": (0.1, 5.0, 0.1),
    "v_rest": (-90.0, -40.0, 1.0),
    "Na_out": (1.0, 500.0, 1.0),
    "Na_in": (1.0, 100.0, 1.0),
    "K_out": (1.0, 50.0, 0.5),
    "K_in": (1.0, 300.0, 1.0),
    "Ca_out": (0.1, 20.0, 0.1),
    "Ca_in": (0.00001, 0.01, 0.00001),
    "T": (273.15, 323.15, 0.5),  # 0°C to 50°C
    # Per-channel conductances — derived from the channel registry.
    **{ch.g_max_field: ch.g_max_range for ch in CHANNELS},
}

# Fixed color for the Current Clamp voltage trace.  Matches the blue used for
# I_total in Voltage Clamp so the primary response trace looks consistent across
# both modes.
CC_VOLTAGE_COLOR: str = "#1f77b4"

# Color for the stimulus/command trace in both clamp modes.  A neutral gray
# keeps the stimulus visually unobtrusive relative to the physiologically
# meaningful channel and gating-variable traces.
STIMULUS_COLOR: str | None = "#888888"

# Per-channel trace colors (used in the Voltage Clamp overlay plot).
# Keys are simulation column names (``"INa"``, ``"IK"``, ``"Ih"``, …) so a
# Sweep's ``channel_currents`` dict can be looked up directly.  The
# ``"total_current"`` key keys the summed-current trace.
CHANNEL_COLORS: dict[str, str] = {
    "total_current": "#1f77b4",  # blue
    **{ch.column_name: ch.current_color for ch in CHANNELS},
}

# Gating variable name → unique color (each variable gets its own distinct
# color).  HH-classic gates (n / m / h) are listed explicitly; every other
# gate's color comes from the channel registry.
GATING_VAR_COLORS: dict[str, str] = {
    "n": "#1f77b4",  # blue
    "m": "#ff7f0e",  # orange
    "h": "#2ca02c",  # green
    **{gv: color for ch in CHANNELS for gv, color in ch.gating_var_colors.items()},
}

# Distinct-hue palette for oscilloscope-style stored traces.  The rgba opacity
# keeps them visually subdued so they serve as background references without
# competing with the live current_sweeps overlays, while the varied hues make
# each stored sweep distinguishable.
STORED_TRACE_COLORS: list[str] = [
    "rgba(255, 140, 0, 0.45)",  # orange
    "rgba(0, 170, 160, 0.45)",  # teal
    "rgba(160, 80, 200, 0.45)",  # purple
    "rgba(220, 60, 100, 0.45)",  # rose
    "rgba(60, 180, 60, 0.45)",  # green
    "rgba(60, 120, 220, 0.45)",  # blue
    "rgba(210, 50, 50, 0.45)",  # red
    "rgba(200, 170, 30, 0.45)",  # gold
    "rgba(0, 190, 220, 0.45)",  # cyan
    "rgba(200, 60, 180, 0.45)",  # magenta
]

# Neutral gray palette for multi-sweep I-V protocol trace coloring so each
# sweep is visually distinct from the colored channel / gating-variable traces.
SWEEP_COLORS: list[str] = [
    "#888888",
    "#666666",
    "#aaaaaa",
    "#555555",
    "#999999",
    "#777777",
    "#bbbbbb",
    "#444444",
    "#9e9e9e",
    "#707070",
]

# Sweep highlight / dim styling for the interactive sweep-selection feature.
HIGHLIGHT_DIM_OPACITY: float = 0.35
"""Opacity applied to non-selected sweeps after a click selection.

Dimmed sweeps still need to read as context for the selected one.  Against the
dark-mode background the unselected traces have less contrast to spend than on
white, so this sits high enough to stay legible there while keeping a clear
separation from the selected sweep at full opacity.
"""

HIGHLIGHT_HOVER_WIDTH: float = 4.0
"""Line width (px) applied to the hovered sweep during a hover preview."""

HIGHLIGHT_DIM_WIDTH: float = 0.5
"""Line width (px) applied to dimmed (non-selected) sweep traces."""

# Dark-mode axis color overrides — single source of truth shared between the
# Python-side _LAYOUT_DARK dict (trace_display.py) and the server-generated JS
# snippet in _build_fetch_figure_js (state/simulation.py).  Both sites import
# this constant so a single edit propagates to both.
DARK_AXIS_STYLE: dict[str, str] = {
    "gridcolor": "rgba(255,255,255,0.1)",
    "linecolor": "rgba(255,255,255,0.25)",
    "zerolinecolor": "rgba(255,255,255,0.2)",
    "tickcolor": "rgba(255,255,255,0.4)",
}
