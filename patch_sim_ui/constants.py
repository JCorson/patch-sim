"""UI constants: parameter ranges and colour palettes."""

# Re-export domain constants from core so existing UI imports continue to work.
from patch_sim.constants import (  # noqa: F401
    CURRENT_CLAMP,
    CURRENT_PROTOCOLS,
    VOLTAGE_CLAMP,
    VOLTAGE_PROTOCOLS,
)
from patch_sim_ui.channels import ADDITIONAL_CHANNELS

# Slider ranges for neuron parameters
PARAM_RANGES: dict[str, tuple[float, float, float]] = {
    # name: (min, max, step)
    "g_Na": (0.0, 300.0, 1.0),
    "g_K": (0.0, 100.0, 0.5),
    "g_NaL": (0.0, 2.0, 0.01),
    "g_KL": (0.0, 2.0, 0.01),
    "C_m": (0.1, 5.0, 0.1),
    "v_rest": (-90.0, -40.0, 1.0),
    "Na_out": (1.0, 500.0, 1.0),
    "Na_in": (1.0, 100.0, 1.0),
    "K_out": (1.0, 50.0, 0.5),
    "K_in": (1.0, 300.0, 1.0),
    "Ca_out": (0.1, 20.0, 0.1),
    "Ca_in": (0.00001, 0.01, 0.00001),
    "T": (273.15, 323.15, 0.5),  # 0°C to 50°C
    # Cell membrane area in cm².  Range covers tiny FS interneurons
    # (~3e-6 cm²) up to Purkinje cells with full dendritic trees (~3e-4 cm²).
    "area_cm2": (1e-7, 5e-4, 1e-7),
    # Additional channel conductances — derived from channel registry.
    **{ch.g_max_field: ch.g_max_range for ch in ADDITIONAL_CHANNELS},
}

# Fixed colour for the Current Clamp voltage trace.  Matches the blue used for
# I_total in Voltage Clamp so the primary response trace looks consistent across
# both modes.
CC_VOLTAGE_COLOR: str = "#1f77b4"

# Colour for the stimulus/command trace in both clamp modes.  A neutral grey
# keeps the stimulus visually unobtrusive relative to the physiologically
# meaningful channel and gating-variable traces.
STIMULUS_COLOR: str | None = "#888888"

# Fixed colours per ion current channel (used in Voltage Clamp overlay plot).
# Additional channel colours are derived from the channel registry.
CHANNEL_COLORS: dict[str, str] = {
    # Classic HH channels
    "total_current": "#1f77b4",  # blue
    "sodium_current": "#ff7f0e",  # orange
    "potassium_current": "#2ca02c",  # green
    "na_leak_current": "#7f7f7f",  # grey
    "k_leak_current": "#bcbd22",  # olive
    # Additional channels
    **{ch.current_key: ch.current_color for ch in ADDITIONAL_CHANNELS},
}

# Gating variable name → unique colour (each variable gets its own distinct colour).
# Additional channel gating variable colours are derived from the channel registry.
GATING_VAR_COLORS: dict[str, str] = {
    # Classic HH gating variables
    "n": "#1f77b4",  # blue
    "m": "#ff7f0e",  # orange
    "h": "#2ca02c",  # green
    # Additional channel gating variables
    **{
        gv: color
        for ch in ADDITIONAL_CHANNELS
        for gv, color in ch.gating_var_colors.items()
    },
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

# Neutral grey palette for multi-sweep I-V protocol trace coloring so each
# sweep is visually distinct from the coloured channel / gating-variable traces.
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
HIGHLIGHT_DIM_OPACITY: float = 0.15
"""Opacity applied to non-selected sweeps after a click selection."""

HIGHLIGHT_HOVER_WIDTH: float = 4.0
"""Line width (px) applied to the hovered sweep during a hover preview."""

HIGHLIGHT_DIM_WIDTH: float = 0.5
"""Line width (px) applied to dimmed (non-selected) sweep traces."""

# Dark-mode axis colour overrides — single source of truth shared between the
# Python-side _LAYOUT_DARK dict (trace_display.py) and the server-generated JS
# snippet in _build_fetch_figure_js (state/simulation.py).  Both sites import
# this constant so a single edit propagates to both.
DARK_AXIS_STYLE: dict[str, str] = {
    "gridcolor": "rgba(255,255,255,0.1)",
    "linecolor": "rgba(255,255,255,0.25)",
    "zerolinecolor": "rgba(255,255,255,0.2)",
    "tickcolor": "rgba(255,255,255,0.4)",
}
