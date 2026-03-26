"""UI constants: parameter ranges and colour palettes."""

#: Clamp mode identifier strings used throughout state, plotting, and UI.
CURRENT_CLAMP: str = "Current Clamp"
VOLTAGE_CLAMP: str = "Voltage Clamp"

# Slider ranges for neuron parameters
PARAM_RANGES: dict[str, tuple[float, float, float]] = {
    # name: (min, max, step)
    "g_Na": (0.0, 300.0, 1.0),
    "g_K": (0.0, 100.0, 0.5),
    "g_L": (0.0, 2.0, 0.01),
    "C_m": (0.1, 5.0, 0.1),
    "v_rest": (-90.0, -40.0, 1.0),
    "Na_out": (1.0, 500.0, 1.0),
    "Na_in": (1.0, 100.0, 1.0),
    "K_out": (1.0, 50.0, 0.5),
    "K_in": (1.0, 300.0, 1.0),
    "Cl_out": (1.0, 300.0, 1.0),
    "Cl_in": (1.0, 100.0, 1.0),
    "Ca_out": (0.1, 20.0, 0.1),
    "Ca_in": (0.00001, 0.01, 0.00001),
    "T": (273.15, 323.15, 0.5),  # 0°C to 50°C
    # Additional channel conductances
    "ih_g_max": (0.0, 1.0, 0.01),
    "ika_g_max": (0.0, 100.0, 0.1),
    "inap_g_max": (0.0, 5.0, 0.01),
    "inar_g_max": (0.0, 5.0, 0.01),
    "im_g_max": (0.0, 5.0, 0.01),
    "ikir_g_max": (0.0, 2.0, 0.01),
    "ikca_g_max": (0.0, 10.0, 0.1),
    "ical_g_max": (0.0, 5.0, 0.01),
    "icat_g_max": (0.0, 5.0, 0.01),
    "ican_g_max": (0.0, 5.0, 0.01),
}

# Current clamp protocol types
CURRENT_PROTOCOLS: list[str] = [
    "Step",
    "Ramp",
    "Pulse Train",
    "Sinusoidal",
    "Chirp",
    "Noise",
]

# Voltage clamp protocol types
VOLTAGE_PROTOCOLS: list[str] = [
    "Step",
    "Ramp",
    "Pulse Train",
    "I-V Curve",
]

# Fixed colour for the Current Clamp voltage trace.  Matches the blue used for
# I_total in Voltage Clamp so the primary response trace looks consistent across
# both modes.
CC_VOLTAGE_COLOR: str = "#1f77b4"

# Colour for the stimulus/command trace in both clamp modes.  A neutral grey
# keeps the stimulus visually unobtrusive relative to the physiologically
# meaningful channel and gating-variable traces.
STIMULUS_COLOR: str | None = "#888888"

# Fixed colours per ion current channel (used in Voltage Clamp overlay plot).
CHANNEL_COLORS: dict[str, str] = {
    "total_current": "#1f77b4",  # blue
    "sodium_current": "#ff7f0e",  # orange
    "potassium_current": "#2ca02c",  # green
    "leak_current": "#7f7f7f",  # grey
    "Ih": "#d62728",  # red
    "IKa": "#9467bd",  # purple
    "INaP": "#e377c2",  # pink
    "INaR": "#bcbd22",  # olive
    "IM": "#17becf",  # cyan
    "IKir": "#8c564b",  # brown
    "IKCa": "#ff9896",  # light red
    "ICaL": "#aec7e8",  # light blue
    "ICaT": "#98df8a",  # light green
    "ICaN": "#c5b0d5",  # light purple
}

# Gating variable name → unique colour (each variable gets its own distinct colour).
GATING_VAR_COLORS: dict[str, str] = {
    # Classic HH gating variables
    "n": "#1f77b4",  # blue
    "m": "#ff7f0e",  # orange
    "h": "#2ca02c",  # green
    # Ih
    "s": "#d62728",  # red
    "hr": "#9467bd",  # purple
    # IKa
    "a": "#8c564b",  # brown
    "b": "#e377c2",  # pink
    # INaP
    "p": "#7f7f7f",  # grey
    # INaR
    "r": "#bcbd22",  # olive
    # IM
    "w": "#17becf",  # cyan
    # IKir
    "kir": "#aec7e8",  # light blue
    # IKCa
    "q": "#ffbb78",  # light orange
    # ICaL
    "d": "#98df8a",  # light green
    "f": "#ff9896",  # light red
    # ICaT
    "dt": "#c5b0d5",  # light purple
    "ft": "#c49c94",  # light brown
    # ICaN
    "dn": "#f7b6d2",  # light pink
    "fn": "#dbdb8d",  # light olive
}

# Faded amber/orange palette for oscilloscope-style stored traces.  The rgba
# opacity keeps them visually subdued so they serve as background references
# without competing with the live current_sweeps or saved_sweeps overlays.
STORED_TRACE_COLORS: list[str] = [
    "rgba(255, 140, 0, 0.45)",  # amber
    "rgba(230, 100, 0, 0.45)",  # dark orange
    "rgba(255, 165, 0, 0.45)",  # orange
    "rgba(200, 80, 0, 0.45)",  # burnt orange
    "rgba(255, 120, 30, 0.45)",  # tangerine
    "rgba(180, 60, 0, 0.45)",  # deep amber
    "rgba(255, 180, 50, 0.45)",  # golden
    "rgba(210, 110, 10, 0.45)",  # ochre
    "rgba(240, 130, 20, 0.45)",  # saffron
    "rgba(190, 90, 20, 0.45)",  # copper
]

# Neutral grey palette for sweep overlays so saved "reference" traces are
# visually distinct from the coloured channel / gating-variable traces.
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
