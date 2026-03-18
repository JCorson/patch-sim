"""UI constants: parameter ranges and colour palettes."""

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
    "Activation",
]

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

# Colour palette for sweep overlays (10 distinct colours)
SWEEP_COLORS: list[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
