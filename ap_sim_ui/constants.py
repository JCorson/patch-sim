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
