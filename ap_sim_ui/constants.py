"""UI constants: defaults, parameter ranges, and colour palettes."""

# Default sampling frequency (Hz) — 10 kHz keeps browser data size manageable
DEFAULT_SAMPLING_FREQUENCY: float = 10000.0

# Default neuron parameters (match HodgkinHuxley dataclass defaults)
DEFAULT_G_NA: float = 120.0
DEFAULT_G_K: float = 36.0
DEFAULT_G_L: float = 0.3
DEFAULT_C_M: float = 1.0
DEFAULT_V_REST: float = -65.0
DEFAULT_NA_OUT: float = 145.0
DEFAULT_NA_IN: float = 15.0
DEFAULT_K_OUT: float = 5.0
DEFAULT_K_IN: float = 140.0
DEFAULT_CL_OUT: float = 120.0
DEFAULT_CL_IN: float = 10.0
DEFAULT_T: float = 310.15  # Kelvin (37°C)

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

# Plotly subplot row labels
RESPONSE_ROW: int = 1
GATING_ROW: int = 2
STIMULUS_ROW: int = 3
