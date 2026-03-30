"""Default neuron parameter values and domain constants for the Hodgkin-Huxley model."""

# Default neuron parameters (classic Hodgkin-Huxley values)
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
DEFAULT_CA_OUT: float = 2.0  # mM extracellular Ca2+ (physiological)
DEFAULT_CA_IN: float = 0.0001  # mM intracellular Ca2+ (physiological resting)

# Additional channel defaults
DEFAULT_G_IH: float = 0.1  # HCN/Ih maximum conductance in mS/cm²
# Ih Na+ permeability relative to K+ (GHK); yields ~-30 mV at default concentrations
DEFAULT_IH_P_NA: float = 0.289

DEFAULT_G_IKA: float = 5.0  # A-type K+ maximum conductance in mS/cm²

DEFAULT_G_NAP: float = 0.5  # Persistent Na+ maximum conductance in mS/cm²

DEFAULT_G_NAR: float = 0.3  # Resurgent Na+ maximum conductance in mS/cm²

DEFAULT_G_IM: float = 0.5  # Muscarinic K+ maximum conductance in mS/cm²

DEFAULT_G_IKIR: float = 0.2  # Inward rectifier K+ maximum conductance in mS/cm²

DEFAULT_G_IKCA: float = 1.0  # Calcium-activated K+ maximum conductance in mS/cm²

DEFAULT_G_ICAL: float = 0.5  # L-type Ca2+ maximum conductance in mS/cm²
DEFAULT_G_ICAT: float = 0.3  # T-type Ca2+ maximum conductance in mS/cm²
DEFAULT_G_ICAN: float = 0.3  # N-type Ca2+ maximum conductance in mS/cm²

# Calcium dynamics defaults
DEFAULT_ALPHA_CA: float = 1e-4  # mM / (µA/cm² · ms)
DEFAULT_TAU_CA: float = 200.0  # ms
DEFAULT_CA_REST: float = 1e-4  # mM

#: Clamp mode identifier strings.
CURRENT_CLAMP: str = "Current Clamp"
VOLTAGE_CLAMP: str = "Voltage Clamp"

#: Current clamp protocol type names.
CURRENT_PROTOCOLS: list[str] = [
    "Step",
    "Ramp",
    "Pulse Train",
    "Sinusoidal",
    "Chirp",
    "Noise",
]

#: Voltage clamp protocol type names.
VOLTAGE_PROTOCOLS: list[str] = [
    "Step",
    "Ramp",
    "Pulse Train",
]

DEFAULT_NEURON_PARAMS: dict[str, float] = {
    "g_Na": DEFAULT_G_NA,
    "g_K": DEFAULT_G_K,
    "g_L": DEFAULT_G_L,
    "C_m": DEFAULT_C_M,
    "v_rest": DEFAULT_V_REST,
    "Na_out": DEFAULT_NA_OUT,
    "Na_in": DEFAULT_NA_IN,
    "K_out": DEFAULT_K_OUT,
    "K_in": DEFAULT_K_IN,
    "Cl_out": DEFAULT_CL_OUT,
    "Cl_in": DEFAULT_CL_IN,
    "Ca_out": DEFAULT_CA_OUT,
    "Ca_in": DEFAULT_CA_IN,
    "T": DEFAULT_T,
}
