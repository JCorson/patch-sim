"""Default neuron parameter values for the Hodgkin-Huxley model."""

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

# Optional channel defaults
DEFAULT_G_IH: float = 0.1  # HCN/Ih maximum conductance in mS/cm²
DEFAULT_E_IH: float = -30.0  # Ih reversal potential in mV (mixed Na/K cation)

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
    "T": DEFAULT_T,
}
