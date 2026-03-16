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

# Additional channel defaults
DEFAULT_G_IH: float = 0.1  # HCN/Ih maximum conductance in mS/cm²
DEFAULT_E_IH: float = -30.0  # Ih reversal potential in mV (mixed Na/K cation)

DEFAULT_G_IKA: float = 5.0  # A-type K+ maximum conductance in mS/cm²
DEFAULT_E_IKA: float = -77.0  # A-type K+ reversal potential in mV

DEFAULT_G_NAP: float = 0.5  # Persistent Na+ maximum conductance in mS/cm²
DEFAULT_E_NAP: float = 60.0  # Persistent Na+ reversal potential in mV

DEFAULT_G_NAR: float = 0.3  # Resurgent Na+ maximum conductance in mS/cm²
DEFAULT_E_NAR: float = 60.0  # Resurgent Na+ reversal potential in mV

DEFAULT_G_IM: float = 0.5  # Muscarinic K+ maximum conductance in mS/cm²
DEFAULT_E_IM: float = -77.0  # Muscarinic K+ reversal potential in mV

DEFAULT_G_IKIR: float = 0.2  # Inward rectifier K+ maximum conductance in mS/cm²
DEFAULT_E_IKIR: float = -77.0  # Inward rectifier K+ reversal potential in mV

DEFAULT_G_IKCA: float = 1.0  # Calcium-activated K+ maximum conductance in mS/cm²
DEFAULT_E_IKCA: float = -77.0  # Calcium-activated K+ reversal potential in mV

DEFAULT_G_ICAL: float = 0.5  # L-type Ca2+ maximum conductance in mS/cm²
DEFAULT_E_ICAL: float = 120.0  # L-type Ca2+ reversal potential in mV

DEFAULT_G_ICAT: float = 0.3  # T-type Ca2+ maximum conductance in mS/cm²
DEFAULT_E_ICAT: float = 120.0  # T-type Ca2+ reversal potential in mV

DEFAULT_G_ICAN: float = 0.3  # N-type Ca2+ maximum conductance in mS/cm²
DEFAULT_E_ICAN: float = 120.0  # N-type Ca2+ reversal potential in mV

# Calcium dynamics defaults
DEFAULT_ALPHA_CA: float = 1e-4  # mM / (µA/cm² · ms)
DEFAULT_TAU_CA: float = 200.0  # ms
DEFAULT_CA_REST: float = 1e-4  # mM

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
