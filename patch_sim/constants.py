"""Default neuron parameter values and domain constants for the Hodgkin-Huxley model."""

# Default neuron parameters (Hodgkin-Huxley squid giant axon values)
#
# DEFAULT_K_OUT = 4.0 mM is the physiological mammalian ACSF value
# (E_K ≈ −95 mV at 37 °C).  All mammalian presets rely on this default.
# SQUID_GIANT_AXON overrides it to 7.8 mM (HH52 seawater, E_K ≈ −77 mV).
#
# The passive leak is split into Na⁺ and K⁺ background conductances:
#   DEFAULT_G_NAL = 0.054 mS/cm²  (Na⁺ leak, NALCN-type)
#   DEFAULT_G_KL  = 0.246 mS/cm²  (K⁺ leak, TREK/TRAAK-type)
# These values are chosen so that g_NaL + g_KL = 0.3 mS/cm² (preserving τ_m)
# and I_NaL + I_KL at V = −65 mV equals the old chloride-leak current
# (g_L_old × (−65 − E_L_old)), ensuring v_rest remains at −65 mV.
# Mammalian presets each override g_NaL and g_KL explicitly and are retuned
# for their own K_out=4.0 value.
DEFAULT_G_NA: float = 120.0
DEFAULT_G_K: float = 36.0
DEFAULT_G_NAL: float = 0.054
DEFAULT_G_KL: float = 0.246
DEFAULT_C_M: float = 1.0
DEFAULT_V_REST: float = -65.0
DEFAULT_NA_OUT: float = 97.4
DEFAULT_NA_IN: float = 15.0
DEFAULT_K_OUT: float = 4.0
DEFAULT_K_IN: float = 140.0
DEFAULT_T: float = 310.15  # Kelvin (37°C)
DEFAULT_Q10: float = 3.0  # Dimensionless Q10 temperature coefficient
DEFAULT_T_REF: float = 295.15  # Kelvin (22°C) — HH52 experimental reference temperature
DEFAULT_CA_OUT: float = 2.0  # mM extracellular Ca2+ (physiological)
DEFAULT_CA_IN: float = 0.0001  # mM intracellular Ca2+ (physiological resting)

# Additional channel defaults
DEFAULT_G_IH: float = 0.1  # HCN/Ih maximum conductance in mS/cm²
# Ih Na+ permeability relative to K+ (GHK); yields ~-30 mV at default concentrations
DEFAULT_IH_P_NA: float = 0.289
DEFAULT_G_IKA: float = 5.0  # A-type K+ maximum conductance in mS/cm²
DEFAULT_G_IKV31: float = 40.0  # Kv3.1-type K+ maximum conductance in mS/cm²
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
    "g_NaL": DEFAULT_G_NAL,
    "g_KL": DEFAULT_G_KL,
    "C_m": DEFAULT_C_M,
    "v_rest": DEFAULT_V_REST,
    "Na_out": DEFAULT_NA_OUT,
    "Na_in": DEFAULT_NA_IN,
    "K_out": DEFAULT_K_OUT,
    "K_in": DEFAULT_K_IN,
    "Ca_out": DEFAULT_CA_OUT,
    "Ca_in": DEFAULT_CA_IN,
    "T": DEFAULT_T,
    "Q10": DEFAULT_Q10,
    "T_ref": DEFAULT_T_REF,
}

# Neuron preset names
SQUID_GIANT_AXON: str = "Squid Giant Axon (Classic HH)"
FAST_SPIKING_INTERNEURON: str = "Fast-Spiking Interneuron"
CORTICAL_PYRAMIDAL: str = "Cortical Pyramidal Neuron"
PURKINJE: str = "Purkinje Neuron"
DOPAMINERGIC: str = "SNc Dopaminergic Neuron"
THALAMIC_RELAY: str = "Thalamic Relay Neuron"
CA1_PYRAMIDAL: str = "Hippocampal CA1 Pyramidal Neuron"
STN: str = "Subthalamic Nucleus Neuron"
TRN: str = "Thalamic Reticular Nucleus Neuron"

# Protocol preset names
ACTION_POTENTIAL: str = "Action Potential"
SUBTHRESHOLD_RESPONSE: str = "Subthreshold Response"
REPETITIVE_FIRING: str = "Repetitive Firing"
FI_CURVE: str = "F-I Curve"
IV_CURVE: str = "I-V Curve"
NA_CHANNEL_ACTIVATION: str = "Na+ Channel Activation"
FREQUENCY_RESPONSE: str = "Frequency Response"
HYPERPOLARIZATION_STEPS: str = "Hyperpolarization Steps"
