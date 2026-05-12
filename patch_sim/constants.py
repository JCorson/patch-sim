"""Default neuron parameter values and domain constants for the Hodgkin-Huxley model."""

# Default neuron parameters.
#
# DEFAULT_K_OUT = 4.0 mM is the physiological mammalian ACSF value
# (E_K ≈ −95 mV at 37 °C).  All mammalian presets rely on this default.
# SQUID_GIANT_AXON overrides it to 7.8 mM (HH52 seawater, E_K ≈ −77 mV).
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
DEFAULT_G_MSKV: float = 1.8  # Mainen-Sejnowski Kv max conductance, mS/cm² (#311)
DEFAULT_G_NAP: float = 0.5  # Persistent Na+ maximum conductance in mS/cm²
# SNc-specific INaP (Drion 2011 V½=-65 mV); see make_snc_inap_channel.
DEFAULT_G_NAP_SNC: float = 0.04
DEFAULT_G_NAR: float = 0.3  # Resurgent Na+ maximum conductance in mS/cm²
DEFAULT_G_IM: float = 0.5  # Muscarinic K+ maximum conductance in mS/cm²
DEFAULT_G_IKIR: float = 0.2  # Inward rectifier K+ maximum conductance in mS/cm²
DEFAULT_G_IKCA: float = 1.0  # Calcium-activated K+ maximum conductance in mS/cm²
DEFAULT_G_ICAL: float = 0.5  # L-type Ca2+ maximum conductance in mS/cm²
DEFAULT_G_CAV13: float = 0.1  # Cav1.3 LVA L-type Ca2+ maximum conductance in mS/cm²
DEFAULT_G_ICAT: float = 0.3  # T-type Ca2+ maximum conductance in mS/cm²
DEFAULT_G_ICAN: float = 0.3  # N-type Ca2+ maximum conductance in mS/cm²
DEFAULT_G_SK: float = 1.0  # Small-conductance Ca2+-activated K+ in mS/cm²
DEFAULT_G_KATP: float = 0.5  # ATP-sensitive K+ maximum conductance in mS/cm²

# Calcium dynamics defaults
DEFAULT_ALPHA_CA: float = 1e-4  # mM / (µA/cm² · ms)
DEFAULT_TAU_CA: float = 200.0  # ms
DEFAULT_CA_REST: float = 1e-4  # mM

#: Clamp mode identifier strings.
CURRENT_CLAMP: str = "Current Clamp"
VOLTAGE_CLAMP: str = "Voltage Clamp"

#: Protocol-type name for the chirp (linear frequency-sweep) current-clamp
#: protocol; a member of :data:`CURRENT_PROTOCOLS`.  Referenced by name in the
#: analysis layer (which runs impedance analysis only for this protocol) and
#: the UI, so it gets its own constant rather than a bare string literal.
CHIRP_PROTOCOL: str = "Chirp"

#: Current clamp protocol type names.
CURRENT_PROTOCOLS: list[str] = [
    "Step",
    "Ramp",
    "Pulse Train",
    "Sinusoidal",
    CHIRP_PROTOCOL,
    "Noise",
]

#: Voltage clamp protocol type names.
VOLTAGE_PROTOCOLS: list[str] = [
    "Step",
    "Ramp",
    "Pulse Train",
]

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
