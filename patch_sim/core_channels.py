"""Core Hodgkin-Huxley channel factory functions.

This module provides the six classic HH rate functions as module-level
callables and four factory functions that bundle them into IonChannel objects.

Rate functions all follow the ``(V: float, ca_i: float) -> float`` signature so
they can be used directly as :class:`~patch_sim.channels.GatingVariable` rate
functions.  The ``ca_i`` argument is accepted but ignored; it exists only for
interface compatibility with calcium-sensitive gating variables.

The passive leak is split into two non-specific conductances:
- :func:`make_na_leak_channel`: Na⁺ leak, reversal via Nernst equation for Na⁺.
- :func:`make_k_leak_channel`: K⁺ leak, reversal via Nernst equation for K⁺.

This mirrors the biophysical reality of background channels (TREK/TRAAK K⁺
channels + persistent Na⁺ leak) and removes the unphysiological dependence on
intracellular [Cl⁻] that arose when using a single chloride-Nernst leak.
"""

from .channels import GatingVariable, IonChannel, IonSpecies, NernstSpec
from .utils import safe_exp

# Threshold for detecting near-singularity in GHK-style rate equations.
# When the denominator voltage term is within this tolerance of zero, the
# L'Hôpital limit is used instead to avoid division by zero.
SINGULARITY_THRESHOLD = 1e-6


def alpha_n(V: float, ca_i: float) -> float:
    """Forward rate for potassium channel activation gate n.

    Has a removable singularity at V = −55 mV; the L'Hôpital limit (0.1) is
    returned when ``|V + 55| < SINGULARITY_THRESHOLD``.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    if abs(V + 55) < SINGULARITY_THRESHOLD:
        return 0.1
    denominator = 1 - safe_exp(-(V + 55) / 10)
    return 0.01 * (V + 55) / denominator


def beta_n(V: float, ca_i: float) -> float:
    """Backward rate for potassium channel activation gate n.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return 0.125 * safe_exp(-(V + 65) / 80)


def alpha_m(V: float, ca_i: float) -> float:
    """Forward rate for sodium channel activation gate m.

    Has a removable singularity at V = −40 mV; the L'Hôpital limit (1.0) is
    returned when ``|V + 40| < SINGULARITY_THRESHOLD``.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    if abs(V + 40) < SINGULARITY_THRESHOLD:
        return 1.0
    denominator = 1 - safe_exp(-(V + 40) / 10)
    return 0.1 * (V + 40) / denominator


def beta_m(V: float, ca_i: float) -> float:
    """Backward rate for sodium channel activation gate m.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return 4.0 * safe_exp(-(V + 65) / 18)


def alpha_h(V: float, ca_i: float) -> float:
    """Forward rate for sodium inactivation gate h.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    return 0.07 * safe_exp(-(V + 65) / 20)


def beta_h(V: float, ca_i: float) -> float:
    """Backward rate for sodium inactivation gate h.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return 1 / (1 + safe_exp(-(V + 35) / 10))


def make_na_channel(g_max: float) -> IonChannel:
    """Create the fast sodium channel (Na⁺).

    Uses the classic Hodgkin-Huxley kinetics: activation gate *m* (power 3)
    and inactivation gate *h* (power 1).  The reversal potential is computed
    dynamically via the Nernst equation for Na⁺.

    Gating variable names are ``"m"`` and ``"h"``, matching the simulation
    result fields used throughout the simulator.

    Args:
        g_max: Maximum conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the fast Na⁺
        channel.
    """
    return IonChannel(
        name="Na",
        g_max=g_max,
        gating_variables=(
            GatingVariable(name="m", power=3, alpha=alpha_m, beta=beta_m),
            GatingVariable(name="h", power=1, alpha=alpha_h, beta=beta_h),
        ),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )


def make_k_channel(g_max: float) -> IonChannel:
    """Create the delayed rectifier potassium channel (K⁺).

    Uses the classic HH activation gate *n* (power 4).  The reversal potential
    is computed dynamically via the Nernst equation for K⁺.

    The gating variable name is ``"n"``, matching the simulation result field
    used throughout the simulator.

    Args:
        g_max: Maximum conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the delayed
        rectifier K⁺ channel.
    """
    return IonChannel(
        name="K",
        g_max=g_max,
        gating_variables=(
            GatingVariable(name="n", power=4, alpha=alpha_n, beta=beta_n),
        ),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )


def make_na_leak_channel(g_max: float) -> IonChannel:
    """Create the sodium leak channel (Na⁺ background conductance).

    No gating variables — conductance is always *g_max*.  The reversal
    potential is computed dynamically via the Nernst equation for Na⁺,
    representing persistent sodium leak channels (e.g. NALCN).

    Args:
        g_max: Maximum (and constant) conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the Na⁺ leak
        channel with current field ``INaL``.
    """
    return IonChannel(
        name="NaL",
        g_max=g_max,
        gating_variables=(),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )


def make_k_leak_channel(g_max: float) -> IonChannel:
    """Create the potassium leak channel (K⁺ background conductance).

    No gating variables — conductance is always *g_max*.  The reversal
    potential is computed dynamically via the Nernst equation for K⁺,
    representing two-pore-domain K⁺ background channels (e.g. TREK, TRAAK).

    Args:
        g_max: Maximum (and constant) conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the K⁺ leak
        channel with current field ``IKL``.
    """
    return IonChannel(
        name="KL",
        g_max=g_max,
        gating_variables=(),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )


# ---------------------------------------------------------------------------
# Pospischil et al. (2008) cortical pyramidal (RS) Na⁺/K⁺ rate functions
#
# Source: Pospischil M. et al. (2008) Minimal Hodgkin-Huxley type models for
# different classes of cortical and thalamic neurons. Biol. Cybern. 99:427-441.
# Rate functions adopted from Traub & Miles (1991) / Huguenard & McCormick
# (1992); VT shifts all thresholds to match cortical RS neuron firing.
# ---------------------------------------------------------------------------

#: Voltage threshold parameter (mV) for cortical RS neurons (Pospischil 2008).
#: Shifts all rate function reference voltages from the original Traub-Miles
#: values to match the cortical pyramidal cell firing threshold.
POSPISCHIL_VT: float = -56.2


def pospischil_alpha_m(V: float, ca_i: float) -> float:
    """Forward rate for Pospischil Na⁺ activation gate m.

    Adopted from Traub & Miles (1991).  Has a removable singularity at
    V = VT + 13 = −43.2 mV; the L'Hôpital limit (1.28) is returned when
    ``|V − VT − 13| < SINGULARITY_THRESHOLD``.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    x = V - POSPISCHIL_VT - 13
    if abs(x) < SINGULARITY_THRESHOLD:
        return 1.28
    return -0.32 * x / (safe_exp(-x / 4) - 1)


def pospischil_beta_m(V: float, ca_i: float) -> float:
    """Backward rate for Pospischil Na⁺ activation gate m.

    Adopted from Traub & Miles (1991).  Has a removable singularity at
    V = VT + 40 = −16.2 mV; the L'Hôpital limit (1.4) is returned when
    ``|V − VT − 40| < SINGULARITY_THRESHOLD``.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    x = V - POSPISCHIL_VT - 40
    if abs(x) < SINGULARITY_THRESHOLD:
        return 1.4
    return 0.28 * x / (safe_exp(x / 5) - 1)


def pospischil_alpha_h(V: float, ca_i: float) -> float:
    """Forward rate for Pospischil Na⁺ inactivation gate h.

    Adopted from Traub & Miles (1991).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    return 0.128 * safe_exp(-(V - POSPISCHIL_VT - 17) / 18)


def pospischil_beta_h(V: float, ca_i: float) -> float:
    """Backward rate for Pospischil Na⁺ inactivation gate h.

    Adopted from Traub & Miles (1991).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return 4.0 / (1 + safe_exp(-(V - POSPISCHIL_VT - 40) / 5))


def pospischil_alpha_n(V: float, ca_i: float) -> float:
    """Forward rate for Pospischil K⁺ delayed-rectifier activation gate n.

    Adopted from Traub & Miles (1991).  Has a removable singularity at
    V = VT + 15 = −41.2 mV; the L'Hôpital limit (0.16) is returned when
    ``|V − VT − 15| < SINGULARITY_THRESHOLD``.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    x = V - POSPISCHIL_VT - 15
    if abs(x) < SINGULARITY_THRESHOLD:
        return 0.16
    return -0.032 * x / (safe_exp(-x / 5) - 1)


def pospischil_beta_n(V: float, ca_i: float) -> float:
    """Backward rate for Pospischil K⁺ delayed-rectifier activation gate n.

    Adopted from Traub & Miles (1991).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return 0.5 * safe_exp(-(V - POSPISCHIL_VT - 10) / 40)


def make_pospischil_na_channel(g_max: float) -> IonChannel:
    """Create the Pospischil cortical RS fast sodium channel (Na⁺).

    Uses Pospischil et al. (2008) Traub-Miles kinetics with VT = −56.2 mV:
    activation gate *m* (power 3) and inactivation gate *h* (power 1).
    The reversal potential is computed dynamically via the Nernst equation
    for Na⁺.

    Intended for use as the ``na_channel_factory`` of the Cortical Pyramidal
    preset to match the Pospischil RS neuron model.

    Args:
        g_max: Maximum conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the
        Pospischil cortical RS fast Na⁺ channel.
    """
    return IonChannel(
        name="Na",
        g_max=g_max,
        gating_variables=(
            GatingVariable(
                name="m", power=3, alpha=pospischil_alpha_m, beta=pospischil_beta_m
            ),
            GatingVariable(
                name="h", power=1, alpha=pospischil_alpha_h, beta=pospischil_beta_h
            ),
        ),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )


def make_pospischil_k_channel(g_max: float) -> IonChannel:
    """Create the Pospischil cortical RS delayed-rectifier potassium channel (K⁺).

    Uses Pospischil et al. (2008) Traub-Miles kinetics with VT = −56.2 mV:
    activation gate *n* (power 4).  The reversal potential is computed
    dynamically via the Nernst equation for K⁺.

    Intended for use as the ``k_channel_factory`` of the Cortical Pyramidal
    preset to match the Pospischil RS neuron model.

    Args:
        g_max: Maximum conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the
        Pospischil cortical RS delayed-rectifier K⁺ channel.
    """
    return IonChannel(
        name="K",
        g_max=g_max,
        gating_variables=(
            GatingVariable(
                name="n", power=4, alpha=pospischil_alpha_n, beta=pospischil_beta_n
            ),
        ),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )


# ---------------------------------------------------------------------------
# McCormick & Huguenard (1992) thalamic relay Na⁺/K⁺ rate functions
#
# Source: McCormick, D.A. & Huguenard, J.R. (1992) A model of the
# electrophysiological properties of thalamocortical relay neurons.
# J. Neurophysiol. 68:1384–1400.
#
# Parameterisation: Pospischil, M. et al. (2008) Minimal Hodgkin-Huxley type
# models for different classes of cortical and thalamic neurons.
# Biol. Cybern. 99:427–441, Table 2 (TC cell).
#
# Rate functions use the same Traub-Miles analytical form as the Pospischil
# cortical RS factories above (pospischil_alpha_m / _beta_m / _alpha_h /
# _beta_h / _alpha_n / _beta_n) — the only difference is the voltage
# threshold VT = −52 mV here vs VT = −56.2 mV for cortical RS.  VT = −52 mV
# matches the firing threshold of guinea-pig dorsal LGN relay neurons
# recorded by McCormick & Huguenard (1992) at 36 °C.
# ---------------------------------------------------------------------------

#: Voltage threshold parameter (mV) for thalamic relay cells.
#: Pospischil et al. (2008), Table 2 (TC model): VT = −52 mV.
THALAMIC_RELAY_VT: float = -52.0


def thalamic_relay_alpha_m(V: float, ca_i: float) -> float:
    """Forward rate for thalamic relay Na⁺ activation gate m.

    Traub-Miles form parameterised for thalamic relay cells (VT = −52 mV).
    Has a removable singularity at V = VT + 13 = −39 mV; the L'Hôpital
    limit (1.28) is returned when ``|V − VT − 13| < SINGULARITY_THRESHOLD``.

    Reference: McCormick & Huguenard (1992), J. Neurophysiol. 68:1384;
    Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (TC model).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    x = V - THALAMIC_RELAY_VT - 13
    if abs(x) < SINGULARITY_THRESHOLD:
        return 1.28
    return -0.32 * x / (safe_exp(-x / 4) - 1)


def thalamic_relay_beta_m(V: float, ca_i: float) -> float:
    """Backward rate for thalamic relay Na⁺ activation gate m.

    Traub-Miles form parameterised for thalamic relay cells (VT = −52 mV).
    Has a removable singularity at V = VT + 40 = −12 mV; the L'Hôpital
    limit (1.4) is returned when ``|V − VT − 40| < SINGULARITY_THRESHOLD``.

    Reference: McCormick & Huguenard (1992), J. Neurophysiol. 68:1384;
    Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (TC model).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    x = V - THALAMIC_RELAY_VT - 40
    if abs(x) < SINGULARITY_THRESHOLD:
        return 1.4
    return 0.28 * x / (safe_exp(x / 5) - 1)


def thalamic_relay_alpha_h(V: float, ca_i: float) -> float:
    """Forward rate for thalamic relay Na⁺ inactivation gate h.

    Traub-Miles form parameterised for thalamic relay cells (VT = −52 mV).

    Reference: McCormick & Huguenard (1992), J. Neurophysiol. 68:1384;
    Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (TC model).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    return 0.128 * safe_exp(-(V - THALAMIC_RELAY_VT - 17) / 18)


def thalamic_relay_beta_h(V: float, ca_i: float) -> float:
    """Backward rate for thalamic relay Na⁺ inactivation gate h.

    Traub-Miles form parameterised for thalamic relay cells (VT = −52 mV).

    Reference: McCormick & Huguenard (1992), J. Neurophysiol. 68:1384;
    Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (TC model).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return 4.0 / (1 + safe_exp(-(V - THALAMIC_RELAY_VT - 40) / 5))


def thalamic_relay_alpha_n(V: float, ca_i: float) -> float:
    """Forward rate for thalamic relay K⁺ delayed-rectifier activation gate n.

    Traub-Miles form parameterised for thalamic relay cells (VT = −52 mV).
    Has a removable singularity at V = VT + 15 = −37 mV; the L'Hôpital
    limit (0.16) is returned when ``|V − VT − 15| < SINGULARITY_THRESHOLD``.

    Reference: McCormick & Huguenard (1992), J. Neurophysiol. 68:1384;
    Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (TC model).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    x = V - THALAMIC_RELAY_VT - 15
    if abs(x) < SINGULARITY_THRESHOLD:
        return 0.16
    return -0.032 * x / (safe_exp(-x / 5) - 1)


def thalamic_relay_beta_n(V: float, ca_i: float) -> float:
    """Backward rate for thalamic relay K⁺ delayed-rectifier activation gate n.

    Traub-Miles form parameterised for thalamic relay cells (VT = −52 mV).

    Reference: McCormick & Huguenard (1992), J. Neurophysiol. 68:1384;
    Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (TC model).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return 0.5 * safe_exp(-(V - THALAMIC_RELAY_VT - 10) / 40)


def make_thalamic_relay_na_channel(g_max: float) -> IonChannel:
    """Create the thalamic relay fast sodium channel (Na⁺).

    Uses Traub-Miles kinetics with VT = −52 mV, parameterised for the
    thalamocortical relay (TC) cell model of Pospischil et al. (2008).
    Rate-equation half-points match McCormick & Huguenard (1992) recordings
    of guinea-pig dorsal LGN relay neurons at 36 °C.

    Intended as the ``na_channel_factory`` of the Thalamic Relay preset.
    Compared with the default HH52 Na⁺ channel (fitted to squid axon at
    22 °C), the Traub-Miles form with VT = −52 mV shifts the activation
    threshold ~13 mV depolarised and slows inactivation, preventing the
    ~5.2× Q10 overcorrection that caused premature Na⁺ inactivation.

    Reference: McCormick & Huguenard (1992), J. Neurophysiol. 68:1384;
    Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (TC model).

    Args:
        g_max: Maximum conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the thalamic
        relay fast Na⁺ channel.
    """
    return IonChannel(
        name="Na",
        g_max=g_max,
        gating_variables=(
            GatingVariable(
                name="m",
                power=3,
                alpha=thalamic_relay_alpha_m,
                beta=thalamic_relay_beta_m,
            ),
            GatingVariable(
                name="h",
                power=1,
                alpha=thalamic_relay_alpha_h,
                beta=thalamic_relay_beta_h,
            ),
        ),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )


def make_thalamic_relay_k_channel(g_max: float) -> IonChannel:
    """Create the thalamic relay delayed-rectifier potassium channel (K⁺).

    Uses Traub-Miles kinetics with VT = −52 mV, parameterised for the
    thalamocortical relay (TC) cell model of Pospischil et al. (2008).
    Rate-equation half-points match McCormick & Huguenard (1992) recordings
    of guinea-pig dorsal LGN relay neurons at 36 °C.

    Intended as the ``k_channel_factory`` of the Thalamic Relay preset.

    Reference: McCormick & Huguenard (1992), J. Neurophysiol. 68:1384;
    Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (TC model).

    Args:
        g_max: Maximum conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the thalamic
        relay delayed-rectifier K⁺ channel.
    """
    return IonChannel(
        name="K",
        g_max=g_max,
        gating_variables=(
            GatingVariable(
                name="n",
                power=4,
                alpha=thalamic_relay_alpha_n,
                beta=thalamic_relay_beta_n,
            ),
        ),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )


# ---------------------------------------------------------------------------
# Huguenard & Prince (1992) / Destexhe et al. (1994) TRN Na⁺/K⁺ rate functions
#
# Primary sources:
#   Huguenard, J.R. & Prince, D.A. (1992) A novel T-type current underlies
#   prolonged Ca²⁺-dependent burst firing in GABAergic neurons of rat thalamic
#   reticular nucleus. J. Neurosci. 12:3804–3817.
#
#   Destexhe, A. et al. (1994) A model of spindle rhythmicity in the isolated
#   thalamic reticular nucleus. J. Neurophysiol. 72:803–818.
#
# Parameterisation:
#   Pospischil, M. et al. (2008) Minimal Hodgkin-Huxley type models for
#   different classes of cortical and thalamic neurons.
#   Biol. Cybern. 99:427–441, Table 2 (RE cell, VT = −67 mV).
#
# Rate functions use the same Traub-Miles analytical form as the Pospischil
# cortical RS and thalamic relay factories.  The only difference is the voltage
# threshold VT = −67 mV here (vs −56.2 mV for cortical RS and −52 mV for TC).
# VT = −67 mV matches the firing threshold of rat TRN cells recorded by
# Huguenard & Prince (1992) at 36 °C.
# ---------------------------------------------------------------------------

#: Voltage threshold parameter (mV) for thalamic reticular nucleus cells.
#: Pospischil et al. (2008), Table 2 (RE model): VT = −67 mV.
TRN_VT: float = -67.0


def trn_alpha_m(V: float, ca_i: float) -> float:
    """Forward rate for TRN Na⁺ activation gate m.

    Traub-Miles form parameterised for thalamic reticular nucleus cells
    (VT = −67 mV).  Has a removable singularity at V = VT + 13 = −54 mV;
    the L'Hôpital limit (1.28) is returned when
    ``|V − VT − 13| < SINGULARITY_THRESHOLD``.

    Reference: Huguenard & Prince (1992), J. Neurosci. 12:3804;
    Destexhe et al. (1994), J. Neurophysiol. 72:803;
    Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (RE model).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    x = V - TRN_VT - 13
    if abs(x) < SINGULARITY_THRESHOLD:
        return 1.28
    return -0.32 * x / (safe_exp(-x / 4) - 1)


def trn_beta_m(V: float, ca_i: float) -> float:
    """Backward rate for TRN Na⁺ activation gate m.

    Traub-Miles form parameterised for thalamic reticular nucleus cells
    (VT = −67 mV).  Has a removable singularity at V = VT + 40 = −27 mV;
    the L'Hôpital limit (1.4) is returned when
    ``|V − VT − 40| < SINGULARITY_THRESHOLD``.

    Reference: Huguenard & Prince (1992), J. Neurosci. 12:3804;
    Destexhe et al. (1994), J. Neurophysiol. 72:803;
    Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (RE model).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    x = V - TRN_VT - 40
    if abs(x) < SINGULARITY_THRESHOLD:
        return 1.4
    return 0.28 * x / (safe_exp(x / 5) - 1)


def trn_alpha_h(V: float, ca_i: float) -> float:
    """Forward rate for TRN Na⁺ inactivation gate h.

    Traub-Miles form parameterised for thalamic reticular nucleus cells
    (VT = −67 mV).

    Reference: Huguenard & Prince (1992), J. Neurosci. 12:3804;
    Destexhe et al. (1994), J. Neurophysiol. 72:803;
    Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (RE model).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    return 0.128 * safe_exp(-(V - TRN_VT - 17) / 18)


def trn_beta_h(V: float, ca_i: float) -> float:
    """Backward rate for TRN Na⁺ inactivation gate h.

    Traub-Miles form parameterised for thalamic reticular nucleus cells
    (VT = −67 mV).

    Reference: Huguenard & Prince (1992), J. Neurosci. 12:3804;
    Destexhe et al. (1994), J. Neurophysiol. 72:803;
    Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (RE model).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return 4.0 / (1 + safe_exp(-(V - TRN_VT - 40) / 5))


def trn_alpha_n(V: float, ca_i: float) -> float:
    """Forward rate for TRN K⁺ delayed-rectifier activation gate n.

    Traub-Miles form parameterised for thalamic reticular nucleus cells
    (VT = −67 mV).  Has a removable singularity at V = VT + 15 = −52 mV;
    the L'Hôpital limit (0.16) is returned when
    ``|V − VT − 15| < SINGULARITY_THRESHOLD``.

    Reference: Huguenard & Prince (1992), J. Neurosci. 12:3804;
    Destexhe et al. (1994), J. Neurophysiol. 72:803;
    Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (RE model).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    x = V - TRN_VT - 15
    if abs(x) < SINGULARITY_THRESHOLD:
        return 0.16
    return -0.032 * x / (safe_exp(-x / 5) - 1)


def trn_beta_n(V: float, ca_i: float) -> float:
    """Backward rate for TRN K⁺ delayed-rectifier activation gate n.

    Traub-Miles form parameterised for thalamic reticular nucleus cells
    (VT = −67 mV).

    Reference: Huguenard & Prince (1992), J. Neurosci. 12:3804;
    Destexhe et al. (1994), J. Neurophysiol. 72:803;
    Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (RE model).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return 0.5 * safe_exp(-(V - TRN_VT - 10) / 40)


def make_trn_na_channel(g_max: float) -> IonChannel:
    """Create the TRN fast sodium channel (Na⁺).

    Uses Traub-Miles kinetics with VT = −67 mV, parameterised for the
    thalamic reticular nucleus (RE) cell model of Pospischil et al. (2008).
    Rate-equation half-points match Huguenard & Prince (1992) recordings
    of rat TRN cells at 36 °C.

    Intended as the ``na_channel_factory`` of the TRN preset.  Compared with
    the default HH52 Na⁺ channel (fitted to squid axon at 22 °C), the
    Traub-Miles form with VT = −67 mV shifts the activation threshold ~13 mV
    depolarised and slows inactivation, preventing the ~5.2× Q10 overcorrection
    that caused premature Na⁺ inactivation.

    Reference: Huguenard & Prince (1992), J. Neurosci. 12:3804;
    Destexhe et al. (1994), J. Neurophysiol. 72:803;
    Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (RE model).
    Kinetics recorded at 36 °C — use T_ref = 309.15 K with this factory.

    Args:
        g_max: Maximum conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the TRN fast
        Na⁺ channel.
    """
    return IonChannel(
        name="Na",
        g_max=g_max,
        gating_variables=(
            GatingVariable(name="m", power=3, alpha=trn_alpha_m, beta=trn_beta_m),
            GatingVariable(name="h", power=1, alpha=trn_alpha_h, beta=trn_beta_h),
        ),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )


def make_trn_k_channel(g_max: float) -> IonChannel:
    """Create the TRN delayed-rectifier potassium channel (K⁺).

    Uses Traub-Miles kinetics with VT = −67 mV, parameterised for the
    thalamic reticular nucleus (RE) cell model of Pospischil et al. (2008).
    Rate-equation half-points match Huguenard & Prince (1992) recordings
    of rat TRN cells at 36 °C.

    Intended as the ``k_channel_factory`` of the TRN preset.

    Reference: Huguenard & Prince (1992), J. Neurosci. 12:3804;
    Destexhe et al. (1994), J. Neurophysiol. 72:803;
    Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (RE model).
    Kinetics recorded at 36 °C — use T_ref = 309.15 K with this factory.

    Args:
        g_max: Maximum conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the TRN
        delayed-rectifier K⁺ channel.
    """
    return IonChannel(
        name="K",
        g_max=g_max,
        gating_variables=(
            GatingVariable(name="n", power=4, alpha=trn_alpha_n, beta=trn_beta_n),
        ),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )


# ---------------------------------------------------------------------------
# De Schutter & Bower (1994) Purkinje cell Na⁺/K⁺ rate functions
#
# Source: De Schutter, E. & Bower, J.M. (1994) An active membrane model of
# the cerebellar Purkinje cell I. Simulation of current clamps in slice.
# J. Neurophysiol. 71:375–400.
#
# Rate functions use the same Traub-Miles analytical form as the other
# cell-type-specific factories.  VT = −58 mV matches the somatic Na⁺
# activation threshold of guinea-pig cerebellar Purkinje neurons at 32 °C
# (the recording temperature of De Schutter & Bower 1994).
# ---------------------------------------------------------------------------

#: Voltage threshold parameter (mV) for cerebellar Purkinje cells.
#: Matches the somatic NaF activation threshold from De Schutter & Bower (1994).
PURKINJE_VT: float = -58.0


def purkinje_alpha_m(V: float, ca_i: float) -> float:
    """Forward rate for Purkinje Na⁺ activation gate m.

    Traub-Miles form parameterised for cerebellar Purkinje cells (VT = −58 mV).
    Has a removable singularity at V = VT + 13 = −45 mV; the L'Hôpital
    limit (1.28) is returned when ``|V − VT − 13| < SINGULARITY_THRESHOLD``.

    Reference: De Schutter & Bower (1994), J. Neurophysiol. 71:375.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    x = V - PURKINJE_VT - 13
    if abs(x) < SINGULARITY_THRESHOLD:
        return 1.28
    return -0.32 * x / (safe_exp(-x / 4) - 1)


def purkinje_beta_m(V: float, ca_i: float) -> float:
    """Backward rate for Purkinje Na⁺ activation gate m.

    Traub-Miles form parameterised for cerebellar Purkinje cells (VT = −58 mV).
    Has a removable singularity at V = VT + 40 = −18 mV; the L'Hôpital
    limit (1.4) is returned when ``|V − VT − 40| < SINGULARITY_THRESHOLD``.

    Reference: De Schutter & Bower (1994), J. Neurophysiol. 71:375.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    x = V - PURKINJE_VT - 40
    if abs(x) < SINGULARITY_THRESHOLD:
        return 1.4
    return 0.28 * x / (safe_exp(x / 5) - 1)


def purkinje_alpha_h(V: float, ca_i: float) -> float:
    """Forward rate for Purkinje Na⁺ inactivation gate h.

    Traub-Miles form parameterised for cerebellar Purkinje cells (VT = −58 mV).

    Reference: De Schutter & Bower (1994), J. Neurophysiol. 71:375.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    return 0.128 * safe_exp(-(V - PURKINJE_VT - 17) / 18)


def purkinje_beta_h(V: float, ca_i: float) -> float:
    """Backward rate for Purkinje Na⁺ inactivation gate h.

    Traub-Miles form parameterised for cerebellar Purkinje cells (VT = −58 mV).

    Reference: De Schutter & Bower (1994), J. Neurophysiol. 71:375.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return 4.0 / (1 + safe_exp(-(V - PURKINJE_VT - 40) / 5))


def purkinje_alpha_n(V: float, ca_i: float) -> float:
    """Forward rate for Purkinje K⁺ delayed-rectifier activation gate n.

    Traub-Miles form parameterised for cerebellar Purkinje cells (VT = −58 mV).
    Has a removable singularity at V = VT + 15 = −43 mV; the L'Hôpital
    limit (0.16) is returned when ``|V − VT − 15| < SINGULARITY_THRESHOLD``.

    Reference: De Schutter & Bower (1994), J. Neurophysiol. 71:375.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    x = V - PURKINJE_VT - 15
    if abs(x) < SINGULARITY_THRESHOLD:
        return 0.16
    return -0.032 * x / (safe_exp(-x / 5) - 1)


def purkinje_beta_n(V: float, ca_i: float) -> float:
    """Backward rate for Purkinje K⁺ delayed-rectifier activation gate n.

    Traub-Miles form parameterised for cerebellar Purkinje cells (VT = −58 mV).

    Reference: De Schutter & Bower (1994), J. Neurophysiol. 71:375.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return 0.5 * safe_exp(-(V - PURKINJE_VT - 10) / 40)


def make_purkinje_na_channel(g_max: float) -> IonChannel:
    """Create the cerebellar Purkinje fast sodium channel (Na⁺).

    Uses Traub-Miles kinetics with VT = −58 mV to match the somatic NaF
    activation threshold of mammalian cerebellar Purkinje neurons recorded
    by De Schutter & Bower (1994) at 32 °C.

    Intended as the ``na_channel_factory`` of the Purkinje preset.  Compared
    with the default HH52 Na⁺ channel (fitted to squid axon at 22 °C), the
    Traub-Miles form with VT = −58 mV places the activation half-point near
    −45 mV and prevents the ~5.2× Q10 overcorrection that caused premature
    Na⁺ inactivation.

    Reference: De Schutter & Bower (1994), J. Neurophysiol. 71:375.
    Kinetics recorded at 32 °C — use T_ref = 305.15 K with this factory.

    Args:
        g_max: Maximum conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the Purkinje
        fast Na⁺ channel.
    """
    return IonChannel(
        name="Na",
        g_max=g_max,
        gating_variables=(
            GatingVariable(
                name="m", power=3, alpha=purkinje_alpha_m, beta=purkinje_beta_m
            ),
            GatingVariable(
                name="h", power=1, alpha=purkinje_alpha_h, beta=purkinje_beta_h
            ),
        ),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )


def make_purkinje_k_channel(g_max: float) -> IonChannel:
    """Create the cerebellar Purkinje delayed-rectifier potassium channel (K⁺).

    Uses Traub-Miles kinetics with VT = −58 mV to match the somatic KDR
    activation threshold of mammalian cerebellar Purkinje neurons recorded
    by De Schutter & Bower (1994) at 32 °C.

    Intended as the ``k_channel_factory`` of the Purkinje preset.

    Reference: De Schutter & Bower (1994), J. Neurophysiol. 71:375.
    Kinetics recorded at 32 °C — use T_ref = 305.15 K with this factory.

    Args:
        g_max: Maximum conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the Purkinje
        delayed-rectifier K⁺ channel.
    """
    return IonChannel(
        name="K",
        g_max=g_max,
        gating_variables=(
            GatingVariable(
                name="n", power=4, alpha=purkinje_alpha_n, beta=purkinje_beta_n
            ),
        ),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )


# ---------------------------------------------------------------------------
# Otsuka et al. (2004) STN channel kinetics
# ---------------------------------------------------------------------------
# Reference: Otsuka, T. et al. (2004). Conductance-based model of the
# voltage-dependent generation of a plateau potential in subthalamic neurons.
# J. Neurophysiol. 92, 255–264.
#
# All rate functions are derived from steady-state (x_inf) and time-constant
# (tau_x) formulations via:
#   alpha_x = x_inf / tau_x
#   beta_x  = (1 − x_inf) / tau_x


def _stn_alpha_m(V: float, ca_i: float) -> float:
    """Forward rate for STN Na⁺ activation gate m (Otsuka et al. 2004).

    Derived from m_inf(V) = 1/(1 + exp(−(V + 40)/8)) and τ_m = 0.2 ms.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    m_inf = 1.0 / (1.0 + safe_exp(-(V + 40.0) / 8.0))
    return m_inf / 0.2


def _stn_beta_m(V: float, ca_i: float) -> float:
    """Backward rate for STN Na⁺ activation gate m (Otsuka et al. 2004).

    Derived from m_inf(V) = 1/(1 + exp(−(V + 40)/8)) and τ_m = 0.2 ms.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    m_inf = 1.0 / (1.0 + safe_exp(-(V + 40.0) / 8.0))
    return (1.0 - m_inf) / 0.2


def _stn_alpha_h(V: float, ca_i: float) -> float:
    """Forward rate for STN Na⁺ inactivation gate h (Otsuka et al. 2004).

    Derived from:
      h_inf(V)   = 1/(1 + exp((V + 45.5)/6.4))
      1/τ_h(V)   = 0.128·exp(−(V + 38)/18) + 4/(1 + exp(−(V + 15)/5))

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    h_inf = 1.0 / (1.0 + safe_exp((V + 45.5) / 6.4))
    inv_tau_h = 0.128 * safe_exp(-(V + 38.0) / 18.0) + 4.0 / (
        1.0 + safe_exp(-(V + 15.0) / 5.0)
    )
    return h_inf * inv_tau_h


def _stn_beta_h(V: float, ca_i: float) -> float:
    """Backward rate for STN Na⁺ inactivation gate h (Otsuka et al. 2004).

    Derived from:
      h_inf(V)   = 1/(1 + exp((V + 45.5)/6.4))
      1/τ_h(V)   = 0.128·exp(−(V + 38)/18) + 4/(1 + exp(−(V + 15)/5))

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    h_inf = 1.0 / (1.0 + safe_exp((V + 45.5) / 6.4))
    inv_tau_h = 0.128 * safe_exp(-(V + 38.0) / 18.0) + 4.0 / (
        1.0 + safe_exp(-(V + 15.0) / 5.0)
    )
    return (1.0 - h_inf) * inv_tau_h


def _stn_alpha_n(V: float, ca_i: float) -> float:
    """Forward rate for STN K⁺ DR activation gate n (Otsuka et al. 2004).

    Derived from:
      n_inf(V) = 1/(1 + exp(−(V + 41)/14))
      τ_n(V)   = 0.25 + 10.75/(exp(−(V + 51)/12) + exp((V + 51)/15))

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    n_inf = 1.0 / (1.0 + safe_exp(-(V + 41.0) / 14.0))
    tau_n = 0.25 + 10.75 / (safe_exp(-(V + 51.0) / 12.0) + safe_exp((V + 51.0) / 15.0))
    return n_inf / tau_n


def _stn_beta_n(V: float, ca_i: float) -> float:
    """Backward rate for STN K⁺ DR activation gate n (Otsuka et al. 2004).

    Derived from:
      n_inf(V) = 1/(1 + exp(−(V + 41)/14))
      τ_n(V)   = 0.25 + 10.75/(exp(−(V + 51)/12) + exp((V + 51)/15))

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    n_inf = 1.0 / (1.0 + safe_exp(-(V + 41.0) / 14.0))
    tau_n = 0.25 + 10.75 / (safe_exp(-(V + 51.0) / 12.0) + safe_exp((V + 51.0) / 15.0))
    return (1.0 - n_inf) / tau_n


def make_stn_na_channel(g_max: float) -> IonChannel:
    """Create the STN high-threshold sodium channel (Otsuka et al. 2004).

    Uses high-threshold activation kinetics specific to subthalamic nucleus
    neurons.  Compared with the classic HH52 channel, the activation half-point
    is the same (−40 mV) but the slope is gentler (8 mV vs ~10 mV in HH52) and
    τ_m is fixed at 0.2 ms, giving a faster, more sharply threshold-dependent
    activation.  The inactivation half-point shifts to −45.5 mV with a 6.4 mV
    slope.

    Gating variable names are ``"m"`` (activation, power 3) and ``"h"``
    (inactivation, power 1), matching the simulation result fields.

    Reference: Otsuka et al. (2004), J. Neurophysiol. 92, 255–264, Table 1.

    Args:
        g_max: Maximum conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the STN Na⁺
        channel.
    """
    return IonChannel(
        name="Na",
        g_max=g_max,
        gating_variables=(
            GatingVariable(name="m", power=3, alpha=_stn_alpha_m, beta=_stn_beta_m),
            GatingVariable(name="h", power=1, alpha=_stn_alpha_h, beta=_stn_beta_h),
        ),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )


def make_stn_k_channel(g_max: float) -> IonChannel:
    """Create the STN fast delayed-rectifier potassium channel (Otsuka et al. 2004).

    Uses fast DR kinetics specific to subthalamic nucleus neurons.  The
    activation half-point is −41 mV (vs −55 mV in HH52) with a slope of
    14 mV, and the voltage-dependent time constant peaks at ~5.6 ms near
    −51 mV and decays at more depolarised or hyperpolarised potentials.

    Uses a single activation gate ``"n"`` with power 4, matching the simulation
    result field used throughout the simulator.

    Reference: Otsuka et al. (2004), J. Neurophysiol. 92, 255–264, Table 1.

    Args:
        g_max: Maximum conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the STN K⁺ DR
        channel.
    """
    return IonChannel(
        name="K",
        g_max=g_max,
        gating_variables=(
            GatingVariable(name="n", power=4, alpha=_stn_alpha_n, beta=_stn_beta_n),
        ),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )
