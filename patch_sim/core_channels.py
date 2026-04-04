"""Core Hodgkin-Huxley channel factory functions.

This module provides the six classic HH rate functions as module-level
callables and three factory functions that bundle them into IonChannel objects.

Rate functions all follow the ``(V: float, ca_i: float) -> float`` signature so
they can be used directly as :class:`~patch_sim.channels.GatingVariable` rate
functions.  The ``ca_i`` argument is accepted but ignored; it exists only for
interface compatibility with calcium-sensitive gating variables.
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


def make_leak_channel(g_max: float) -> IonChannel:
    """Create the passive leak channel.

    No gating variables — conductance is always *g_max*.  The reversal
    potential is computed dynamically via the Nernst equation for Cl⁻.

    Args:
        g_max: Maximum (and constant) conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the leak channel.
    """
    return IonChannel(
        name="leak",
        g_max=g_max,
        gating_variables=(),
        reversal_spec=NernstSpec(IonSpecies.CHLORIDE),
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
