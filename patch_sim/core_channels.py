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

    Gating variable names are ``"sodium_activation"`` and
    ``"sodium_inactivation"``, matching the DataFrame column names used
    throughout the simulator.

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
            GatingVariable(
                name="sodium_activation", power=3, alpha=alpha_m, beta=beta_m
            ),
            GatingVariable(
                name="sodium_inactivation", power=1, alpha=alpha_h, beta=beta_h
            ),
        ),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )


def make_k_channel(g_max: float) -> IonChannel:
    """Create the delayed rectifier potassium channel (K⁺).

    Uses the classic HH activation gate *n* (power 4).  The reversal potential
    is computed dynamically via the Nernst equation for K⁺.

    The gating variable name is ``"potassium_activation"``, matching the
    DataFrame column name used throughout the simulator.

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
            GatingVariable(
                name="potassium_activation", power=4, alpha=alpha_n, beta=beta_n
            ),
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
