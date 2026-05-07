"""Hodgkin-Huxley (1952) squid axon channel factories and rate functions.

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

from ..rates import VoltageOnlyFn
from ..utils import safe_exp
from ._traub_miles import SINGULARITY_THRESHOLD
from .base import GatingVariable, IonChannel, IonSpecies, NernstSpec

__all__ = [
    "SINGULARITY_THRESHOLD",
    "alpha_n",
    "beta_n",
    "alpha_m",
    "beta_m",
    "alpha_h",
    "beta_h",
    "make_na_channel",
    "make_k_channel",
    "make_na_leak_channel",
    "make_k_leak_channel",
]


def _alpha_n_impl(V: float, ca_i: float) -> float:
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


alpha_n = VoltageOnlyFn(_alpha_n_impl)


def _beta_n_impl(V: float, ca_i: float) -> float:
    """Backward rate for potassium channel activation gate n.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return 0.125 * safe_exp(-(V + 65) / 80)


beta_n = VoltageOnlyFn(_beta_n_impl)


def _alpha_m_impl(V: float, ca_i: float) -> float:
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


alpha_m = VoltageOnlyFn(_alpha_m_impl)


def _beta_m_impl(V: float, ca_i: float) -> float:
    """Backward rate for sodium channel activation gate m.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return 4.0 * safe_exp(-(V + 65) / 18)


beta_m = VoltageOnlyFn(_beta_m_impl)


def _alpha_h_impl(V: float, ca_i: float) -> float:
    """Forward rate for sodium inactivation gate h.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    return 0.07 * safe_exp(-(V + 65) / 20)


alpha_h = VoltageOnlyFn(_alpha_h_impl)


def _beta_h_impl(V: float, ca_i: float) -> float:
    """Backward rate for sodium inactivation gate h.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return 1 / (1 + safe_exp(-(V + 35) / 10))


beta_h = VoltageOnlyFn(_beta_h_impl)


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
