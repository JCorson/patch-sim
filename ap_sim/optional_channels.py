"""Concrete optional ion channel implementations.

These channels can be added to a HodgkinHuxley model via the
``optional_channels`` argument to extend the classic three-channel HH model
with additional biophysical mechanisms.
"""

from .channels import BaseIonChannel, GatingVariable
from .constants import DEFAULT_E_IH, DEFAULT_E_IKA, DEFAULT_G_IH, DEFAULT_G_IKA
from .utils import safe_exp


def _alpha_r(V: float) -> float:
    """Forward rate for Ih gating variable r (Destexhe-style HCN kinetics).

    The Ih current is activated by hyperpolarization; alpha_r increases as
    membrane voltage becomes more negative.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Forward rate alpha_r in 1/ms.
    """
    return safe_exp(-14.59 - 0.086 * V)


def _beta_r(V: float) -> float:
    """Backward rate for Ih gating variable r (Destexhe-style HCN kinetics).

    Args:
        V: Membrane voltage in mV.

    Returns:
        Backward rate beta_r in 1/ms.
    """
    return safe_exp(-1.87 + 0.0701 * V)


_SINGULARITY_TOL: float = 1e-7


def _alpha_a(V: float) -> float:
    """Forward rate for IKa activation gating variable a (Traub & Miles 1991).

    Uses a Boltzmann-style rate shifted to the absolute voltage convention
    (-65 mV resting).  A singularity guard replaces the 0/0 form at
    V = -51.9 mV with the analytic limit.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Forward rate alpha_a in 1/ms.
    """
    x = -51.9 - V
    if abs(x) < _SINGULARITY_TOL:
        return 0.02 * 10.0  # L'Hôpital limit: coefficient * denominator scale
    return 0.02 * x / (safe_exp(x / 10.0) - 1.0)


def _beta_a(V: float) -> float:
    """Backward rate for IKa activation gating variable a (Traub & Miles 1991).

    A singularity guard replaces the 0/0 form at V = -24.9 mV with the
    analytic limit.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Backward rate beta_a in 1/ms.
    """
    x = V + 24.9
    if abs(x) < _SINGULARITY_TOL:
        return 0.0175 * 10.0  # L'Hôpital limit
    return 0.0175 * x / (safe_exp(x / 10.0) - 1.0)


def _alpha_b(V: float) -> float:
    """Forward rate for IKa inactivation gating variable b (Traub & Miles 1991).

    Args:
        V: Membrane voltage in mV.

    Returns:
        Forward rate alpha_b in 1/ms.
    """
    return 0.0016 * safe_exp(-(V + 73.0) / 18.0)


def _beta_b(V: float) -> float:
    """Backward rate for IKa inactivation gating variable b (Traub & Miles 1991).

    Args:
        V: Membrane voltage in mV.

    Returns:
        Backward rate beta_b in 1/ms.
    """
    return 0.05 / (1.0 + safe_exp(-(V + 13.0) / 10.0))


def make_ika_channel(
    g_max: float = DEFAULT_G_IKA,
    e_rev: float = DEFAULT_E_IKA,
) -> BaseIonChannel:
    """Create an IKa (A-type K⁺) ion channel.

    IKa is a fast-inactivating, transient K⁺ current that delays the first
    spike and controls firing rate at stimulus onset.  It uses two gating
    variables: ``a`` (activation, power 1) and ``b`` (inactivation, power 1).

    Kinetics follow Traub & Miles (1991) hippocampal neuron models, shifted by
    -65 mV to match this codebase's absolute voltage convention.

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~ap_sim.constants.DEFAULT_G_IKA`.
        e_rev: Reversal potential in mV. Defaults to
            :data:`~ap_sim.constants.DEFAULT_E_IKA`.

    Returns:
        A :class:`~ap_sim.channels.BaseIonChannel` representing the IKa current.
    """
    a_var = GatingVariable(name="a", power=1, alpha=_alpha_a, beta=_beta_a)
    b_var = GatingVariable(name="b", power=1, alpha=_alpha_b, beta=_beta_b)
    return BaseIonChannel(
        name="IKa",
        g_max=g_max,
        gating_variables=(a_var, b_var),
        e_rev=e_rev,
    )


def make_ih_channel(
    g_max: float = DEFAULT_G_IH,
    e_rev: float = DEFAULT_E_IH,
) -> BaseIonChannel:
    """Create an Ih (HCN/funny current) ion channel.

    The Ih channel is hyperpolarization-activated and carries a mixed Na⁺/K⁺
    cation current. It is responsible for the depolarising sag potential seen
    during sustained hyperpolarisation in many neuron types.

    Kinetics follow Destexhe et al. (1993) with a single gating variable ``r``
    (power 1).  At resting potential (~-65 mV) the channel is largely closed;
    deep hyperpolarisation (~-100 mV) activates it strongly.

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~ap_sim.constants.DEFAULT_G_IH`.
        e_rev: Reversal potential in mV. Defaults to
            :data:`~ap_sim.constants.DEFAULT_E_IH`.

    Returns:
        A :class:`~ap_sim.channels.BaseIonChannel` representing the Ih current.
    """
    r_var = GatingVariable(name="r", power=1, alpha=_alpha_r, beta=_beta_r)
    return BaseIonChannel(
        name="Ih",
        g_max=g_max,
        gating_variables=(r_var,),
        e_rev=e_rev,
    )
