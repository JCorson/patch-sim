"""Concrete optional ion channel implementations.

These channels can be added to a HodgkinHuxley model via the
``optional_channels`` argument to extend the classic three-channel HH model
with additional biophysical mechanisms.
"""

from .channels import BaseIonChannel, GatingVariable
from .constants import DEFAULT_E_IH, DEFAULT_G_IH
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
