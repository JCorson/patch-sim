"""Otsuka et al. (2004) STN (subthalamic nucleus) channel factories.

Reference: Otsuka, T. et al. (2004). Conductance-based model of the
voltage-dependent generation of a plateau potential in subthalamic neurons.
J. Neurophysiol. 92, 255–264.

All rate functions are derived from steady-state (x_inf) and time-constant
(tau_x) formulations via:
  alpha_x = x_inf / tau_x
  beta_x  = (1 − x_inf) / tau_x
"""

from ..electrochemistry import boltzmann_cosh_rates
from ..rates import VoltageOnlyFn
from ..utils import safe_exp
from .base import GatingVariable, IonChannel, IonSpecies, NernstSpec

__all__ = [
    "make_stn_na_channel",
]


def _stn_alpha_m_impl(V: float, ca_i: float) -> float:
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


_stn_alpha_m = VoltageOnlyFn(_stn_alpha_m_impl)


def _stn_beta_m_impl(V: float, ca_i: float) -> float:
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


_stn_beta_m = VoltageOnlyFn(_stn_beta_m_impl)


def _stn_alpha_h_impl(V: float, ca_i: float) -> float:
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


_stn_alpha_h = VoltageOnlyFn(_stn_alpha_h_impl)


def _stn_beta_h_impl(V: float, ca_i: float) -> float:
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


_stn_beta_h = VoltageOnlyFn(_stn_beta_h_impl)


# ---------------------------------------------------------------------------
# STN fast Na⁺ slow voltage-dependent inactivation gate ``sNa`` (opt-in).
# ---------------------------------------------------------------------------
#
# The Otsuka 2004 fast Na⁺ kinetics retain h_inf ≈ 0.01 at −15 mV; scaled by
# g_Na = 30 mS/cm² in the STN preset this still drives ~10–20 µA/cm² inward
# at the depol-block plateau (#324) which is enough to sustain a residual
# plateau even after INaP slow inactivation has eliminated the persistent
# Na⁺ component.  Real fast Na⁺ channels also undergo slow voltage-dependent
# inactivation on top of the fast h gate (Fleidervish & Gutnick 1996;
# Mickus, Jung & Spruston 1999; Do & Bean 2003 demonstrate the same in STN
# pacemaker channels).  This second gate only engages on sustained
# depolarisation and provides the missing escape route at the −15 mV
# attractor.
#
# Parameters: V½ = −50 mV (depolarised end of the −60 to −50 mV slow-
# inactivation V½ range reported by Fleidervish & Gutnick 1996 / Mickus
# et al. 1999); slope 8 mV (mid-range 6–10 mV); inverted Boltzmann so the
# gate is open at hyperpolarised potentials.  At V = −65 mV s_inf ≈ 0.87
# (autonomous pacemaking is minimally perturbed); at V = −15 mV s_inf ≈ 0.01
# (residual h-tail abolished).  τ_scale = 200 ms / τ_floor = 20 ms is
# reduced from literature (~seconds) because in STN the gate co-acts with
# INaP sNaP and K_ATP to recover within ~1 s.
#
# Gate name ``sNa`` was chosen to avoid colliding with ``s`` (INaR
# activation) and ``sNaP`` (INaP slow inactivation).  Slow inactivation is
# opt-in (default off) so presets calibrated against the no-slow-
# inactivation Otsuka kinetics keep their existing phenotypes.

_stn_alpha_sNa, _stn_beta_sNa = boltzmann_cosh_rates(
    half=-50.0,
    slope=8.0,
    tau_scale=200.0,
    tau_floor=20.0,
    inverted=True,
)


def make_stn_na_channel(g_max: float) -> IonChannel:
    """Create the STN high-threshold sodium channel (Otsuka et al. 2004).

    Uses high-threshold activation kinetics specific to subthalamic nucleus
    neurons.  Compared with the classic HH52 channel, the activation
    half-point is the same (−40 mV) but the slope is gentler (8 mV vs
    ~10 mV in HH52) and τ_m is fixed at 0.2 ms, giving a faster, more
    sharply threshold-dependent activation.  The inactivation half-point
    shifts to −45.5 mV with a 6.4 mV slope.

    The channel always includes three gates: ``"m"`` (activation, power 3),
    ``"h"`` (fast inactivation, power 1), and ``"sNa"`` (slow voltage-
    dependent inactivation, power 1; V½ = −50 mV, slope 8 mV, inverted
    Boltzmann).  The slow gate is mostly available at hyperpolarised
    potentials and decays towards zero on sustained depolarisation,
    providing the escape mechanism that lets the membrane repolarise after
    a prolonged suprathreshold step (#324).  The simulation column is
    named ``sNa`` rather than ``s`` to avoid colliding with
    :func:`~patch_sim.channels.auxiliary.make_inar_channel`'s activation
    gate.

    References:
        - Otsuka et al. (2004), J. Neurophysiol. 92:255 (fast Na kinetics).
        - Fleidervish & Gutnick (1996), J. Physiol. 493:83 (slow Na
          inactivation in cortical pyramidal cells).
        - Mickus, Jung & Spruston (1999), Biophys. J. 76:846 (slow Na
          inactivation in CA1 pyramidal cells).
        - Do & Bean (2003), Neuron 39:109 (Na pacemaker channels in STN).

    Args:
        g_max: Maximum conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the STN Na⁺
        channel.
    """
    m_var = GatingVariable(name="m", power=3, alpha=_stn_alpha_m, beta=_stn_beta_m)
    h_var = GatingVariable(name="h", power=1, alpha=_stn_alpha_h, beta=_stn_beta_h)
    s_var = GatingVariable(
        name="sNa", power=1, alpha=_stn_alpha_sNa, beta=_stn_beta_sNa
    )
    return IonChannel(
        name="Na",
        g_max=g_max,
        gating_variables=(m_var, h_var, s_var),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )
