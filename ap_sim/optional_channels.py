"""Concrete optional ion channel implementations.

These channels can be added to a HodgkinHuxley model via the
``optional_channels`` argument to extend the classic three-channel HH model
with additional biophysical mechanisms.
"""

import math

from .channels import BaseIonChannel, GatingVariable
from .constants import (
    DEFAULT_E_IH,
    DEFAULT_E_IKA,
    DEFAULT_E_NAP,
    DEFAULT_E_NAR,
    DEFAULT_G_IH,
    DEFAULT_G_IKA,
    DEFAULT_G_NAP,
    DEFAULT_G_NAR,
)
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


# ---------------------------------------------------------------------------
# INaP — Persistent Na⁺ channel (Magistretti & Alonso 1999)
# ---------------------------------------------------------------------------

_NAP_HALF: float = -52.6  # Half-activation voltage in mV
_NAP_SLOPE: float = 4.6  # Activation slope in mV
_NAP_TAU_SCALE: float = 6.0  # Time-constant scale in ms
_NAP_TAU_FLOOR: float = 0.1  # Minimum time constant in ms


def _nap_m_inf(V: float) -> float:
    """Steady-state activation of INaP at voltage V.

    Uses a Boltzmann function with half-activation at -52.6 mV.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Steady-state open probability in [0, 1].
    """
    return 1.0 / (1.0 + safe_exp(-(V - _NAP_HALF) / _NAP_SLOPE))


def _nap_tau(V: float) -> float:
    """Voltage-dependent time constant for INaP activation.

    Uses a cosh-based expression with a floor to prevent near-zero values.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Time constant in ms (floored at 0.1 ms).
    """
    tau = _NAP_TAU_SCALE / math.cosh((V - _NAP_HALF) / (2.0 * _NAP_SLOPE))
    return max(tau, _NAP_TAU_FLOOR)


def _alpha_p(V: float) -> float:
    """Forward rate for INaP gating variable p (Magistretti & Alonso 1999).

    Derived from Boltzmann steady state and voltage-dependent time constant:
    alpha_p = m_inf / tau.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Forward rate alpha_p in 1/ms.
    """
    return _nap_m_inf(V) / _nap_tau(V)


def _beta_p(V: float) -> float:
    """Backward rate for INaP gating variable p (Magistretti & Alonso 1999).

    Derived from Boltzmann steady state and voltage-dependent time constant:
    beta_p = (1 - m_inf) / tau.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Backward rate beta_p in 1/ms.
    """
    return (1.0 - _nap_m_inf(V)) / _nap_tau(V)


def make_inap_channel(
    g_max: float = DEFAULT_G_NAP,
    e_rev: float = DEFAULT_E_NAP,
) -> BaseIonChannel:
    """Create an INaP (persistent Na⁺) ion channel.

    INaP is a non-inactivating Na⁺ current active near the resting potential.
    It amplifies subthreshold depolarisations and can lower the threshold for
    action potential generation.  It uses a single gating variable ``p``
    (power 1).

    Kinetics follow Magistretti & Alonso (1999), with a Boltzmann activation
    centred at -52.6 mV and a cosh-based time constant.

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~ap_sim.constants.DEFAULT_G_NAP`.
        e_rev: Reversal potential in mV. Defaults to
            :data:`~ap_sim.constants.DEFAULT_E_NAP`.

    Returns:
        A :class:`~ap_sim.channels.BaseIonChannel` representing the INaP current.
    """
    p_var = GatingVariable(name="p", power=1, alpha=_alpha_p, beta=_beta_p)
    return BaseIonChannel(
        name="INaP",
        g_max=g_max,
        gating_variables=(p_var,),
        e_rev=e_rev,
    )


# ---------------------------------------------------------------------------
# INaR — Resurgent Na⁺ channel (Raman & Bean 2001, simplified)
# ---------------------------------------------------------------------------

_NAR_S_HALF: float = -42.0  # Half-activation for s in mV
_NAR_S_SLOPE: float = 5.0  # Activation slope for s in mV
_NAR_S_TAU_SCALE: float = 0.5  # Time-constant scale for s in ms
_NAR_S_TAU_FLOOR: float = 0.05  # Minimum tau_s in ms

_NAR_HR_HALF: float = -55.0  # Half-unblocking for hr in mV
_NAR_HR_SLOPE: float = 8.0  # Unblocking slope for hr in mV
_NAR_HR_TAU_A: float = 150.0  # Asymmetric tau_hr numerator in ms
_NAR_HR_TAU_HALF: float = -40.0  # Voltage of tau_hr half-rise in mV
_NAR_HR_TAU_SLOPE: float = 10.0  # Slope of tau_hr sigmoid in mV
_NAR_HR_TAU_OFFSET: float = 1.0  # Additive offset for tau_hr in ms


def _nar_s_inf(V: float) -> float:
    """Steady-state activation of INaR variable s at voltage V.

    Boltzmann function with half-activation at -42.0 mV.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Steady-state open probability in [0, 1].
    """
    return 1.0 / (1.0 + safe_exp(-(V - _NAR_S_HALF) / _NAR_S_SLOPE))


def _nar_tau_s(V: float) -> float:
    """Voltage-dependent time constant for INaR activation variable s.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Time constant in ms (floored at 0.05 ms).
    """
    tau = _NAR_S_TAU_SCALE / math.cosh((V - _NAR_S_HALF) / (2.0 * _NAR_S_SLOPE))
    return max(tau, _NAR_S_TAU_FLOOR)


def _alpha_s(V: float) -> float:
    """Forward rate for INaR activation gating variable s.

    Derived as alpha_s = s_inf / tau_s.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Forward rate alpha_s in 1/ms.
    """
    return _nar_s_inf(V) / _nar_tau_s(V)


def _beta_s(V: float) -> float:
    """Backward rate for INaR activation gating variable s.

    Derived as beta_s = (1 - s_inf) / tau_s.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Backward rate beta_s in 1/ms.
    """
    return (1.0 - _nar_s_inf(V)) / _nar_tau_s(V)


def _nar_hr_inf(V: float) -> float:
    """Steady-state unblocking variable hr of INaR at voltage V.

    High at hyperpolarised potentials (channel unblocked), low at depolarised
    potentials (channel blocked).  Half-point at -55.0 mV.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Steady-state unblocking probability in [0, 1].
    """
    return 1.0 / (1.0 + safe_exp((V - _NAR_HR_HALF) / _NAR_HR_SLOPE))


def _nar_tau_hr(V: float) -> float:
    """Asymmetric voltage-dependent time constant for INaR unblocking variable hr.

    Slow at depolarised potentials (slow blocking), intermediate at
    hyperpolarised potentials (faster unblocking).

    Args:
        V: Membrane voltage in mV.

    Returns:
        Time constant in ms.
    """
    return (
        _NAR_HR_TAU_A / (1.0 + safe_exp(-(V - _NAR_HR_TAU_HALF) / _NAR_HR_TAU_SLOPE))
        + _NAR_HR_TAU_OFFSET
    )


def _alpha_hr(V: float) -> float:
    """Forward rate for INaR unblocking gating variable hr.

    Derived as alpha_hr = hr_inf / tau_hr.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Forward rate alpha_hr in 1/ms.
    """
    return _nar_hr_inf(V) / _nar_tau_hr(V)


def _beta_hr(V: float) -> float:
    """Backward rate for INaR unblocking gating variable hr.

    Derived as beta_hr = (1 - hr_inf) / tau_hr.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Backward rate beta_hr in 1/ms.
    """
    return (1.0 - _nar_hr_inf(V)) / _nar_tau_hr(V)


def make_inar_channel(
    g_max: float = DEFAULT_G_NAR,
    e_rev: float = DEFAULT_E_NAR,
) -> BaseIonChannel:
    """Create an INaR (resurgent Na⁺) ion channel.

    INaR produces a transient inward Na⁺ current on membrane repolarisation,
    known as the resurgent current.  The mechanism relies on two gating
    variables: ``s`` (activation, power 1) and ``hr`` (unblocking, power 1).

    At rest ``s`` is low and ``hr`` is high (channel closed but unblocked).
    During an action potential ``s`` opens quickly while ``hr`` slowly blocks
    the channel.  On repolarisation ``hr`` recovers before ``s`` deactivates,
    producing a transient resurgent current.

    Kinetics follow a simplified Raman & Bean (2001) two-variable alpha/beta
    model.

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~ap_sim.constants.DEFAULT_G_NAR`.
        e_rev: Reversal potential in mV. Defaults to
            :data:`~ap_sim.constants.DEFAULT_E_NAR`.

    Returns:
        A :class:`~ap_sim.channels.BaseIonChannel` representing the INaR current.
    """
    s_var = GatingVariable(name="s", power=1, alpha=_alpha_s, beta=_beta_s)
    hr_var = GatingVariable(name="hr", power=1, alpha=_alpha_hr, beta=_beta_hr)
    return BaseIonChannel(
        name="INaR",
        g_max=g_max,
        gating_variables=(s_var, hr_var),
        e_rev=e_rev,
    )
