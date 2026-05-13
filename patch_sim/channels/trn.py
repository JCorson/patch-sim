"""Huguenard & Prince (1992) / Destexhe et al. (1994) TRN channel factories.

Primary sources:
  Huguenard, J.R. & Prince, D.A. (1992) A novel T-type current underlies
  prolonged Ca²⁺-dependent burst firing in GABAergic neurons of rat thalamic
  reticular nucleus. J. Neurosci. 12:3804–3817.

  Destexhe, A. et al. (1994) A model of spindle rhythmicity in the isolated
  thalamic reticular nucleus. J. Neurophysiol. 72:803–818.

Parameterisation:
  Pospischil, M. et al. (2008) Minimal Hodgkin-Huxley type models for
  different classes of cortical and thalamic neurons.
  Biol. Cybern. 99:427–441, Table 2 (RE cell, VT = −67 mV).
"""

import dataclasses

from ..constants import DEFAULT_G_ICAT
from ..rates import VoltageOnlyFn, VoltageOnlyRate
from ..utils import safe_exp
from ._traub_miles import (
    _traub_miles_alpha_h,
    _traub_miles_alpha_m,
    _traub_miles_alpha_n,
    _traub_miles_beta_h,
    _traub_miles_beta_m,
    _traub_miles_beta_n,
)
from .auxiliary import _alpha_dt, _beta_dt
from .base import GatingVariable, IonChannel, IonSpecies, NernstSpec

__all__ = [
    "TRN_VT",
    "trn_alpha_m",
    "trn_beta_m",
    "trn_alpha_h",
    "trn_beta_h",
    "trn_alpha_n",
    "trn_beta_n",
    "make_trn_na_channel",
    "make_trn_k_channel",
    "make_trn_icat_channel",
]

#: Voltage threshold parameter (mV) for thalamic reticular nucleus cells.
#: Pospischil et al. (2008), Table 2 (RE model): VT = −67 mV.
TRN_VT: float = -67.0


def _trn_alpha_m_impl(V: float, ca_i: float) -> float:
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
    return _traub_miles_alpha_m(V, TRN_VT)


trn_alpha_m = VoltageOnlyFn(_trn_alpha_m_impl)


def _trn_beta_m_impl(V: float, ca_i: float) -> float:
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
    return _traub_miles_beta_m(V, TRN_VT)


trn_beta_m = VoltageOnlyFn(_trn_beta_m_impl)


def _trn_alpha_h_impl(V: float, ca_i: float) -> float:
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
    return _traub_miles_alpha_h(V, TRN_VT)


trn_alpha_h = VoltageOnlyFn(_trn_alpha_h_impl)


def _trn_beta_h_impl(V: float, ca_i: float) -> float:
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
    return _traub_miles_beta_h(V, TRN_VT)


trn_beta_h = VoltageOnlyFn(_trn_beta_h_impl)


def _trn_alpha_n_impl(V: float, ca_i: float) -> float:
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
    return _traub_miles_alpha_n(V, TRN_VT)


trn_alpha_n = VoltageOnlyFn(_trn_alpha_n_impl)


def _trn_beta_n_impl(V: float, ca_i: float) -> float:
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
    return _traub_miles_beta_n(V, TRN_VT)


trn_beta_n = VoltageOnlyFn(_trn_beta_n_impl)


@dataclasses.dataclass(frozen=True)
class _ShiftedTrnAlphaH(VoltageOnlyRate):
    """Picklable Traub-Miles ``alpha_h`` rate with h V½ shifted depolarized.

    The shift evaluates the underlying Traub-Miles ``alpha_h`` at
    ``V − h_v_half_shift``, so at any membrane voltage the gate behaves as
    if V were ``h_v_half_shift`` mV more hyperpolarized — larger alpha
    (faster recovery from inactivation) at the LTS-plateau voltages.

    Implemented as a frozen dataclass (not a closure-wrapping
    :class:`~patch_sim.rates.VoltageOnlyFn`) so that
    :func:`~patch_sim.clamp_simulations.simulate_batch` can pickle the
    instance when handing it to a worker process.

    Attributes:
        h_v_half_shift: Depolarized shift of h V½ in mV.
    """

    h_v_half_shift: float

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate alpha_h at ``V − h_v_half_shift`` against ``TRN_VT``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

        Returns:
            Forward rate in 1/ms.
        """
        return _traub_miles_alpha_h(V - self.h_v_half_shift, TRN_VT)


@dataclasses.dataclass(frozen=True)
class _ShiftedTrnBetaH(VoltageOnlyRate):
    """Picklable Traub-Miles ``beta_h`` rate with h V½ shifted depolarized.

    Pairs with :class:`_ShiftedTrnAlphaH`: evaluating beta_h at
    ``V − h_v_half_shift`` yields smaller beta at the LTS-plateau voltages
    (slower inactivation), reinforcing the alpha_h gain.

    Attributes:
        h_v_half_shift: Depolarized shift of h V½ in mV.
    """

    h_v_half_shift: float

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate beta_h at ``V − h_v_half_shift`` against ``TRN_VT``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

        Returns:
            Backward rate in 1/ms.
        """
        return _traub_miles_beta_h(V - self.h_v_half_shift, TRN_VT)


def make_trn_na_channel(g_max: float, h_v_half_shift: float = 0.0) -> IonChannel:
    """Create the TRN fast sodium channel (Na⁺).

    Uses Traub-Miles kinetics with VT = −67 mV, parameterised for the
    thalamic reticular nucleus (RE) cell model of Pospischil et al. (2008).
    Rate-equation half-points match Huguenard & Prince (1992) recordings
    of rat TRN cells at 36 °C.

    Used as the TRN preset's Na⁺ channel.  Compared with the default HH52
    Na⁺ channel (fitted to squid axon at 22 °C), the Traub-Miles form with
    VT = −67 mV shifts the activation threshold ~13 mV depolarized and slows
    inactivation, preventing the ~5.2× Q10 overcorrection that caused
    premature Na⁺ inactivation.

    The optional ``h_v_half_shift`` parameter shifts the h-gate V½
    depolarized while leaving m and n kinetics on the shared TRN_VT.  This
    encodes an isoform-level kinetic difference (NaV1.6 vs NaV1.2;
    Rush et al. 2005, J. Physiol. 564:803; Hatch et al. 2017,
    J. Neurosci. 37:1641) — TRN expresses both isoforms, and the effective
    h V½ depends on isoform mix.  A positive shift accelerates recovery
    from inactivation, breaking the early-LTS-rise depol-block transient
    on REPETITIVE_FIRING cold start (#348).

    Reference: Huguenard & Prince (1992), J. Neurosci. 12:3804;
    Destexhe et al. (1994), J. Neurophysiol. 72:803;
    Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (RE model);
    Rush et al. (2005), J. Physiol. 564:803 (NaV1.6 vs NaV1.2 h kinetics);
    Hatch et al. (2017), J. Neurosci. 37:1641 (TRN NaV1.6/1.2 mix).
    Kinetics recorded at 36 °C — use T_ref = 309.15 K with this factory.

    Args:
        g_max: Maximum conductance in mS/cm².
        h_v_half_shift: Depolarized shift of the h-gate V½ in mV.  Positive
            values accelerate recovery from inactivation (NaV1.6-like).
            Defaults to 0.0 (unshifted Pospischil RE kinetics).

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the TRN fast
        Na⁺ channel.
    """
    if h_v_half_shift == 0.0:
        alpha_h = trn_alpha_h
        beta_h = trn_beta_h
    else:
        alpha_h = _ShiftedTrnAlphaH(h_v_half_shift=h_v_half_shift)
        beta_h = _ShiftedTrnBetaH(h_v_half_shift=h_v_half_shift)
    return IonChannel(
        name="Na",
        g_max=g_max,
        gating_variables=(
            GatingVariable(name="m", power=3, alpha=trn_alpha_m, beta=trn_beta_m),
            GatingVariable(name="h", power=1, alpha=alpha_h, beta=beta_h),
        ),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )


def make_trn_k_channel(g_max: float) -> IonChannel:
    """Create the TRN delayed-rectifier potassium channel (K⁺).

    Uses Traub-Miles kinetics with VT = −67 mV, parameterised for the
    thalamic reticular nucleus (RE) cell model of Pospischil et al. (2008).
    Rate-equation half-points match Huguenard & Prince (1992) recordings
    of rat TRN cells at 36 °C.

    Used as the TRN preset's K⁺ channel.

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
# TRN-specific ICaT — sigmoid-shaped inactivation tau
# ---------------------------------------------------------------------------
# Huguenard & Prince (1992), J. Neurosci. 12:3804 record TRN low-threshold
# spike (LTS) bursts of 5–15 Na⁺ spikes at 200–600 Hz on hyperpolarizing-step
# release.  Reproducing that spike count requires the LTS plateau to last
# long enough to fit 5+ Na⁺/K⁺ AP cycles, which means an ICaT inactivation
# tau in the 100–250 ms range at LTS-plateau voltages (V > −56 mV).
#
# The default Destexhe (1994) cosh-shaped tau peaks at the half-inactivation
# voltage (−80 mV → 20 ms) and decays at depolarized V (≈4 ms at −40 mV,
# floored at 2 ms by −20 mV), so the LTS plateau collapses in 5–10 ms — too
# fast.  Increasing g_T to compensate is not viable: the window-current slope
# conductance grows linearly and beyond g_T ≈ 4 mS/cm² the cell autonomously
# bursts at rest.
#
# This factory replaces the cosh tau with a sigmoid tau that is small at
# hyperpolarized V (rest stability — fast equilibration of ft prevents
# positive-feedback runaway from the window current) and large at LTS-plateau
# V (sustained plateau for 5+ Na⁺ spikes).  ``ft_inf(V)`` is bit-identical to
# the Destexhe default so the existing ft_inf-at-rest invariants are
# preserved.
_TRN_FT_HALF: float = -80.0  # Half-inactivation voltage for ft in mV
_TRN_FT_SLOPE: float = -9.0  # Inactivation slope for ft in mV (Destexhe 1994)
# tau_ft sigmoid parameters.  TAU_MIN matches the Destexhe (1994) cosh-tau
# value at v_rest, preserving rest dynamics.  TAU_MAX is set so the LTS
# plateau (V > −56 mV after ICaT activation) can sustain 5+ Na⁺/K⁺ AP cycles
# (200–600 Hz).  V_HALF and TAU_SLOPE position a smooth transition between
# the rest and plateau regimes around the ICaT activation knee.
_TRN_FT_TAU_MIN: float = 20.0  # ft tau at hyperpolarized V in ms
_TRN_FT_TAU_MAX: float = 200.0  # ft tau at LTS-plateau V in ms
_TRN_FT_TAU_VHALF: float = -50.0  # Sigmoid midpoint for tau_ft in mV
_TRN_FT_TAU_SLOPE: float = 5.0  # Sigmoid slope for tau_ft in mV


def _trn_ft_inf(V: float) -> float:
    """Steady-state inactivation of the TRN ICaT ``ft`` gate at voltage V.

    Bit-identical to the Destexhe (1994) default used by
    :func:`make_icat_channel`: half-point −80 mV, slope −9 mV.  At V = −80 mV
    (TRN v_rest), ``ft_inf = 0.50`` — half de-inactivated, enabling the
    post-inhibitory rebound burst.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Steady-state inactivation probability in [0, 1].
    """
    return 1.0 / (1.0 + safe_exp(-(V - _TRN_FT_HALF) / _TRN_FT_SLOPE))


def _trn_tau_ft(V: float) -> float:
    """Sigmoid voltage-dependent time constant for the TRN ICaT ``ft`` gate.

    Small at hyperpolarized V (≈ ``_TRN_FT_TAU_MIN`` = 20 ms) and large at
    depolarized V (≈ ``_TRN_FT_TAU_MAX`` = 200 ms), with a smooth sigmoid
    transition centered at V = −50 mV (slope 5 mV).  This shape preserves
    rest stability at −80 mV (fast ft equilibration) while sustaining the
    LTS plateau long enough for 5+ Na⁺ spikes (slow ft inactivation at
    plateau voltages of −30 to −10 mV).

    Args:
        V: Membrane voltage in mV.

    Returns:
        Time constant in ms.
    """
    sigmoid = 1.0 / (1.0 + safe_exp(-(V - _TRN_FT_TAU_VHALF) / _TRN_FT_TAU_SLOPE))
    return _TRN_FT_TAU_MIN + (_TRN_FT_TAU_MAX - _TRN_FT_TAU_MIN) * sigmoid


def _alpha_ft_trn_impl(V: float, ca_i: float) -> float:
    """Forward rate for the TRN ICaT inactivation gate ft.

    Derived as ``alpha_ft = ft_inf / tau_ft``.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

    Returns:
        Forward rate alpha_ft in 1/ms.
    """
    return _trn_ft_inf(V) / _trn_tau_ft(V)


_alpha_ft_trn = VoltageOnlyFn(_alpha_ft_trn_impl)


def _beta_ft_trn_impl(V: float, ca_i: float) -> float:
    """Backward rate for the TRN ICaT inactivation gate ft.

    Derived as ``beta_ft = (1 - ft_inf) / tau_ft``.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

    Returns:
        Backward rate beta_ft in 1/ms.
    """
    return (1.0 - _trn_ft_inf(V)) / _trn_tau_ft(V)


_beta_ft_trn = VoltageOnlyFn(_beta_ft_trn_impl)


def make_trn_icat_channel(
    g_max: float = DEFAULT_G_ICAT,
) -> IonChannel:
    """Create the TRN-tuned ICaT (T-type Ca²⁺) channel.

    Variant of :func:`make_icat_channel` whose inactivation time constant
    ``tau_ft(V)`` is sigmoid-shaped rather than cosh-shaped: small (20 ms) at
    hyperpolarized V and large (200 ms) at LTS-plateau V, with a smooth
    transition centered at −50 mV.  This sustains the low-threshold spike
    plateau long enough to support the 5–15 Na⁺ spike, 200–600 Hz rebound
    burst that defines TRN burst mode (Huguenard & Prince 1992) while
    preserving rest stability at −80 mV.

    Activation half-point and slope are unchanged from the global ICaT
    (−56 mV / 6.2 mV).  Inactivation half-point and slope are unchanged
    (−80 mV / −9 mV), so ``ft_inf(V)`` is bit-identical to the Destexhe
    (1994) default — the existing ft_inf-at-rest invariants for the TRN
    preset continue to hold.

    The reversal potential is computed dynamically from the neuron's Ca²⁺
    concentrations using the Nernst equation.

    Reference: Huguenard & Prince (1992), J. Neurosci. 12:3804;
    Destexhe et al. (1994), J. Neurophysiol. 72:803;
    Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (RE).

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~patch_sim.constants.DEFAULT_G_ICAT`.

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the
        TRN ICaT current.
    """
    dt_var = GatingVariable(name="dt", power=2, alpha=_alpha_dt, beta=_beta_dt)
    ft_var = GatingVariable(name="ft", power=1, alpha=_alpha_ft_trn, beta=_beta_ft_trn)
    return IonChannel(
        name="CaT",
        g_max=g_max,
        gating_variables=(dt_var, ft_var),
        reversal_spec=NernstSpec(IonSpecies.CALCIUM),
        carries_calcium=True,
    )
