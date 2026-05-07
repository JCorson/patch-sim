"""Mainen & Sejnowski (1996) cortical pyramidal Na⁺/Kv channel factories.

Source: Mainen, Z.F. & Sejnowski, T.J. (1996) Influence of dendritic structure
on firing pattern in model neocortical neurons.  Nature 382:363–366.
Canonical kinetic parameters from the kv.mod and na.mod ModelDB entries (e.g.
accession 2488).

Temperature pre-scale: rate constants are pre-scaled from the published 23 °C
kinetics to 34 °C using Q10 = 2.3 (factor 2.3^((34-23)/10) = 2.3^1.1 ≈ 2.55),
matching the CORTICAL_PYRAMIDAL preset's T_ref = 307.15 K (34 °C).
"""

from ..constants import DEFAULT_G_MSKV
from ..rates import VoltageOnlyFn
from ..utils import safe_exp
from ._traub_miles import SINGULARITY_THRESHOLD
from .base import GatingVariable, IonChannel, IonSpecies, NernstSpec

__all__ = [
    "MAINEN_SEJNOWSKI_KV_VHALF",
    "MAINEN_SEJNOWSKI_KV_PRESCALE",
    "MAINEN_SEJNOWSKI_NA_THA",
    "mainen_sejnowski_alpha_m",
    "mainen_sejnowski_alpha_h",
    "mainen_sejnowski_alpha_n",
    "mainen_sejnowski_beta_m",
    "mainen_sejnowski_beta_h",
    "mainen_sejnowski_beta_n",
    "make_mainen_sejnowski_na_channel",
    "make_mainen_sejnowski_kv_channel",
]

#: Voltage parameter (mV) for the Mainen-Sejnowski Kv channel — the location
#: of the removable singularity in the α_n / β_n rate functions, taken from
#: the canonical kv.mod ``tha`` parameter.  The rate-function prefactors
#: differ by 10× (α: 0.02, β: 0.002), so n_inf passes through 0.5 at
#: V ≈ +4 mV, *not* at this constant; this is still a high-threshold
#: delayed rectifier (essentially closed below 0 mV).
MAINEN_SEJNOWSKI_KV_VHALF: float = 25.0

#: Temperature pre-scale factor applied to the published 23 °C M-S Kv rate
#: constants so that they represent the channel at 34 °C, the
#: ``CORTICAL_PYRAMIDAL`` preset's ``T_ref``.  Computed as
#: ``2.3 ** ((34 - 23) / 10) = 2.3 ** 1.1 ≈ 2.55`` using the published
#: Q10 = 2.3 from Mainen & Sejnowski (1996).
MAINEN_SEJNOWSKI_KV_PRESCALE: float = 2.55

#: Pre-scaled forward-rate prefactor at 34 °C (= 0.02 / ms × 2.55).
_MAINEN_SEJNOWSKI_ALPHA_PREFACTOR: float = 0.02 * MAINEN_SEJNOWSKI_KV_PRESCALE

#: Pre-scaled backward-rate prefactor at 34 °C (= 0.002 / ms × 2.55).
_MAINEN_SEJNOWSKI_BETA_PREFACTOR: float = 0.002 * MAINEN_SEJNOWSKI_KV_PRESCALE

#: Slope parameter (mV) shared by α_n and β_n.
_MAINEN_SEJNOWSKI_KV_SLOPE: float = 9.0


def _mainen_sejnowski_alpha_n_impl(V: float, ca_i: float) -> float:
    """Forward rate for Mainen-Sejnowski Kv activation gate n.

    Implements ``α_n(V) = A · (V - 25) / (1 - exp(-(V - 25) / 9))`` where
    ``A = 0.02 / ms × 2.55`` bakes in the 23 → 34 °C Q10 = 2.3 pre-scale.
    Has a removable singularity at V = 25 mV; the L'Hôpital limit
    ``A · 9 ≈ 0.459 / ms`` is returned when
    ``|V - 25| < SINGULARITY_THRESHOLD``.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    x = V - MAINEN_SEJNOWSKI_KV_VHALF
    if abs(x) < SINGULARITY_THRESHOLD:
        return _MAINEN_SEJNOWSKI_ALPHA_PREFACTOR * _MAINEN_SEJNOWSKI_KV_SLOPE
    denom = 1.0 - safe_exp(-x / _MAINEN_SEJNOWSKI_KV_SLOPE)
    return _MAINEN_SEJNOWSKI_ALPHA_PREFACTOR * x / denom


mainen_sejnowski_alpha_n = VoltageOnlyFn(_mainen_sejnowski_alpha_n_impl)


def _mainen_sejnowski_beta_n_impl(V: float, ca_i: float) -> float:
    """Backward rate for Mainen-Sejnowski Kv activation gate n.

    Implements ``β_n(V) = -B · (V - 25) / (1 - exp((V - 25) / 9))`` where
    ``B = 0.002 / ms × 2.55`` bakes in the 23 → 34 °C Q10 = 2.3 pre-scale.
    The leading minus combines with the ``(V - 25)`` factor and the
    denominator sign so β_n is positive for all V.  Has a removable
    singularity at V = 25 mV; the L'Hôpital limit
    ``B · 9 ≈ 0.0459 / ms`` is returned when
    ``|V - 25| < SINGULARITY_THRESHOLD``.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    x = V - MAINEN_SEJNOWSKI_KV_VHALF
    if abs(x) < SINGULARITY_THRESHOLD:
        return _MAINEN_SEJNOWSKI_BETA_PREFACTOR * _MAINEN_SEJNOWSKI_KV_SLOPE
    denom = 1.0 - safe_exp(x / _MAINEN_SEJNOWSKI_KV_SLOPE)
    return -_MAINEN_SEJNOWSKI_BETA_PREFACTOR * x / denom


mainen_sejnowski_beta_n = VoltageOnlyFn(_mainen_sejnowski_beta_n_impl)


def make_mainen_sejnowski_kv_channel(g_max: float = DEFAULT_G_MSKV) -> IonChannel:
    """Create the Mainen-Sejnowski cortical pyramidal Kv delayed-rectifier.

    Implements the high-threshold delayed-rectifier K⁺ channel from Mainen &
    Sejnowski (1996) — single activation gate ``n`` with power 1, essentially
    closed at rest (n_inf passes through 0.5 at V ≈ +4 mV), opens rapidly
    only above ~0 mV during the AP upstroke, and deactivates on a τ ~1–2 ms
    scale.  Designed to broaden the cortical pyramidal AP into the literature
    1.0–2.5 ms half-width band that the Pospischil n^4 form cannot reach on
    its own at any g_K (issue #311).

    The channel name is ``"Kv"`` (not ``"K"``) and the gating-variable name
    is ``"nKv"`` so the channel can be added via a preset's ``channels``
    list without colliding with a primary delayed-rectifier produced by
    ``k_channel_factory``.  In ``CORTICAL_PYRAMIDAL`` this M-S Kv is the
    only active K conductance — Pospischil's K factory is kept wired up
    for structural symmetry but pinned to ``g_K=0``.

    Rate constants are pre-scaled from the published 23 °C kinetics to 34 °C
    using Q10 = 2.3, so the channel matches the ``CORTICAL_PYRAMIDAL``
    preset's ``T_ref = 307.15 K``.  The reversal potential is computed
    dynamically via the Nernst equation for K⁺.

    Refs: Mainen, Z.F. & Sejnowski, T.J. (1996) Nature 382:363; canonical
    kv.mod ModelDB kinetics.

    Args:
        g_max: Maximum conductance in mS/cm². Defaults to
            :data:`~patch_sim.constants.DEFAULT_G_MSKV`.

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the
        Mainen-Sejnowski Kv delayed-rectifier K⁺ channel.
    """
    return IonChannel(
        name="Kv",
        g_max=g_max,
        gating_variables=(
            GatingVariable(
                name="nKv",
                power=1,
                alpha=mainen_sejnowski_alpha_n,
                beta=mainen_sejnowski_beta_n,
            ),
        ),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )


#: m-gate activation V_½ (mV) for Mainen-Sejnowski Na (na.mod ``tha``).
MAINEN_SEJNOWSKI_NA_THA: float = -35.0

#: m-gate activation slope (mV) for Mainen-Sejnowski Na (na.mod ``qa``).
_MAINEN_SEJNOWSKI_NA_QA: float = 9.0

#: Pre-scaled m-gate forward-rate prefactor at 34 °C (= 0.182 / ms × 2.55).
_MAINEN_SEJNOWSKI_NA_RA: float = 0.182 * MAINEN_SEJNOWSKI_KV_PRESCALE

#: Pre-scaled m-gate backward-rate prefactor at 34 °C (= 0.124 / ms × 2.55).
_MAINEN_SEJNOWSKI_NA_RB: float = 0.124 * MAINEN_SEJNOWSKI_KV_PRESCALE

#: h-gate τ alpha-branch V_½ (mV) (na.mod ``thi1``).
_MAINEN_SEJNOWSKI_NA_THI1: float = -50.0

#: h-gate τ beta-branch V_½ (mV) (na.mod ``thi2``).
_MAINEN_SEJNOWSKI_NA_THI2: float = -75.0

#: h-gate τ slope (mV) (na.mod ``qi``).
_MAINEN_SEJNOWSKI_NA_QI: float = 5.0

#: Pre-scaled h-gate τ alpha-branch prefactor at 34 °C (= 0.024 / ms × 2.55).
_MAINEN_SEJNOWSKI_NA_RD: float = 0.024 * MAINEN_SEJNOWSKI_KV_PRESCALE

#: Pre-scaled h-gate τ beta-branch prefactor at 34 °C (= 0.0091 / ms × 2.55).
_MAINEN_SEJNOWSKI_NA_RG: float = 0.0091 * MAINEN_SEJNOWSKI_KV_PRESCALE

#: h-gate steady-state Boltzmann V_½ (mV) (na.mod ``thinf``).  Combined
#: with ``qinf=6.2`` gives h_inf ≈ 7×10⁻⁴ at V = -20 mV — far smaller
#: than the Pospischil h_inf ≈ 0.034 at the same voltage.  M-S Na
#: therefore carries only a tiny window current at sustained depolarised
#: V relative to Pospischil Na.
_MAINEN_SEJNOWSKI_NA_THINF: float = -65.0

#: h-gate steady-state Boltzmann slope (mV) (na.mod ``qinf``).
_MAINEN_SEJNOWSKI_NA_QINF: float = 6.2


def _ms_na_trap(V: float, th: float, prefactor: float, q: float) -> float:
    """Singularity-guarded ``a · (V - th) / (1 - exp(-(V - th) / q))``.

    Mirrors the ``trap0`` helper in the canonical na.mod entry.  Has a
    removable singularity at V = th; the L'Hôpital limit ``a · q`` is
    returned when ``|V - th| < SINGULARITY_THRESHOLD``.

    Args:
        V: Membrane voltage in mV.
        th: Half-voltage parameter in mV.
        prefactor: Rate-equation prefactor in 1/ms (already pre-scaled).
        q: Slope parameter in mV.

    Returns:
        Rate in 1/ms.
    """
    x = V - th
    if abs(x) < SINGULARITY_THRESHOLD:
        return prefactor * q
    return prefactor * x / (1.0 - safe_exp(-x / q))


def _mainen_sejnowski_alpha_m_impl(V: float, ca_i: float) -> float:
    """Forward rate for Mainen-Sejnowski Na activation gate m.

    ``α_m(V) = R_a · (V - tha) / (1 - exp(-(V - tha) / q_a))`` with
    R_a = 0.182 / ms × 2.55 (23 → 34 °C pre-scale) and tha = -35 mV.
    Has a removable singularity at V = -35 mV; the L'Hôpital limit
    R_a · q_a ≈ 4.179 / ms is returned within the threshold.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    return _ms_na_trap(
        V, MAINEN_SEJNOWSKI_NA_THA, _MAINEN_SEJNOWSKI_NA_RA, _MAINEN_SEJNOWSKI_NA_QA
    )


mainen_sejnowski_alpha_m = VoltageOnlyFn(_mainen_sejnowski_alpha_m_impl)


def _mainen_sejnowski_beta_m_impl(V: float, ca_i: float) -> float:
    """Backward rate for Mainen-Sejnowski Na activation gate m.

    Per the na.mod entry, β_m takes the same trap0 form with the voltage
    sign flipped: ``β_m(V) = trap0(-V, -tha, R_b, q_a)``.  Singularity at
    -V = -tha, i.e. V = -35 mV — same physical voltage as α_m's.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return _ms_na_trap(
        -V,
        -MAINEN_SEJNOWSKI_NA_THA,
        _MAINEN_SEJNOWSKI_NA_RB,
        _MAINEN_SEJNOWSKI_NA_QA,
    )


mainen_sejnowski_beta_m = VoltageOnlyFn(_mainen_sejnowski_beta_m_impl)


def _ms_na_h_inf(V: float) -> float:
    """Steady-state h gate value for Mainen-Sejnowski Na.

    ``h_inf(V) = 1 / (1 + exp((V - thinf) / qinf))`` with thinf = -65 mV
    and qinf = 6.2 mV.  Approaches 1 at hyperpolarised voltages (Na
    available) and 0 at depolarised voltages (Na inactivated).

    Args:
        V: Membrane voltage in mV.

    Returns:
        Steady-state h in [0, 1].
    """
    return 1.0 / (
        1.0 + safe_exp((V - _MAINEN_SEJNOWSKI_NA_THINF) / _MAINEN_SEJNOWSKI_NA_QINF)
    )


def _ms_na_h_tau(V: float) -> float:
    """Time constant of the h gate for Mainen-Sejnowski Na.

    Computed from the alpha/beta sum of the trap0 functions evaluated at
    thi1 (alpha branch, voltage-positive) and thi2 (beta branch,
    voltage-negative); see na.mod.  Returns τ in ms.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Time constant in ms.
    """
    a = _ms_na_trap(
        V, _MAINEN_SEJNOWSKI_NA_THI1, _MAINEN_SEJNOWSKI_NA_RD, _MAINEN_SEJNOWSKI_NA_QI
    )
    b = _ms_na_trap(
        -V,
        -_MAINEN_SEJNOWSKI_NA_THI2,
        _MAINEN_SEJNOWSKI_NA_RG,
        _MAINEN_SEJNOWSKI_NA_QI,
    )
    return 1.0 / (a + b)


def _mainen_sejnowski_alpha_h_impl(V: float, ca_i: float) -> float:
    """Forward rate for Mainen-Sejnowski Na inactivation gate h.

    Derived as ``α_h = h_inf / τ_h`` from the M-S 1996 decoupled
    inf/tau formulation.  Mirrors the convention used by
    :func:`_alpha_hr_impl` for INaR.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    return _ms_na_h_inf(V) / _ms_na_h_tau(V)


mainen_sejnowski_alpha_h = VoltageOnlyFn(_mainen_sejnowski_alpha_h_impl)


def _mainen_sejnowski_beta_h_impl(V: float, ca_i: float) -> float:
    """Backward rate for Mainen-Sejnowski Na inactivation gate h.

    Derived as ``β_h = (1 - h_inf) / τ_h``.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return (1.0 - _ms_na_h_inf(V)) / _ms_na_h_tau(V)


mainen_sejnowski_beta_h = VoltageOnlyFn(_mainen_sejnowski_beta_h_impl)


def make_mainen_sejnowski_na_channel(g_max: float) -> IonChannel:
    """Create the Mainen-Sejnowski cortical pyramidal fast Na⁺ channel.

    Implements the fast Na⁺ kinetics from Mainen & Sejnowski (1996) via
    activation gate ``m`` (power 3) and inactivation gate ``h`` (power 1).
    Distinctive features relative to the Pospischil Na channel:

    * **Stronger steady-state inactivation at depolarised V** — h_inf
      passes through 0.5 at V = -65 mV (vs -56 for Pospischil) and
      collapses to ~7×10⁻⁴ at V = -20 mV (vs ~0.034 for Pospischil), so
      the steady-state Na window current at sustained depolarised V is
      ~50× smaller than Pospischil's.

    Currently no preset wires this Na factory in — it is provided as a
    building block alongside :func:`make_mainen_sejnowski_kv_channel`
    (issue #311) for any future cortical-style preset that needs the
    weaker depolarised-V window current.

    Rate constants are pre-scaled from the published 23 °C kinetics to
    34 °C using Q10 = 2.3, matching the cortical pyramidal preset's
    ``T_ref = 307.15 K``.  Reversal potential is computed dynamically via
    the Nernst equation for Na⁺.

    Refs: Mainen, Z.F. & Sejnowski, T.J. (1996) Nature 382:363; canonical
    na.mod ModelDB kinetics.

    Args:
        g_max: Maximum conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the
        Mainen-Sejnowski cortical pyramidal Na⁺ channel.
    """
    return IonChannel(
        name="Na",
        g_max=g_max,
        gating_variables=(
            GatingVariable(
                name="m",
                power=3,
                alpha=mainen_sejnowski_alpha_m,
                beta=mainen_sejnowski_beta_m,
            ),
            GatingVariable(
                name="h",
                power=1,
                alpha=mainen_sejnowski_alpha_h,
                beta=mainen_sejnowski_beta_h,
            ),
        ),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )
