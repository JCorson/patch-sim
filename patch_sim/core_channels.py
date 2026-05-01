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
from .constants import DEFAULT_G_MSKV
from .rates import VoltageOnlyFn
from .utils import safe_exp

__all__ = [
    # Module constant
    "SINGULARITY_THRESHOLD",
    # Classic Hodgkin-Huxley rate functions and factories
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
    # Pospischil et al. (2008) cortical RS
    "POSPISCHIL_VT",
    "pospischil_alpha_m",
    "pospischil_beta_m",
    "pospischil_alpha_h",
    "pospischil_beta_h",
    "pospischil_alpha_n",
    "pospischil_beta_n",
    "make_pospischil_na_channel",
    "make_pospischil_k_channel",
    # Mainen & Sejnowski (1996) cortical pyramidal Na⁺/Kv
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
    # McCormick & Huguenard (1992) thalamic relay
    "THALAMIC_RELAY_VT",
    "thalamic_relay_alpha_m",
    "thalamic_relay_beta_m",
    "thalamic_relay_alpha_h",
    "thalamic_relay_beta_h",
    "thalamic_relay_alpha_n",
    "thalamic_relay_beta_n",
    "make_thalamic_relay_na_channel",
    "make_thalamic_relay_k_channel",
    # Huguenard & Prince (1992) / Destexhe et al. (1994) TRN
    "TRN_VT",
    "trn_alpha_m",
    "trn_beta_m",
    "trn_alpha_h",
    "trn_beta_h",
    "trn_alpha_n",
    "trn_beta_n",
    "make_trn_na_channel",
    "make_trn_k_channel",
    # De Schutter & Bower (1994) Purkinje
    "PURKINJE_VT",
    "purkinje_alpha_m",
    "purkinje_beta_m",
    "purkinje_alpha_h",
    "purkinje_beta_h",
    "purkinje_alpha_n",
    "purkinje_beta_n",
    "make_purkinje_na_channel",
    "make_purkinje_k_channel",
    # Otsuka et al. (2004) STN
    "make_stn_na_channel",
    "make_stn_k_channel",
    # Komendantov et al. (2004) midbrain dopaminergic (SNc/VTA)
    "DOPAMINERGIC_VT",
    "dopaminergic_alpha_m",
    "dopaminergic_beta_m",
    "dopaminergic_alpha_h",
    "dopaminergic_beta_h",
    "dopaminergic_alpha_n",
    "dopaminergic_beta_n",
    "make_dopaminergic_na_channel",
    "make_dopaminergic_k_channel",
]

# Threshold for detecting near-singularity in GHK-style rate equations.
# When the denominator voltage term is within this tolerance of zero, the
# L'Hôpital limit is used instead to avoid division by zero.
SINGULARITY_THRESHOLD = 1e-6


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


# ---------------------------------------------------------------------------
# Shared Traub-Miles (1991) analytical rate-function helpers
#
# All four cell-type-specific families below (Pospischil, Thalamic Relay, TRN,
# Purkinje) use the same six Traub-Miles analytical forms — they differ only in
# the voltage threshold parameter ``vt``.  These private helpers encode each
# form once; the public wrappers delegate to them with the appropriate VT
# constant.
# ---------------------------------------------------------------------------


def _traub_miles_alpha_m(V: float, vt: float) -> float:
    """Traub-Miles forward rate for Na⁺ activation gate m.

    Has a removable singularity at V = vt + 13; the L'Hôpital limit (1.28)
    is returned when ``|V − vt − 13| < SINGULARITY_THRESHOLD``.

    Args:
        V: Membrane voltage in mV.
        vt: Cell-type voltage threshold in mV.

    Returns:
        Forward rate in 1/ms.
    """
    x = V - vt - 13
    if abs(x) < SINGULARITY_THRESHOLD:
        return 1.28
    return -0.32 * x / (safe_exp(-x / 4) - 1)


def _traub_miles_beta_m(V: float, vt: float) -> float:
    """Traub-Miles backward rate for Na⁺ activation gate m.

    Has a removable singularity at V = vt + 40; the L'Hôpital limit (1.4)
    is returned when ``|V − vt − 40| < SINGULARITY_THRESHOLD``.

    Args:
        V: Membrane voltage in mV.
        vt: Cell-type voltage threshold in mV.

    Returns:
        Backward rate in 1/ms.
    """
    x = V - vt - 40
    if abs(x) < SINGULARITY_THRESHOLD:
        return 1.4
    return 0.28 * x / (safe_exp(x / 5) - 1)


def _traub_miles_alpha_h(V: float, vt: float) -> float:
    """Traub-Miles forward rate for Na⁺ inactivation gate h.

    Args:
        V: Membrane voltage in mV.
        vt: Cell-type voltage threshold in mV.

    Returns:
        Forward rate in 1/ms.
    """
    return 0.128 * safe_exp(-(V - vt - 17) / 18)


def _traub_miles_beta_h(V: float, vt: float) -> float:
    """Traub-Miles backward rate for Na⁺ inactivation gate h.

    Args:
        V: Membrane voltage in mV.
        vt: Cell-type voltage threshold in mV.

    Returns:
        Backward rate in 1/ms.
    """
    return 4.0 / (1 + safe_exp(-(V - vt - 40) / 5))


def _traub_miles_alpha_n(V: float, vt: float) -> float:
    """Traub-Miles forward rate for K⁺ delayed-rectifier activation gate n.

    Has a removable singularity at V = vt + 15; the L'Hôpital limit (0.16)
    is returned when ``|V − vt − 15| < SINGULARITY_THRESHOLD``.

    Args:
        V: Membrane voltage in mV.
        vt: Cell-type voltage threshold in mV.

    Returns:
        Forward rate in 1/ms.
    """
    x = V - vt - 15
    if abs(x) < SINGULARITY_THRESHOLD:
        return 0.16
    return -0.032 * x / (safe_exp(-x / 5) - 1)


def _traub_miles_beta_n(V: float, vt: float) -> float:
    """Traub-Miles backward rate for K⁺ delayed-rectifier activation gate n.

    Args:
        V: Membrane voltage in mV.
        vt: Cell-type voltage threshold in mV.

    Returns:
        Backward rate in 1/ms.
    """
    return 0.5 * safe_exp(-(V - vt - 10) / 40)


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


def _pospischil_alpha_m_impl(V: float, ca_i: float) -> float:
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
    return _traub_miles_alpha_m(V, POSPISCHIL_VT)


pospischil_alpha_m = VoltageOnlyFn(_pospischil_alpha_m_impl)


def _pospischil_beta_m_impl(V: float, ca_i: float) -> float:
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
    return _traub_miles_beta_m(V, POSPISCHIL_VT)


pospischil_beta_m = VoltageOnlyFn(_pospischil_beta_m_impl)


def _pospischil_alpha_h_impl(V: float, ca_i: float) -> float:
    """Forward rate for Pospischil Na⁺ inactivation gate h.

    Adopted from Traub & Miles (1991).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    return _traub_miles_alpha_h(V, POSPISCHIL_VT)


pospischil_alpha_h = VoltageOnlyFn(_pospischil_alpha_h_impl)


def _pospischil_beta_h_impl(V: float, ca_i: float) -> float:
    """Backward rate for Pospischil Na⁺ inactivation gate h.

    Adopted from Traub & Miles (1991).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return _traub_miles_beta_h(V, POSPISCHIL_VT)


pospischil_beta_h = VoltageOnlyFn(_pospischil_beta_h_impl)


def _pospischil_alpha_n_impl(V: float, ca_i: float) -> float:
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
    return _traub_miles_alpha_n(V, POSPISCHIL_VT)


pospischil_alpha_n = VoltageOnlyFn(_pospischil_alpha_n_impl)


def _pospischil_beta_n_impl(V: float, ca_i: float) -> float:
    """Backward rate for Pospischil K⁺ delayed-rectifier activation gate n.

    Adopted from Traub & Miles (1991).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return _traub_miles_beta_n(V, POSPISCHIL_VT)


pospischil_beta_n = VoltageOnlyFn(_pospischil_beta_n_impl)


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
# Mainen & Sejnowski (1996) cortical pyramidal Kv delayed rectifier
#
# Source: Mainen, Z.F. & Sejnowski, T.J. (1996) Influence of dendritic structure
# on firing pattern in model neocortical neurons.  Nature 382:363–366.
# Canonical kinetic parameters from the kv.mod ModelDB entry (e.g. accession
# 2488), which expresses a high-threshold delayed-rectifier (rate-function
# singularity at V = +25 mV; n_inf 50% point at V ≈ +4 mV) widely used in
# cortical pyramidal cell models.
#
# Used by ``CORTICAL_PYRAMIDAL`` as the sole delayed rectifier (replacing
# Pospischil Kd) to broaden the AP half-width into the literature 1.0–2.5 ms
# band (McCormick et al. 1985), where the Pospischil Traub-Miles n^4 form
# alone caps half-width at ~0.5 ms because τ_n ≈ 0.3 ms near AP peak
# (issue #311).  Mainen-Sejnowski Kv is closed at rest, opens rapidly only
# on the upstroke, and deactivates on a τ ~1–2 ms scale.
#
# This channel is named ``"Kv"`` (not ``"K"``) and is wired in via the
# preset's ``channels`` list rather than its ``k_channel_factory``.  The
# ``k_channel_factory`` remains pointed at ``make_pospischil_k_channel``
# but with ``g_K=0`` so the Pospischil delayed rectifier is structurally
# present (consistent with FSI / CA1 / other Pospischil-Na presets) but
# contributes no current — keeping the cortical preset's na/k factory
# pair conceptually aligned with the rest of the family while the active
# K conductance comes entirely from M-S Kv.
#
# Temperature pre-scale: Mainen & Sejnowski measured at 23 °C with Q10 = 2.3.
# The cortical pyramidal preset uses T_ref = 307.15 K (34 °C, Pospischil
# reference) so the rate constants are pre-scaled from 23 °C to 34 °C with
# the M-S Q10 = 2.3 (factor 2.3^((34-23)/10) = 2.3^1.1 ≈ 2.55), keeping
# Pospischil Na (34 °C ref) and M-S Kv compatible under one shared T_ref.
# ---------------------------------------------------------------------------

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


def _thalamic_relay_alpha_m_impl(V: float, ca_i: float) -> float:
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
    return _traub_miles_alpha_m(V, THALAMIC_RELAY_VT)


thalamic_relay_alpha_m = VoltageOnlyFn(_thalamic_relay_alpha_m_impl)


def _thalamic_relay_beta_m_impl(V: float, ca_i: float) -> float:
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
    return _traub_miles_beta_m(V, THALAMIC_RELAY_VT)


thalamic_relay_beta_m = VoltageOnlyFn(_thalamic_relay_beta_m_impl)


def _thalamic_relay_alpha_h_impl(V: float, ca_i: float) -> float:
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
    return _traub_miles_alpha_h(V, THALAMIC_RELAY_VT)


thalamic_relay_alpha_h = VoltageOnlyFn(_thalamic_relay_alpha_h_impl)


def _thalamic_relay_beta_h_impl(V: float, ca_i: float) -> float:
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
    return _traub_miles_beta_h(V, THALAMIC_RELAY_VT)


thalamic_relay_beta_h = VoltageOnlyFn(_thalamic_relay_beta_h_impl)


def _thalamic_relay_alpha_n_impl(V: float, ca_i: float) -> float:
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
    return _traub_miles_alpha_n(V, THALAMIC_RELAY_VT)


thalamic_relay_alpha_n = VoltageOnlyFn(_thalamic_relay_alpha_n_impl)


def _thalamic_relay_beta_n_impl(V: float, ca_i: float) -> float:
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
    return _traub_miles_beta_n(V, THALAMIC_RELAY_VT)


thalamic_relay_beta_n = VoltageOnlyFn(_thalamic_relay_beta_n_impl)


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


def _purkinje_alpha_m_impl(V: float, ca_i: float) -> float:
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
    return _traub_miles_alpha_m(V, PURKINJE_VT)


purkinje_alpha_m = VoltageOnlyFn(_purkinje_alpha_m_impl)


def _purkinje_beta_m_impl(V: float, ca_i: float) -> float:
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
    return _traub_miles_beta_m(V, PURKINJE_VT)


purkinje_beta_m = VoltageOnlyFn(_purkinje_beta_m_impl)


def _purkinje_alpha_h_impl(V: float, ca_i: float) -> float:
    """Forward rate for Purkinje Na⁺ inactivation gate h.

    Traub-Miles form parameterised for cerebellar Purkinje cells (VT = −58 mV).

    Reference: De Schutter & Bower (1994), J. Neurophysiol. 71:375.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    return _traub_miles_alpha_h(V, PURKINJE_VT)


purkinje_alpha_h = VoltageOnlyFn(_purkinje_alpha_h_impl)


def _purkinje_beta_h_impl(V: float, ca_i: float) -> float:
    """Backward rate for Purkinje Na⁺ inactivation gate h.

    Traub-Miles form parameterised for cerebellar Purkinje cells (VT = −58 mV).

    Reference: De Schutter & Bower (1994), J. Neurophysiol. 71:375.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return _traub_miles_beta_h(V, PURKINJE_VT)


purkinje_beta_h = VoltageOnlyFn(_purkinje_beta_h_impl)


def _purkinje_alpha_n_impl(V: float, ca_i: float) -> float:
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
    return _traub_miles_alpha_n(V, PURKINJE_VT)


purkinje_alpha_n = VoltageOnlyFn(_purkinje_alpha_n_impl)


def _purkinje_beta_n_impl(V: float, ca_i: float) -> float:
    """Backward rate for Purkinje K⁺ delayed-rectifier activation gate n.

    Traub-Miles form parameterised for cerebellar Purkinje cells (VT = −58 mV).

    Reference: De Schutter & Bower (1994), J. Neurophysiol. 71:375.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return _traub_miles_beta_n(V, PURKINJE_VT)


purkinje_beta_n = VoltageOnlyFn(_purkinje_beta_n_impl)


def make_purkinje_na_channel(g_max: float) -> IonChannel:
    """Create the cerebellar Purkinje fast sodium channel (Na⁺).

    Uses Traub-Miles kinetics with VT = −58 mV to match the somatic NaF
    activation threshold of mammalian cerebellar Purkinje neurons recorded
    by De Schutter & Bower (1994) at 32 °C.

    Intended as the ``na_channel_factory`` of the Purkinje preset.  Compared
    with the default HH52 Na⁺ channel (fitted to squid axon at 22 °C), the
    Traub-Miles form with VT = −58 mV places the α_m singularity (and
    approximate activation inflection) near −45 mV and prevents the ~5.2×
    Q10 overcorrection that caused premature Na⁺ inactivation.

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


def _stn_alpha_n_impl(V: float, ca_i: float) -> float:
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


_stn_alpha_n = VoltageOnlyFn(_stn_alpha_n_impl)


def _stn_beta_n_impl(V: float, ca_i: float) -> float:
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


_stn_beta_n = VoltageOnlyFn(_stn_beta_n_impl)


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


# ---------------------------------------------------------------------------
# Komendantov et al. (2004) midbrain dopaminergic (SNc/VTA) Na⁺/K⁺ rate functions
#
# Primary source:
#   Komendantov, A.O., Komendantova, O.G., Johnson, S.W. & Canavier, C.C.
#   (2004) A modeling study suggests complementary roles for GABAA and NMDA
#   receptors and the SK channel in regulating the firing pattern in midbrain
#   dopamine neurons. J. Neurophysiol. 91:346–357.
#
# Kinetic basis:
#   Canavier, C.C. (1999) Sodium dynamics underlying burst firing and putative
#   mechanisms for the regulation of the firing pattern in midbrain dopamine
#   neurons: a computational approach. J. Comput. Neurosci. 6:49–69.
#
# Kinetics reported for somatic SNc DA neurons; recording temperature ~35 °C.
#
# The Canavier (1999) / Komendantov (2004) Na⁺ rate functions are:
#   α_m = 0.32*(V+54)/(1−exp(−(V+54)/4))
#   β_m = 0.28*(V+27)/(exp((V+27)/5)−1)
#   α_n = 0.032*(V+52)/(1−exp(−(V+52)/5))
# These are algebraically identical to the Traub-Miles form with VT = −67 mV:
#   α_m(V, VT) = −0.32*(V−VT−13)/(exp(−(V−VT−13)/4)−1)
# because VT+13 = −54 → VT = −67.  The same VT applies to all six rate
# functions.  VT = −67 mV produces m_inf ≈ 5.6% at −60 mV, which is the
# window Na⁺ current that allows Ih-driven post-hyperpolarization rebound
# spiking observed in SNc DA neurons.
# ---------------------------------------------------------------------------

#: Voltage threshold parameter (mV) for midbrain dopaminergic (SNc/VTA) cells.
#: Derived from the Canavier (1999) / Komendantov (2004) rate functions which
#: use α_m = 0.32*(V+54)/..., equivalent to Traub-Miles with VT+13 = −54 mV,
#: i.e. VT = −67 mV.  This VT gives m_inf ≈ 5.6% at the −60 mV resting
#: potential — the window Na⁺ current that supports Ih-driven rebound spiking
#: observed in SNc DA neurons in slice (Wilson & Callaway 2000).
DOPAMINERGIC_VT: float = -67.0


def _dopaminergic_alpha_m_impl(V: float, ca_i: float) -> float:
    """Forward rate for dopaminergic Na⁺ activation gate m.

    Traub-Miles form parameterised for midbrain dopaminergic (SNc/VTA) cells
    (VT = −67 mV).  Has a removable singularity at V = VT + 13 = −54 mV; the
    L'Hôpital limit (1.28) is returned when
    ``|V − VT − 13| < SINGULARITY_THRESHOLD``.

    Algebraically equivalent to α_m = 0.32*(V+54)/(1−exp(−(V+54)/4)) from
    Canavier (1999) / Komendantov et al. (2004).

    Reference: Canavier (1999), J. Comput. Neurosci. 6:49;
    Komendantov et al. (2004), J. Neurophysiol. 91:346.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    return _traub_miles_alpha_m(V, DOPAMINERGIC_VT)


dopaminergic_alpha_m = VoltageOnlyFn(_dopaminergic_alpha_m_impl)


def _dopaminergic_beta_m_impl(V: float, ca_i: float) -> float:
    """Backward rate for dopaminergic Na⁺ activation gate m.

    Traub-Miles form parameterised for midbrain dopaminergic (SNc/VTA) cells
    (VT = −67 mV).  Has a removable singularity at V = VT + 40 = −27 mV; the
    L'Hôpital limit (1.4) is returned when
    ``|V − VT − 40| < SINGULARITY_THRESHOLD``.

    Algebraically equivalent to β_m = 0.28*(V+27)/(exp((V+27)/5)−1) from
    Canavier (1999) / Komendantov et al. (2004).

    Reference: Canavier (1999), J. Comput. Neurosci. 6:49;
    Komendantov et al. (2004), J. Neurophysiol. 91:346.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return _traub_miles_beta_m(V, DOPAMINERGIC_VT)


dopaminergic_beta_m = VoltageOnlyFn(_dopaminergic_beta_m_impl)


def _dopaminergic_alpha_h_impl(V: float, ca_i: float) -> float:
    """Forward rate for dopaminergic Na⁺ inactivation gate h.

    Traub-Miles form parameterised for midbrain dopaminergic (SNc/VTA) cells
    (VT = −67 mV).

    Reference: Canavier (1999), J. Comput. Neurosci. 6:49;
    Komendantov et al. (2004), J. Neurophysiol. 91:346.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    return _traub_miles_alpha_h(V, DOPAMINERGIC_VT)


dopaminergic_alpha_h = VoltageOnlyFn(_dopaminergic_alpha_h_impl)


def _dopaminergic_beta_h_impl(V: float, ca_i: float) -> float:
    """Backward rate for dopaminergic Na⁺ inactivation gate h.

    Traub-Miles form parameterised for midbrain dopaminergic (SNc/VTA) cells
    (VT = −67 mV).

    Reference: Canavier (1999), J. Comput. Neurosci. 6:49;
    Komendantov et al. (2004), J. Neurophysiol. 91:346.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return _traub_miles_beta_h(V, DOPAMINERGIC_VT)


dopaminergic_beta_h = VoltageOnlyFn(_dopaminergic_beta_h_impl)


def _dopaminergic_alpha_n_impl(V: float, ca_i: float) -> float:
    """Forward rate for dopaminergic K⁺ delayed-rectifier activation gate n.

    Traub-Miles form parameterised for midbrain dopaminergic (SNc/VTA) cells
    (VT = −67 mV).  Has a removable singularity at V = VT + 15 = −52 mV; the
    L'Hôpital limit (0.16) is returned when
    ``|V − VT − 15| < SINGULARITY_THRESHOLD``.

    Algebraically equivalent to α_n = 0.032*(V+52)/(1−exp(−(V+52)/5)) from
    Canavier (1999) / Komendantov et al. (2004).

    Reference: Canavier (1999), J. Comput. Neurosci. 6:49;
    Komendantov et al. (2004), J. Neurophysiol. 91:346.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Forward rate in 1/ms.
    """
    return _traub_miles_alpha_n(V, DOPAMINERGIC_VT)


dopaminergic_alpha_n = VoltageOnlyFn(_dopaminergic_alpha_n_impl)


def _dopaminergic_beta_n_impl(V: float, ca_i: float) -> float:
    """Backward rate for dopaminergic K⁺ delayed-rectifier activation gate n.

    Traub-Miles form parameterised for midbrain dopaminergic (SNc/VTA) cells
    (VT = −67 mV).

    Reference: Canavier (1999), J. Comput. Neurosci. 6:49;
    Komendantov et al. (2004), J. Neurophysiol. 91:346.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (accepted but ignored).

    Returns:
        Backward rate in 1/ms.
    """
    return _traub_miles_beta_n(V, DOPAMINERGIC_VT)


dopaminergic_beta_n = VoltageOnlyFn(_dopaminergic_beta_n_impl)


def make_dopaminergic_na_channel(g_max: float) -> IonChannel:
    """Create the midbrain dopaminergic fast sodium channel (Na⁺).

    Uses Traub-Miles kinetics with VT = −67 mV, directly equivalent to the
    Canavier (1999) / Komendantov et al. (2004) rate functions (α_m =
    0.32*(V+54)/..., α_n = 0.032*(V+52)/...) for SNc DA neurons at ~35 °C.

    Intended as the ``na_channel_factory`` of the SNc Dopaminergic preset.
    Compared with the default HH52 Na⁺ channel (fitted to squid axon at
    22 °C), these kinetics give m_inf ≈ 5.6% at the −60 mV resting potential
    and prevent the ~5.2× Q10 overcorrection that distorts Na⁺ inactivation
    under the default reference temperature.

    Reference: Canavier (1999), J. Comput. Neurosci. 6:49;
    Komendantov et al. (2004), J. Neurophysiol. 91:346.
    Kinetics at ~35 °C — use T_ref = 308.15 K with this factory.

    Args:
        g_max: Maximum conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the
        dopaminergic fast Na⁺ channel.
    """
    return IonChannel(
        name="Na",
        g_max=g_max,
        gating_variables=(
            GatingVariable(
                name="m",
                power=3,
                alpha=dopaminergic_alpha_m,
                beta=dopaminergic_beta_m,
            ),
            GatingVariable(
                name="h",
                power=1,
                alpha=dopaminergic_alpha_h,
                beta=dopaminergic_beta_h,
            ),
        ),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )


def make_dopaminergic_k_channel(g_max: float) -> IonChannel:
    """Create the midbrain dopaminergic delayed-rectifier potassium channel (K⁺).

    Uses Traub-Miles kinetics with VT = −67 mV, directly equivalent to the
    Canavier (1999) / Komendantov et al. (2004) α_n = 0.032*(V+52)/...
    rate function for SNc DA neurons at ~35 °C.

    Intended as the ``k_channel_factory`` of the SNc Dopaminergic preset.

    Reference: Canavier (1999), J. Comput. Neurosci. 6:49;
    Komendantov et al. (2004), J. Neurophysiol. 91:346.
    Kinetics at ~35 °C — use T_ref = 308.15 K with this factory.

    Args:
        g_max: Maximum conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the
        dopaminergic delayed-rectifier K⁺ channel.
    """
    return IonChannel(
        name="K",
        g_max=g_max,
        gating_variables=(
            GatingVariable(
                name="n",
                power=4,
                alpha=dopaminergic_alpha_n,
                beta=dopaminergic_beta_n,
            ),
        ),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )
