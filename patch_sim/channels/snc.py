"""SNc dopaminergic channel factories (Komendantov et al. 2004).

Primary source:
  Komendantov, A.O., Komendantova, O.G., Johnson, S.W. & Canavier, C.C.
  (2004) A modeling study suggests complementary roles for GABAA and NMDA
  receptors and the SK channel in regulating the firing pattern in midbrain
  dopamine neurons. J. Neurophysiol. 91:346–357.

Kinetic basis:
  Canavier, C.C. (1999) Sodium dynamics underlying burst firing and putative
  mechanisms for the regulation of the firing pattern in midbrain dopamine
  neurons: a computational approach. J. Comput. Neurosci. 6:49–69.

Includes SNc-specific INaP (Drion et al. 2011) and Cav1.3 (Putzier et al.
2009) factories that are cell-type-specific to the SNc Dopaminergic preset.
"""

from ..constants import DEFAULT_G_CAV13, DEFAULT_G_NAP_SNC
from ..electrochemistry import boltzmann_cosh_rates
from ..rates import VoltageOnlyFn
from ._traub_miles import (
    _traub_miles_alpha_h,
    _traub_miles_alpha_m,
    _traub_miles_alpha_n,
    _traub_miles_beta_h,
    _traub_miles_beta_m,
    _traub_miles_beta_n,
)
from .base import GatingVariable, IonChannel, IonSpecies, NernstSpec

__all__ = [
    "DOPAMINERGIC_VT",
    "dopaminergic_alpha_m",
    "dopaminergic_beta_m",
    "dopaminergic_alpha_h",
    "dopaminergic_beta_h",
    "dopaminergic_alpha_n",
    "dopaminergic_beta_n",
    "make_dopaminergic_na_channel",
    "make_dopaminergic_k_channel",
    "make_snc_inap_channel",
    "make_cav13_channel",
]

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


# ---------------------------------------------------------------------------
# Dopaminergic SNc fast Na⁺ slow voltage-dependent inactivation gate ``sNa_da``.
# ---------------------------------------------------------------------------
#
# Khaliq & Bean (2010) directly demonstrated cumulative slow inactivation of
# the fast Na⁺ current in SNc dopaminergic neurons; Tucker, Hagiwara &
# Williams (2012) confirmed the same in midbrain DA cells.  The Canavier
# (1999) / Komendantov et al. (2004) Traub-Miles kinetics carry only the
# fast h gate and therefore retain a residual h-tail under sustained
# depolarisation.  Adding the slow gate makes the cell biologically more
# accurate and guards against pathological depol-block plateaus.
#
# Parameters mirror the STN, Pospischil and Purkinje sNa gates: V½ = −50 mV
# (within the −47 mV midpoint reported by Khaliq & Bean 2010), slope 8 mV,
# inverted Boltzmann so the gate is open at hyperpolarised potentials;
# τ_scale = 200 ms / τ_floor = 20 ms.  At the SNc oscillator's hyperpolarised
# trough (≈ −75 mV) s_inf > 0.93; at v_rest = −55 mV s_inf ≈ 0.65 (lower
# margin than STN/Purkinje because SNc rests more depolarised); at the
# −15 mV plateau s_inf < 0.02 (residual h-tail abolished).
#
# Gate name ``sNa_da`` follows the SNc-specific ``pSNc`` namespacing already
# used by :func:`make_snc_inap_channel`; it avoids colliding with the bare
# ``sNa`` of the Purkinje / STN / Pospischil Na factories should a hypothetical
# preset ever combine them.
#
# Slow inactivation is always on here because ``make_dopaminergic_na_channel``
# is dedicated to the SNc Dopaminergic preset — there is no other caller
# whose calibration could be perturbed by enabling the gate.

_dopaminergic_alpha_sNa, _dopaminergic_beta_sNa = boltzmann_cosh_rates(
    half=-50.0,
    slope=8.0,
    tau_scale=200.0,
    tau_floor=20.0,
    inverted=True,
)


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

    The channel exposes three gates: activation ``m`` (power 3), fast
    inactivation ``h`` (power 1), and slow voltage-dependent inactivation
    ``sNa_da`` (power 1; V½ = −50 mV, slope 8 mV, inverted Boltzmann).  The
    slow gate is mostly available at hyperpolarised potentials and decays
    towards zero on sustained depolarisation, providing the escape route
    from depol-block plateaus reported in real SNc DA neurons (Khaliq &
    Bean 2010).  The column is named ``sNa_da`` rather than ``sNa`` to
    match the SNc-specific ``pSNc`` namespacing convention used by
    :func:`make_snc_inap_channel`.

    References:
        - Canavier (1999), J. Comput. Neurosci. 6:49; Komendantov et al.
          (2004), J. Neurophysiol. 91:346 — m, h, n kinetics at ~35 °C
          (use T_ref = 308.15 K with this factory).
        - Khaliq & Bean (2010), J. Neurosci. 30:7401 (slow Na inactivation in
          SNc dopaminergic neurons — primary source for the sNa_da gate).
        - Tucker, Hagiwara & Williams (2012), J. Neurophysiol. 108:2492
          (confirms slow Na inactivation in midbrain DA neurons).

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
            GatingVariable(
                name="sNa_da",
                power=1,
                alpha=_dopaminergic_alpha_sNa,
                beta=_dopaminergic_beta_sNa,
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


# ---------------------------------------------------------------------------
# INaP_SNc — SNc-specific persistent Na⁺ (Drion et al. 2011)
# ---------------------------------------------------------------------------
# Drion et al. 2011 (PLOS Comp Biol) reconciles Putzier 2009 (Cav1.3 essential)
# with Guzman/Surmeier 2009 (Cav1.3 dispensable, INaP essential): the two
# subthreshold negative-slope conductances are interchangeable drivers of SNc
# DA pacemaking, and an honest model needs both.  The Drion fit places SNc
# INaP V½ at −65 mV (well below firing threshold), so this channel carries a
# subthreshold ramp-up current rather than a near-threshold amplifier.
# This is structurally distinct from the Magistretti & Alonso 1999 entorhinal
# fit (V½ = −52.6 mV) used by ``make_inap_channel``, which is preserved for
# cortical/hippocampal presets.
#
# Activation gate ``pSNc`` is paired with an always-on slow inactivation gate
# ``sNaP_snc`` (V½ = −55 mV, slope 7 mV, inverted Boltzmann; τ_scale 200 ms /
# τ_floor 20 ms).  Magistretti & Alonso (1999) §"Slow inactivation" report
# the entorhinal INaP slow gate at V½ in the −47 to −54 mV range; the SNc
# Drion fit shifts INaP activation V½ ~12 mV more hyperpolarised (−65 mV
# vs −52.6 mV).  Applying the same shift strictly would place the slow gate
# at ≈ −57 mV; V½ = −55 mV is chosen as a round number just past the
# hyperpolarised edge of the Magistretti-Alonso band, preserving the
# activation/slow-inactivation overlap that underlies the INaP escape from
# sustained depolarisation.  Slow inactivation in SNc
# DA Na⁺ currents is documented by Khaliq & Bean (2010) and Tucker, Hagiwara
# & Williams (2012); the gate makes the cell biologically more accurate.
# Always on because ``make_snc_inap_channel`` is dedicated to the SNc
# Dopaminergic preset — there is no other caller whose calibration could be
# perturbed by enabling the gate.
#
# Gate name ``sNaP_snc`` mirrors the SNc-specific ``pSNc`` namespacing and
# avoids colliding with the bare ``sNaP`` of ``make_inap_channel`` should a
# hypothetical preset combine the two factories.
#
# Reference:
#   Drion, Massotte, Sepulchre, Seutin (2011), PLOS Comp Biol 7:e1002050
#     (activation kinetics).
#   Magistretti & Alonso (1999), J. Gen. Physiol. 114:491
#     (slow inactivation V½/slope band).
#   Khaliq & Bean (2010), J. Neurosci. 30:7401 (slow Na inactivation, SNc).
#   Tucker, Hagiwara & Williams (2012), J. Neurophysiol. 108:2492.
_alpha_p_snc, _beta_p_snc = boltzmann_cosh_rates(
    half=-65.0, slope=5.0, tau_scale=5.0, tau_floor=0.1
)

_alpha_sNaP_snc, _beta_sNaP_snc = boltzmann_cosh_rates(
    half=-55.0,
    slope=7.0,
    tau_scale=200.0,
    tau_floor=20.0,
    inverted=True,
)


def make_snc_inap_channel(
    g_max: float = DEFAULT_G_NAP_SNC,
) -> IonChannel:
    """Create an SNc-specific INaP (persistent Na⁺) channel.

    Drion et al. 2011 fit SNc DA INaP at V½ = −65 mV (slope k = 5 mV), well
    below firing threshold.  In this regime INaP carries a subthreshold
    ramp-up current that, together with Cav1.3, drives the inter-spike
    depolarisation in the Putzier+Drion reconciliation of SNc pacemaking.

    The channel exposes two gates: activation ``pSNc`` (power 1) and slow
    voltage-dependent inactivation ``sNaP_snc`` (power 1; V½ = −55 mV,
    slope 7 mV, inverted Boltzmann).  The slow gate is always on, mirroring
    the Khaliq & Bean (2010) and Tucker, Hagiwara & Williams (2012)
    observations of slow Na inactivation in SNc DA neurons.  Both gate
    columns are SNc-namespaced (``pSNc``, ``sNaP_snc``) so they do not
    collide with the ``p``/``sNaP`` gates of
    :func:`~patch_sim.channels.auxiliary.make_inap_channel`.

    The reversal potential is computed dynamically from the neuron's Na⁺
    concentrations using the Nernst equation.

    References:
        - Drion et al. (2011), PLOS Comp Biol 7:e1002050 (activation V½).
        - Magistretti & Alonso (1999), J. Gen. Physiol. 114:491 (slow
          inactivation V½/slope band).
        - Khaliq & Bean (2010), J. Neurosci. 30:7401 (slow Na inactivation
          in SNc DA — primary source for the sNaP_snc gate).
        - Tucker, Hagiwara & Williams (2012), J. Neurophysiol. 108:2492.

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~patch_sim.constants.DEFAULT_G_NAP_SNC`.

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the SNc INaP
        current.
    """
    p_var = GatingVariable(name="pSNc", power=1, alpha=_alpha_p_snc, beta=_beta_p_snc)
    s_var = GatingVariable(
        name="sNaP_snc",
        power=1,
        alpha=_alpha_sNaP_snc,
        beta=_beta_sNaP_snc,
    )
    return IonChannel(
        name="NaP_SNc",
        g_max=g_max,
        gating_variables=(p_var, s_var),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )


# ---------------------------------------------------------------------------
# Cav1.3 — LVA L-type Ca²⁺ channel (Putzier 2009 SNc pacemaker driver)
# ---------------------------------------------------------------------------
# Cav1.3 (α1D) is the low-voltage-activated isoform of the L-type Ca²⁺ family.
# Activation midpoint sits well below the canonical Cav1.2 ICaL, so a small
# persistent window current flows at sub-threshold voltages near the SNc DA
# pacemaker potential.  This window current is the slow inward driver of the
# inter-spike depolarisation that ramps from the post-AP AHP back to threshold
# (Putzier et al. 2009).  Inactivation is slow (τ ≈ 200 ms) and incomplete, so
# the residual window persists through the inter-spike interval.
#
# Activation kinetics use Putzier et al. 2009 dynamic-clamp Boltzmann fit:
# half = −31.1 mV, slope k = 5.35 mV.  Putzier's central thesis is that this
# specific V½ is what drives pacemaking — shifting V½ by ±20 mV abolishes
# rescue, demonstrating that the negative-slope conductance window at this V½
# is the load-bearing mechanism (not Ca²⁺ flux per se).
#
# References:
#   Putzier, Kullmann, Roeper (2009), J. Neurosci. 29:15414 — Cav1.3 V½
#       drives SNc pacemaking; dynamic-clamp Boltzmann fit V½ = -31.1 mV,
#       k = 5.35 mV
#   Xu & Lipscombe (2001), J. Neurosci. 21:5944 — Cav1.3 vs Cav1.2 kinetics
#   Wolfart et al. (2001), J. Neurosci. 21:3443 — SNc Cav1.3/SK pairing
_alpha_dL13, _beta_dL13 = boltzmann_cosh_rates(
    half=-31.1, slope=5.35, tau_scale=2.0, tau_floor=0.5
)
_alpha_fL13, _beta_fL13 = boltzmann_cosh_rates(
    half=-50.0, slope=-8.0, tau_scale=200.0, tau_floor=20.0
)


def make_cav13_channel(
    g_max: float = DEFAULT_G_CAV13,
) -> IonChannel:
    """Create a Cav1.3 (LVA L-type Ca²⁺) ion channel.

    Cav1.3 is the low-voltage-activated isoform of the neuronal L-type Ca²⁺
    family.  Activation follows the Putzier et al. 2009 dynamic-clamp
    Boltzmann fit (half = −31.1 mV, slope k = 5.35 mV) — the specific V½ at
    which a persistent sub-threshold window current drives SNc autonomous
    pacemaking.  Putzier's central result is that shifting V½ by ±20 mV
    abolishes pacemaker rescue, so this V½ is load-bearing.

    Inactivation is slow (τ ≈ 200 ms) and incomplete (half ≈ −50 mV, slope
    −8 mV), so the residual window persists through the full inter-spike
    interval.  Two gating variables: ``dL13`` (activation, power 1) and
    ``fL13`` (inactivation, power 1).  Activation power is 1 (not 2 as in
    HVA Cav1.2) — Putzier-class models treat the LVA L-type as a single-gate
    activator since the steepness is set by the Boltzmann slope, not gate
    multiplication.

    Because Cav1.3 carries Ca²⁺, ``carries_calcium=True`` is set so the
    simulation loop accumulates its contribution for the Ca²⁺ ODE and
    :meth:`~patch_sim.channels.IonChannel.compute_current` uses live ``ca_i``
    for a dynamic E_Ca.

    The reversal potential is computed dynamically from the neuron's Ca²⁺
    concentrations using the Nernst equation.

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~patch_sim.constants.DEFAULT_G_CAV13`.

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the Cav1.3
        current.
    """
    d_var = GatingVariable(name="dL13", power=1, alpha=_alpha_dL13, beta=_beta_dL13)
    f_var = GatingVariable(name="fL13", power=1, alpha=_alpha_fL13, beta=_beta_fL13)
    return IonChannel(
        name="Cav1.3",
        g_max=g_max,
        gating_variables=(d_var, f_var),
        reversal_spec=NernstSpec(IonSpecies.CALCIUM),
        carries_calcium=True,
    )
