"""Pospischil et al. (2008) cortical RS Na⁺/K⁺ channel factories.

Source: Pospischil M. et al. (2008) Minimal Hodgkin-Huxley type models for
different classes of cortical and thalamic neurons. Biol. Cybern. 99:427-441.
Rate functions adopted from Traub & Miles (1991) / Huguenard & McCormick
(1992); VT shifts all thresholds to match cortical RS neuron firing.
"""

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
    "POSPISCHIL_VT",
    "pospischil_alpha_m",
    "pospischil_beta_m",
    "pospischil_alpha_h",
    "pospischil_beta_h",
    "pospischil_alpha_n",
    "pospischil_beta_n",
    "make_nav12_channel",
    "make_nav11_channel",
    "make_pospischil_k_channel",
]

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


# ---------------------------------------------------------------------------
# Pospischil-base fast Na⁺ slow voltage-dependent inactivation gates,
# parameterised per Nav isoform.
# ---------------------------------------------------------------------------
#
# The Pospischil 2008 / Traub-Miles fast Na⁺ kinetics retain a small but
# non-zero h-availability at depolarised plateau voltages.  All neuronal
# Nav isoforms also undergo a second, slow voltage-dependent inactivation
# process on top of the fast h gate — Fleidervish & Gutnick (1996),
# J. Physiol. 493:83 first demonstrated it in cortical pyramidal cells —
# but the kinetics differ markedly between isoforms.
#
# Two gate parameter sets are defined here, used by the isoform-named
# fast-Na factories below:
#
#   * Nav1.2 (somatic in cortical / CA1 pyramidal): V½ = −50 mV, slope
#     8 mV, τ_scale = 200 ms, τ_floor = 20 ms. Matches Fleidervish &
#     Gutnick (1996) within rounding; the gate fully equilibrates within
#     ~1 s of sustained depolarisation, providing the depolarisation-block
#     escape route required by the cortical/CA1 presets (#327, #328).
#
#   * Nav1.1 (PV+ fast-spiking interneurons): V½ = −45 mV, slope 8 mV,
#     τ_scale = 50000 ms, τ_floor = 5000 ms — biologically much weaker
#     than Nav1.2 so the gate barely engages at the 100–500 Hz firing
#     rates typical of FSI.  Patel et al. 2015 (PLOS ONE 10:e0133485)
#     report Nav1.1 slow inactivation V½ ≈ −68 mV in HEK293; native
#     PV+ interneurons express Nav1.1 with β1/β2 subunits and lipid
#     environments that depolarise V½ by ~20 mV, putting the native
#     V½ in the −45 mV range used here.  τ_floor = 5000 ms (effectively
#     a near-constant gate over 1 s) ensures that per-spike inactivation
#     closure during a high-frequency train accumulates only a few
#     percent of the rest availability rather than collapsing the cell
#     after a handful of APs.  Captures the biological fact that Nav1.1
#     has slow inactivation while not dominating FSI behaviour.  Note:
#     the activation/fast-inactivation kinetics are still
#     Pospischil/Traub-Miles, not isoform-specific HEK fits.  A
#     higher-fidelity isoform-fitted overhaul is tracked separately.
#
# Distinct gate names ``sNa11`` / ``sNa12`` keep the simulation result
# schema unambiguous if both factories are ever used in the same network,
# and avoid collision with the other slow-inactivation gate names already
# in use: ``s`` (INaR activation), ``sNa`` (STN and Purkinje fast-Na slow
# inactivation; reused because the two presets never coexist on the same
# neuron), ``sNa_da`` (SNc dopaminergic fast-Na slow inactivation),
# ``sNaP`` (INaP slow inactivation), and ``sNaP_snc`` (SNc INaP slow
# inactivation).

_nav12_alpha_sNa, _nav12_beta_sNa = boltzmann_cosh_rates(
    half=-50.0,
    slope=8.0,
    tau_scale=200.0,
    tau_floor=20.0,
    inverted=True,
)

_nav11_alpha_sNa, _nav11_beta_sNa = boltzmann_cosh_rates(
    half=-45.0,
    slope=8.0,
    tau_scale=50000.0,
    tau_floor=5000.0,
    inverted=True,
)


def _make_pospischil_base_gates() -> tuple[GatingVariable, GatingVariable]:
    """Construct the shared Pospischil/Traub-Miles (m, h) fast-Na gates.

    Returns:
        Tuple of activation gate ``m`` (power 3) and fast-inactivation
        gate ``h`` (power 1) using Pospischil rate functions.
    """
    m_var = GatingVariable(
        name="m", power=3, alpha=pospischil_alpha_m, beta=pospischil_beta_m
    )
    h_var = GatingVariable(
        name="h", power=1, alpha=pospischil_alpha_h, beta=pospischil_beta_h
    )
    return m_var, h_var


def make_nav12_channel(g_max: float) -> IonChannel:
    """Create a Nav1.2-flavoured fast sodium channel (cortical pyramidal somatic).

    Composes Pospischil et al. (2008) Traub-Miles activation/fast-inactivation
    kinetics (gates ``m`` power 3 and ``h`` power 1, VT = −56.2 mV) with a
    Fleidervish-style slow voltage-dependent inactivation gate ``sNa12``
    (V½ = −50 mV, slope 8 mV, inverted Boltzmann; τ_scale = 200 ms /
    τ_floor = 20 ms).  At V = −70 mV (cortical rest) sNa12_inf ≈ 0.92; at
    V = −15 mV sNa12_inf ≈ 0.01.  The slow gate provides the
    depolarisation-block escape route required by cortical/CA1 presets
    (#327, #328) and matches Fleidervish & Gutnick's native cortical
    pyramidal recordings within rounding.

    The activation/fast-inactivation kinetics are Pospischil/Traub-Miles
    rather than isoform-specific Nav1.2 HEK fits — a fuller isoform-fitted
    overhaul is tracked as a follow-up.  The "Nav1.2" naming reflects the
    cell-type-functional intent (cortical pyramidal somatic Na⁺) rather
    than a strict molecular claim.

    The reversal potential is computed dynamically via the Nernst equation
    for Na⁺.

    References:
        - Pospischil et al. (2008), Biol. Cybern. 99:427 (cortical RS Na
          activation/fast-inactivation kinetics).
        - Fleidervish & Gutnick (1996), J. Physiol. 493:83 (slow Na
          inactivation, cortical pyramidal — primary source).
        - Mickus, Jung & Spruston (1999), Biophys. J. 76:846 (slow Na
          inactivation, CA1 pyramidal).

    Args:
        g_max: Maximum conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing a
        Nav1.2-flavoured cortical pyramidal somatic fast Na⁺ channel.
    """
    m_var, h_var = _make_pospischil_base_gates()
    s_var = GatingVariable(
        name="sNa12",
        power=1,
        alpha=_nav12_alpha_sNa,
        beta=_nav12_beta_sNa,
    )
    return IonChannel(
        name="Na",
        g_max=g_max,
        gating_variables=(m_var, h_var, s_var),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )


def make_nav11_channel(g_max: float) -> IonChannel:
    """Create a Nav1.1-flavoured fast sodium channel (PV+ fast-spiking interneuron).

    Composes Pospischil et al. (2008) Traub-Miles activation/fast-inactivation
    kinetics (gates ``m`` power 3 and ``h`` power 1, VT = −56.2 mV) with a
    weak Nav1.1-flavoured slow voltage-dependent inactivation gate
    ``sNa11`` (V½ = −45 mV, slope 8 mV, inverted Boltzmann; τ_scale =
    50000 ms / τ_floor = 5000 ms).  The slow τ floor (5 s) ensures the
    gate barely engages at the 100–500 Hz firing rates typical of FSI,
    capturing the biological fact that Nav1.1 has slow inactivation
    (Patel et al. 2015 report Nav1.1 slow inactivation V½ ≈ −68 mV in
    HEK293; native PV+ interneurons express Nav1.1 with β1/β2 subunits
    that depolarise V½ by ~20 mV) without dominating FSI behaviour.

    The activation/fast-inactivation kinetics are Pospischil/Traub-Miles
    rather than isoform-specific Nav1.1 HEK fits — a fuller isoform-fitted
    overhaul is tracked as a follow-up.  The "Nav1.1" naming reflects the
    cell-type-functional intent (FS-interneuron fast Na⁺) rather than a
    strict molecular claim.

    The reversal potential is computed dynamically via the Nernst equation
    for Na⁺.

    References:
        - Pospischil et al. (2008), Biol. Cybern. 99:427 (cortical RS Na
          activation/fast-inactivation kinetics).
        - Patel, Barbosa, Xiao & Cummins (2015), PLOS ONE 10:e0133485
          (Nav1.1 vs Nav1.6 isoform comparison; slow-inactivation V½).
        - Hu & Jonas (2014), Nat. Neurosci. 17:686 (Nav1.1 in PV+
          interneurons; sustained high-frequency firing).

    Args:
        g_max: Maximum conductance in mS/cm².

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing a
        Nav1.1-flavoured fast-spiking-interneuron fast Na⁺ channel.
    """
    m_var, h_var = _make_pospischil_base_gates()
    s_var = GatingVariable(
        name="sNa11",
        power=1,
        alpha=_nav11_alpha_sNa,
        beta=_nav11_beta_sNa,
    )
    return IonChannel(
        name="Na",
        g_max=g_max,
        gating_variables=(m_var, h_var, s_var),
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
