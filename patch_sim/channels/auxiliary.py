"""Cell-agnostic auxiliary ion channel factories.

These channels can be added to any neuron model's ``channels`` tuple.
They are cell-type-agnostic (not tuned for a specific neuron type).
"""

from ..constants import (
    DEFAULT_G_ICAL,
    DEFAULT_G_ICAN,
    DEFAULT_G_ICAT,
    DEFAULT_G_IH,
    DEFAULT_G_IKA,
    DEFAULT_G_IKCA,
    DEFAULT_G_IKIR,
    DEFAULT_G_IKV31,
    DEFAULT_G_IM,
    DEFAULT_G_KATP,
    DEFAULT_G_NAP,
    DEFAULT_G_NAR,
    DEFAULT_G_SK,
    DEFAULT_IH_P_NA,
)
from ..electrochemistry import boltzmann_cosh_rates
from ..rates import CalciumDependentFn, VoltageOnlyFn
from ..utils import safe_cosh, safe_exp
from .base import (
    GatingVariable,
    GoldmanSpec,
    IonChannel,
    IonSpecies,
    NernstSpec,
)

__all__ = [
    "make_ika_channel",
    "make_ikv31_channel",
    "make_ih_channel",
    "make_inap_channel",
    "make_inar_channel",
    "make_im_channel",
    "make_katp_channel",
    "make_ikir_channel",
    "make_ikca_channel",
    "make_ical_channel",
    "make_icat_channel",
    "make_ican_channel",
    "make_sk_channel",
]


def _alpha_r_impl(V: float, ca_i: float) -> float:
    """Forward rate for Ih gating variable r (Destexhe-style HCN kinetics).

    The Ih current is activated by hyperpolarization; alpha_r increases as
    membrane voltage becomes more negative.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

    Returns:
        Forward rate alpha_r in 1/ms.
    """
    return safe_exp(-14.59 - 0.086 * V)


_alpha_r = VoltageOnlyFn(_alpha_r_impl)


def _beta_r_impl(V: float, ca_i: float) -> float:
    """Backward rate for Ih gating variable r (Destexhe-style HCN kinetics).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

    Returns:
        Backward rate beta_r in 1/ms.
    """
    return safe_exp(-1.87 + 0.0701 * V)


_beta_r = VoltageOnlyFn(_beta_r_impl)


_SINGULARITY_TOL: float = 1e-7


def _alpha_a_impl(V: float, ca_i: float) -> float:
    """Forward rate for IKa activation gating variable a (Traub & Miles 1991).

    Uses a Boltzmann-style rate shifted to the absolute voltage convention
    (-65 mV resting).  A singularity guard replaces the 0/0 form at
    V = -51.9 mV with the analytic limit.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

    Returns:
        Forward rate alpha_a in 1/ms.
    """
    x = -51.9 - V
    if abs(x) < _SINGULARITY_TOL:
        return 0.02 * 10.0  # L'Hôpital limit: coefficient * denominator scale
    return 0.02 * x / (safe_exp(x / 10.0) - 1.0)


_alpha_a = VoltageOnlyFn(_alpha_a_impl)


def _beta_a_impl(V: float, ca_i: float) -> float:
    """Backward rate for IKa activation gating variable a (Traub & Miles 1991).

    A singularity guard replaces the 0/0 form at V = -24.9 mV with the
    analytic limit.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

    Returns:
        Backward rate beta_a in 1/ms.
    """
    x = V + 24.9
    if abs(x) < _SINGULARITY_TOL:
        return 0.0175 * 10.0  # L'Hôpital limit
    return 0.0175 * x / (safe_exp(x / 10.0) - 1.0)


_beta_a = VoltageOnlyFn(_beta_a_impl)


def _alpha_b_impl(V: float, ca_i: float) -> float:
    """Forward rate for IKa inactivation gating variable b (Traub & Miles 1991).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

    Returns:
        Forward rate alpha_b in 1/ms.
    """
    return 0.0016 * safe_exp(-(V + 73.0) / 18.0)


_alpha_b = VoltageOnlyFn(_alpha_b_impl)


def _beta_b_impl(V: float, ca_i: float) -> float:
    """Backward rate for IKa inactivation gating variable b (Traub & Miles 1991).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

    Returns:
        Backward rate beta_b in 1/ms.
    """
    return 0.05 / (1.0 + safe_exp(-(V + 13.0) / 10.0))


_beta_b = VoltageOnlyFn(_beta_b_impl)


def make_ika_channel(
    g_max: float = DEFAULT_G_IKA,
) -> IonChannel:
    """Create an IKa (A-type K⁺) ion channel.

    IKa is a fast-inactivating, transient K⁺ current that delays the first
    spike and controls firing rate at stimulus onset.  It uses two gating
    variables: ``a`` (activation, power 1) and ``b`` (inactivation, power 1).

    Kinetics follow Traub & Miles (1991) hippocampal neuron models, shifted by
    -65 mV to match this codebase's absolute voltage convention.

    The reversal potential is computed dynamically from the neuron's K⁺
    concentrations using the Nernst equation.

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~patch_sim.constants.DEFAULT_G_IKA`.

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the IKa current.
    """
    a_var = GatingVariable(name="a", power=1, alpha=_alpha_a, beta=_beta_a)
    b_var = GatingVariable(name="b", power=1, alpha=_alpha_b, beta=_beta_b)
    return IonChannel(
        name="Ka",
        g_max=g_max,
        gating_variables=(a_var, b_var),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )


_ikv31_alpha_nk, _ikv31_beta_nk = boltzmann_cosh_rates(
    half=-12.4,
    slope=11.8,
    tau_scale=4.0,
    tau_floor=0.2,
)


def make_ikv31_channel(
    g_max: float = DEFAULT_G_IKV31,
) -> IonChannel:
    """Create an IKv31 (Kv3.1-type K⁺) ion channel.

    IKv31 is a high-threshold, fast-deactivating delayed-rectifier K⁺ current
    that is the hallmark conductance of fast-spiking interneurons.  Its high
    activation threshold (~−12 mV V₁/₂) means it is virtually closed at rest,
    preventing the spurious hyperpolarization that a low-threshold K⁺ channel
    would produce.  Fast deactivation enables rapid repolarization and supports
    high-frequency firing without adaptation.

    Uses a single activation gate ``nk`` with power 2 (n-squared kinetics from
    Erisir et al. 1999).  The reversal potential is computed dynamically from
    the neuron's K⁺ concentrations via the Nernst equation.

    Kinetics follow Erisir et al. (1999), J. Neurophysiol. 82:2476, expressed
    with Boltzmann/cosh rate functions:

    * V₁/₂ = −12.4 mV, slope = 11.8 mV
    * τ_scale = 4.0 ms, τ_floor = 0.2 ms

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~patch_sim.constants.DEFAULT_G_IKV31`.

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the IKv31
        current.
    """
    nk_var = GatingVariable(
        name="nk",
        power=2,
        alpha=_ikv31_alpha_nk,
        beta=_ikv31_beta_nk,
    )
    return IonChannel(
        name="Kv31",
        g_max=g_max,
        gating_variables=(nk_var,),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )


def make_ih_channel(
    g_max: float = DEFAULT_G_IH,
    p_na: float = DEFAULT_IH_P_NA,
) -> IonChannel:
    """Create an Ih (HCN/funny current) ion channel.

    The Ih channel is hyperpolarization-activated and carries a mixed Na⁺/K⁺
    cation current. It is responsible for the depolarizing sag potential seen
    during sustained hyperpolarization in many neuron types.

    Kinetics follow Destexhe et al. (1993) with a single gating variable ``r``
    (power 1).  At resting potential (~-65 mV) the channel is largely closed;
    deep hyperpolarization (~-100 mV) activates it strongly.

    The reversal potential is computed dynamically from the neuron's Na⁺ and
    K⁺ concentrations using the Goldman-Hodgkin-Katz equation, parameterised by
    the Na⁺ permeability relative to K⁺ (``p_na``).

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~patch_sim.constants.DEFAULT_G_IH`.
        p_na: Na⁺ permeability relative to K⁺ (dimensionless). A value of
            ~0.289 yields a reversal potential near -30 mV at default HH ion
            concentrations.  Defaults to
            :data:`~patch_sim.constants.DEFAULT_IH_P_NA`.

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the Ih current.
    """
    r_var = GatingVariable(name="r", power=1, alpha=_alpha_r, beta=_beta_r)
    return IonChannel(
        name="h",
        g_max=g_max,
        gating_variables=(r_var,),
        reversal_spec=GoldmanSpec(
            permeabilities=(
                (IonSpecies.SODIUM, p_na),
                (IonSpecies.POTASSIUM, 1.0),
            )
        ),
    )


# ---------------------------------------------------------------------------
# INaP — Persistent Na⁺ channel (Magistretti & Alonso 1999)
# ---------------------------------------------------------------------------
#
# Activation gate ``p`` follows the standard Magistretti & Alonso 1999 kinetics
# (fast Boltzmann, V½ = −52.6 mV, slope 4.6 mV).
#
# Slow inactivation gate ``sNaP`` is also from Magistretti & Alonso 1999
# §"Slow inactivation": V½ = −45 mV, slope 7 mV (inverted Boltzmann;
# values picked from the depolarized end of the experimental −47 to −54
# mV / k = 7-10 mV spread to leave near-rest availability high,
# s_inf(−65 mV) ≈ 0.94).  The paper reports τ on the order of seconds at
# the V½ peak; the implementation uses a faster tau_scale (≈200 ms peak)
# so the gate produces a useful escape from a depolarization plateau
# within a single sustained step rather than over multiple seconds.  The
# gate is part of the standard topology because native INaP biologically
# undergoes this slow inactivation (Magistretti & Alonso 1999).
#
# In STN this gate co-acts with the fast-Na slow inactivation gate baked
# into ``make_stn_na_channel`` and with ``make_katp_channel`` (#324); the
# three together collapse the −15 mV plateau cleanly.  Cortical pyramidal
# (#327), CA1 pyramidal (#328), and Purkinje (#329) combine ``sNaP`` with
# their own fast-Na slow-inactivation gate (built into
# ``make_nav12_channel`` for the cortical/CA1 cases, ``make_purkinje_na_channel``
# for Purkinje) but no K_ATP — autonomous-pacemaker metabolic safety
# isn't biologically motivated there (or, for Purkinje, Carter & Bean
# 2009 demonstrate depol-block recovery without it), and the two slow-
# inactivation gates suffice.  SNc dopaminergic (#330) ships its slow-
# inactivation gates via the SNc-specific factories
# ``make_dopaminergic_na_channel`` (sNa_da gate, Khaliq & Bean 2010) and
# ``make_snc_inap_channel`` (sNaP_snc gate, V½ shifted to track the
# Drion 2011 SNc INaP fit) rather than this entorhinal factory.  The
# chosen τ is therefore robust to the channel cocktail the gate happens
# to ship in.
#
# The slow-inactivation gate is named ``sNaP`` rather than ``s`` because the
# gating-state dictionary is keyed by gate name only and ``make_inar_channel``
# already declares an ``s`` gate.  Sharing the name would alias the two
# channels' gating variables.
#
# Slow inactivation is opt-in (default off) so that presets tuned without it
# keep their existing phenotypes.  Enable it on presets where depolarization-
# block recovery matters — STN (#324), cortical pyramidal (#327), CA1
# pyramidal (#328), and Purkinje (#329).

_alpha_p, _beta_p = boltzmann_cosh_rates(
    half=-52.6, slope=4.6, tau_scale=6.0, tau_floor=0.1
)

_alpha_sNaP, _beta_sNaP = boltzmann_cosh_rates(
    half=-45.0,
    slope=7.0,
    tau_scale=200.0,
    tau_floor=20.0,
    inverted=True,
)


def make_inap_channel(g_max: float = DEFAULT_G_NAP) -> IonChannel:
    """Create an INaP (persistent Na⁺) ion channel.

    INaP is a sustained Na⁺ current active near the resting potential that
    amplifies subthreshold depolarizations and can lower the threshold for
    action potential generation.

    The channel always includes two gates: an activation gate ``p``
    (Magistretti & Alonso 1999 kinetics, V½ = −52.6 mV) and a slow
    voltage-dependent inactivation gate ``sNaP`` (also Magistretti &
    Alonso 1999, §"Slow inactivation"; V½ = −45 mV, slope 7 mV, inverted
    Boltzmann; τ_scale = 200 ms / τ_floor = 20 ms).  The slow gate is
    mostly available at hyperpolarized potentials and decays toward zero
    during sustained depolarization, providing the escape mechanism that
    lets the membrane repolarize after a prolonged suprathreshold step.
    Native INaP biologically undergoes this slow inactivation (Magistretti
    & Alonso 1999), so the gate is part of the standard topology rather
    than an opt-in feature.  The simulation column is named ``sNaP``
    rather than ``s`` to avoid colliding with :func:`make_inar_channel`'s
    activation gate, which is also named ``s``.

    The reversal potential is computed dynamically from the neuron's Na⁺
    concentrations using the Nernst equation.

    References:
        - Magistretti & Alonso (1999), J. Gen. Physiol. 114:491
          (entorhinal INaP activation and slow inactivation).

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~patch_sim.constants.DEFAULT_G_NAP`.

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the INaP current.
    """
    p_var = GatingVariable(name="p", power=1, alpha=_alpha_p, beta=_beta_p)
    s_var = GatingVariable(name="sNaP", power=1, alpha=_alpha_sNaP, beta=_beta_sNaP)
    return IonChannel(
        name="NaP",
        g_max=g_max,
        gating_variables=(p_var, s_var),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )


# ---------------------------------------------------------------------------
# INaR — Resurgent Na⁺ channel (Raman & Bean 2001, simplified)
# ---------------------------------------------------------------------------

_NAR_HR_HALF: float = -55.0  # Half-unblocking for hr in mV
_NAR_HR_SLOPE: float = 8.0  # Unblocking slope for hr in mV
_NAR_HR_TAU_A: float = 150.0  # Asymmetric tau_hr numerator in ms
_NAR_HR_TAU_HALF: float = -40.0  # Voltage of tau_hr half-rise in mV
_NAR_HR_TAU_SLOPE: float = 10.0  # Slope of tau_hr sigmoid in mV
_NAR_HR_TAU_OFFSET: float = 1.0  # Additive offset for tau_hr in ms

_alpha_s, _beta_s = boltzmann_cosh_rates(
    half=-42.0, slope=5.0, tau_scale=0.5, tau_floor=0.05
)


def _nar_hr_inf(V: float) -> float:
    """Steady-state unblocking variable hr of INaR at voltage V.

    High at hyperpolarized potentials (channel unblocked), low at depolarized
    potentials (channel blocked).  Half-point at -55.0 mV.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Steady-state unblocking probability in [0, 1].
    """
    return 1.0 / (1.0 + safe_exp((V - _NAR_HR_HALF) / _NAR_HR_SLOPE))


def _nar_tau_hr(V: float) -> float:
    """Asymmetric voltage-dependent time constant for INaR unblocking variable hr.

    Slow at depolarized potentials (slow blocking), intermediate at
    hyperpolarized potentials (faster unblocking).

    Args:
        V: Membrane voltage in mV.

    Returns:
        Time constant in ms.
    """
    return (
        _NAR_HR_TAU_A / (1.0 + safe_exp(-(V - _NAR_HR_TAU_HALF) / _NAR_HR_TAU_SLOPE))
        + _NAR_HR_TAU_OFFSET
    )


def _alpha_hr_impl(V: float, ca_i: float) -> float:
    """Forward rate for INaR unblocking gating variable hr.

    Derived as alpha_hr = hr_inf / tau_hr.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

    Returns:
        Forward rate alpha_hr in 1/ms.
    """
    return _nar_hr_inf(V) / _nar_tau_hr(V)


_alpha_hr = VoltageOnlyFn(_alpha_hr_impl)


def _beta_hr_impl(V: float, ca_i: float) -> float:
    """Backward rate for INaR unblocking gating variable hr.

    Derived as beta_hr = (1 - hr_inf) / tau_hr.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

    Returns:
        Backward rate beta_hr in 1/ms.
    """
    return (1.0 - _nar_hr_inf(V)) / _nar_tau_hr(V)


_beta_hr = VoltageOnlyFn(_beta_hr_impl)


def make_inar_channel(
    g_max: float = DEFAULT_G_NAR,
) -> IonChannel:
    """Create an INaR (resurgent Na⁺) ion channel.

    INaR produces a transient inward Na⁺ current on membrane repolarization,
    known as the resurgent current.  The mechanism relies on two gating
    variables: ``s`` (activation, power 1) and ``hr`` (unblocking, power 1).

    At rest ``s`` is low and ``hr`` is high (channel closed but unblocked).
    During an action potential ``s`` opens quickly while ``hr`` slowly blocks
    the channel.  On repolarization ``hr`` recovers before ``s`` deactivates,
    producing a transient resurgent current.

    Kinetics follow a simplified Raman & Bean (2001) two-variable alpha/beta
    model.

    The reversal potential is computed dynamically from the neuron's Na⁺
    concentrations using the Nernst equation.

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~patch_sim.constants.DEFAULT_G_NAR`.

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the INaR current.
    """
    s_var = GatingVariable(name="s", power=1, alpha=_alpha_s, beta=_beta_s)
    hr_var = GatingVariable(name="hr", power=1, alpha=_alpha_hr, beta=_beta_hr)
    return IonChannel(
        name="NaR",
        g_max=g_max,
        gating_variables=(s_var, hr_var),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )


# ---------------------------------------------------------------------------
# I_M — Muscarinic K⁺ channel (Adams et al. 1982 / Traub & Miles 1991)
# ---------------------------------------------------------------------------

_alpha_w, _beta_w = boltzmann_cosh_rates(
    half=-35.0, slope=10.0, tau_scale=1000.0, tau_floor=10.0, tau_rate=6.6
)


def make_im_channel(
    g_max: float = DEFAULT_G_IM,
) -> IonChannel:
    """Create an IM (muscarinic K⁺) ion channel.

    IM is a slow, non-inactivating K⁺ current that is suppressed by
    muscarinic receptor activation.  It contributes to spike-frequency
    adaptation and the medium afterhyperpolarization.  It uses a single
    gating variable ``w`` (power 1).

    Kinetics follow Adams et al. (1982) / Traub & Miles (1991) with a
    Boltzmann activation centered at -35 mV and a slow cosh-based time
    constant (~150 ms near -35 mV).

    The reversal potential is computed dynamically from the neuron's K⁺
    concentrations using the Nernst equation.

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~patch_sim.constants.DEFAULT_G_IM`.

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the IM current.
    """
    w_var = GatingVariable(name="w", power=1, alpha=_alpha_w, beta=_beta_w)
    return IonChannel(
        name="M",
        g_max=g_max,
        gating_variables=(w_var,),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )


# ---------------------------------------------------------------------------
# I_K_ATP — ATP-sensitive K⁺ channel (Kir6.x; phenomenological proxy)
# ---------------------------------------------------------------------------
#
# Real K_ATP channels (Kir6.x/SUR) are gated by intracellular ATP/ADP
# stoichiometry, opening when [ATP] falls relative to [ADP] during
# metabolic stress.  In STN cells they contribute to depolarization-block
# recovery and to the membrane response under sustained suprathreshold
# drive (Stanford & Lacey 1996; Bevan & Wilson 1999).  Modeling them
# faithfully would require an ATP/ADP state variable coupled to firing
# activity (Erecińska & Silver 1989); since the simulator doesn't track
# metabolic state, this factory uses a voltage-driven slow-activation
# proxy instead.  The channel reaches the same depol-block-rescue
# endpoint without adding a new ODE.
#
# Parameters: V½ = −25 mV (above autonomous threshold ~−40 mV, so
# subthreshold tonic pacemaking is undisturbed but the channel engages
# strongly at the −15 mV depol-block plateau); slope 8 mV; τ_scale 400 ms,
# τ_floor 50 ms (slow activation reflects the ATP-depletion timescale
# during sustained spiking).  At V = −15 mV kATP_inf ≈ 0.78, so at
# g_max = 0.5 mS/cm² the channel produces ~0.5·0.78·75 ≈ 29 µA/cm² outward
# K⁺ drive — comfortably exceeds the residual ~10–20 µA/cm² fast-Na drive
# at the plateau.

_alpha_kATP, _beta_kATP = boltzmann_cosh_rates(
    half=-25.0,
    slope=8.0,
    tau_scale=400.0,
    tau_floor=50.0,
)


def make_katp_channel(
    g_max: float = DEFAULT_G_KATP,
) -> IonChannel:
    """Create a K_ATP (ATP-sensitive K⁺) ion channel.

    Real K_ATP channels are metabolically gated (Kir6.x/SUR octamers
    activated by low [ATP]/[ADP]); this factory models them
    phenomenologically with voltage-driven slow activation, since the
    simulator does not track metabolic state.  The channel engages at
    sustained depolarization (V½ = −25 mV, well above autonomous
    pacemaking threshold) with a slow time constant (~400 ms peak),
    providing an outward K⁺ drive that helps the membrane escape
    depolarization block (#324 in STN).

    Uses a single gating variable ``kATP`` (power 1).  The reversal
    potential is computed dynamically from the neuron's K⁺ concentrations
    via the Nernst equation.

    References:
        - Stanford & Lacey (1996), J. Neurophysiol. 75:1714 (K_ATP in STN
          and SNc).
        - Bevan & Wilson (1999), J. Neurosci. 19:7617 (STN spontaneous
          activity).
        - Hahn & McIntyre (2010), J. Comput. Neurosci. 28:425 (STN model
          including K_ATP).
        - Erecińska & Silver (1989) (ATP/ADP dynamics during firing).

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~patch_sim.constants.DEFAULT_G_KATP`.

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the K_ATP
        current.
    """
    katp_var = GatingVariable(name="kATP", power=1, alpha=_alpha_kATP, beta=_beta_kATP)
    return IonChannel(
        name="KATP",
        g_max=g_max,
        gating_variables=(katp_var,),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )


# ---------------------------------------------------------------------------
# I_Kir — Inward rectifier K⁺ channel (Hagiwara & Takahashi 1974)
# ---------------------------------------------------------------------------

_alpha_kir, _beta_kir = boltzmann_cosh_rates(
    half=-80.0, slope=12.0, tau_scale=10.0, tau_floor=0.5, inverted=True
)


def make_ikir_channel(
    g_max: float = DEFAULT_G_IKIR,
) -> IonChannel:
    """Create an IKir (inward rectifier K⁺) ion channel.

    IKir is a K⁺ channel that is most active at hyperpolarized potentials
    (inward rectification).  It helps stabilize the resting potential and
    contributes a large conductance near the K⁺ equilibrium potential.
    It uses a single gating variable ``kir`` (power 1).

    Kinetics follow Hagiwara & Takahashi (1974) / Steephen & Bhalla (2009),
    with an inverted Boltzmann activation centered at -80 mV and a fast
    cosh-based time constant.

    The reversal potential is computed dynamically from the neuron's K⁺
    concentrations using the Nernst equation.

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~patch_sim.constants.DEFAULT_G_IKIR`.

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the IKir current.
    """
    kir_var = GatingVariable(name="kir", power=1, alpha=_alpha_kir, beta=_beta_kir)
    return IonChannel(
        name="Kir",
        g_max=g_max,
        gating_variables=(kir_var,),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )


# ---------------------------------------------------------------------------
# I_KCa — Calcium-activated K⁺ channel (simplified BK-like)
# ---------------------------------------------------------------------------

_IKCA_HILL_KD: float = 0.001  # Half-saturation Ca²⁺ concentration in mM (K_d)
_IKCA_V_HALF: float = -20.0  # Half-activation voltage in mV
_IKCA_V_SLOPE: float = 10.0  # Voltage activation slope in mV
_IKCA_TAU_SCALE: float = 10.0  # Time-constant numerator in ms
_IKCA_TAU_COSH_SCALE: float = 20.0  # Voltage scale in cosh denominator in mV
_IKCA_TAU_FLOOR: float = 1.0  # Minimum time constant in ms


def _ikca_q_inf(V: float, ca_i: float) -> float:
    """Steady-state activation of IKCa at voltage V and [Ca²⁺]ᵢ.

    Combines a linear Hill function (n=1, K_d=0.001 mM) with a Boltzmann
    voltage factor.  Zero calcium gives zero activation regardless of voltage.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM.

    Returns:
        Steady-state open probability in [0, 1].
    """
    hill = ca_i / (ca_i + _IKCA_HILL_KD)
    boltzmann = 1.0 / (1.0 + safe_exp(-(V - _IKCA_V_HALF) / _IKCA_V_SLOPE))
    return hill * boltzmann


def _ikca_tau(V: float) -> float:
    """Voltage-dependent time constant for IKCa gating variable q.

    Uses a cosh-based expression with a floor to prevent near-zero values.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Time constant in ms (floored at 1 ms).
    """
    tau = _IKCA_TAU_SCALE / safe_cosh((V - _IKCA_V_HALF) / _IKCA_TAU_COSH_SCALE)
    return max(tau, _IKCA_TAU_FLOOR)


def _alpha_q_impl(V: float, ca_i: float) -> float:
    """Forward rate for IKCa gating variable q.

    Derived as alpha_q = q_inf(V, ca_i) / tau(V).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM.

    Returns:
        Forward rate alpha_q in 1/ms.
    """
    return _ikca_q_inf(V, ca_i) / _ikca_tau(V)


_alpha_q = CalciumDependentFn(_alpha_q_impl)


def _beta_q_impl(V: float, ca_i: float) -> float:
    """Backward rate for IKCa gating variable q.

    Derived as beta_q = (1 - q_inf(V, ca_i)) / tau(V).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM.

    Returns:
        Backward rate beta_q in 1/ms.
    """
    return (1.0 - _ikca_q_inf(V, ca_i)) / _ikca_tau(V)


_beta_q = CalciumDependentFn(_beta_q_impl)


def make_ikca_channel(
    g_max: float = DEFAULT_G_IKCA,
) -> IonChannel:
    """Create an IKCa (calcium-activated K⁺) ion channel.

    IKCa is a BK-like K⁺ channel activated by both membrane depolarization
    and elevated intracellular Ca²⁺.  It contributes to spike repolarization
    and the afterhyperpolarization following Ca²⁺ entry.  It uses a single
    gating variable ``q`` (power 1) whose kinetics depend on both voltage and
    [Ca²⁺]ᵢ.

    Note: IKCa is calcium-*activated* but carries K⁺, not Ca²⁺.
    ``carries_calcium`` is ``False``.

    Kinetics use a Hill function (K_d = 0.001 mM, n = 1) multiplied by a
    Boltzmann voltage factor (half-activation at -20 mV) for the steady state,
    and a cosh-based voltage-dependent time constant.

    The reversal potential is computed dynamically from the neuron's K⁺
    concentrations using the Nernst equation.

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~patch_sim.constants.DEFAULT_G_IKCA`.

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the IKCa current.
    """
    q_var = GatingVariable(name="q", power=1, alpha=_alpha_q, beta=_beta_q)
    return IonChannel(
        name="KCa",
        g_max=g_max,
        gating_variables=(q_var,),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )


# ---------------------------------------------------------------------------
# ICaL — L-type Ca²⁺ channel (high-voltage activated, slow inactivating)
# ---------------------------------------------------------------------------

_alpha_d, _beta_d = boltzmann_cosh_rates(
    half=-10.0, slope=6.2, tau_scale=1.0, tau_floor=0.1
)
_alpha_f, _beta_f = boltzmann_cosh_rates(
    half=-35.0, slope=-9.0, tau_scale=50.0, tau_floor=5.0
)


def make_ical_channel(
    g_max: float = DEFAULT_G_ICAL,
) -> IonChannel:
    """Create an ICaL (L-type Ca²⁺) ion channel.

    ICaL is a high-voltage-activated Ca²⁺ channel with slow voltage-dependent
    inactivation.  It is the dominant source of Ca²⁺ influx during action
    potentials in many neuron types.  It uses two gating variables: ``d``
    (activation, power 2) and ``f`` (inactivation, power 1).

    Because ICaL carries Ca²⁺, ``carries_calcium=True`` is set so that
    the simulation loop accumulates its contribution for the Ca²⁺ ODE
    and :meth:`~patch_sim.channels.IonChannel.compute_current` uses live
    ``ca_i`` for a dynamic E_Ca.

    Kinetics use Boltzmann-cosh rate functions with activation centered at
    -10 mV (slope 6.2 mV) and inactivation centered at -35 mV (slope -9 mV,
    inverted).

    The reversal potential is computed dynamically from the neuron's Ca²⁺
    concentrations using the Nernst equation.

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~patch_sim.constants.DEFAULT_G_ICAL`.

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the ICaL current.
    """
    d_var = GatingVariable(name="d", power=2, alpha=_alpha_d, beta=_beta_d)
    f_var = GatingVariable(name="f", power=1, alpha=_alpha_f, beta=_beta_f)
    return IonChannel(
        name="CaL",
        g_max=g_max,
        gating_variables=(d_var, f_var),
        reversal_spec=NernstSpec(IonSpecies.CALCIUM),
        carries_calcium=True,
    )


# ---------------------------------------------------------------------------
# ICaT — T-type Ca²⁺ channel (low-voltage activated, transient)
# ---------------------------------------------------------------------------

_alpha_dt, _beta_dt = boltzmann_cosh_rates(
    half=-56.0, slope=6.2, tau_scale=1.0, tau_floor=0.1
)
_alpha_ft, _beta_ft = boltzmann_cosh_rates(
    half=-80.0, slope=-9.0, tau_scale=20.0, tau_floor=2.0
)


def make_icat_channel(
    g_max: float = DEFAULT_G_ICAT,
) -> IonChannel:
    """Create an ICaT (T-type Ca²⁺) ion channel.

    ICaT is a low-voltage-activated, transient Ca²⁺ channel.  It activates
    near the resting potential and contributes to burst firing and
    oscillatory behavior (e.g. in thalamic neurons).  It uses two gating
    variables: ``dt`` (activation, power 2) and ``ft`` (inactivation, power 1).

    Kinetics follow Destexhe et al. (1994) with activation half-point at
    -56 mV and inactivation half-point at -80 mV.

    The reversal potential is computed dynamically from the neuron's Ca²⁺
    concentrations using the Nernst equation.

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~patch_sim.constants.DEFAULT_G_ICAT`.

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the ICaT current.
    """
    dt_var = GatingVariable(name="dt", power=2, alpha=_alpha_dt, beta=_beta_dt)
    ft_var = GatingVariable(name="ft", power=1, alpha=_alpha_ft, beta=_beta_ft)
    return IonChannel(
        name="CaT",
        g_max=g_max,
        gating_variables=(dt_var, ft_var),
        reversal_spec=NernstSpec(IonSpecies.CALCIUM),
        carries_calcium=True,
    )


# ---------------------------------------------------------------------------
# ICaN — N-type Ca²⁺ channel (high-voltage activated)
# ---------------------------------------------------------------------------

_alpha_dn, _beta_dn = boltzmann_cosh_rates(
    half=-20.0, slope=6.2, tau_scale=1.0, tau_floor=0.1
)
_alpha_fn, _beta_fn = boltzmann_cosh_rates(
    half=-40.0, slope=-9.0, tau_scale=30.0, tau_floor=3.0
)


def make_ican_channel(
    g_max: float = DEFAULT_G_ICAN,
) -> IonChannel:
    """Create an ICaN (N-type Ca²⁺) ion channel.

    ICaN is a high-voltage-activated Ca²⁺ channel that inactivates more
    rapidly than ICaL.  It is widely expressed in neuronal dendrites and
    presynaptic terminals and contributes to neurotransmitter release.
    It uses two gating variables: ``dn`` (activation, power 2) and
    ``fn`` (inactivation, power 1).

    Kinetics use Boltzmann-cosh rate functions with activation centered at
    -20 mV and inactivation centered at -40 mV.

    The reversal potential is computed dynamically from the neuron's Ca²⁺
    concentrations using the Nernst equation.

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~patch_sim.constants.DEFAULT_G_ICAN`.

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the ICaN current.
    """
    dn_var = GatingVariable(name="dn", power=2, alpha=_alpha_dn, beta=_beta_dn)
    fn_var = GatingVariable(name="fn", power=1, alpha=_alpha_fn, beta=_beta_fn)
    return IonChannel(
        name="CaN",
        g_max=g_max,
        gating_variables=(dn_var, fn_var),
        reversal_spec=NernstSpec(IonSpecies.CALCIUM),
        carries_calcium=True,
    )


# ---------------------------------------------------------------------------
# I_SK — Small-conductance Ca²⁺-activated K⁺ channel (apamin-sensitive)
# ---------------------------------------------------------------------------
# SK channels are Ca²⁺-gated with no voltage dependence: a Hill function on
# intracellular [Ca²⁺] sets the open probability.  In SNc DA neurons SK
# (mainly SK3) couples to Cav1.3 Ca²⁺ entry and shapes the medium AHP that
# follows each pacemaker spike — gating the inter-spike interval and pinning
# the firing regularity that distinguishes tonic SNc DA pacemaking from the
# bursting seen when SK is blocked with apamin (Wolfart et al. 2001).
#
# Half-activation [Ca²⁺]_i = 0.3 µM (3e-4 mM) with Hill coefficient n = 4.
# Drion et al. 2011 use this K_d in the reconciled SNc DA model; it sits
# between the high-affinity SK2 fit of Hirschberg et al. 1998 (~0.3 µM) and
# the SK3 fit of Bond et al. 2004 (~0.5 µM), and is the value that gives the
# tightest ISI regularity in the Cav1.3↔SK loop.  The kinetics are fast
# (τ ≈ 10 ms) — fast enough to follow each AP's Ca²⁺ transient and shape
# the medium AHP that immediately follows.
#
# References:
#   Wolfart et al. (2001), J. Neurosci. 21:3443 — SK gates SNc tonic firing
#   Hirschberg et al. (1998), J. Gen. Physiol. 111:565 — SK2 Ca²⁺ K_d
#   Bond et al. (2004), J. Neurosci. 24:5301 — SK2/SK3 Ca²⁺ sensitivity
#   Drion et al. (2011), PLOS Comp Biol 7:e1002050 — SNc K_d = 0.3 µM
_SK_HILL_KD: float = 3e-4  # Half-activation [Ca²⁺]_i in mM (= 0.3 µM)
_SK_HILL_N: int = 4
_SK_TAU: float = 10.0  # Time constant in ms


def _sk_q_inf(ca_i: float) -> float:
    """Steady-state activation of the SK gating variable at [Ca²⁺]_i.

    Hill function with K_d = 0.3 µM and Hill coefficient n = 4.  No voltage
    dependence — SK is purely Ca²⁺-gated.

    Args:
        ca_i: Intracellular Ca²⁺ concentration in mM.

    Returns:
        Steady-state open probability in [0, 1].
    """
    ca_n = ca_i**_SK_HILL_N
    kd_n = _SK_HILL_KD**_SK_HILL_N
    return ca_n / (ca_n + kd_n)


def _alpha_qSK_impl(V: float, ca_i: float) -> float:
    """Forward rate for the SK gating variable.

    Derived as ``alpha = q_inf(ca_i) / tau``.  Voltage independent.

    Args:
        V: Membrane voltage in mV (ignored).
        ca_i: Intracellular Ca²⁺ concentration in mM.

    Returns:
        Forward rate in 1/ms.
    """
    return _sk_q_inf(ca_i) / _SK_TAU


_alpha_qSK = CalciumDependentFn(_alpha_qSK_impl)


def _beta_qSK_impl(V: float, ca_i: float) -> float:
    """Backward rate for the SK gating variable.

    Derived as ``beta = (1 - q_inf(ca_i)) / tau``.  Voltage independent.

    Args:
        V: Membrane voltage in mV (ignored).
        ca_i: Intracellular Ca²⁺ concentration in mM.

    Returns:
        Backward rate in 1/ms.
    """
    return (1.0 - _sk_q_inf(ca_i)) / _SK_TAU


_beta_qSK = CalciumDependentFn(_beta_qSK_impl)


def make_sk_channel(
    g_max: float = DEFAULT_G_SK,
) -> IonChannel:
    """Create an SK (small-conductance Ca²⁺-activated K⁺) ion channel.

    SK is gated purely by intracellular Ca²⁺ (no voltage dependence) via a
    Hill function (K_d = 0.3 µM, Hill coefficient n = 4 — Drion et al. 2011).
    The kinetics are fast (τ ≈ 10 ms), so the channel follows each AP's Ca²⁺
    transient and shapes the medium afterhyperpolarization.

    In SNc DA neurons SK couples tightly to Cav1.3 Ca²⁺ entry: each spike
    loads Ca²⁺, SK opens to produce the medium AHP, the AHP closes Cav1.3,
    Ca²⁺ is buffered and SK closes — and the Cav1.3 sub-threshold window
    current then ramps the cell back to threshold.  This Cav1.3↔SK loop is
    the Putzier 2009 mechanism for tonic SNc pacemaking.

    Note: SK is calcium-*activated* but carries K⁺, not Ca²⁺;
    ``carries_calcium`` is therefore ``False``.

    The reversal potential is computed dynamically from the neuron's K⁺
    concentrations using the Nernst equation.

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~patch_sim.constants.DEFAULT_G_SK`.

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the SK current.
    """
    qSK_var = GatingVariable(name="qSK", power=1, alpha=_alpha_qSK, beta=_beta_qSK)
    return IonChannel(
        name="SK",
        g_max=g_max,
        gating_variables=(qSK_var,),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )
