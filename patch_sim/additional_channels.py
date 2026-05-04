"""Concrete additional ion channel implementations.

These channels can be added to a :class:`~patch_sim.Neuron` via the
``additional_channels`` argument to extend the classic three-channel model
with additional biophysical mechanisms.
"""

from .channels import (
    GatingVariable,
    GoldmanSpec,
    IonChannel,
    IonSpecies,
    NernstSpec,
)
from .constants import (
    DEFAULT_G_CAV13,
    DEFAULT_G_ICAL,
    DEFAULT_G_ICAN,
    DEFAULT_G_ICAT,
    DEFAULT_G_IH,
    DEFAULT_G_IKA,
    DEFAULT_G_IKCA,
    DEFAULT_G_IKIR,
    DEFAULT_G_IKV31,
    DEFAULT_G_IM,
    DEFAULT_G_NAP,
    DEFAULT_G_NAR,
    DEFAULT_G_SK,
    DEFAULT_IH_P_NA,
)
from .electrochemistry import boltzmann_cosh_rates
from .rates import CalciumDependentFn, VoltageOnlyFn
from .utils import safe_cosh, safe_exp


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

_alpha_p, _beta_p = boltzmann_cosh_rates(
    half=-52.6, slope=4.6, tau_scale=6.0, tau_floor=0.1
)


def make_inap_channel(
    g_max: float = DEFAULT_G_NAP,
) -> IonChannel:
    """Create an INaP (persistent Na⁺) ion channel.

    INaP is a non-inactivating Na⁺ current active near the resting potential.
    It amplifies subthreshold depolarizations and can lower the threshold for
    action potential generation.  It uses a single gating variable ``p``
    (power 1).

    Kinetics follow Magistretti & Alonso (1999), with a Boltzmann activation
    centred at -52.6 mV and a cosh-based time constant.

    The reversal potential is computed dynamically from the neuron's Na⁺
    concentrations using the Nernst equation.

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~patch_sim.constants.DEFAULT_G_NAP`.

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the INaP current.
    """
    p_var = GatingVariable(name="p", power=1, alpha=_alpha_p, beta=_beta_p)
    return IonChannel(
        name="NaP",
        g_max=g_max,
        gating_variables=(p_var,),
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
    Boltzmann activation centred at -35 mV and a slow cosh-based time
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
    with an inverted Boltzmann activation centred at -80 mV and a fast
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

    Kinetics use Boltzmann-cosh rate functions with activation centred at
    -10 mV (slope 6.2 mV) and inactivation centred at -35 mV (slope -9 mV,
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
# Cav1.3 — LVA L-type Ca²⁺ channel (Putzier 2009 SNc pacemaker driver)
# ---------------------------------------------------------------------------
# Cav1.3 (α1D) is the low-voltage-activated isoform of the L-type Ca²⁺ family.
# Activation midpoint is shifted ~25 mV negative of the canonical Cav1.2 ICaL
# (half ≈ −45 mV vs −10 mV), so a small persistent window current flows at
# sub-threshold voltages near the SNc DA resting potential (−60 to −50 mV).
# This window current is the slow inward driver of the inter-spike "pacemaker
# potential" that ramps from the post-AP AHP back to threshold (Putzier et
# al. 2009, fig 5).  Inactivation is slow (τ ≈ 200 ms) and incomplete, so the
# residual window persists through the inter-spike interval.
#
# References:
#   Putzier, Kullmann, Roeper (2009), J. Neurosci. 29:15531 — Cav1.3 is
#       the dominant pacemaker driver in SNc DA neurons
#   Xu & Lipscombe (2001), J. Neurosci. 21:5944 — Cav1.3 vs Cav1.2 kinetics
#   Wolfart et al. (2001), J. Neurosci. 21:3443 — SNc Cav1.3/SK pairing
_alpha_dL13, _beta_dL13 = boltzmann_cosh_rates(
    half=-33.0, slope=4.0, tau_scale=2.0, tau_floor=0.5
)
_alpha_fL13, _beta_fL13 = boltzmann_cosh_rates(
    half=-50.0, slope=-8.0, tau_scale=200.0, tau_floor=20.0
)


def make_cav13_channel(
    g_max: float = DEFAULT_G_CAV13,
) -> IonChannel:
    """Create a Cav1.3 (LVA L-type Ca²⁺) ion channel.

    Cav1.3 is the low-voltage-activated isoform of the neuronal L-type Ca²⁺
    family.  Its activation midpoint is shifted ~25 mV negative of the
    canonical Cav1.2 (half ≈ −45 mV, slope 5 mV) so a persistent sub-threshold
    window current flows near the SNc DA resting potential.  This window
    current drives the slow inter-spike depolarisation that defines SNc
    autonomous pacemaking (Putzier et al. 2009).

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
    oscillatory behaviour (e.g. in thalamic neurons).  It uses two gating
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


# Thalamic-relay-specific ICaT inactivation: 5× slower than the Destexhe
# (1994) global default (tau_scale=20 ms).  McCormick & Huguenard (1992) report
# tau_h_T ≈ 25–40 ms in the depolarised range used during the LTS plateau,
# which is necessary to sustain the plateau long enough for 3–7 Na⁺ spikes
# (issue #287).  Activation kinetics (dt half-point, slope, and tau) are
# unchanged from the global ICaT.
_alpha_ft_tc, _beta_ft_tc = boltzmann_cosh_rates(
    half=-80.0, slope=-9.0, tau_scale=100.0, tau_floor=2.0
)


def make_thalamic_relay_icat_channel(
    g_max: float = DEFAULT_G_ICAT,
) -> IonChannel:
    """Create the Thalamic-Relay-tuned ICaT (T-type Ca²⁺) channel.

    Variant of :func:`make_icat_channel` with slower inactivation kinetics
    (``ft`` tau_scale = 100 ms vs the global default 20 ms) that match the
    McCormick & Huguenard (1992) recordings of guinea-pig dorsal LGN relay
    neurons.  The slower inactivation sustains the low-threshold spike (LTS)
    plateau long enough to support a multi-spike burst (3–7 Na⁺ spikes at
    200–500 Hz) on hyperpolarising-step release — the defining feature of
    TC burst mode (Sherman & Guillery 1996; Llinás & Jahnsen 1982).

    Activation half-point and slope are unchanged from the global ICaT
    (-56 mV / 6.2 mV).  Inactivation half-point also unchanged (-80 mV /
    -9 mV slope).

    The reversal potential is computed dynamically from the neuron's Ca²⁺
    concentrations using the Nernst equation.

    Reference: McCormick & Huguenard (1992), J. Neurophysiol. 68:1384;
    Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (TC).

    Args:
        g_max: Maximum conductance in mS/cm². Must be non-negative.
            Defaults to :data:`~patch_sim.constants.DEFAULT_G_ICAT`.

    Returns:
        An :class:`~patch_sim.channels.IonChannel` representing the
        Thalamic-Relay ICaT current.
    """
    dt_var = GatingVariable(name="dt", power=2, alpha=_alpha_dt, beta=_beta_dt)
    ft_var = GatingVariable(name="ft", power=1, alpha=_alpha_ft_tc, beta=_beta_ft_tc)
    return IonChannel(
        name="CaT",
        g_max=g_max,
        gating_variables=(dt_var, ft_var),
        reversal_spec=NernstSpec(IonSpecies.CALCIUM),
        carries_calcium=True,
    )


# ---------------------------------------------------------------------------
# TRN-specific ICaT — sigmoid-shaped inactivation tau
# ---------------------------------------------------------------------------
# Huguenard & Prince (1992), J. Neurosci. 12:3804 record TRN low-threshold
# spike (LTS) bursts of 5–15 Na⁺ spikes at 200–600 Hz on hyperpolarising-step
# release.  Reproducing that spike count requires the LTS plateau to last
# long enough to fit 5+ Na⁺/K⁺ AP cycles, which means an ICaT inactivation
# tau in the 100–250 ms range at LTS-plateau voltages (V > −56 mV).
#
# The default Destexhe (1994) cosh-shaped tau peaks at the half-inactivation
# voltage (−80 mV → 20 ms) and decays at depolarised V (≈4 ms at −40 mV,
# floored at 2 ms by −20 mV), so the LTS plateau collapses in 5–10 ms — too
# fast.  Increasing g_T to compensate is not viable: the window-current slope
# conductance grows linearly and beyond g_T ≈ 4 mS/cm² the cell autonomously
# bursts at rest.
#
# This factory replaces the cosh tau with a sigmoid tau that is small at
# hyperpolarised V (rest stability — fast equilibration of ft prevents
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
_TRN_FT_TAU_MIN: float = 20.0  # ft tau at hyperpolarised V in ms
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

    Small at hyperpolarised V (≈ ``_TRN_FT_TAU_MIN`` = 20 ms) and large at
    depolarised V (≈ ``_TRN_FT_TAU_MAX`` = 200 ms), with a smooth sigmoid
    transition centred at V = −50 mV (slope 5 mV).  This shape preserves
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
    hyperpolarised V and large (200 ms) at LTS-plateau V, with a smooth
    transition centred at −50 mV.  This sustains the low-threshold spike
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

    Kinetics use Boltzmann-cosh rate functions with activation centred at
    -20 mV and inactivation centred at -40 mV.

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
# Half-activation [Ca²⁺]_i = 0.5 µM (5e-4 mM) with Hill coefficient n = 4
# matches the steady-state Ca²⁺ sensitivity reported for SK2/SK3 in expression
# systems (Bond et al. 2004) and used in the Putzier 2009 SNc model.  The
# kinetics are fast (τ ≈ 10 ms) — fast enough to follow each AP's Ca²⁺
# transient and shape the medium AHP that immediately follows.
#
# References:
#   Wolfart et al. (2001), J. Neurosci. 21:3443 — SK gates SNc tonic firing
#   Bond et al. (2004), J. Neurosci. 24:5301 — SK2 Ca²⁺ sensitivity
#   Putzier, Kullmann, Roeper (2009), J. Neurosci. 29:15531 — SNc model
_SK_HILL_KD: float = 5e-4  # Half-activation [Ca²⁺]_i in mM (= 0.5 µM)
_SK_HILL_N: int = 4
_SK_TAU: float = 10.0  # Time constant in ms


def _sk_q_inf(ca_i: float) -> float:
    """Steady-state activation of the SK gating variable at [Ca²⁺]_i.

    Hill function with K_d = 0.5 µM and Hill coefficient n = 4.  No voltage
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
    Hill function (K_d = 0.5 µM, Hill coefficient n = 4).  The kinetics are
    fast (τ ≈ 10 ms), so the channel follows each AP's Ca²⁺ transient and
    shapes the medium afterhyperpolarisation.

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
