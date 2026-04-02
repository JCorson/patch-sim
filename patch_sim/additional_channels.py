"""Concrete additional ion channel implementations.

These channels can be added to a HodgkinHuxley model via the
``additional_channels`` argument to extend the classic three-channel HH model
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
    DEFAULT_G_ICAL,
    DEFAULT_G_ICAN,
    DEFAULT_G_ICAT,
    DEFAULT_G_IH,
    DEFAULT_G_IKA,
    DEFAULT_G_IKCA,
    DEFAULT_G_IKIR,
    DEFAULT_G_IM,
    DEFAULT_G_NAP,
    DEFAULT_G_NAR,
    DEFAULT_IH_P_NA,
)
from .electrochemistry import boltzmann_cosh_rates
from .utils import safe_cosh, safe_exp


def _alpha_r(V: float, ca_i: float) -> float:
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


def _beta_r(V: float, ca_i: float) -> float:
    """Backward rate for Ih gating variable r (Destexhe-style HCN kinetics).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

    Returns:
        Backward rate beta_r in 1/ms.
    """
    return safe_exp(-1.87 + 0.0701 * V)


_SINGULARITY_TOL: float = 1e-7


def _alpha_a(V: float, ca_i: float) -> float:
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


def _beta_a(V: float, ca_i: float) -> float:
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


def _alpha_b(V: float, ca_i: float) -> float:
    """Forward rate for IKa inactivation gating variable b (Traub & Miles 1991).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

    Returns:
        Forward rate alpha_b in 1/ms.
    """
    return 0.0016 * safe_exp(-(V + 73.0) / 18.0)


def _beta_b(V: float, ca_i: float) -> float:
    """Backward rate for IKa inactivation gating variable b (Traub & Miles 1991).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

    Returns:
        Backward rate beta_b in 1/ms.
    """
    return 0.05 / (1.0 + safe_exp(-(V + 13.0) / 10.0))


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


def _alpha_hr(V: float, ca_i: float) -> float:
    """Forward rate for INaR unblocking gating variable hr.

    Derived as alpha_hr = hr_inf / tau_hr.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

    Returns:
        Forward rate alpha_hr in 1/ms.
    """
    return _nar_hr_inf(V) / _nar_tau_hr(V)


def _beta_hr(V: float, ca_i: float) -> float:
    """Backward rate for INaR unblocking gating variable hr.

    Derived as beta_hr = (1 - hr_inf) / tau_hr.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

    Returns:
        Backward rate beta_hr in 1/ms.
    """
    return (1.0 - _nar_hr_inf(V)) / _nar_tau_hr(V)


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


def _alpha_q(V: float, ca_i: float) -> float:
    """Forward rate for IKCa gating variable q.

    Derived as alpha_q = q_inf(V, ca_i) / tau(V).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM.

    Returns:
        Forward rate alpha_q in 1/ms.
    """
    return _ikca_q_inf(V, ca_i) / _ikca_tau(V)


def _beta_q(V: float, ca_i: float) -> float:
    """Backward rate for IKCa gating variable q.

    Derived as beta_q = (1 - q_inf(V, ca_i)) / tau(V).

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM.

    Returns:
        Backward rate beta_q in 1/ms.
    """
    return (1.0 - _ikca_q_inf(V, ca_i)) / _ikca_tau(V)


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
    :meth:`~patch_sim.hodgkin_huxley.HodgkinHuxley.calcium_current` sums its
    contribution automatically.

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
