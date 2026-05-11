"""De Schutter & Bower (1994) cerebellar Purkinje cell channel factories.

Source: De Schutter, E. & Bower, J.M. (1994) An active membrane model of
the cerebellar Purkinje cell I. Simulation of current clamps in slice.
J. Neurophysiol. 71:375–400.

Rate functions use the same Traub-Miles analytical form as the other
cell-type-specific factories.  VT = −58 mV matches the somatic Na⁺
activation threshold of guinea-pig cerebellar Purkinje neurons at 32 °C
(the recording temperature of De Schutter & Bower 1994).
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
    "PURKINJE_VT",
    "purkinje_alpha_m",
    "purkinje_beta_m",
    "purkinje_alpha_h",
    "purkinje_beta_h",
    "purkinje_alpha_n",
    "purkinje_beta_n",
    "make_purkinje_na_channel",
    "make_purkinje_k_channel",
]

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


# ---------------------------------------------------------------------------
# Purkinje fast Na⁺ slow voltage-dependent inactivation gate ``sNa``.
# ---------------------------------------------------------------------------
#
# Cerebellar Purkinje neurons are intrinsic high-frequency pacemakers; their
# Na⁺ channels undergo cumulative slow inactivation on top of the fast h gate.
# This was directly demonstrated in cerebellar Purkinje cells by Carter & Bean
# (2009), Neuron 64:898 (the companion paper to Do & Bean 2003 on STN
# pacemaker channels).  Without this second slow gate the De Schutter & Bower
# (1994) Traub-Miles formulation can pin the membrane on a non-physiological
# depol-block plateau under sustained climbing-fiber-style drive (#329).
#
# Parameters mirror the STN and Pospischil sNa gates: V½ = −50 mV, slope
# 8 mV, inverted Boltzmann so the gate is open at hyperpolarized potentials;
# τ_scale = 200 ms / τ_floor = 20 ms.  At V = −65 mV (Purkinje rest)
# s_inf ≈ 0.87 (autonomous pacemaking is minimally perturbed); at V = −15 mV
# s_inf ≈ 0.01 (residual h-tail abolished).
#
# Gate name ``sNa`` is shared with STN and Pospischil presets; it never
# coexists in the same neuron with those factories, and the choice avoids
# colliding with ``s`` (INaR activation) and ``sNaP`` (INaP slow
# inactivation), both of which are present in the Purkinje preset.
#
# Unlike the Pospischil and STN factories, slow inactivation is always on
# here because ``make_purkinje_na_channel`` is dedicated to the Purkinje
# preset — there is no other caller whose calibration could be perturbed by
# enabling the gate.

_purkinje_alpha_sNa, _purkinje_beta_sNa = boltzmann_cosh_rates(
    half=-50.0,
    slope=8.0,
    tau_scale=200.0,
    tau_floor=20.0,
    inverted=True,
)


def make_purkinje_na_channel(g_max: float) -> IonChannel:
    """Create the cerebellar Purkinje fast sodium channel (Na⁺).

    Uses Traub-Miles kinetics with VT = −58 mV to match the somatic NaF
    activation threshold of mammalian cerebellar Purkinje neurons recorded
    by De Schutter & Bower (1994) at 32 °C.

    Used as the Purkinje preset's Na⁺ channel.  Compared with the default
    HH52 Na⁺ channel (fitted to squid axon at 22 °C), the Traub-Miles form
    with VT = −58 mV places the α_m singularity (and approximate activation
    inflection) near −45 mV and prevents the ~5.2× Q10 overcorrection that
    caused premature Na⁺ inactivation.

    The channel exposes three gates: activation ``m`` (power 3), fast
    inactivation ``h`` (power 1), and slow voltage-dependent inactivation
    ``sNa`` (power 1; V½ = −50 mV, slope 8 mV, inverted Boltzmann).  The
    slow gate is mostly available at hyperpolarized potentials and decays
    toward zero on sustained depolarization, providing the escape route
    from depol-block plateaus reported in real Purkinje cells (#329).  The
    column is named ``sNa`` rather than ``s`` to avoid colliding with
    :func:`~patch_sim.channels.auxiliary.make_inar_channel`'s activation
    gate, which also appears in the Purkinje preset.

    References:
        - De Schutter & Bower (1994), J. Neurophysiol. 71:375 (m and h
          kinetics, recorded at 32 °C — use T_ref = 305.15 K).
        - Carter & Bean (2009), Neuron 64:898 (slow Na inactivation in
          cerebellar Purkinje cells — primary source for the sNa gate).
        - Do & Bean (2003), Neuron 39:109 (Na pacemaker channels,
          comparative).

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
            GatingVariable(
                name="sNa",
                power=1,
                alpha=_purkinje_alpha_sNa,
                beta=_purkinje_beta_sNa,
            ),
        ),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )


def make_purkinje_k_channel(g_max: float) -> IonChannel:
    """Create the cerebellar Purkinje delayed-rectifier potassium channel (K⁺).

    Uses Traub-Miles kinetics with VT = −58 mV to match the somatic KDR
    activation threshold of mammalian cerebellar Purkinje neurons recorded
    by De Schutter & Bower (1994) at 32 °C.

    Used as the Purkinje preset's K⁺ channel.

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
