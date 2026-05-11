"""McCormick & Huguenard (1992) thalamic relay channel factories.

Source: McCormick, D.A. & Huguenard, J.R. (1992) A model of the
electrophysiological properties of thalamocortical relay neurons.
J. Neurophysiol. 68:1384–1400.

Parameterisation: Pospischil, M. et al. (2008) Minimal Hodgkin-Huxley type
models for different classes of cortical and thalamic neurons.
Biol. Cybern. 99:427–441, Table 2 (TC cell).
"""

from ..constants import DEFAULT_G_ICAT
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
from .auxiliary import _alpha_dt, _beta_dt
from .base import GatingVariable, IonChannel, IonSpecies, NernstSpec

__all__ = [
    "THALAMIC_RELAY_VT",
    "thalamic_relay_alpha_m",
    "thalamic_relay_beta_m",
    "thalamic_relay_alpha_h",
    "thalamic_relay_beta_h",
    "thalamic_relay_alpha_n",
    "thalamic_relay_beta_n",
    "make_thalamic_relay_na_channel",
    "make_thalamic_relay_k_channel",
    "make_thalamic_relay_icat_channel",
]

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

    Used as the Thalamic Relay preset's Na⁺ channel.  Compared with the
    default HH52 Na⁺ channel (fitted to squid axon at 22 °C), the Traub-Miles
    form with VT = −52 mV shifts the activation threshold ~13 mV depolarized
    and slows inactivation, preventing the ~5.2× Q10 overcorrection that
    caused premature Na⁺ inactivation.

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

    Used as the Thalamic Relay preset's K⁺ channel.

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


# Thalamic-relay-specific ICaT inactivation: 5× slower than the Destexhe
# (1994) global default (tau_scale=20 ms).  McCormick & Huguenard (1992) report
# tau_h_T ≈ 25–40 ms in the depolarized range used during the LTS plateau,
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
    200–500 Hz) on hyperpolarizing-step release — the defining feature of
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
