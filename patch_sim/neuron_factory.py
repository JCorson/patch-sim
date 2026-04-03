"""NeuronConfig/ChannelConfig dataclasses and make_neuron() factory.

Provides a declarative, UI-free way to describe and instantiate conductance-based
neurons with optional additional channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .additional_channels import (
    make_ical_channel,
    make_ican_channel,
    make_icat_channel,
    make_ih_channel,
    make_ika_channel,
    make_ikca_channel,
    make_ikir_channel,
    make_im_channel,
    make_inap_channel,
    make_inar_channel,
)
from .calcium import CalciumDynamics
from .channels import GoldmanSpec, IonChannel, IonSpecies, NernstSpec
from .constants import (
    DEFAULT_C_M,
    DEFAULT_CA_IN,
    DEFAULT_CA_OUT,
    DEFAULT_CL_IN,
    DEFAULT_CL_OUT,
    DEFAULT_G_K,
    DEFAULT_G_L,
    DEFAULT_G_NA,
    DEFAULT_K_IN,
    DEFAULT_K_OUT,
    DEFAULT_NA_IN,
    DEFAULT_NA_OUT,
    DEFAULT_T,
    DEFAULT_V_REST,
)
from .core_channels import make_k_channel, make_leak_channel, make_na_channel
from .neuron import Neuron


@dataclass(frozen=True)
class ChannelConfig:
    """Declarative description of one additional ion channel.

    Attributes:
        factory: Channel factory function (e.g. ``make_ih_channel``).
        g_max: Maximum conductance in mS/cm².
        extra_kwargs: Additional keyword arguments forwarded to *factory*.
    """

    factory: Callable[..., IonChannel]
    g_max: float
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NeuronConfig:
    """Declarative description of a conductance-based neuron configuration.

    All fields mirror the corresponding :class:`~patch_sim.Neuron` constructor
    parameters.  ``channels`` holds zero or more :class:`ChannelConfig`
    entries describing the additional channels to attach.

    Attributes:
        g_Na: Maximum sodium conductance in mS/cm².
        g_K: Maximum potassium conductance in mS/cm².
        g_L: Leak conductance in mS/cm².
        C_m: Membrane capacitance in µF/cm².
        v_rest: Resting membrane potential in mV.
        Na_out: Extracellular sodium concentration in mM.
        Na_in: Intracellular sodium concentration in mM.
        K_out: Extracellular potassium concentration in mM.
        K_in: Intracellular potassium concentration in mM.
        Cl_out: Extracellular chloride concentration in mM.
        Cl_in: Intracellular chloride concentration in mM.
        Ca_out: Extracellular calcium concentration in mM.
        Ca_in: Intracellular calcium concentration in mM.
        T: Temperature in Kelvin.
        na_channel_factory: Factory for the Na⁺ core channel. Defaults to HH52
            squid axon kinetics.
        k_channel_factory: Factory for the K⁺ core channel. Defaults to HH52
            squid axon kinetics.
        leak_channel_factory: Factory for the leak core channel. Defaults to
            HH52 squid axon kinetics.
        channels: Tuple of additional channel configs to include.
    """

    g_Na: float = DEFAULT_G_NA
    g_K: float = DEFAULT_G_K
    g_L: float = DEFAULT_G_L
    C_m: float = DEFAULT_C_M
    v_rest: float = DEFAULT_V_REST
    Na_out: float = DEFAULT_NA_OUT
    Na_in: float = DEFAULT_NA_IN
    K_out: float = DEFAULT_K_OUT
    K_in: float = DEFAULT_K_IN
    Cl_out: float = DEFAULT_CL_OUT
    Cl_in: float = DEFAULT_CL_IN
    Ca_out: float = DEFAULT_CA_OUT
    Ca_in: float = DEFAULT_CA_IN
    T: float = DEFAULT_T
    na_channel_factory: Callable[[float], IonChannel] = field(default=make_na_channel)
    k_channel_factory: Callable[[float], IonChannel] = field(default=make_k_channel)
    leak_channel_factory: Callable[[float], IonChannel] = field(
        default=make_leak_channel
    )
    channels: tuple[ChannelConfig, ...] = ()


#: Maps short channel names to their factory functions.
CHANNEL_REGISTRY: dict[str, Callable[..., IonChannel]] = {
    "ih": make_ih_channel,
    "ika": make_ika_channel,
    "inap": make_inap_channel,
    "inar": make_inar_channel,
    "im": make_im_channel,
    "ikir": make_ikir_channel,
    "ikca": make_ikca_channel,
    "ical": make_ical_channel,
    "icat": make_icat_channel,
    "ican": make_ican_channel,
}


def _needs_calcium(channels: tuple[IonChannel, ...]) -> bool:
    """Return True if any channel carries calcium ions.

    Args:
        channels: Built IonChannel instances to inspect.

    Returns:
        True if at least one channel uses ``IonSpecies.CALCIUM``.
    """
    for ch in channels:
        spec = ch.reversal_spec
        if isinstance(spec, NernstSpec) and spec.species is IonSpecies.CALCIUM:
            return True
        if isinstance(spec, GoldmanSpec):
            for species, _ in spec.permeabilities:
                if species is IonSpecies.CALCIUM:
                    return True
    return False


def make_neuron(config: NeuronConfig) -> Neuron:
    """Build a :class:`~patch_sim.Neuron` from a :class:`NeuronConfig`.

    Automatically detects whether any additional channel requires calcium
    dynamics and attaches a :class:`CalciumDynamics` instance when needed.

    Args:
        config: Declarative neuron configuration.

    Returns:
        A fully constructed :class:`~patch_sim.Neuron` instance.
    """
    built_channels = tuple(
        cc.factory(g_max=cc.g_max, **cc.extra_kwargs) for cc in config.channels
    )
    calcium_dynamics = CalciumDynamics() if _needs_calcium(built_channels) else None
    return Neuron(
        g_Na=config.g_Na,
        g_K=config.g_K,
        g_L=config.g_L,
        C_m=config.C_m,
        v_rest=config.v_rest,
        Na_out=config.Na_out,
        Na_in=config.Na_in,
        K_out=config.K_out,
        K_in=config.K_in,
        Cl_out=config.Cl_out,
        Cl_in=config.Cl_in,
        Ca_out=config.Ca_out,
        Ca_in=config.Ca_in,
        T=config.T,
        na_channel_factory=config.na_channel_factory,
        k_channel_factory=config.k_channel_factory,
        leak_channel_factory=config.leak_channel_factory,
        additional_channels=built_channels,
        calcium_dynamics=calcium_dynamics,
    )
