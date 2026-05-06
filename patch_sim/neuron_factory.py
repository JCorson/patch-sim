"""NeuronConfig/ChannelConfig dataclasses and make_neuron() factory.

Provides a declarative, UI-free way to describe and instantiate conductance-based
neurons with optional additional channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .additional_channels import (
    make_cav13_channel,
    make_ical_channel,
    make_ican_channel,
    make_icat_channel,
    make_ih_channel,
    make_ika_channel,
    make_ikca_channel,
    make_ikir_channel,
    make_ikv31_channel,
    make_im_channel,
    make_inap_channel,
    make_inar_channel,
    make_katp_channel,
    make_sk_channel,
)
from .calcium import CalciumDynamics
from .channels import GoldmanSpec, IonChannel, IonSpecies, NernstSpec
from .constants import (
    DEFAULT_C_M,
    DEFAULT_CA_IN,
    DEFAULT_CA_OUT,
    DEFAULT_G_K,
    DEFAULT_G_KL,
    DEFAULT_G_NA,
    DEFAULT_G_NAL,
    DEFAULT_K_IN,
    DEFAULT_K_OUT,
    DEFAULT_NA_IN,
    DEFAULT_NA_OUT,
    DEFAULT_Q10,
    DEFAULT_T,
    DEFAULT_T_REF,
    DEFAULT_V_REST,
)
from .core_channels import (
    make_k_channel,
    make_k_leak_channel,
    make_mainen_sejnowski_kv_channel,
    make_na_channel,
    make_na_leak_channel,
)
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
        g_NaL: Na⁺ leak conductance in mS/cm².
        g_KL: K⁺ leak conductance in mS/cm².
        C_m: Membrane capacitance in µF/cm².
        v_rest: Resting membrane potential in mV.
        Na_out: Extracellular sodium concentration in mM.
        Na_in: Intracellular sodium concentration in mM.
        K_out: Extracellular potassium concentration in mM.
        K_in: Intracellular potassium concentration in mM.
        Ca_out: Extracellular calcium concentration in mM.
        Ca_in: Intracellular calcium concentration in mM.
        T: Temperature in Kelvin.
        Q10: Q10 temperature coefficient for gating kinetics (dimensionless).
        T_ref: Reference temperature in Kelvin for Q10 scaling.
        na_channel_factory: Factory for the Na⁺ core channel. Defaults to HH52
            squid axon kinetics.
        k_channel_factory: Factory for the K⁺ core channel. Defaults to HH52
            squid axon kinetics.
        na_leak_channel_factory: Factory for the Na⁺ leak channel. Defaults to
            make_na_leak_channel.
        k_leak_channel_factory: Factory for the K⁺ leak channel. Defaults to
            make_k_leak_channel.
        channels: Tuple of additional channel configs to include.
        calcium_dynamics: Optional per-preset :class:`~patch_sim.CalciumDynamics`
            override.  When set, ``make_neuron`` uses it directly instead of
            constructing ``CalciumDynamics()`` with module-level defaults.
            Raises :class:`ValueError` if provided for a configuration whose
            channels carry no Ca²⁺.  ``None`` (default) falls back to
            auto-detection: a default ``CalciumDynamics()`` is created whenever
            at least one channel carries Ca²⁺, and ``None`` is passed otherwise.
        area_cm2: Total membrane surface area in cm².  Forwarded onto the
            built :class:`~patch_sim.Neuron` as a physical attribute.  Not
            consumed by the ODE solver — single-compartment HH dynamics are
            scale-invariant in the per-area units used elsewhere — but read
            by the analysis layer (e.g. :func:`~patch_sim.run_membrane_test`)
            to report passive properties in absolute units (MΩ, pF) alongside
            the per-area values.  ``None`` (default) means only per-area
            density values are reported.  cm² is the natural unit here for
            consistency with mS/cm², µF/cm², µA/cm² used elsewhere in the
            model; representative values fall in the ``1e-6`` to ``3e-4``
            cm² range for typical mammalian neurons.
    """

    g_Na: float = DEFAULT_G_NA
    g_K: float = DEFAULT_G_K
    g_NaL: float = DEFAULT_G_NAL
    g_KL: float = DEFAULT_G_KL
    C_m: float = DEFAULT_C_M
    v_rest: float = DEFAULT_V_REST
    Na_out: float = DEFAULT_NA_OUT
    Na_in: float = DEFAULT_NA_IN
    K_out: float = DEFAULT_K_OUT
    K_in: float = DEFAULT_K_IN
    Ca_out: float = DEFAULT_CA_OUT
    Ca_in: float = DEFAULT_CA_IN
    T: float = DEFAULT_T
    Q10: float = DEFAULT_Q10
    T_ref: float = DEFAULT_T_REF
    na_channel_factory: Callable[[float], IonChannel] = field(default=make_na_channel)
    k_channel_factory: Callable[[float], IonChannel] = field(default=make_k_channel)
    na_leak_channel_factory: Callable[[float], IonChannel] = field(
        default=make_na_leak_channel
    )
    k_leak_channel_factory: Callable[[float], IonChannel] = field(
        default=make_k_leak_channel
    )
    channels: tuple[ChannelConfig, ...] = ()
    calcium_dynamics: CalciumDynamics | None = None
    area_cm2: float | None = None

    def __post_init__(self) -> None:
        """Validate Q10, T_ref, and area_cm2 on construction.

        Raises:
            ValueError: If Q10 is not positive, T_ref is not positive, or
                ``area_cm2`` is provided and not strictly positive.
        """
        if self.Q10 <= 0:
            raise ValueError("Q10 must be positive.")
        if self.T_ref <= 0:
            raise ValueError("T_ref must be positive (in Kelvin).")
        if self.area_cm2 is not None and self.area_cm2 <= 0:
            raise ValueError("area_cm2 must be positive when provided.")


#: Maps short channel names to their factory functions.
CHANNEL_REGISTRY: dict[str, Callable[..., IonChannel]] = {
    "ih": make_ih_channel,
    "ika": make_ika_channel,
    "ikv31": make_ikv31_channel,
    "mskv": make_mainen_sejnowski_kv_channel,
    "inap": make_inap_channel,
    "inar": make_inar_channel,
    "im": make_im_channel,
    "katp": make_katp_channel,
    "ikir": make_ikir_channel,
    "ikca": make_ikca_channel,
    "ical": make_ical_channel,
    "cav13": make_cav13_channel,
    "icat": make_icat_channel,
    "ican": make_ican_channel,
    "sk": make_sk_channel,
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
    if config.calcium_dynamics is not None:
        if not _needs_calcium(built_channels):
            raise ValueError(
                "calcium_dynamics provided but none of the configured channels "
                "carry Ca²⁺ — remove the override or add a Ca²⁺ channel."
            )
        calcium_dynamics: CalciumDynamics | None = config.calcium_dynamics
    elif _needs_calcium(built_channels):
        calcium_dynamics = CalciumDynamics()
    else:
        calcium_dynamics = None
    return Neuron(
        g_Na=config.g_Na,
        g_K=config.g_K,
        g_NaL=config.g_NaL,
        g_KL=config.g_KL,
        C_m=config.C_m,
        v_rest=config.v_rest,
        Na_out=config.Na_out,
        Na_in=config.Na_in,
        K_out=config.K_out,
        K_in=config.K_in,
        Ca_out=config.Ca_out,
        Ca_in=config.Ca_in,
        T=config.T,
        Q10=config.Q10,
        T_ref=config.T_ref,
        na_channel_factory=config.na_channel_factory,
        k_channel_factory=config.k_channel_factory,
        na_leak_channel_factory=config.na_leak_channel_factory,
        k_leak_channel_factory=config.k_leak_channel_factory,
        additional_channels=built_channels,
        calcium_dynamics=calcium_dynamics,
        area_cm2=config.area_cm2,
    )
