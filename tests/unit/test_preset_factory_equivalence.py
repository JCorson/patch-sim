"""Transitional equivalence tests between preset factories and make_neuron.

These tests guarantee that each ``make_*`` preset factory in
``patch_sim.presets`` produces a :class:`~patch_sim.Neuron` identical to the
one returned by ``make_neuron(NEURON_PRESETS[name])``.  The tests are
deliberately temporary — they are deleted once the ``NEURON_PRESETS`` dict
is itself converted to map names to factories (issue #337).
"""

from __future__ import annotations

import pytest

from patch_sim.constants import (
    CA1_PYRAMIDAL,
    CORTICAL_PYRAMIDAL,
    DOPAMINERGIC,
    FAST_SPIKING_INTERNEURON,
    PURKINJE,
    SQUID_GIANT_AXON,
    STN,
    THALAMIC_RELAY,
    TRN,
)
from patch_sim.neuron_factory import make_neuron
from patch_sim.presets import (
    NEURON_PRESETS,
    make_ca1_pyramidal,
    make_cortical_pyramidal,
    make_dopaminergic,
    make_fast_spiking_interneuron,
    make_purkinje,
    make_squid_giant_axon,
    make_stn,
    make_thalamic_relay,
    make_trn,
)

_PRESET_FACTORIES = {
    SQUID_GIANT_AXON: make_squid_giant_axon,
    FAST_SPIKING_INTERNEURON: make_fast_spiking_interneuron,
    CORTICAL_PYRAMIDAL: make_cortical_pyramidal,
    PURKINJE: make_purkinje,
    DOPAMINERGIC: make_dopaminergic,
    THALAMIC_RELAY: make_thalamic_relay,
    CA1_PYRAMIDAL: make_ca1_pyramidal,
    STN: make_stn,
    TRN: make_trn,
}

_SCALAR_FIELDS = (
    "g_Na",
    "g_K",
    "g_NaL",
    "g_KL",
    "C_m",
    "v_rest",
    "Na_out",
    "Na_in",
    "K_out",
    "K_in",
    "Ca_out",
    "Ca_in",
    "T",
    "Q10",
    "T_ref",
    "area_cm2",
)


@pytest.mark.parametrize("preset_name", list(_PRESET_FACTORIES))
def test_factory_matches_make_neuron(preset_name: str) -> None:
    """Factory output must equal make_neuron(NEURON_PRESETS[name]).

    Verifies scalar fields, channel factories, additional channel composition,
    and calcium-dynamics parameters round-trip identically through the new
    factory path.
    """
    factory = _PRESET_FACTORIES[preset_name]
    via_factory = factory()
    via_make_neuron = make_neuron(NEURON_PRESETS[preset_name])

    for field in _SCALAR_FIELDS:
        assert getattr(via_factory, field) == getattr(via_make_neuron, field), (
            f"{preset_name}: {field} mismatch — factory={getattr(via_factory, field)} "
            f"vs make_neuron={getattr(via_make_neuron, field)}"
        )

    assert via_factory.na_channel_factory is via_make_neuron.na_channel_factory
    assert via_factory.k_channel_factory is via_make_neuron.k_channel_factory
    assert (
        via_factory.na_leak_channel_factory is via_make_neuron.na_leak_channel_factory
    )
    assert via_factory.k_leak_channel_factory is via_make_neuron.k_leak_channel_factory

    assert len(via_factory.additional_channels) == len(
        via_make_neuron.additional_channels
    )
    for ch_factory, ch_ref in zip(
        via_factory.additional_channels,
        via_make_neuron.additional_channels,
        strict=True,
    ):
        assert ch_factory.name == ch_ref.name, (
            f"{preset_name}: channel name mismatch — {ch_factory.name} vs {ch_ref.name}"
        )
        assert ch_factory.g_max == ch_ref.g_max, (
            f"{preset_name}: channel {ch_factory.name} g_max mismatch — "
            f"{ch_factory.g_max} vs {ch_ref.g_max}"
        )

    factory_ca = via_factory.calcium_dynamics
    ref_ca = via_make_neuron.calcium_dynamics
    if ref_ca is None:
        assert factory_ca is None, (
            f"{preset_name}: factory has unexpected CalciumDynamics"
        )
    else:
        assert factory_ca is not None, f"{preset_name}: factory missing CalciumDynamics"
        assert factory_ca.alpha_ca == pytest.approx(ref_ca.alpha_ca)
        assert factory_ca.tau_ca == pytest.approx(ref_ca.tau_ca)
        assert factory_ca.ca_rest == pytest.approx(ref_ca.ca_rest)
        assert factory_ca.ca_init == pytest.approx(ref_ca.ca_init)
