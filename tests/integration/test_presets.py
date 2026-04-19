"""Tests for patch_sim.presets — preset data structures and build_protocol_from_preset.

Focuses on the preset catalogue itself and verifies that neuron-specific
protocol adjustments are actually applied (not just that the function runs
without error).
"""

import numpy as np
import pytest

import patch_sim
from patch_sim.additional_channels import make_icat_channel
from patch_sim.analysis.membrane_test import run_membrane_test
from patch_sim.constants import (
    ACTION_POTENTIAL,
    CA1_PYRAMIDAL,
    CORTICAL_PYRAMIDAL,
    DOPAMINERGIC,
    FAST_SPIKING_INTERNEURON,
    PURKINJE,
    REPETITIVE_FIRING,
    SQUID_GIANT_AXON,
    STN,
    THALAMIC_RELAY,
    TRN,
)
from patch_sim.core_channels import (
    make_pospischil_k_channel,
    make_pospischil_na_channel,
)
from patch_sim.neuron_factory import make_neuron
from patch_sim.presets import (
    NEURON_PRESET_NAMES,
    NEURON_PRESETS,
    NEURON_PROTOCOL_ADJUSTMENTS,
    PROTOCOL_PRESET_NAMES,
    PROTOCOL_PRESETS,
    build_protocol_from_preset,
)

SAMPLING_FREQUENCY = 10_000.0  # Hz — fast enough for correct shapes


# ---------------------------------------------------------------------------
# Preset catalogue integrity
# ---------------------------------------------------------------------------


def test_neuron_preset_names_matches_neuron_presets_keys() -> None:
    """NEURON_PRESET_NAMES is exactly the keys of NEURON_PRESETS in order."""
    assert NEURON_PRESET_NAMES == list(NEURON_PRESETS.keys())


def test_protocol_preset_names_matches_protocol_presets_keys() -> None:
    """PROTOCOL_PRESET_NAMES is exactly the keys of PROTOCOL_PRESETS in order."""
    assert PROTOCOL_PRESET_NAMES == list(PROTOCOL_PRESETS.keys())


def test_neuron_protocol_adjustments_keys_are_known_neuron_presets() -> None:
    """Every key in NEURON_PROTOCOL_ADJUSTMENTS names a known neuron preset."""
    for neuron_name in NEURON_PROTOCOL_ADJUSTMENTS:
        assert neuron_name in NEURON_PRESETS, (
            f"Adjustment key {neuron_name!r} is not a known neuron preset"
        )


def test_neuron_protocol_adjustment_protocol_keys_are_known() -> None:
    """Every protocol name in NEURON_PROTOCOL_ADJUSTMENTS is a known preset."""
    for neuron_name, proto_map in NEURON_PROTOCOL_ADJUSTMENTS.items():
        for proto_name in proto_map:
            assert proto_name in PROTOCOL_PRESETS, (
                f"Adjustment for neuron {neuron_name!r} references unknown "
                f"protocol preset {proto_name!r}"
            )


# ---------------------------------------------------------------------------
# build_protocol_from_preset — each preset produces valid output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", PROTOCOL_PRESET_NAMES)
def test_build_protocol_from_preset_produces_valid_output(preset_name: str) -> None:
    """Each protocol preset returns a non-empty 2-D finite ndarray."""
    result = build_protocol_from_preset(
        preset_name, sampling_frequency=SAMPLING_FREQUENCY
    )
    assert isinstance(result, np.ndarray) and result.ndim == 2 and result.shape[0] > 0
    assert bool(np.all(np.isfinite(result))), (
        f"Non-finite values in preset '{preset_name}'"
    )


def test_build_protocol_from_preset_unknown_raises_key_error() -> None:
    """A non-existent preset name raises KeyError."""
    with pytest.raises(KeyError, match="Unknown protocol preset"):
        build_protocol_from_preset("NoSuchPreset")


# ---------------------------------------------------------------------------
# Neuron-specific protocol adjustments are applied
# ---------------------------------------------------------------------------


def _protocol_total_samples(preset_name: str, neuron_name: str | None) -> int:
    """Return the total number of samples in the first sweep of a preset."""
    result = build_protocol_from_preset(
        preset_name,
        neuron_preset=neuron_name,
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    return result[0].size


def test_neuron_protocol_adjustments_change_stimulus_duration() -> None:
    """Neuron adjustments that change stimulus_duration produce a different length.

    The Dopaminergic Neuron adjusts Repetitive Firing from 180 ms to 480 ms —
    the longer duration must result in a longer stimulus array.
    """
    base_samples = _protocol_total_samples(REPETITIVE_FIRING, None)
    adjusted_samples = _protocol_total_samples(REPETITIVE_FIRING, DOPAMINERGIC)
    assert adjusted_samples > base_samples, (
        "Dopaminergic Neuron adjustment should produce a longer array"
    )


def test_neuron_protocol_adjustments_not_applied_for_unknown_neuron() -> None:
    """Passing a neuron preset with no adjustments returns the base-preset output."""
    # "Squid Giant Axon (Classic HH)" has no adjustments defined.
    base_samples = _protocol_total_samples(REPETITIVE_FIRING, None)
    squid_samples = _protocol_total_samples(REPETITIVE_FIRING, SQUID_GIANT_AXON)
    assert squid_samples == base_samples


@pytest.mark.parametrize(
    "neuron_name",
    [
        n
        for n in NEURON_PRESET_NAMES
        if n in NEURON_PROTOCOL_ADJUSTMENTS
        and REPETITIVE_FIRING in NEURON_PROTOCOL_ADJUSTMENTS[n]
    ],
)
def test_repetitive_firing_adjusted_for_all_configured_neurons(
    neuron_name: str,
) -> None:
    """Every neuron with a Repetitive Firing adjustment produces a valid protocol."""
    result = build_protocol_from_preset(
        REPETITIVE_FIRING,
        neuron_preset=neuron_name,
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    assert isinstance(result, np.ndarray) and result.shape[0] == 1
    assert result[0].size > 0 and bool(np.all(np.isfinite(result[0])))


def test_caller_overrides_take_precedence_over_neuron_adjustments() -> None:
    """Caller-supplied overrides win over both the base preset and neuron adjustments.

    Dopaminergic Neuron sets stimulus_duration=480; override to 50 ms.
    The resulting array should match a plain 50 ms stimulus.
    """
    base_50 = build_protocol_from_preset(
        REPETITIVE_FIRING,
        sampling_frequency=SAMPLING_FREQUENCY,
        pre_stimulus_duration=10.0,
        stimulus_duration=50.0,
        post_stimulus_duration=10.0,
    )
    overridden = build_protocol_from_preset(
        REPETITIVE_FIRING,
        neuron_preset=DOPAMINERGIC,
        sampling_frequency=SAMPLING_FREQUENCY,
        pre_stimulus_duration=10.0,
        stimulus_duration=50.0,
        post_stimulus_duration=10.0,
    )
    assert base_50[0].size == overridden[0].size


# ---------------------------------------------------------------------------
# Top-level patch_sim re-export
# ---------------------------------------------------------------------------


def test_build_protocol_from_preset_exported_from_patch_sim() -> None:
    """build_protocol_from_preset is accessible directly from the patch_sim package."""
    assert callable(patch_sim.build_protocol_from_preset)
    result = patch_sim.build_protocol_from_preset(
        ACTION_POTENTIAL, sampling_frequency=SAMPLING_FREQUENCY
    )
    assert isinstance(result, np.ndarray) and result.shape[0] == 1


# ---------------------------------------------------------------------------
# Presets that use Pospischil factories (mammalian kinetics at 34 °C)
# ---------------------------------------------------------------------------


def test_cortical_pyramidal_uses_pospischil_na_factory() -> None:
    """Cortical Pyramidal preset wires the Pospischil Na⁺ channel factory."""
    config = NEURON_PRESETS[CORTICAL_PYRAMIDAL]
    assert config.na_channel_factory is make_pospischil_na_channel


def test_cortical_pyramidal_uses_pospischil_k_factory() -> None:
    """Cortical Pyramidal preset wires the Pospischil K⁺ channel factory."""
    config = NEURON_PRESETS[CORTICAL_PYRAMIDAL]
    assert config.k_channel_factory is make_pospischil_k_channel


def test_fsi_uses_pospischil_na_factory() -> None:
    """FSI preset wires the Pospischil Na⁺ channel factory (issue #231)."""
    config = NEURON_PRESETS[FAST_SPIKING_INTERNEURON]
    assert config.na_channel_factory is make_pospischil_na_channel


def test_fsi_uses_pospischil_k_factory() -> None:
    """FSI preset wires the Pospischil K⁺ channel factory (issue #231)."""
    config = NEURON_PRESETS[FAST_SPIKING_INTERNEURON]
    assert config.k_channel_factory is make_pospischil_k_channel


def test_ca1_uses_pospischil_na_factory() -> None:
    """CA1 preset wires the Pospischil Na⁺ channel factory (issue #231)."""
    config = NEURON_PRESETS[CA1_PYRAMIDAL]
    assert config.na_channel_factory is make_pospischil_na_channel


def test_ca1_uses_pospischil_k_factory() -> None:
    """CA1 preset wires the Pospischil K⁺ channel factory (issue #231)."""
    config = NEURON_PRESETS[CA1_PYRAMIDAL]
    assert config.k_channel_factory is make_pospischil_k_channel


# ---------------------------------------------------------------------------
# Passive membrane properties are physiologically differentiated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset_name, tau_lo, tau_hi, rin_lo, rin_hi",
    [
        # (preset_name, τ_m_min_ms, τ_m_max_ms, R_in_min_kΩcm², R_in_max_kΩcm²)
        # Passive properties are measured on the channel-blocked neuron, so
        # R_in = 1/(g_NaL+g_KL) and τ_m = C_m/(g_NaL+g_KL).
        (SQUID_GIANT_AXON, 2.5, 4.5, 2.5, 4.5),  # g_total=0.3  τ_m≈3.3 ms
        (FAST_SPIKING_INTERNEURON, 0.4, 1.0, 0.4, 1.0),  # g_total=1.5  τ_m≈0.67 ms
        (CORTICAL_PYRAMIDAL, 17.0, 23.0, 17.0, 23.0),  # g_total=0.05 τ_m≈20 ms
        (PURKINJE, 45.0, 55.0, 45.0, 55.0),  # g_total=0.02 τ_m≈50 ms
        (DOPAMINERGIC, 2.5, 4.5, 2.5, 4.5),  # g_total=0.3  τ_m≈3.3 ms
        (THALAMIC_RELAY, 8.0, 12.0, 8.0, 12.0),  # g_total=0.1  τ_m≈10 ms
        (CA1_PYRAMIDAL, 17.0, 23.0, 17.0, 23.0),  # g_total=0.05 τ_m≈20 ms
        (STN, 2.5, 5.5, 2.5, 5.5),  # g_total=0.25 τ_m≈4 ms
        (TRN, 10.0, 15.0, 10.0, 15.0),  # g_total=0.08 τ_m≈12.5 ms
    ],
)
def test_preset_passive_properties_in_physiological_range(
    preset_name: str,
    tau_lo: float,
    tau_hi: float,
    rin_lo: float,
    rin_hi: float,
) -> None:
    """Each preset's τ_m and R_in fall within the expected physiological range.

    Passive properties are extracted from a channel-blocked copy of the neuron
    so R_in = 1/(g_NaL+g_KL) and τ_m = C_m/(g_NaL+g_KL) exactly.  Each preset's
    bounds in the parametrize
    table are set per-preset to encompass the analytically expected value with
    enough margin to tolerate minor numerical artefacts in the exponential fit.

    Args:
        preset_name: Key in NEURON_PRESETS.
        tau_lo: Lower bound for the membrane time constant in ms.
        tau_hi: Upper bound for the membrane time constant in ms.
        rin_lo: Lower bound for the input resistance in kΩ·cm².
        rin_hi: Upper bound for the input resistance in kΩ·cm².
    """
    neuron = make_neuron(NEURON_PRESETS[preset_name])
    props = run_membrane_test(neuron)
    assert props is not None, f"Preset '{preset_name}': run_membrane_test returned None"
    assert tau_lo <= props.time_constant <= tau_hi, (
        f"Preset '{preset_name}': τ_m = {props.time_constant:.2f} ms "
        f"outside [{tau_lo}, {tau_hi}]"
    )
    assert rin_lo <= props.input_resistance <= rin_hi, (
        f"Preset '{preset_name}': R_in = {props.input_resistance:.2f} kΩ·cm² "
        f"outside [{rin_lo}, {rin_hi}]"
    )


# ---------------------------------------------------------------------------
# TRN burst-enabling conditions
# ---------------------------------------------------------------------------


def test_trn_preset_vrest_is_physiological() -> None:
    """TRN v_rest must be −77 mV — the physiological slice value.

    This is the primary claim of the K_out=4.0 mM retuning: with E_K ≈ −95 mV
    the K⁺ leak has 18 mV of outward driving force at rest, allowing a small
    Na⁺ leak to balance ICaT window current at −77 mV.  If v_rest drifts, the
    ICaT inactivation gate ft will shift and burst firing will be impaired.
    """
    config = NEURON_PRESETS[TRN]
    assert config.v_rest == pytest.approx(-77.0)


def test_trn_icat_ft_inf_at_vrest_enables_burst_firing() -> None:
    """ICaT inactivation gate ft_inf at TRN v_rest must be ≈ 0.42.

    ft_inf ≈ 0.42 at −77 mV means ICaT is substantially de-inactivated at
    rest, enabling the post-inhibitory rebound burst and burst character on
    depolarising steps that define TRN firing.  Previously (K_out=7.8 mM,
    E_K ≈ −77 mV) v_rest settled at −66 mV where ft_inf ≈ 0.17 — too
    inactivated for reliable burst firing.
    """
    channel = make_icat_channel()
    ft_var = next(gv for gv in channel.gating_variables if gv.name == "ft")
    v_rest = NEURON_PRESETS[TRN].v_rest
    alpha = ft_var.alpha(v_rest, 0.0)
    beta = ft_var.beta(v_rest, 0.0)
    ft_inf = alpha / (alpha + beta)
    assert ft_inf == pytest.approx(0.42, abs=0.02)
