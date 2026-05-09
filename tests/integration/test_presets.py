"""Tests for patch_sim.presets — preset data structures and build_protocol_from_preset.

Focuses on the preset catalogue itself and verifies that neuron-specific
protocol adjustments are actually applied (not just that the function runs
without error).
"""

import numpy as np
import pytest

import patch_sim
from patch_sim.analysis.membrane_test import run_membrane_test
from patch_sim.channels import (
    make_dopaminergic_k_channel,
    make_dopaminergic_na_channel,
    make_nav11_channel,
    make_nav12_channel,
    make_pospischil_k_channel,
    make_purkinje_k_channel,
    make_purkinje_na_channel,
    make_thalamic_relay_k_channel,
    make_thalamic_relay_na_channel,
    make_trn_icat_channel,
    make_trn_k_channel,
    make_trn_na_channel,
)
from patch_sim.channels.base import IonChannel
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
from patch_sim.neuron import Neuron
from patch_sim.presets import (
    NEURON_PRESET_NAMES,
    NEURON_PRESETS,
    NEURON_PROTOCOL_ADJUSTMENTS,
    PROTOCOL_PRESET_NAMES,
    PROTOCOL_PRESETS,
    build_protocol_from_preset,
)

SAMPLING_FREQUENCY = 10_000.0  # Hz — fast enough for correct shapes


def _kinetics_match(actual: IonChannel, factory) -> None:
    """Assert that ``actual``'s gate kinetics match a channel built by ``factory``.

    Compares gate names (set-equal) and each gate's alpha/beta function output
    at four representative voltages.  Used as a structural replacement for
    factory-identity assertions after issue #320.
    """
    reference = factory(g_max=1.0)
    actual_gates = {gv.name: gv for gv in actual.gating_variables}
    expected_gates = {gv.name: gv for gv in reference.gating_variables}
    assert actual_gates.keys() == expected_gates.keys(), (
        f"gate names differ: actual={set(actual_gates)} expected={set(expected_gates)}"
    )
    for V in (-65.0, -30.0, 0.0, 30.0):
        for name, gv_a in actual_gates.items():
            gv_e = expected_gates[name]
            assert gv_a.alpha(V, 0.0) == pytest.approx(gv_e.alpha(V, 0.0)), (
                f"gate {name!r} alpha differs at V={V}"
            )
            assert gv_a.beta(V, 0.0) == pytest.approx(gv_e.beta(V, 0.0)), (
                f"gate {name!r} beta differs at V={V}"
            )


def _channel_by_name(neuron: Neuron, name: str) -> IonChannel:
    """Return the unique channel with ``name`` on ``neuron``.

    Raises:
        KeyError: If no channel with that name is present.
    """
    by_name = {ch.name: ch for ch in neuron.channels}
    return by_name[name]


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

    The SNc Dopaminergic Neuron adjusts Repetitive Firing from 180 ms to 3000 ms —
    the longer duration must result in a longer stimulus array.
    """
    base_samples = _protocol_total_samples(REPETITIVE_FIRING, None)
    adjusted_samples = _protocol_total_samples(REPETITIVE_FIRING, DOPAMINERGIC)
    assert adjusted_samples > base_samples, (
        "SNc Dopaminergic Neuron adjustment should produce a longer array"
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

    Dopaminergic Neuron sets stimulus_duration=3000; override to 50 ms.
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
    """Cortical Pyramidal preset wires the Nav1.2-flavoured Na⁺ channel factory.

    make_nav12_channel always includes the sNa12 slow inactivation gate
    (#327 depol-block recovery) — no factory wrapping needed.
    """
    neuron = NEURON_PRESETS[CORTICAL_PYRAMIDAL]()
    _kinetics_match(_channel_by_name(neuron, "Na"), make_nav12_channel)


def test_cortical_pyramidal_uses_mainen_sejnowski_kv_via_channels() -> None:
    """Cortical Pyramidal includes Mainen-Sejnowski Kv via channels list (#311)."""
    neuron = NEURON_PRESETS[CORTICAL_PYRAMIDAL]()
    channel_names = {ch.name for ch in neuron.channels}
    assert "Kv" in channel_names
    # M-S Kv is the sole delayed rectifier — the HH-style core "K" channel
    # is omitted from the channels list entirely (issue #311 / #320).
    assert "K" not in channel_names


def test_fsi_uses_nav11_factory() -> None:
    """FSI preset wires the Nav1.1-flavoured Na⁺ channel factory (issue #231)."""
    neuron = NEURON_PRESETS[FAST_SPIKING_INTERNEURON]()
    _kinetics_match(_channel_by_name(neuron, "Na"), make_nav11_channel)


def test_fsi_uses_pospischil_k_factory() -> None:
    """FSI preset wires the Pospischil K⁺ channel factory (issue #231)."""
    neuron = NEURON_PRESETS[FAST_SPIKING_INTERNEURON]()
    _kinetics_match(_channel_by_name(neuron, "K"), make_pospischil_k_channel)


def test_ca1_uses_nav12_factory() -> None:
    """CA1 preset wires the Nav1.2-flavoured Na⁺ channel factory (issue #231).

    make_nav12_channel always includes the sNa12 slow inactivation gate
    (#328 depol-block recovery) — no factory wrapping needed.
    """
    neuron = NEURON_PRESETS[CA1_PYRAMIDAL]()
    _kinetics_match(_channel_by_name(neuron, "Na"), make_nav12_channel)


def test_ca1_uses_pospischil_k_factory() -> None:
    """CA1 preset wires the Pospischil K⁺ channel factory (issue #231)."""
    neuron = NEURON_PRESETS[CA1_PYRAMIDAL]()
    _kinetics_match(_channel_by_name(neuron, "K"), make_pospischil_k_channel)


def test_thalamic_relay_uses_mh92_na_factory() -> None:
    """Thalamic Relay preset wires the McCormick-Huguenard Na⁺ factory (issue #241)."""
    neuron = NEURON_PRESETS[THALAMIC_RELAY]()
    _kinetics_match(_channel_by_name(neuron, "Na"), make_thalamic_relay_na_channel)


def test_thalamic_relay_uses_mh92_k_factory() -> None:
    """Thalamic Relay preset wires the McCormick-Huguenard K⁺ factory (issue #241)."""
    neuron = NEURON_PRESETS[THALAMIC_RELAY]()
    _kinetics_match(_channel_by_name(neuron, "K"), make_thalamic_relay_k_channel)


def test_thalamic_relay_t_ref_is_mccormick_huguenard_recording_temp() -> None:
    """Thalamic Relay T_ref matches McCormick & Huguenard (1992) 36 °C recording temp.

    Prevents regression to the HH52 default (295.15 K = 22 °C) which applied
    a ~5.2× Q10 overcorrection and distorted Na⁺ inactivation kinetics.
    """
    assert NEURON_PRESETS[THALAMIC_RELAY]().T_ref == pytest.approx(309.15)


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
        (PURKINJE, 18.0, 28.0, 18.0, 28.0),  # g_total=0.044 τ_m≈22.7 ms
        # DOPAMINERGIC post-#318 review: g_total = g_NaL + g_KL = 0.040 mS/cm²
        # (split 0.012 / 0.028 to keep V_leak ≈ −45 mV) gives τ_m ≈ 25 ms,
        # within the SNc DA literature band (20–30 ms; Wolfart et al. 2001).
        (DOPAMINERGIC, 20.0, 30.0, 20.0, 30.0),
        (THALAMIC_RELAY, 5.0, 9.0, 5.0, 9.0),  # g_total=0.15  τ_m≈6.7 ms
        (CA1_PYRAMIDAL, 17.0, 23.0, 17.0, 23.0),  # g_total=0.05 τ_m≈20 ms
        (STN, 22.0, 28.0, 22.0, 28.0),  # g_total=0.04 τ_m≈25 ms
        (TRN, 10.0, 15.0, 10.0, 15.0),  # g_total=0.07 τ_m≈14.3 ms
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
    neuron = NEURON_PRESETS[preset_name]()
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
    """TRN v_rest must be −80 mV — a stable, physiologically valid resting potential.

    −80 mV is chosen over −77 mV because the ICaT window current at −77 mV
    (ft_inf ≈ 0.42) creates ~3.1 mS/cm² of negative conductance — far exceeding
    g_total = 0.07 mS/cm² and making −77 mV an unstable equilibrium in the
    steady-state conductance picture.  At −80 mV, the ft gate is at its
    half-inactivation point (ft_inf = 0.50) and the ICaT negative conductance
    falls to ~0.038 mS/cm², which g_total safely overcomes.  −80 mV is within
    the physiological range reported by Huguenard & Prince (1992) for TRN cells.
    """
    assert NEURON_PRESETS[TRN]().v_rest == pytest.approx(-80.0)


def test_trn_icat_ft_inf_at_vrest_enables_burst_firing() -> None:
    """ICaT inactivation gate ft_inf at TRN v_rest must be ≈ 0.50.

    ft_inf = 0.50 at −80 mV (the ICaT half-inactivation voltage) means ICaT
    is half de-inactivated at rest, enabling the post-inhibitory rebound burst
    and burst character on depolarising steps that define TRN firing.
    Previously (K_out=7.8 mM, E_K ≈ −77 mV) v_rest settled at −66 mV where
    ft_inf ≈ 0.17 — too inactivated for reliable burst firing.

    The TRN preset uses :func:`make_trn_icat_channel` (issue #295), whose
    ``ft_inf`` is bit-identical to the Destexhe (1994) default — the
    sigmoid-tau modification touches only ``tau_ft``.
    """
    channel = make_trn_icat_channel()
    ft_var = next(gv for gv in channel.gating_variables if gv.name == "ft")
    v_rest = NEURON_PRESETS[TRN]().v_rest
    alpha = ft_var.alpha(v_rest, 0.0)
    beta = ft_var.beta(v_rest, 0.0)
    ft_inf = alpha / (alpha + beta)
    assert ft_inf == pytest.approx(0.50, abs=0.02)


def test_trn_uses_huguenard_na_factory() -> None:
    """TRN preset must wire make_trn_na_channel as its Na⁺ channel factory.

    Verifies that the Huguenard & Prince (1992) / Pospischil (2008) RE-cell
    Na⁺ kinetics (VT = −67 mV, recorded at 36 °C) are used instead of the
    default HH52 squid axon kinetics.
    """
    _kinetics_match(_channel_by_name(NEURON_PRESETS[TRN](), "Na"), make_trn_na_channel)


def test_trn_uses_huguenard_k_factory() -> None:
    """TRN preset must wire make_trn_k_channel as its K⁺ channel factory.

    Verifies that the Huguenard & Prince (1992) / Pospischil (2008) RE-cell
    K⁺ kinetics (VT = −67 mV, recorded at 36 °C) are used instead of the
    default HH52 squid axon kinetics.
    """
    _kinetics_match(_channel_by_name(NEURON_PRESETS[TRN](), "K"), make_trn_k_channel)


def test_trn_t_ref_is_huguenard_recording_temp() -> None:
    """TRN preset T_ref must be 309.15 K (36 °C), the HP92 recording temperature.

    Setting T_ref = 309.15 K limits the Q10 correction from ~5.2× (22→37 °C)
    to ~1.12× (36→37 °C), preserving the published Huguenard & Prince kinetics
    and preventing premature Na⁺ inactivation at physiological temperature.
    """
    assert NEURON_PRESETS[TRN]().T_ref == pytest.approx(309.15)


def test_trn_includes_ikca_channel() -> None:
    """TRN preset must wire IKCa for Ca²⁺-driven AHP shaping (issue #286).

    Huguenard & Prince (1992) describe IKCa as the dominant K⁺ current
    converting ICaT-mediated Ca²⁺ entry into outward current that shapes
    inter-spike intervals during tonic firing and contributes to LTS-burst
    termination.  Without IKCa there is no biophysical mechanism to convert
    [Ca²⁺]ᵢ rises into inter-burst pauses — confirmed by Pospischil 2008
    (Biol. Cybern. 99:427) Table 2 RE column.

    This test asserts only that an IKCa channel is wired into the TRN preset
    with a positive g_max; the full HP92 rebound-burst phenotype (5–15
    spike, 200–600 Hz) is verified separately in
    ``test_trn_step_release_produces_hp92_rebound_burst``.
    """
    neuron = NEURON_PRESETS[TRN]()
    ikca_channels = [c for c in neuron.channels if c.name == "KCa"]
    assert len(ikca_channels) == 1, (
        f"TRN preset must include exactly one IKCa channel, found {len(ikca_channels)}"
    )
    assert ikca_channels[0].g_max > 0.0, (
        f"TRN IKCa g_max must be positive, got {ikca_channels[0].g_max}"
    )


def test_trn_uses_trn_icat_factory() -> None:
    """TRN preset must wire the sigmoid-tau ICaT variant (issue #295).

    The default :func:`make_icat_channel` cosh-shaped tau collapses the LTS
    plateau in 5–10 ms, which is too fast to fit the 5–15 Na⁺ spikes of the
    HP92 rebound burst.  :func:`make_trn_icat_channel` replaces only the
    inactivation time constant with a sigmoid shape; ft_inf is bit-identical.

    Distinguishes the two variants behaviorally: at V = −50 mV the TRN
    sigmoid-tau gives tau_ft ≈ 110 ms versus ≈ 7 ms for the default
    cosh-shaped tau — a >15× difference.
    """
    neuron = NEURON_PRESETS[TRN]()
    icat_channels = [c for c in neuron.channels if c.name == "CaT"]
    assert len(icat_channels) == 1, (
        f"TRN preset must include exactly one CaT channel; found {len(icat_channels)}"
    )
    icat = icat_channels[0]
    assert icat.g_max > 0.0
    # Verify sigmoid-tau: at V=-50 mV the TRN factory gives tau_ft >> 50 ms.
    ft_var = next(gv for gv in icat.gating_variables if gv.name == "ft")
    v_test = -50.0
    tau_ft = 1.0 / (ft_var.alpha(v_test, 0.0) + ft_var.beta(v_test, 0.0))
    assert tau_ft > 50.0, (
        f"TRN ICaT tau_ft at V=-50 mV should be >50 ms (sigmoid shape), "
        f"got {tau_ft:.1f} ms — the default cosh-shaped factory may have "
        "been wired instead of make_trn_icat_channel."
    )


# ---------------------------------------------------------------------------
# Purkinje preset uses De Schutter & Bower (1994) factories (issue #243)
# ---------------------------------------------------------------------------


def test_purkinje_uses_dschutter_bower_na_factory() -> None:
    """Purkinje preset wires the De Schutter & Bower (1994) Na⁺ channel factory.

    Verifies that Traub-Miles kinetics (VT = −58 mV, recorded at 32 °C) are
    used instead of the default HH52 squid axon kinetics, which applied a
    ~5.2× Q10 overcorrection inappropriate for a mammalian Purkinje cell.
    """
    _kinetics_match(
        _channel_by_name(NEURON_PRESETS[PURKINJE](), "Na"), make_purkinje_na_channel
    )


def test_purkinje_uses_dschutter_bower_k_factory() -> None:
    """Purkinje preset wires the De Schutter & Bower (1994) K⁺ channel factory.

    Verifies that Traub-Miles kinetics (VT = −58 mV, recorded at 32 °C) are
    used instead of the default HH52 squid axon kinetics.
    """
    _kinetics_match(
        _channel_by_name(NEURON_PRESETS[PURKINJE](), "K"), make_purkinje_k_channel
    )


def test_purkinje_t_ref_is_dschutter_bower_recording_temp() -> None:
    """Purkinje preset T_ref must be 305.15 K (32 °C), the DSB94 recording temperature.

    Setting T_ref = 305.15 K limits the Q10 correction from ~5.2× (22→37 °C)
    to ~1.73× (32→37 °C), preserving the published De Schutter & Bower kinetics
    and preventing distorted Na⁺ inactivation at physiological temperature.
    """
    assert NEURON_PRESETS[PURKINJE]().T_ref == pytest.approx(305.15)


def test_purkinje_has_nap_channel() -> None:
    """Purkinje preset includes an INaP (persistent Na⁺) channel.

    INaP shifts the zero-current equilibrium from −82.3 mV to the physiological
    pacemaking range and amplifies subthreshold depolarizations.
    Ref: Raman & Bean (1999), J. Neurosci. 19:4663.
    """
    channel_names = [ch.name for ch in NEURON_PRESETS[PURKINJE]().channels]
    assert "NaP" in channel_names


def test_purkinje_has_nar_channel() -> None:
    """Purkinje preset includes an INaR (resurgent Na⁺) channel.

    INaR enables the fast repriming and high-frequency repetitive firing
    that characterises climbing-fibre-driven complex spikes in cerebellar
    Purkinje cells.  Climbing-fibre input is not modelled in this
    single-compartment preset; INaR contributes here only by shaping
    intrinsic tonic spiking.
    Ref: Raman & Bean (1997), Neuron 19:881.
    """
    channel_names = [ch.name for ch in NEURON_PRESETS[PURKINJE]().channels]
    assert "NaR" in channel_names


def test_purkinje_nap_conductance_physiological() -> None:
    """INaP conductance in Purkinje preset is in the physiological range.

    Expected range: [0.01, 0.5] mS/cm².  Values below 0.01 have negligible
    effect on resting potential; values above 0.5 produce pathologically large
    persistent inward currents.
    """
    nap_channels = [
        ch for ch in NEURON_PRESETS[PURKINJE]().channels if ch.name == "NaP"
    ]
    assert len(nap_channels) == 1
    assert 0.01 <= nap_channels[0].g_max <= 0.5


def test_purkinje_vrest_in_pacemaking_range() -> None:
    """Purkinje v_rest is in the in-vivo pacemaking range [−65.5, −55] mV.

    Häusser & Clark (1997, J. Neurosci. 17:2358) report spontaneous pacemaking
    near −55 to −65 mV in vivo.  Raman & Bean (1999) attribute this to INaP/INaR
    subthreshold window currents.  The lower bound is widened to −65.5 mV to
    accommodate the 0.5 mV equilibrium-finding tolerance.
    """
    assert -65.5 <= NEURON_PRESETS[PURKINJE]().v_rest <= -55.0


def test_purkinje_nar_conductance_physiological() -> None:
    """INaR conductance in Purkinje preset is in the physiological range.

    Expected range: [0.01, 0.5] mS/cm².  Consistent bound as for INaP;
    guards against silent regressions to non-physiological values.
    Ref: Raman & Bean (1997), Neuron 19:881.
    """
    nar_channels = [
        ch for ch in NEURON_PRESETS[PURKINJE]().channels if ch.name == "NaR"
    ]
    assert len(nar_channels) == 1
    assert 0.01 <= nar_channels[0].g_max <= 0.5


def test_purkinje_has_ih_channel() -> None:
    """Purkinje preset includes an Ih (hyperpolarization-activated) channel.

    Ih activates during the post-AP AHP and drives recovery to threshold,
    enabling spontaneous autonomous pacemaking without external current.
    Ref: Destexhe et al. (1993), J. Neurophysiol. 70:1385.
    """
    channel_names = [ch.name for ch in NEURON_PRESETS[PURKINJE]().channels]
    assert "h" in channel_names


def test_purkinje_ih_conductance_physiological() -> None:
    """Ih conductance in Purkinje preset is in the physiological range.

    Expected range: [0.1, 5.0] mS/cm².  Values below 0.1 are insufficient
    to drive recovery from the post-AP AHP; values above 5.0 dominate
    membrane current and produce unrealistically fast pacemaking.
    """
    ih_channels = [ch for ch in NEURON_PRESETS[PURKINJE]().channels if ch.name == "h"]
    assert len(ih_channels) == 1
    assert 0.1 <= ih_channels[0].g_max <= 5.0


# ---------------------------------------------------------------------------
# SNc Dopaminergic preset uses Canavier/Komendantov (1999/2004) factories (#244)
# ---------------------------------------------------------------------------


def test_dopaminergic_uses_komendantov_na_factory() -> None:
    """SNc Dopaminergic preset wires the Canavier/Komendantov Na⁺ channel factory.

    Verifies that Komendantov (2004) kinetics (VT=-67 mV, T_ref=308.15 K) are
    used instead of the HH52 squid axon kinetics, which applied a ~5.2×
    Q10 overcorrection inappropriate for an SNc dopaminergic neuron.
    """
    neuron = NEURON_PRESETS[DOPAMINERGIC]()
    _kinetics_match(_channel_by_name(neuron, "Na"), make_dopaminergic_na_channel)


def test_dopaminergic_uses_komendantov_k_factory() -> None:
    """SNc Dopaminergic preset wires the Canavier/Komendantov K⁺ channel factory.

    Verifies that Komendantov (2004) kinetics (VT=-67 mV, T_ref=308.15 K) are
    used instead of the HH52 squid axon kinetics.
    """
    neuron = NEURON_PRESETS[DOPAMINERGIC]()
    _kinetics_match(_channel_by_name(neuron, "K"), make_dopaminergic_k_channel)


def test_dopaminergic_t_ref_is_komendantov_recording_temp() -> None:
    """SNc Dopaminergic T_ref matches Komendantov (2004) 35 °C recording temperature.

    Prevents regression to the HH52 default (295.15 K = 22 °C) which applied
    a ~5.2× Q10 overcorrection and produced excessively fast Na⁺ gating.
    Komendantov (2004) recorded at 35 °C = 308.15 K.
    """
    assert NEURON_PRESETS[DOPAMINERGIC]().T_ref == pytest.approx(308.15)


# ---------------------------------------------------------------------------
# Cell surface area metadata
# ---------------------------------------------------------------------------


# Representative areas from the issue table (cm²).  Squid axon is intentionally
# left as None because the HH52 preparation is reported per-area.
_EXPECTED_AREAS_CM2: dict[str, float | None] = {
    SQUID_GIANT_AXON: None,
    FAST_SPIKING_INTERNEURON: 3e-6,
    CORTICAL_PYRAMIDAL: 20e-6,
    PURKINJE: 250e-6,
    DOPAMINERGIC: 50e-6,
    THALAMIC_RELAY: 12e-6,
    CA1_PYRAMIDAL: 25e-6,
    STN: 7e-6,
    TRN: 7e-6,
}


@pytest.mark.parametrize(
    ("preset_name", "expected_area"),
    list(_EXPECTED_AREAS_CM2.items()),
)
def test_presets_have_expected_area_cm2(
    preset_name: str, expected_area: float | None
) -> None:
    """Each preset carries the area_cm2 listed in the issue table."""
    neuron = NEURON_PRESETS[preset_name]()
    if expected_area is None:
        assert neuron.area_cm2 is None
    else:
        assert neuron.area_cm2 == pytest.approx(expected_area)


@pytest.mark.parametrize(
    "preset_name",
    [name for name, area in _EXPECTED_AREAS_CM2.items() if area is not None],
)
def test_preset_with_area_yields_finite_absolute_passive_properties(
    preset_name: str,
) -> None:
    """Each preset with area_cm2 set produces finite absolute MΩ / pF outputs.

    The membrane test should return well-defined R_n in MΩ and C in pF for
    every preset that opts into the absolute-units pathway.  Hard physiological
    bounds are intentionally lax (presets vary by orders of magnitude) — this
    test guards against silent regressions where the conversion returns
    None / inf / negative values.
    """
    neuron = NEURON_PRESETS[preset_name]()
    props = run_membrane_test(neuron)
    assert props is not None
    assert props.input_resistance_mohm is not None
    assert props.membrane_capacitance_pf is not None
    assert props.input_resistance_mohm > 0
    assert props.membrane_capacitance_pf > 0
    # 0.5–10000 MΩ and 0.1–1000 pF cover every plausible single-compartment
    # preparation; tighter bands would break on legitimate preset edits.
    assert 0.5 < props.input_resistance_mohm < 10000.0
    assert 0.1 < props.membrane_capacitance_pf < 1000.0
    # τ_m identity: τ [ms] = R_n [MΩ] × C [pF] / 1000.  This is exact (the
    # absolute values are derived from the same density values via the same
    # area), so this is a tight regression guard with biological meaning.
    tau_from_absolute = (
        props.input_resistance_mohm * props.membrane_capacitance_pf / 1000.0
    )
    assert tau_from_absolute == pytest.approx(props.time_constant, rel=1e-9)
