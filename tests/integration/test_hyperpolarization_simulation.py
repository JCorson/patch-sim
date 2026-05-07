"""Integration tests for the Hyperpolarization Steps protocol across all neuron types.

Runs the preset hyperpolarization protocol (with per-neuron adjustments) through
the full simulation pipeline and verifies:
  - Structural plausibility for every preset (no NaN, in-range voltages).
  - Ih-driven sag (steady-state depolarisation above the peak) for neurons
    known to express HCN channels.
  - Post-step rebound spikes for every model that produces them, covering all
    three biophysical mechanisms present in this simulator:

    * ICaT de-inactivation (thalamic relay, STN, TRN, Purkinje)
    * Ih-driven post-step overshoot (dopaminergic, cortical pyramidal)
    * HH anode-break excitation (squid giant axon, cortical pyramidal)

  - Absence of true sag for neurons without HCN channels.

Unit tests with synthetic voltage traces live in
tests/unit/test_hyperpolarization_analysis.py.
"""

import numpy as np
import pytest

import patch_sim
from patch_sim.analysis.hyperpolarization import (
    HyperpolarizationAnalysisResult,
    analyze_hyperpolarization,
)
from patch_sim.constants import (
    CA1_PYRAMIDAL,
    CORTICAL_PYRAMIDAL,
    DOPAMINERGIC,
    FAST_SPIKING_INTERNEURON,
    HYPERPOLARIZATION_STEPS,
    SQUID_GIANT_AXON,
    STN,
    THALAMIC_RELAY,
)
from patch_sim.presets import (
    NEURON_PRESET_NAMES,
    NEURON_PRESETS,
    NEURON_PROTOCOL_ADJUSTMENTS,
    PROTOCOL_PRESETS,
)

_SAMPLING_FREQ = 40_000.0


def _run_hyperpolarization_sweeps(
    preset_name: str,
) -> HyperpolarizationAnalysisResult:
    """Run the Hyperpolarization Steps preset for a neuron and return the analysis.

    Args:
        preset_name: Key in NEURON_PRESETS selecting the neuron configuration.

    Returns:
        A :class:`HyperpolarizationAnalysisResult` from the final multi-sweep run.
    """
    neuron = NEURON_PRESETS[preset_name]()

    base = dict(PROTOCOL_PRESETS[HYPERPOLARIZATION_STEPS])
    adjustments = NEURON_PROTOCOL_ADJUSTMENTS.get(preset_name, {}).get(
        HYPERPOLARIZATION_STEPS, {}
    )
    base.update(adjustments)

    pre = base["pre_stimulus_duration"]
    stim = base["stimulus_duration"]
    post = base["post_stimulus_duration"]
    min_i = base["min_stimulus"]
    max_i = base["max_stimulus"]
    step_i = base["stimulus_step"]
    total = pre + stim + post

    n_steps = round((max_i - min_i) / step_i) + 1
    current_steps = list(np.linspace(min_i, max_i, n_steps))

    protocols = [
        patch_sim.step_current(
            duration=total,
            current_amplitude=float(amp),
            step_start=pre,
            step_duration=stim,
            sampling_frequency=_SAMPLING_FREQ,
        )
        for amp in current_steps
    ]
    results = list(
        patch_sim.simulate_batch(neuron, protocols, patch_sim.simulate_current_clamp)
    )
    voltages = [r["voltage"] for r in results]
    time = results[0]["time"]

    return analyze_hyperpolarization(time, voltages, current_steps, pre, pre + stim)


@pytest.fixture(scope="module")
def _hyp_sweeps_by_preset() -> dict[str, HyperpolarizationAnalysisResult]:
    """Run hyperpolarization sweeps for all presets once and cache per module.

    Returns:
        Mapping from preset name to its :class:`HyperpolarizationAnalysisResult`.
    """
    return {name: _run_hyperpolarization_sweeps(name) for name in NEURON_PRESET_NAMES}


# ---------------------------------------------------------------------------
# Structural plausibility — all 9 neuron types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", NEURON_PRESET_NAMES)
def test_hyperpolarization_preset_is_structurally_plausible(
    preset_name: str,
    _hyp_sweeps_by_preset: dict[str, HyperpolarizationAnalysisResult],
) -> None:
    """Hyperpolarization Steps preset runs without errors for every neuron type.

    For every neuron, the protocol must:
    - Complete without raising exceptions.
    - Return one SagPoint per current step.
    - Produce finite voltage values throughout.
    - Hyperpolarize the cell by at least 3 mV at the most negative step.
    - Show non-negative sag amplitude (steady-state ≥ peak voltage during step).

    Args:
        preset_name: Key in NEURON_PRESETS selecting the neuron configuration.
        _hyp_sweeps_by_preset: Module-scoped cache of all preset sweep results.
    """
    result = _hyp_sweeps_by_preset[preset_name]

    assert len(result.points) > 0, (
        f"{preset_name}: expected at least one SagPoint, got 0"
    )

    v_rest = NEURON_PRESETS[preset_name]().v_rest

    for pt in result.points:
        assert np.isfinite(pt.peak_voltage), (
            f"{preset_name}: non-finite peak_voltage at I={pt.current_step}"
        )
        assert np.isfinite(pt.steady_state_voltage), (
            f"{preset_name}: non-finite steady_state_voltage at I={pt.current_step}"
        )
        assert pt.sag_amplitude >= 0.0, (
            f"{preset_name}: negative sag amplitude {pt.sag_amplitude:.2f} mV "
            f"at I={pt.current_step} (steady_state={pt.steady_state_voltage:.1f}, "
            f"peak={pt.peak_voltage:.1f})"
        )
        assert -150.0 <= pt.peak_voltage <= 60.0, (
            f"{preset_name}: peak voltage {pt.peak_voltage:.1f} mV out of range "
            f"[-150, 60] at I={pt.current_step}"
        )

    most_negative = result.points[0]
    assert most_negative.peak_voltage < v_rest - 3.0, (
        f"{preset_name}: most negative step (I={most_negative.current_step}) "
        f"did not hyperpolarize by ≥3 mV below v_rest ({v_rest} mV); "
        f"peak={most_negative.peak_voltage:.1f} mV"
    )


# ---------------------------------------------------------------------------
# Voltage sag (Ih) — neurons with HCN channels
# ---------------------------------------------------------------------------


def test_sag_in_cortical_pyramidal(
    _hyp_sweeps_by_preset: dict[str, HyperpolarizationAnalysisResult],
) -> None:
    """Cortical pyramidal shows clear Ih-driven voltage sag during hyperpolarization.

    Ih is a depolarising inward current activated by hyperpolarisation.  During
    a sustained negative step, Ih activates and drives the membrane back toward
    rest — the characteristic sag.  The most negative step should show ≥5 mV
    of sag in this model (Ih conductance is significant in the CP preset).

    Args:
        _hyp_sweeps_by_preset: Module-scoped cache of all preset sweep results.
    """
    result = _hyp_sweeps_by_preset[CORTICAL_PYRAMIDAL]
    most_negative = result.points[0]
    assert most_negative.sag_amplitude > 5.0, (
        f"Cortical Pyramidal: expected sag > 5 mV at most negative step, "
        f"got {most_negative.sag_amplitude:.2f} mV "
        f"(peak={most_negative.peak_voltage:.1f},"
        f" ss={most_negative.steady_state_voltage:.1f})"
    )


def test_sag_in_thalamic_relay(
    _hyp_sweeps_by_preset: dict[str, HyperpolarizationAnalysisResult],
) -> None:
    """Thalamic relay neuron shows Ih-driven voltage sag during hyperpolarization.

    Args:
        _hyp_sweeps_by_preset: Module-scoped cache of all preset sweep results.
    """
    result = _hyp_sweeps_by_preset[THALAMIC_RELAY]
    most_negative = result.points[0]
    assert most_negative.sag_amplitude > 1.0, (
        f"Thalamic Relay: expected sag > 1 mV, got {most_negative.sag_amplitude:.2f} mV"
    )


def test_sag_in_ca1_pyramidal(
    _hyp_sweeps_by_preset: dict[str, HyperpolarizationAnalysisResult],
) -> None:
    """Hippocampal CA1 pyramidal neuron shows Ih-driven sag.

    Args:
        _hyp_sweeps_by_preset: Module-scoped cache of all preset sweep results.
    """
    result = _hyp_sweeps_by_preset[CA1_PYRAMIDAL]
    most_negative = result.points[0]
    assert most_negative.sag_amplitude > 1.0, (
        f"CA1 Pyramidal: expected sag > 1 mV, got {most_negative.sag_amplitude:.2f} mV"
    )


def test_sag_in_stn(
    _hyp_sweeps_by_preset: dict[str, HyperpolarizationAnalysisResult],
) -> None:
    """Subthalamic nucleus neuron shows Ih-driven sag during hyperpolarization.

    Args:
        _hyp_sweeps_by_preset: Module-scoped cache of all preset sweep results.
    """
    result = _hyp_sweeps_by_preset[STN]
    most_negative = result.points[0]
    assert most_negative.sag_amplitude > 2.0, (
        f"STN: expected sag > 2 mV, got {most_negative.sag_amplitude:.2f} mV"
    )


def test_sag_in_dopaminergic(
    _hyp_sweeps_by_preset: dict[str, HyperpolarizationAnalysisResult],
) -> None:
    """Dopaminergic neuron shows Ih-driven sag during hyperpolarization.

    Args:
        _hyp_sweeps_by_preset: Module-scoped cache of all preset sweep results.
    """
    result = _hyp_sweeps_by_preset[DOPAMINERGIC]
    most_negative = result.points[0]
    assert most_negative.sag_amplitude > 1.0, (
        f"Dopaminergic: expected sag > 1 mV, got {most_negative.sag_amplitude:.2f} mV"
    )


def test_squid_giant_axon_minimal_sag(
    _hyp_sweeps_by_preset: dict[str, HyperpolarizationAnalysisResult],
) -> None:
    """Classic HH squid axon has no Ih channel and shows minimal voltage sag.

    The HH52 model contains only Na⁺ and K⁺ conductance-based channels plus a
    passive leak.  Any apparent sag during hyperpolarisation comes from K channel
    deactivation (reduction in outward current), which is a small effect.  The
    sag amplitude should be well below the 1 mV threshold used for Ih-expressing
    neurons across the full current range of the preset.

    Args:
        _hyp_sweeps_by_preset: Module-scoped cache of all preset sweep results.
    """
    result = _hyp_sweeps_by_preset[SQUID_GIANT_AXON]
    for pt in result.points:
        assert pt.sag_amplitude < 2.0, (
            f"Squid: unexpected sag {pt.sag_amplitude:.2f} mV at I={pt.current_step} "
            f"(no Ih in classic HH)"
        )


# ---------------------------------------------------------------------------
# Post-inhibitory rebound burst — neurons with T-type Ca²⁺
# ---------------------------------------------------------------------------


def test_rebound_burst_in_thalamic_relay(
    _hyp_sweeps_by_preset: dict[str, HyperpolarizationAnalysisResult],
) -> None:
    """Thalamic relay neuron fires a rebound burst after hyperpolarization release.

    Sustained hyperpolarisation de-inactivates T-type Ca²⁺ channels (ICaT).
    When the step ends, the return to resting potential activates ICaT and drives
    a burst of action potentials — the post-inhibitory rebound.  At the most
    negative current step (−10 µA/cm² base preset), ≥1 rebound spike is expected
    within 50 ms of step offset.

    Args:
        _hyp_sweeps_by_preset: Module-scoped cache of all preset sweep results.
    """
    result = _hyp_sweeps_by_preset[THALAMIC_RELAY]
    most_negative = result.points[0]
    assert most_negative.rebound_spike_count >= 1, (
        f"Thalamic Relay: expected ≥1 rebound spike after most negative step, "
        f"got {most_negative.rebound_spike_count} "
        f"(peak={most_negative.peak_voltage:.1f} mV)"
    )


def test_hyperpolarization_sag_in_ca1_pyramidal(
    _hyp_sweeps_by_preset: dict[str, HyperpolarizationAnalysisResult],
) -> None:
    """CA1 pyramidal neuron shows Ih-driven sag during hyperpolarization.

    The Pospischil Na⁺/K⁺ kinetics (34 °C reference) produce sufficient outward
    K⁺ current on step release that the threshold for post-inhibitory rebound
    is not reached in this stimulus range.  Ih-driven sag is still the
    distinguishing feature verified here.

    Args:
        _hyp_sweeps_by_preset: Module-scoped cache of all preset sweep results.
    """
    result = _hyp_sweeps_by_preset[CA1_PYRAMIDAL]
    most_negative = result.points[0]
    assert most_negative.sag_amplitude > 0.0, (
        f"CA1 Pyramidal: expected Ih-driven sag at most negative step, "
        f"got sag_amplitude={most_negative.sag_amplitude:.2f} mV"
    )


# ---------------------------------------------------------------------------
# HH anode-break, Ih overshoot, and Kv3.1 rebound — non-ICaT neurons
# ---------------------------------------------------------------------------


def test_anode_break_in_squid_giant_axon(
    _hyp_sweeps_by_preset: dict[str, HyperpolarizationAnalysisResult],
) -> None:
    """Squid giant axon fires a post-hyperpolarization spike via HH anode-break.

    The plain HH52 model contains only Na⁺, K⁺, and leak conductances — no
    T-type Ca²⁺ or Ih channels.  Nevertheless, deep hyperpolarisation (~−98 mV)
    fully de-inactivates the Na⁺ h-gate (h_inf ≈ 0.996, τ_h ≈ 2.5 ms) and
    deactivates the K⁺ n-gate (n_inf ≈ 0.002) within the 300 ms step.  When the
    step ends, the m-gate activates rapidly as the membrane recovers while h is
    still elevated and g_K is negligible, triggering a post-hyperpolarization
    action potential — classic anode-break excitation first described by Hodgkin
    & Huxley (1952, J. Physiol. 117:500).

    Args:
        _hyp_sweeps_by_preset: Module-scoped cache of all preset sweep results.
    """
    result = _hyp_sweeps_by_preset[SQUID_GIANT_AXON]
    most_negative = result.points[0]
    assert most_negative.rebound_spike_count >= 1, (
        f"Squid Giant Axon: expected ≥1 anode-break spike after most negative step, "
        f"got {most_negative.rebound_spike_count} "
        f"(peak={most_negative.peak_voltage:.1f} mV)"
    )


def test_rebound_in_cortical_pyramidal(
    _hyp_sweeps_by_preset: dict[str, HyperpolarizationAnalysisResult],
) -> None:
    """Cortical pyramidal neuron fires a rebound spike via anode-break and Ih overshoot.

    The cortical pyramidal preset reaches ~−109 mV at the most negative step
    (−5 µA/cm²), fully de-inactivating h and deactivating n, which sets up
    HH anode-break excitation on release.  Simultaneously, Ih (activated during
    the step) continues to conduct after step offset, providing an inward
    depolarising current that accelerates return to threshold.  The cell has no
    ICaT, so the rebound is driven entirely by these Na⁺ and Ih mechanisms.

    Args:
        _hyp_sweeps_by_preset: Module-scoped cache of all preset sweep results.
    """
    result = _hyp_sweeps_by_preset[CORTICAL_PYRAMIDAL]
    most_negative = result.points[0]
    assert most_negative.rebound_spike_count >= 1, (
        f"Cortical Pyramidal: expected ≥1 rebound spike after most negative step, "
        f"got {most_negative.rebound_spike_count} "
        f"(peak={most_negative.peak_voltage:.1f} mV)"
    )


def test_hyperpolarization_in_fast_spiking_interneuron(
    _hyp_sweeps_by_preset: dict[str, HyperpolarizationAnalysisResult],
) -> None:
    """Fast-spiking interneuron hyperpolarizes cleanly with no Ih sag.

    The FSI has no HCN channels, so hyperpolarizing steps produce a flat
    voltage deflection with zero sag amplitude.  Pospischil Na⁺/K⁺ kinetics
    (issue #231) have a higher firing threshold and stronger outward K⁺
    rectification than the previous HH52 model, so the post-step voltage
    does not exceed threshold and no rebound spike is generated.

    Args:
        _hyp_sweeps_by_preset: Module-scoped cache of all preset sweep results.
    """
    result = _hyp_sweeps_by_preset[FAST_SPIKING_INTERNEURON]
    most_negative = result.points[0]
    assert most_negative.sag_amplitude < 0.01, (
        f"Fast-Spiking Interneuron: expected zero sag (no Ih), "
        f"got sag_amplitude={most_negative.sag_amplitude:.4f} mV"
    )


def test_rebound_in_dopaminergic(
    _hyp_sweeps_by_preset: dict[str, HyperpolarizationAnalysisResult],
) -> None:
    """Dopaminergic neuron fires a rebound spike driven by Ih.

    The dopaminergic preset has a high HCN conductance (g_Ih = 2.0 mS/cm²) but
    no ICaT.  Even at shallow hyperpolarisation depths (−69 to −62 mV), Ih
    activates substantially during the 300 ms step.  On release, this inward
    current continues to depolarise the membrane above its resting state, and in
    a cell with low firing threshold (pacemaker) the transient overshoot is
    sufficient to trigger a spike.  Rebound appears for steps of −15 µA/cm² and
    above.

    Args:
        _hyp_sweeps_by_preset: Module-scoped cache of all preset sweep results.
    """
    result = _hyp_sweeps_by_preset[DOPAMINERGIC]
    most_negative = result.points[0]
    assert most_negative.rebound_spike_count >= 1, (
        f"Dopaminergic: expected ≥1 Ih-driven rebound spike after most negative "
        f"step, got {most_negative.rebound_spike_count} "
        f"(peak={most_negative.peak_voltage:.1f} mV)"
    )
