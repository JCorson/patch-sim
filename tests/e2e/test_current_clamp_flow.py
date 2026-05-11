"""E2E tests for the current-clamp simulation pipeline.

Each test drives: neuron preset → protocol preset → simulate → assert on
sweep count and analysis population.  The Reflex runtime is not needed;
all handlers are called directly on bare state instances.
"""

import pytest

from patch_sim.constants import (
    ACTION_POTENTIAL,
    CORTICAL_PYRAMIDAL,
    DOPAMINERGIC,
    FAST_SPIKING_INTERNEURON,
    FREQUENCY_RESPONSE,
    PURKINJE,
    REPETITIVE_FIRING,
    SQUID_GIANT_AXON,
    THALAMIC_RELAY,
)
from tests.e2e.conftest import (
    StateTree,
    patch_get_state,
    run_flow,
    simulate_and_apply,
)


@pytest.mark.parametrize(
    "neuron_preset",
    [
        SQUID_GIANT_AXON,
        FAST_SPIKING_INTERNEURON,
        CORTICAL_PYRAMIDAL,
        PURKINJE,
        DOPAMINERGIC,
        THALAMIC_RELAY,
    ],
)
async def test_action_potential_preset_produces_sweep(
    state_tree: StateTree, neuron_preset: str
) -> None:
    """Running the Action Potential preset on each neuron produces exactly one sweep."""
    result = await run_flow(
        state_tree,
        neuron_preset=neuron_preset,
        protocol_preset=ACTION_POTENTIAL,
    )

    assert len(result.sweeps) == 1
    assert len(state_tree.sim._current_sweeps) == 1
    assert state_tree.sim.sim_token != ""


@pytest.mark.parametrize(
    "neuron_preset",
    [
        SQUID_GIANT_AXON,
        FAST_SPIKING_INTERNEURON,
        CORTICAL_PYRAMIDAL,
    ],
)
async def test_action_potential_preset_populates_ap_summary(
    state_tree: StateTree, neuron_preset: str
) -> None:
    """The Action Potential preset populates ap_summary with expected keys after run."""
    await run_flow(
        state_tree,
        neuron_preset=neuron_preset,
        protocol_preset=ACTION_POTENTIAL,
    )

    summary = state_tree.analysis.ap_summary
    expected_keys = {
        "spike_count",
        "mean_threshold_voltage",
        "mean_peak_voltage",
        "mean_rise_time",
        "mean_half_width",
        "firing_rate",
    }
    assert expected_keys.issubset(summary.keys())


async def test_squid_action_potential_spikes(state_tree: StateTree) -> None:
    """Squid giant axon with Action Potential preset fires exactly one spike."""
    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=ACTION_POTENTIAL,
    )

    spike_count = int(state_tree.analysis.ap_summary.get("spike_count", "0"))
    assert spike_count == 1


async def test_repetitive_firing_preset_produces_sweep(state_tree: StateTree) -> None:
    """Repetitive Firing protocol produces exactly one sweep."""
    result = await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=REPETITIVE_FIRING,
    )

    assert len(result.sweeps) == 1
    assert len(state_tree.sim._current_sweeps) == 1


async def test_repetitive_firing_populates_phase_plane(state_tree: StateTree) -> None:
    """Repetitive Firing protocol populates phase-plane data after run."""
    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=REPETITIVE_FIRING,
    )

    assert state_tree.analysis.phase_plane_data != {}


async def test_subthreshold_chirp_populates_impedance(state_tree: StateTree) -> None:
    """A subthreshold chirp through the full pipeline populates impedance_data.

    Loads the Frequency Response (chirp) preset on the squid axon, then dials
    the chirp down to a subthreshold amplitude so ``analyze_impedance`` has a
    valid linear response to work with.
    """
    async with patch_get_state(state_tree):
        [_ async for _ in state_tree.neuron.load_neuron_preset(SQUID_GIANT_AXON)]
        [_ async for _ in state_tree.protocol.load_protocol_preset(FREQUENCY_RESPONSE)]
    state_tree.protocol.dc_offset = 0.0
    state_tree.protocol.amplitude = 1.0

    simulate_and_apply(state_tree)

    impedance = state_tree.analysis.impedance_data
    assert impedance != {}
    expected_keys = {
        "frequencies",
        "magnitude",
        "phase",
        "resonance_frequency",
        "quality_factor",
        "peak_impedance",
        "units",
    }
    assert expected_keys.issubset(impedance.keys())
    assert len(impedance["frequencies"]) == len(impedance["magnitude"])
    assert len(impedance["frequencies"]) > 0


async def test_suprathreshold_chirp_leaves_impedance_empty(
    state_tree: StateTree,
) -> None:
    """The default (large) Frequency Response chirp drives spiking → no impedance."""
    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=FREQUENCY_RESPONSE,
    )

    assert state_tree.analysis.impedance_data == {}


async def test_action_potential_preset_leaves_impedance_empty(
    state_tree: StateTree,
) -> None:
    """A non-chirp protocol does not populate impedance_data."""
    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=ACTION_POTENTIAL,
    )

    assert state_tree.analysis.impedance_data == {}


async def test_apply_simulation_clears_previous_analysis(
    state_tree: StateTree,
) -> None:
    """Running a simulation overwrites stale analysis from a previous run."""
    # Seed stale analysis data.
    state_tree.analysis.ap_summary = {"spike_count": "99"}

    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=ACTION_POTENTIAL,
    )

    # Stale value must be gone (fresh analysis written from the new run).
    assert state_tree.analysis.ap_summary.get("spike_count") != "99"
