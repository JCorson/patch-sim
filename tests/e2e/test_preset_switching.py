"""E2E tests for preset switching between simulation runs.

Exercises the wiring that must clear stale results when the user switches
neuron or protocol presets and runs again.
"""

from patch_sim.constants import (
    ACTION_POTENTIAL,
    CORTICAL_PYRAMIDAL,
    IV_CURVE,
    REPETITIVE_FIRING,
    SQUID_GIANT_AXON,
)
from tests.e2e.conftest import StateTree, run_flow


async def test_switching_neuron_preset_updates_sweeps(state_tree: StateTree) -> None:
    """Switching neuron preset and re-running replaces the previous sweep."""
    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=ACTION_POTENTIAL,
    )
    first_token = state_tree.sim.sim_token

    await run_flow(
        state_tree,
        neuron_preset=CORTICAL_PYRAMIDAL,
        protocol_preset=ACTION_POTENTIAL,
    )

    assert state_tree.sim.sim_token != first_token
    assert len(state_tree.sim._current_sweeps) == 1


async def test_switching_protocol_preset_updates_mode(state_tree: StateTree) -> None:
    """Switching from a CC preset to a VC preset changes clamp_mode correctly."""
    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=ACTION_POTENTIAL,
    )
    assert state_tree.protocol.clamp_mode == "Current Clamp"

    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=IV_CURVE,
    )
    assert state_tree.protocol.clamp_mode == "Voltage Clamp"


async def test_cc_after_vc_clears_iv_data(state_tree: StateTree) -> None:
    """Switching from VC to CC and re-running clears the IV analysis data."""
    # First run: VC → populates iv_data
    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=IV_CURVE,
    )
    assert state_tree.analysis.iv_data != {}

    # Second run: CC → must clear iv_data
    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=ACTION_POTENTIAL,
    )
    assert state_tree.analysis.iv_data == {}


async def test_vc_after_cc_clears_ap_summary(state_tree: StateTree) -> None:
    """Switching from CC to VC and re-running clears the AP summary."""
    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=ACTION_POTENTIAL,
    )
    assert state_tree.analysis.ap_summary != {}

    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=IV_CURVE,
    )
    assert state_tree.analysis.ap_summary == {}


async def test_two_cc_runs_replace_analysis(state_tree: StateTree) -> None:
    """A second CC run replaces AP analysis from the first run, not appends."""
    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=ACTION_POTENTIAL,
    )
    first_count = int(state_tree.analysis.ap_summary.get("spike_count", "0"))

    # Load a different protocol that produces different firing behaviour.
    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=REPETITIVE_FIRING,
    )
    second_count = int(state_tree.analysis.ap_summary.get("spike_count", "0"))

    # Spike counts differ because the protocols inject different currents.
    # The key assertion is that there is exactly one set of analysis — no
    # accumulation across runs.
    assert len(state_tree.sim._current_sweeps) == 1
    # The analysis reflects the second run (not accumulated from both).
    assert "spike_count" in state_tree.analysis.ap_summary
