"""E2E tests for stored-trace and reset interactions.

Exercises the wiring between run → store_trace → run again → clear_stored_traces.
"""

from patch_sim.constants import ACTION_POTENTIAL, SQUID_GIANT_AXON
from tests.e2e.conftest import StateTree, run_flow, simulate_and_apply


async def test_do_store_trace_after_run(state_tree: StateTree) -> None:
    """Calling _do_store_trace after a run moves the current sweep to stored_traces."""
    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=ACTION_POTENTIAL,
    )
    assert state_tree.sim.has_result

    state_tree.sim._do_store_trace()

    assert len(state_tree.sim.stored_traces) == 1


async def test_current_sweeps_survive_store(state_tree: StateTree) -> None:
    """_do_store_trace copies the sweep — it does not remove it from _current_sweeps."""
    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=ACTION_POTENTIAL,
    )

    state_tree.sim._do_store_trace()

    assert len(state_tree.sim._current_sweeps) == 1
    assert len(state_tree.sim.stored_traces) == 1


async def test_second_run_does_not_affect_stored_traces(state_tree: StateTree) -> None:
    """A second run without reloading presets leaves stored_traces untouched."""
    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=ACTION_POTENTIAL,
    )
    state_tree.sim._do_store_trace()
    stored_label = state_tree.sim.stored_traces[0].label

    # Run again without reloading presets (which would clear stored traces).
    simulate_and_apply(state_tree)

    assert len(state_tree.sim.stored_traces) == 1
    assert state_tree.sim.stored_traces[0].label == stored_label


async def test_clear_stored_traces_empties_stored_only(state_tree: StateTree) -> None:
    """_do_clear_stored_traces removes stored traces, leaving _current_sweeps intact."""
    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=ACTION_POTENTIAL,
    )
    state_tree.sim._do_store_trace()
    assert len(state_tree.sim.stored_traces) == 1
    assert len(state_tree.sim._current_sweeps) == 1

    state_tree.sim._do_clear_stored_traces()

    assert state_tree.sim.stored_traces == []
    assert len(state_tree.sim._current_sweeps) == 1


async def test_store_trace_requires_prior_run(state_tree: StateTree) -> None:
    """_do_store_trace is a no-op when no simulation has been run."""
    assert not state_tree.sim.has_result

    state_tree.sim._do_store_trace()

    assert state_tree.sim.stored_traces == []


async def test_multiple_stores_accumulate(state_tree: StateTree) -> None:
    """Calling _do_store_trace twice after two runs accumulates two stored traces."""
    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=ACTION_POTENTIAL,
    )
    state_tree.sim._do_store_trace()

    # Re-run without reloading presets so stored_traces are not cleared.
    simulate_and_apply(state_tree)
    state_tree.sim._do_store_trace()

    assert len(state_tree.sim.stored_traces) == 2
