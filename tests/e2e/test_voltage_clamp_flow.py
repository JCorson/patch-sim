"""E2E tests for the voltage-clamp simulation pipeline.

Exercises: VC preset loading → simulate → IV/GV analysis population.
"""

from patch_sim.constants import (
    CORTICAL_PYRAMIDAL,
    IV_CURVE,
    NA_CHANNEL_ACTIVATION,
    SQUID_GIANT_AXON,
)
from tests.e2e.conftest import StateTree, run_flow


async def test_iv_curve_preset_produces_multiple_sweeps(
    state_tree: StateTree,
) -> None:
    """The I-V Curve preset produces more than one sweep (multi-step protocol)."""
    result = await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=IV_CURVE,
    )

    assert len(result.sweeps) > 1
    assert len(state_tree.sim._current_sweeps) > 1


async def test_iv_curve_preset_populates_iv_data(state_tree: StateTree) -> None:
    """The I-V Curve preset populates iv_data after run."""
    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=IV_CURVE,
    )

    iv = state_tree.analysis.iv_data
    assert iv != {}
    assert "voltages" in iv
    assert "peak_inward_currents" in iv
    assert len(iv["voltages"]) > 1


async def test_iv_curve_preset_populates_gv_data(state_tree: StateTree) -> None:
    """The I-V Curve preset populates gv_data for the squid axon (has Na channel)."""
    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=IV_CURVE,
    )

    gv = state_tree.analysis.gv_data
    assert gv != {}
    assert "voltages" in gv
    assert "g_normalized" in gv


async def test_na_channel_activation_preset_produces_multi_sweep(
    state_tree: StateTree,
) -> None:
    """Na+ Channel Activation preset produces multiple sweeps."""
    result = await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=NA_CHANNEL_ACTIVATION,
    )

    assert len(result.sweeps) > 1


async def test_na_channel_activation_on_cortical_pyramidal(
    state_tree: StateTree,
) -> None:
    """Na+ Channel Activation on Cortical Pyramidal runs end-to-end without crashing.

    Cortical Pyramidal omits the HH-style core ``"K"`` channel since #320,
    so its VC simulation has no ``IK`` / ``IKL`` / ``INaL`` columns and no
    ``n`` gate.  The combination of (a) a multi-sweep protocol, (b) default
    visibility flags that try to render those columns, and (c) the
    ``is_multi_sweep`` hover-table path previously raised
    ``IndexError: index 0 is out of bounds for axis 1 with size 0`` from
    ``_build_hover_tables`` and a similar error from the gating-carrier
    fallback.  This exercises the whole pipeline to pin the regression.
    """
    result = await run_flow(
        state_tree,
        neuron_preset=CORTICAL_PYRAMIDAL,
        protocol_preset=NA_CHANNEL_ACTIVATION,
    )

    assert len(result.sweeps) > 1


async def test_voltage_clamp_clears_cc_analysis(state_tree: StateTree) -> None:
    """Running a VC simulation clears any stale current-clamp analysis fields."""
    # Seed stale CC analysis.
    state_tree.analysis.ap_metrics = [{"index": 0, "peak_voltage": "40"}]
    state_tree.analysis.ap_summary = {"spike_count": "5"}

    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=IV_CURVE,
    )

    assert state_tree.analysis.ap_metrics == []
    assert state_tree.analysis.ap_summary == {}


async def test_voltage_clamp_sim_token_is_set(state_tree: StateTree) -> None:
    """Running a voltage-clamp simulation sets sim_token on SimulationState."""
    assert state_tree.sim.sim_token == ""

    await run_flow(
        state_tree,
        neuron_preset=SQUID_GIANT_AXON,
        protocol_preset=IV_CURVE,
    )

    assert state_tree.sim.sim_token != ""
