"""Shared fixtures and helpers for headless e2e pipeline tests.

Each test in tests/e2e/ exercises a full user-facing flow — preset loading,
simulation, and analysis population — by calling real state-handler code
directly, without spinning up a Reflex dev server or browser.

The Reflex runtime guard requires PYTEST_CURRENT_TEST to be set before any
Reflex state classes are imported; this module sets it at import time.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator
from unittest.mock import patch

# Must be set before importing any Reflex or patch_sim_ui modules.
os.environ.setdefault("PYTEST_CURRENT_TEST", "conftest::e2e")

import pytest  # noqa: E402

pytest.importorskip("reflex")

from patch_sim_ui.state import SimulationState  # noqa: E402
from patch_sim_ui.state.analysis import AnalysisState  # noqa: E402
from patch_sim_ui.state.log import LogState  # noqa: E402
from patch_sim_ui.state.neuron import NeuronState  # noqa: E402
from patch_sim_ui.state.protocol import ProtocolState  # noqa: E402
from patch_sim_ui.state.simulation import _compute_simulation, _SimResult  # noqa: E402
from patch_sim_ui.state.visibility import VisibilityState  # noqa: E402

# ---------------------------------------------------------------------------
# State factories
# ---------------------------------------------------------------------------


def make_sim_state() -> SimulationState:
    """Return a fresh SimulationState bypassing the Reflex runtime guard.

    ``_reflex_internal_init=True`` is a private Reflex kwarg that skips the
    runtime-presence check (``State.__init__`` normally raises outside a live
    Reflex app).  If a Reflex upgrade breaks this, search for the kwarg in the
    Reflex source to find the new bypass mechanism.
    """
    return SimulationState(_reflex_internal_init=True)


def make_neuron_state() -> NeuronState:
    """Return a fresh NeuronState bypassing the Reflex runtime guard."""
    return NeuronState(_reflex_internal_init=True)


def make_protocol_state() -> ProtocolState:
    """Return a fresh ProtocolState bypassing the Reflex runtime guard."""
    return ProtocolState(_reflex_internal_init=True)


def make_analysis_state() -> AnalysisState:
    """Return a fresh AnalysisState bypassing the Reflex runtime guard."""
    return AnalysisState(_reflex_internal_init=True)


def make_visibility_state() -> VisibilityState:
    """Return a fresh VisibilityState bypassing the Reflex runtime guard."""
    return VisibilityState(_reflex_internal_init=True)


def make_log_state() -> LogState:
    """Return a fresh LogState bypassing the Reflex runtime guard."""
    return LogState(_reflex_internal_init=True)


# ---------------------------------------------------------------------------
# State tree
# ---------------------------------------------------------------------------


class StateTree:
    """Bundle of all substates for a single e2e test run.

    Attributes:
        sim: Simulation substate.
        neuron: Neuron substate.
        protocol: Protocol substate.
        analysis: Analysis substate.
        visibility: Visibility substate.
        log: Log substate.
    """

    def __init__(self) -> None:
        """Initialise all substates as fresh bare instances."""
        self.sim = make_sim_state()
        self.neuron = make_neuron_state()
        self.protocol = make_protocol_state()
        self.analysis = make_analysis_state()
        self.visibility = make_visibility_state()
        self.log = make_log_state()

    def get_state_fn(self, host_cls: type):
        """Return an async *get_state* replacement for *host_cls*.

        The returned function resolves sibling classes to their instances
        within this tree, and returns a fresh MagicMock for anything else
        (so handlers that access rarely-used siblings don't crash).

        Args:
            host_cls: The state class whose ``get_state`` is being patched.

        Returns:
            An async callable suitable for use with ``patch.object``.
        """
        from unittest.mock import MagicMock

        mapping = {
            SimulationState: self.sim,
            NeuronState: self.neuron,
            ProtocolState: self.protocol,
            AnalysisState: self.analysis,
            VisibilityState: self.visibility,
            LogState: self.log,
        }

        async def _get_state(_self: object, cls: type) -> object:
            """Resolve *cls* to the tree instance or a MagicMock fallback."""
            return mapping.get(cls, MagicMock())

        return _get_state


@pytest.fixture()
def state_tree() -> StateTree:
    """Provide a fresh StateTree for each test.

    Returns:
        A :class:`StateTree` with all substates initialised to defaults.
    """
    return StateTree()


# ---------------------------------------------------------------------------
# get_state patch context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def patch_get_state(tree: StateTree) -> AsyncIterator[None]:
    """Patch ``get_state`` on all substates to resolve within *tree*.

    Args:
        tree: The :class:`StateTree` whose instances should be returned by
            ``get_state`` calls.

    Yields:
        Nothing — used purely for its side effects.
    """
    with (
        patch.object(
            SimulationState, "get_state", new=tree.get_state_fn(SimulationState)
        ),
        patch.object(NeuronState, "get_state", new=tree.get_state_fn(NeuronState)),
        patch.object(ProtocolState, "get_state", new=tree.get_state_fn(ProtocolState)),
        patch.object(AnalysisState, "get_state", new=tree.get_state_fn(AnalysisState)),
        patch.object(
            VisibilityState, "get_state", new=tree.get_state_fn(VisibilityState)
        ),
        patch.object(LogState, "get_state", new=tree.get_state_fn(LogState)),
    ):
        yield


# ---------------------------------------------------------------------------
# High-level flow helper
# ---------------------------------------------------------------------------


async def run_flow(
    tree: StateTree,
    *,
    neuron_preset: str,
    protocol_preset: str,
) -> _SimResult:
    """Drive the full preset-load → simulate → apply pipeline.

    Mirrors the user flow: load a neuron preset, load a protocol preset, build
    the neuron and protocol objects, run :func:`_compute_simulation`, and apply
    results via :meth:`~SimulationState._do_apply_simulation`.

    Args:
        tree: Fully initialised :class:`StateTree` for this test.
        neuron_preset: Name of the neuron preset to load (e.g. ``SQUID_GIANT_AXON``).
        protocol_preset: Protocol preset name (e.g. ``ACTION_POTENTIAL``).

    Returns:
        The :class:`~patch_sim_ui.state.simulation._SimResult` produced by the
        simulation, also already applied to ``tree.sim`` and ``tree.analysis``.
    """
    async with patch_get_state(tree):
        [_ async for _ in tree.neuron.load_neuron_preset(neuron_preset)]
        [_ async for _ in tree.protocol.load_protocol_preset(protocol_preset)]

    neuron = tree.neuron._build_neuron()
    protocols = tree.protocol._build_protocols()
    mode = tree.protocol.clamp_mode

    result = _compute_simulation(
        neuron=neuron,
        protocols=protocols,
        mode=mode,
        stored_traces=list(tree.sim.stored_traces),
        show_hover=tree.sim.show_hover,
        min_stimulus=tree.protocol.min_stimulus,
        max_stimulus=tree.protocol.max_stimulus,
        stimulus_step=tree.protocol.stimulus_step,
        pre_stimulus_duration=tree.protocol.pre_stimulus_duration,
        stimulus_duration=tree.protocol.stimulus_duration,
    )
    tree.sim._do_apply_simulation(result, tree.analysis)
    return result
