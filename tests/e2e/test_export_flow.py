"""E2E tests for the CSV export flow: run -> export_csv -> download store."""

import io
from typing import Any

import numpy as np
import pytest

from patch_sim.constants import (
    ACTION_POTENTIAL,
    IV_CURVE,
    SQUID_GIANT_AXON,
)
from patch_sim_ui.api import exports
from patch_sim_ui.state import SimulationState
from patch_sim_ui.state.simulation import _protocol_export_params
from tests.e2e.conftest import StateTree, patch_get_state, run_flow


class _BackgroundSelf:
    """Stand-in for the ``self`` of a background handler.

    Supports ``async with self`` and forwards attribute access to a bare
    state instance, so a background handler's underlying function can run
    without a live Reflex app.
    """

    def __init__(self, state: SimulationState) -> None:
        """Wrap *state*.

        Args:
            state: Bare state instance to forward to.
        """
        self._state = state

    async def __aenter__(self) -> "_BackgroundSelf":
        """Enter the (no-op) state lock."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Leave the (no-op) state lock."""

    def __getattr__(self, name: str) -> Any:
        """Forward attribute reads to the wrapped state.

        Args:
            name: Attribute name.

        Returns:
            The wrapped state's attribute.
        """
        return getattr(self._state, name)


def _record_context(tree: StateTree) -> None:
    """Record export context for the tree's latest run.

    Args:
        tree: State tree that has just run a simulation.
    """
    tree.sim._record_export_context(
        tree.neuron._build_neuron(),
        neuron_preset=tree.neuron.active_neuron_type,
        protocol_preset=tree.protocol.active_protocol_preset,
        mode=tree.protocol.clamp_mode,
        ptype=tree.protocol.protocol_type,
        protocol_params=_protocol_export_params(tree.protocol),
    )


async def _export(tree: StateTree) -> str:
    """Run export_csv and return the CSV text from the download store.

    Args:
        tree: State tree with a simulation result.

    Returns:
        The exported CSV text.
    """
    exports._STORE.clear()
    async with patch_get_state(tree):
        events = [
            e async for e in SimulationState.export_csv.fn(_BackgroundSelf(tree.sim))
        ]
    assert len(events) == 1
    assert len(exports._STORE) == 1
    token = next(iter(exports._STORE))
    filename, payload = exports.pop(token)
    assert filename.startswith("patch-sim_squid-giant-axon-classic-hh_")
    assert filename.endswith(".csv")
    return payload.decode("utf-8")


def _parse(csv: str) -> np.ndarray:
    """Parse exported CSV text, skipping metadata comments.

    Args:
        csv: Exported CSV text.

    Returns:
        Structured array with one field per column.
    """
    body = "\n".join(ln for ln in csv.splitlines() if not ln.startswith("#"))
    return np.genfromtxt(io.StringIO(body), delimiter=",", names=True)


def _names(arr: np.ndarray) -> tuple[str, ...]:
    """Return the column names of a structured array.

    Args:
        arr: Structured array from :func:`_parse`.

    Returns:
        The field names, or an empty tuple for an unstructured array.
    """
    return arr.dtype.names or ()


async def test_current_clamp_export(state_tree: StateTree) -> None:
    """A CC step exports one row per sample with voltage, current and metadata."""
    await run_flow(
        state_tree, neuron_preset=SQUID_GIANT_AXON, protocol_preset=ACTION_POTENTIAL
    )
    _record_context(state_tree)
    csv = await _export(state_tree)

    sweep = state_tree.sim._current_sweeps[0]
    arr = _parse(csv)
    assert len(arr) == len(sweep.time)
    assert arr["V_mV"] == pytest.approx(sweep.voltage, rel=1e-5)
    assert arr["Istim_uA_cm2"] == pytest.approx(sweep.stimulus, rel=1e-5)
    assert "INa_uA_cm2" in _names(arr)
    assert "# clamp_mode: Current Clamp" in csv
    assert f"# neuron_preset: {SQUID_GIANT_AXON}" in csv
    assert "# g_max " in csv
    csv.encode("ascii")


async def test_multi_sweep_voltage_clamp_export(state_tree: StateTree) -> None:
    """A multi-sweep VC protocol exports suffixed columns and sweep labels."""
    await run_flow(state_tree, neuron_preset=SQUID_GIANT_AXON, protocol_preset=IV_CURVE)
    _record_context(state_tree)
    csv = await _export(state_tree)

    n = len(state_tree.sim._current_sweeps)
    assert n > 1
    arr = _parse(csv)
    assert "Vcmd_mV_s01" in _names(arr)
    assert f"Vcmd_mV_s{n:02d}" in _names(arr)
    assert "V_mV_s01" not in _names(arr)
    assert "# sweep s01: " in csv
    csv.encode("ascii")


async def test_export_without_result_stores_nothing(state_tree: StateTree) -> None:
    """export_csv with no simulation result is a no-op."""
    exports._STORE.clear()
    async with patch_get_state(state_tree):
        events = [
            e
            async for e in SimulationState.export_csv.fn(
                _BackgroundSelf(state_tree.sim)
            )
        ]
    assert events == []
    assert len(exports._STORE) == 0
