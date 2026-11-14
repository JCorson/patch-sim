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
from tests.e2e.conftest import StateTree, patch_get_state


class _BackgroundSelf:
    """Stand-in for the ``self`` of a background handler.

    Supports ``async with self`` and forwards attribute reads and writes to a
    bare state instance, so a background handler's underlying function can
    run without a live Reflex app.
    """

    def __init__(self, state: SimulationState) -> None:
        """Wrap *state*.

        Args:
            state: Bare state instance to forward to.
        """
        object.__setattr__(self, "_state", state)

    async def __aenter__(self) -> "_BackgroundSelf":
        """Enter the (no-op) state lock.

        Returns:
            This stand-in.
        """
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Leave the (no-op) state lock.

        Args:
            *exc: Exception type, value, and traceback (ignored).
        """

    def __getattr__(self, name: str) -> Any:
        """Forward attribute reads to the wrapped state.

        Args:
            name: Attribute name.

        Returns:
            The wrapped state's attribute.
        """
        return getattr(self._state, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute writes to the wrapped state.

        Args:
            name: Attribute name.
            value: Value to assign.
        """
        setattr(self._state, name, value)


async def _run_simulation(
    tree: StateTree, *, neuron_preset: str, protocol_preset: str
) -> None:
    """Load presets, then run the real ``run_simulation`` handler.

    Args:
        tree: State tree to run in.
        neuron_preset: Neuron preset to load.
        protocol_preset: Protocol preset to load.
    """
    async with patch_get_state(tree):
        [_ async for _ in tree.neuron.load_neuron_preset(neuron_preset)]
        [_ async for _ in tree.protocol.load_protocol_preset(protocol_preset)]
        [_ async for _ in SimulationState.run_simulation.fn(_BackgroundSelf(tree.sim))]


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
    filename, payload = exports.get(token)
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
    await _run_simulation(
        state_tree, neuron_preset=SQUID_GIANT_AXON, protocol_preset=ACTION_POTENTIAL
    )
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
    await _run_simulation(
        state_tree, neuron_preset=SQUID_GIANT_AXON, protocol_preset=IV_CURVE
    )
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


async def test_export_header_records_values_that_were_run(
    state_tree: StateTree,
) -> None:
    """Editing the protocol after a run does not change the exported header."""
    await _run_simulation(
        state_tree, neuron_preset=SQUID_GIANT_AXON, protocol_preset=ACTION_POTENTIAL
    )
    ran_duration = state_tree.protocol.stimulus_duration
    state_tree.protocol.stimulus_duration = ran_duration + 123.0

    csv = await _export(state_tree)

    assert f"# stimulus_duration_ms: {ran_duration:g}" in csv
    assert f"# stimulus_duration_ms: {ran_duration + 123.0:g}" not in csv
    assert "# continuous_mode: false" in csv
    assert "# simulated_at: " in csv
    assert "# exported_at: " in csv
