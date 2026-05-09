"""Neuron parameter state for the patch_sim web UI."""

import dataclasses
import hashlib
import logging
from typing import Any, AsyncGenerator

import reflex as rx

import patch_sim
import patch_sim.channels
from patch_sim.presets import NEURON_PRESETS
from patch_sim_ui import presets
from patch_sim_ui.channels import CHANNELS
from patch_sim_ui.state._common import _set_float

_NEURON_FLOAT_FIELDS: list[str] = list(presets.NEURON_CONFIG_SCALAR_FIELDS)

_CHANNEL_FLOAT_FIELDS: list[str] = [ch.g_max_field for ch in CHANNELS]


#: The NeuronState fields that determine the passive-only membrane test result.
#: Only these fields are included in ``neuron_fingerprint`` so that changing
#: active conductances (na_g_max, k_g_max, v_rest, auxiliary channels) does
#: not invalidate the membrane test cache.
#:
#: Ca²⁺ concentrations are omitted: no Ca²⁺ channels survive the passive
#: filter (membrane_test only retains channels with no gating variables), so
#: their reversal potentials carry no current.  Na⁺ and K⁺ concentrations ARE
#: included because E_NaL and E_KL are used to compute the effective E_L for
#: the passive RC circuit.
_PASSIVE_PARAM_FIELDS: list[str] = [
    "nal_g_max",
    "kl_g_max",
    "C_m",
    "Na_out",
    "Na_in",
    "K_out",
    "K_in",
    "T",
]


def _make_neuron_float_setter(field_name: str):
    """Factory returning an async generator setter that chains a membrane test.

    Wraps :func:`~patch_sim_ui.state._common._set_float` and then yields
    ``SimulationState.run_membrane_test_debounced`` so that any change to a
    neuron parameter automatically refreshes the displayed passive properties.
    Uses ``yield`` (not ``return``) because Reflex only chains events yielded
    from generators; returning an event from a sync handler has no effect.

    The fingerprint-based cache inside ``run_membrane_test`` ensures the
    simulation only re-runs when passive-relevant parameters (nal_g_max, kl_g_max,
    C_m, ion concentrations, T) actually change.

    Args:
        field_name: Name of the ``NeuronState`` attribute to update.

    Returns:
        An async generator event handler that accepts
        ``str | list[float] | float``, updates the field, and yields
        ``run_membrane_test_debounced``.
    """

    async def setter(self, value: "str | list[float] | float"):
        """Set the field from an input or slider event and queue a membrane test."""
        # Late import avoids a circular dependency between neuron and simulation.
        from patch_sim_ui.state.simulation import (  # noqa: PLC0415
            SimulationState,
        )

        _set_float(self, field_name, value)
        yield SimulationState.run_membrane_test_debounced

    setter.__name__ = f"set_{field_name}"
    setter.__qualname__ = f"NeuronState.set_{field_name}"
    setter.__doc__ = f"Set {field_name} and queue a membrane test re-run."
    return setter


logger = logging.getLogger(__name__)


class NeuronState(rx.State):
    """State for neuron biophysical parameters and channel configuration.

    Per-channel ``{id}_g_max`` reactive vars are added dynamically at module
    load time (one per entry in :data:`CHANNELS`).  See the
    ``add_var`` loop at the bottom of this module.  Visibility (which slider
    is shown) is driven by ``visible_channel_ids`` from the active preset.
    """

    # Snapshot of slider values overwritten by the active protocol's
    # ``PROTOCOL_NEURON_OVERRIDES`` entry — populated when an override is
    # applied, drained when the user switches to a protocol without an
    # override (or to a different override).  Keeps Na+ Channel Activation
    # reversible so leaving the protocol restores the user's prior values
    # rather than leaving every non-Na slider stuck at 0.
    _protocol_override_snapshot: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Preset label                                                        #
    # ------------------------------------------------------------------ #
    active_neuron_type: str = "Squid Giant Axon (Classic HH)"

    # ------------------------------------------------------------------ #
    # Visible auxiliary channels                                         #
    # ------------------------------------------------------------------ #
    @rx.var
    def visible_channel_ids(self) -> list[str]:
        """UI ids of auxiliary channels the active preset includes.

        Iterated in :data:`CHANNELS` order so the rendered row
        order is deterministic regardless of ``frozenset`` iteration order.
        Drives both the neuron-panel auxiliary-channel rows and the
        sweep-manager trace-visibility checkboxes — single source of truth
        for "this preset uses these channels".

        Returns:
            Ordered list of channel ids (e.g. ``["ih", "inap", "im", "mskv"]``
            for Cortical Pyramidal).  Empty for presets with no auxiliary
            channels (e.g. Squid Giant Axon) or unknown preset names.
        """
        ids = presets.PRESET_CHANNEL_IDS.get(self.active_neuron_type, frozenset())
        return [ch.id for ch in CHANNELS if ch.id in ids]

    @rx.var
    def has_visible_channels(self) -> bool:
        """True if the active preset exposes any auxiliary channels.

        Used by the neuron panel to hide the "Additional Channels" accordion
        entirely when the preset has none (e.g. Squid Giant Axon).

        Returns:
            ``True`` iff :attr:`visible_channel_ids` is non-empty.
        """
        return len(self.visible_channel_ids) > 0

    # ------------------------------------------------------------------ #
    # Derived reversal potentials                                        #
    # ------------------------------------------------------------------ #
    @rx.var
    def E_Na(self) -> float:
        """Sodium reversal potential in mV."""
        return float(patch_sim.nernst_potential(1, self.T, self.Na_out, self.Na_in))

    @rx.var
    def E_K(self) -> float:
        """Potassium reversal potential in mV."""
        return float(patch_sim.nernst_potential(1, self.T, self.K_out, self.K_in))

    @rx.var
    def E_L(self) -> float:
        """Effective leak reversal in mV (g_max-weighted average of E_NaL and E_KL)."""
        e_na = float(patch_sim.nernst_potential(1, self.T, self.Na_out, self.Na_in))
        e_k = float(patch_sim.nernst_potential(1, self.T, self.K_out, self.K_in))
        g_nal = self.nal_g_max
        g_kl = self.kl_g_max
        g_total = g_nal + g_kl
        if g_total <= 0:
            return self.v_rest
        return (g_nal * e_na + g_kl * e_k) / g_total

    @rx.var
    def E_Ca(self) -> float:
        """Calcium reversal potential in mV (z=+2)."""
        return float(patch_sim.nernst_potential(2, self.T, self.Ca_out, self.Ca_in))

    def _compute_fingerprint(self) -> str:
        """Compute a SHA-256 hex digest of the passive membrane parameters.

        Reads the raw state var values directly so that callers always get a
        fresh result reflecting the current state.  This is intentionally a
        plain method, not a ``@rx.var``: Reflex caches ``ComputedVar`` values
        and may return a stale digest if called from a background task before
        the cache has been invalidated by the latest state flush.

        Only hashes the fields that determine the passive-only membrane
        test result (nal_g_max, kl_g_max, C_m, ion concentrations, T).  Active
        conductances are excluded because the membrane test blocks them.

        Returns:
            A hex digest string that changes only when passive parameters change.
        """
        parts = [repr(getattr(self, f)) for f in _PASSIVE_PARAM_FIELDS]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    @rx.var
    def neuron_fingerprint(self) -> str:
        """SHA-256 hex digest of the passive membrane parameters (frontend var).

        Delegates to :meth:`_compute_fingerprint` so the frontend can react
        to passive parameter changes.  Do not call this from background tasks
        — use :meth:`_compute_fingerprint` directly to bypass the
        ``ComputedVar`` cache.

        Returns:
            A hex digest string that changes only when passive parameters change.
        """
        return self._compute_fingerprint()

    # ------------------------------------------------------------------ #
    # Protocol override snapshot/restore                                  #
    # ------------------------------------------------------------------ #
    def _restore_protocol_overrides(self) -> None:
        """Undo the active protocol override by restoring snapshotted values.

        Walks :attr:`_protocol_override_snapshot`, restoring each key whose
        slider still exists on this state instance.  Clears the snapshot so
        a subsequent override takes a fresh capture.  Skips keys that no
        longer correspond to a registered field (e.g. when the user changed
        neuron preset between snapshot and restore).
        """
        for key, value in self._protocol_override_snapshot.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._protocol_override_snapshot = {}

    def _apply_protocol_overrides(self, protocol_name: str | None) -> None:
        """Apply ``PROTOCOL_NEURON_OVERRIDES`` for ``protocol_name``.

        Always restores any previously-snapshotted values first, so the
        new override is taken on top of the user's pre-override slider
        state.  Then snapshots the current values for keys the new override
        touches, applies the override, and stores the snapshot for the
        next call.

        When ``protocol_name`` is ``None`` or has no override entry, this
        method just clears the snapshot (no new override is applied).
        Static dict overrides and callable overrides are both supported;
        a callable is invoked with :attr:`active_neuron_type` so it can
        adapt to whichever channels the active preset has.

        Args:
            protocol_name: Key into ``presets.PROTOCOL_NEURON_OVERRIDES``,
                or ``None`` if no protocol is active.
        """
        self._restore_protocol_overrides()
        if protocol_name is None:
            return
        overrides = presets.PROTOCOL_NEURON_OVERRIDES.get(protocol_name, {})
        if callable(overrides):
            overrides = overrides(self.active_neuron_type)
        snapshot: dict[str, float] = {}
        for key, value in overrides.items():
            if hasattr(self, key):
                snapshot[key] = float(getattr(self, key))
                setattr(self, key, value)
        self._protocol_override_snapshot = snapshot

    # ------------------------------------------------------------------ #
    # Event handlers                                                     #
    # ------------------------------------------------------------------ #
    def _apply_neuron_preset(self, name: str) -> None:
        """Apply neuron preset config to this state instance synchronously.

        Sets all conductances, channel enables, and ``active_neuron_type`` for
        the given preset.  Clears any active protocol-override snapshot
        because the snapshot's keys may reference sliders that don't apply
        to the new preset.  Does not touch any cross-state fields (use
        :meth:`load_neuron_preset` for the full async handler that
        re-applies the active protocol's override on top).

        Args:
            name: Key into ``patch_sim_ui.presets.NEURON_UI_PRESETS``.
                Silently ignored if not found.
        """
        if name not in presets.NEURON_UI_PRESETS:
            return
        # Discard any prior protocol-override snapshot — it referenced the
        # previous preset's sliders and would be misleading after re-seed.
        self._protocol_override_snapshot = {}
        config = presets.NEURON_UI_PRESETS[name]
        for key, value in config.items():
            setattr(self, key, value)
        self.active_neuron_type = name

    async def load_neuron_preset(  # type: ignore[override]  # base class declares -> None; yielding events upgrades this to AsyncGenerator
        self, name: str
    ) -> AsyncGenerator[Any, None]:
        """Load a neuron-type preset and re-apply any active protocol overrides.

        Sets conductances and auxiliary channel configuration for the selected
        neuron type, then re-applies the currently active protocol preset (if
        any) with neuron-specific adjustments from NEURON_PROTOCOL_ADJUSTMENTS
        and PROTOCOL_NEURON_OVERRIDES.  This ensures that switching neuron type
        immediately reflects the correct protocol parameters without requiring
        the user to re-select the protocol.

        Clears current sweeps (unless stored traces exist) and continuous state
        via SimulationState, and syncs the ``_figure_clamp_mode`` and
        ``_label_neuron_type`` shadow copies.  When stored traces are present the
        previous simulation is kept visible so the figure is not cleared
        mid-comparison.

        Args:
            name: Key into ``patch_sim_ui.presets.NEURON_UI_PRESETS``.
                Ignored if not found.
        """
        from patch_sim_ui.state.analysis import AnalysisState
        from patch_sim_ui.state.protocol import ProtocolState
        from patch_sim_ui.state.simulation import SimulationState

        if name not in presets.NEURON_UI_PRESETS:
            logger.debug("load_neuron_preset: unknown preset %r ignored", name)
            return
        logger.info("Loaded neuron preset: %s", name)
        self._apply_neuron_preset(name)
        proto_st = await self.get_state(ProtocolState)
        if proto_st.active_protocol_preset:
            proto_st._apply_protocol_preset(proto_st.active_protocol_preset, name)
            self._apply_protocol_overrides(proto_st.active_protocol_preset)
        sim_st = await self.get_state(SimulationState)
        _cleared = not sim_st.stored_traces
        if _cleared:
            sim_st._current_sweeps = []
            analysis_st = await self.get_state(AnalysisState)
            analysis_st.clear_results()
        sim_st._cont_has_state = False
        sim_st._label_neuron_type = name
        sim_st._figure_clamp_mode = proto_st.clamp_mode
        if _cleared:
            yield rx.call_script(
                "var gd=document.getElementById('ps-trace-plot');"
                "if(gd&&gd.data){Plotly.purge(gd);}"
            )
        yield SimulationState.run_membrane_test

    # ------------------------------------------------------------------ #
    # Numeric field setters                                              #
    # ------------------------------------------------------------------ #
    def _set_float(self, field: str, value: "str | list[float] | float") -> None:
        """Coerce value to float and set the named field.

        Args:
            field: Name of the NeuronState attribute to update.
            value: Raw value from an input or slider event.
        """
        _set_float(self, field, value)

    for _f in _NEURON_FLOAT_FIELDS + _CHANNEL_FLOAT_FIELDS:
        vars()[f"set_{_f}"] = _make_neuron_float_setter(_f)

    # ------------------------------------------------------------------ #
    # Simulation helpers                                                 #
    # ------------------------------------------------------------------ #
    def _build_neuron(self) -> "patch_sim.Neuron":
        """Construct a Neuron from current state parameters.

        Calls the active preset's factory to obtain a fully-configured
        baseline (carrying the preset's full channel list, calcium dynamics,
        and area), then overlays the user's scalar slider values via
        :func:`dataclasses.replace`.  Per-channel ``g_max`` slider values are
        also overlaid onto the baseline's channels so the user can scale
        individual conductances; channel composition and kinetics remain
        preset-baked.

        Returns:
            A :class:`patch_sim.Neuron` whose scalar fields and per-channel
            ``g_max`` reflect the current sliders.
        """
        factory = NEURON_PRESETS.get(self.active_neuron_type)
        baseline = factory() if factory is not None else patch_sim.Neuron()
        scalar_overrides = {
            name: getattr(self, name) for name in presets.NEURON_CONFIG_SCALAR_FIELDS
        }
        rebuilt_channels = tuple(
            dataclasses.replace(ch, g_max=getattr(self, f"{ui_id}_g_max"))
            if (ui_id := presets.CHANNEL_NAME_TO_ID.get(ch.name)) is not None
            else ch
            for ch in baseline.channels
        )
        return dataclasses.replace(
            baseline,
            channels=rebuilt_channels,
            **scalar_overrides,
        )


# Register Neuron scalar fields as NeuronState vars
for _nc_name, _nc_default in presets.NEURON_CONFIG_SCALAR_DEFAULTS.items():
    NeuronState.add_var(_nc_name, float, _nc_default)

# Register one ``{id}_g_max`` reactive var per channel in the registry, seeded
# from the registry default.  Per-channel sliders bind to these vars; values
# are reset by ``neuron_to_ui_state`` whenever a preset is loaded.
for _ch_meta in CHANNELS:
    NeuronState.add_var(
        _ch_meta.g_max_field,
        float,
        presets.DEFAULT_G_MAX[_ch_meta.id],
    )
