"""Neuron parameter state for the patch_sim web UI."""

import hashlib
import logging
from typing import Any, AsyncGenerator

import reflex as rx

import patch_sim
from patch_sim.constants import (
    DEFAULT_G_CAV13,
    DEFAULT_G_ICAL,
    DEFAULT_G_ICAN,
    DEFAULT_G_ICAT,
    DEFAULT_G_IH,
    DEFAULT_G_IKA,
    DEFAULT_G_IKCA,
    DEFAULT_G_IKIR,
    DEFAULT_G_IKV31,
    DEFAULT_G_IM,
    DEFAULT_G_KATP,
    DEFAULT_G_MSKV,
    DEFAULT_G_NAP,
    DEFAULT_G_NAR,
    DEFAULT_G_SK,
)
from patch_sim_ui import presets
from patch_sim_ui.channels import ADDITIONAL_CHANNELS
from patch_sim_ui.state._common import _set_float

_NEURON_FLOAT_FIELDS: list[str] = list(presets.NEURON_CONFIG_SCALAR_FIELDS)

_CHANNEL_FLOAT_FIELDS: list[str] = [ch.g_max_field for ch in ADDITIONAL_CHANNELS]

_NON_VISIBILITY_BOOL_FIELDS: list[str] = [
    ch.enabled_field for ch in ADDITIONAL_CHANNELS
]


def _make_bool_setter(field_name: str, class_name: str = "NeuronState"):
    """Factory returning a bool event handler for ``field_name``.

    Args:
        field_name: Name of the state attribute to update.
        class_name: Owning state class name used in ``__qualname__``.

    Returns:
        An event handler method that sets the bool field.
    """

    def setter(self, value: bool) -> None:
        """Set the field from a checkbox event."""
        setattr(self, field_name, value)

    setter.__name__ = f"set_{field_name}"
    setter.__qualname__ = f"{class_name}.set_{field_name}"
    setter.__doc__ = f"Set {field_name} from a checkbox event."
    return setter


#: The NeuronState fields that determine the passive-only membrane test result.
#: Only these fields are included in ``neuron_fingerprint`` so that changing
#: active conductances (g_Na, g_K, v_rest, auxiliary channels) does not
#: invalidate the membrane test cache.
#:
#: Ca²⁺ concentrations are omitted: no Ca²⁺ channels are present in the
#: passive neuron (g_Na = g_K = 0), so their reversal potentials carry no
#: current.  Na⁺ and K⁺ concentrations ARE included because E_NaL and E_KL
#: are used to compute the effective E_L for the passive RC circuit.
_PASSIVE_PARAM_FIELDS: list[str] = [
    "g_NaL",
    "g_KL",
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
    simulation only re-runs when passive-relevant parameters (g_NaL, g_KL,
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
    """State for neuron biophysical parameters and auxiliary channel configuration."""

    # ------------------------------------------------------------------ #
    # Additional channels                                                 #
    # ------------------------------------------------------------------ #
    ih_enabled: bool = False
    ih_g_max: float = DEFAULT_G_IH
    ika_enabled: bool = False
    ika_g_max: float = DEFAULT_G_IKA
    ikv31_enabled: bool = False
    ikv31_g_max: float = DEFAULT_G_IKV31
    mskv_enabled: bool = False
    mskv_g_max: float = DEFAULT_G_MSKV
    inap_enabled: bool = False
    inap_g_max: float = DEFAULT_G_NAP
    inar_enabled: bool = False
    inar_g_max: float = DEFAULT_G_NAR
    im_enabled: bool = False
    im_g_max: float = DEFAULT_G_IM
    katp_enabled: bool = False
    katp_g_max: float = DEFAULT_G_KATP
    ikir_enabled: bool = False
    ikir_g_max: float = DEFAULT_G_IKIR
    ikca_enabled: bool = False
    ikca_g_max: float = DEFAULT_G_IKCA
    ical_enabled: bool = False
    ical_g_max: float = DEFAULT_G_ICAL
    cav13_enabled: bool = False
    cav13_g_max: float = DEFAULT_G_CAV13
    icat_enabled: bool = False
    icat_g_max: float = DEFAULT_G_ICAT
    ican_enabled: bool = False
    ican_g_max: float = DEFAULT_G_ICAN
    sk_enabled: bool = False
    sk_g_max: float = DEFAULT_G_SK

    # ------------------------------------------------------------------ #
    # Preset label                                                        #
    # ------------------------------------------------------------------ #
    active_neuron_type: str = "Squid Giant Axon (Classic HH)"

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
        """Effective leak reversal in mV (weighted average of E_Na and E_K)."""
        e_na = float(patch_sim.nernst_potential(1, self.T, self.Na_out, self.Na_in))
        e_k = float(patch_sim.nernst_potential(1, self.T, self.K_out, self.K_in))
        g_total = self.g_NaL + self.g_KL
        if g_total <= 0:
            return self.v_rest
        return (self.g_NaL * e_na + self.g_KL * e_k) / g_total

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
        test result (g_NaL, g_KL, C_m, ion concentrations, T).  Active
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
    # Event handlers                                                     #
    # ------------------------------------------------------------------ #
    def _apply_neuron_preset(self, name: str) -> None:
        """Apply neuron preset config to this state instance synchronously.

        Sets all conductances, channel enables, and ``active_neuron_type`` for
        the given preset.  Does not touch any cross-state fields (use
        :meth:`load_neuron_preset` for the full async handler).

        Args:
            name: Key into ``patch_sim_ui.presets.NEURON_UI_PRESETS``.
                Silently ignored if not found.
        """
        if name not in presets.NEURON_UI_PRESETS:
            return
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
            for key, value in presets.PROTOCOL_NEURON_OVERRIDES.get(
                proto_st.active_protocol_preset, {}
            ).items():
                setattr(self, key, value)
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

    for _f in _NON_VISIBILITY_BOOL_FIELDS:
        vars()[f"set_{_f}"] = _make_bool_setter(_f, "NeuronState")

    # ------------------------------------------------------------------ #
    # Simulation helpers                                                 #
    # ------------------------------------------------------------------ #
    def _build_neuron(self) -> "patch_sim.Neuron":
        """Construct a Neuron from current state parameters.

        Returns:
            A :class:`patch_sim.Neuron` configured with the current conductances,
            ion concentrations, and any enabled auxiliary channels.
        """
        # Use the core Na⁺/K⁺ channel factories from the active preset so
        # that presets with non-default kinetics (e.g. Pospischil, STN) are
        # honoured.  Fall back to HH52 defaults for unknown preset names.
        preset_cfg = patch_sim.NEURON_PRESETS.get(self.active_neuron_type)

        # Map channel-name → preset-specific factory variant, so presets that
        # use a non-canonical factory (e.g. Thalamic Relay's slow-inactivating
        # ICaT) keep their kinetics through UI round-trip.  Channels not in
        # the preset fall back to the canonical CHANNEL_REGISTRY factory.
        preset_factory_overrides: dict[str, Any] = {}
        if preset_cfg is not None:
            for cc in preset_cfg.channels:
                name = presets._FACTORY_TO_NAME.get(cc.factory)
                if name is not None:
                    preset_factory_overrides[name] = cc.factory

        channels = tuple(
            patch_sim.ChannelConfig(
                preset_factory_overrides.get(name, factory),
                g_max=getattr(self, f"{name}_g_max"),
            )
            for name, factory in patch_sim.CHANNEL_REGISTRY.items()
            if getattr(self, f"{name}_enabled")
        )

        na_factory = (
            preset_cfg.na_channel_factory if preset_cfg else patch_sim.make_na_channel
        )
        k_factory = (
            preset_cfg.k_channel_factory if preset_cfg else patch_sim.make_k_channel
        )

        scalar_kwargs = {
            name: getattr(self, name) for name in presets.NEURON_CONFIG_SCALAR_FIELDS
        }
        # area_cm2 and calcium_dynamics are sourced from the active preset's
        # NeuronConfig.  area_cm2 is a static physical attribute of the cell,
        # not user-editable.  calcium_dynamics carries the preset's tuned
        # alpha_ca/tau_ca/ca_init — without forwarding them every preset
        # would silently use NeuronConfig's auto-instantiated default
        # (alpha_ca=1e-4, tau_ca=200 ms), which is dramatically off for
        # presets that tune these for fast Ca clearance (e.g. TRN at
        # tau_ca=20 ms — 10× faster than default).
        area_cm2 = preset_cfg.area_cm2 if preset_cfg else None
        calcium_dynamics = preset_cfg.calcium_dynamics if preset_cfg else None
        config = patch_sim.NeuronConfig(
            **scalar_kwargs,
            channels=channels,
            na_channel_factory=na_factory,
            k_channel_factory=k_factory,
            area_cm2=area_cm2,
            calcium_dynamics=calcium_dynamics,
        )
        return patch_sim.make_neuron(config=config)


# Register NeuronConfig scalar fields as NeuronState vars
for _nc_name, _nc_default in presets.NEURON_CONFIG_SCALAR_DEFAULTS.items():
    NeuronState.add_var(_nc_name, float, _nc_default)
