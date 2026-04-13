"""Neuron parameter state for the patch_sim web UI."""

import logging

import reflex as rx

import patch_sim
from patch_sim.constants import (
    DEFAULT_C_M,
    DEFAULT_CA_IN,
    DEFAULT_CA_OUT,
    DEFAULT_CL_IN,
    DEFAULT_CL_OUT,
    DEFAULT_G_ICAL,
    DEFAULT_G_ICAN,
    DEFAULT_G_ICAT,
    DEFAULT_G_IH,
    DEFAULT_G_IKA,
    DEFAULT_G_IKCA,
    DEFAULT_G_IKIR,
    DEFAULT_G_IKV31,
    DEFAULT_G_IM,
    DEFAULT_G_K,
    DEFAULT_G_L,
    DEFAULT_G_NA,
    DEFAULT_G_NAP,
    DEFAULT_G_NAR,
    DEFAULT_K_IN,
    DEFAULT_K_OUT,
    DEFAULT_NA_IN,
    DEFAULT_NA_OUT,
    DEFAULT_T,
    DEFAULT_V_REST,
)
from patch_sim_ui import presets
from patch_sim_ui.state._common import _make_float_setter, _set_float

_NEURON_FLOAT_FIELDS: list[str] = [
    "g_Na",
    "g_K",
    "g_L",
    "C_m",
    "v_rest",
    "Na_out",
    "Na_in",
    "K_out",
    "K_in",
    "Cl_out",
    "Cl_in",
    "Ca_out",
    "Ca_in",
    "T",
]

_CHANNEL_FLOAT_FIELDS: list[str] = [
    "ih_g_max",
    "ika_g_max",
    "ikv31_g_max",
    "inap_g_max",
    "inar_g_max",
    "im_g_max",
    "ikir_g_max",
    "ikca_g_max",
    "ical_g_max",
    "icat_g_max",
    "ican_g_max",
]

_NON_VISIBILITY_BOOL_FIELDS: list[str] = [
    "ih_enabled",
    "ika_enabled",
    "ikv31_enabled",
    "inap_enabled",
    "inar_enabled",
    "im_enabled",
    "ikir_enabled",
    "ikca_enabled",
    "ical_enabled",
    "icat_enabled",
    "ican_enabled",
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


logger = logging.getLogger(__name__)


class NeuronState(rx.State):
    """State for neuron biophysical parameters and auxiliary channel configuration."""

    # ------------------------------------------------------------------ #
    # Neuron parameters                                                   #
    # ------------------------------------------------------------------ #
    g_Na: float = DEFAULT_G_NA
    g_K: float = DEFAULT_G_K
    g_L: float = DEFAULT_G_L
    C_m: float = DEFAULT_C_M
    v_rest: float = DEFAULT_V_REST
    Na_out: float = DEFAULT_NA_OUT
    Na_in: float = DEFAULT_NA_IN
    K_out: float = DEFAULT_K_OUT
    K_in: float = DEFAULT_K_IN
    Cl_out: float = DEFAULT_CL_OUT
    Cl_in: float = DEFAULT_CL_IN
    Ca_out: float = DEFAULT_CA_OUT
    Ca_in: float = DEFAULT_CA_IN
    T: float = DEFAULT_T

    # ------------------------------------------------------------------ #
    # Additional channels                                                 #
    # ------------------------------------------------------------------ #
    ih_enabled: bool = False
    ih_g_max: float = DEFAULT_G_IH
    ika_enabled: bool = False
    ika_g_max: float = DEFAULT_G_IKA
    ikv31_enabled: bool = False
    ikv31_g_max: float = DEFAULT_G_IKV31
    inap_enabled: bool = False
    inap_g_max: float = DEFAULT_G_NAP
    inar_enabled: bool = False
    inar_g_max: float = DEFAULT_G_NAR
    im_enabled: bool = False
    im_g_max: float = DEFAULT_G_IM
    ikir_enabled: bool = False
    ikir_g_max: float = DEFAULT_G_IKIR
    ikca_enabled: bool = False
    ikca_g_max: float = DEFAULT_G_IKCA
    ical_enabled: bool = False
    ical_g_max: float = DEFAULT_G_ICAL
    icat_enabled: bool = False
    icat_g_max: float = DEFAULT_G_ICAT
    ican_enabled: bool = False
    ican_g_max: float = DEFAULT_G_ICAN

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
        """Leak reversal potential in mV."""
        return float(patch_sim.nernst_potential(-1, self.T, self.Cl_out, self.Cl_in))

    @rx.var
    def E_Ca(self) -> float:
        """Calcium reversal potential in mV (z=+2)."""
        return float(patch_sim.nernst_potential(2, self.T, self.Ca_out, self.Ca_in))

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

    async def load_neuron_preset(self, name: str) -> None:
        """Load a neuron-type preset, updating only neuron parameters.

        Sets conductances and auxiliary channel configuration for the selected
        neuron type without touching any protocol settings.  Records the active
        neuron type so that subsequent protocol preset loads can apply
        neuron-specific adjustments via NEURON_PROTOCOL_ADJUSTMENTS.

        Clears current sweeps (unless stored traces exist) and continuous state
        via SimulationState, and syncs the ``_label_neuron_type`` shadow copy
        used by sweep labels.  When stored traces are present the previous
        simulation is kept visible so the figure is not cleared mid-comparison.

        Args:
            name: Key into ``patch_sim_ui.presets.NEURON_UI_PRESETS``.
                Ignored if not found.
        """
        from patch_sim_ui.state.simulation import SimulationState

        if name not in presets.NEURON_UI_PRESETS:
            logger.debug("load_neuron_preset: unknown preset %r ignored", name)
            return
        logger.info("Loaded neuron preset: %s", name)
        self._apply_neuron_preset(name)
        sim_st = await self.get_state(SimulationState)
        if not sim_st.stored_traces:
            sim_st.current_sweeps = []
        sim_st._cont_has_state = False
        sim_st._label_neuron_type = name

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
        vars()[f"set_{_f}"] = _make_float_setter(_f, "NeuronState")

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
        channels = tuple(
            patch_sim.ChannelConfig(factory, g_max=getattr(self, f"{name}_g_max"))
            for name, factory in patch_sim.CHANNEL_REGISTRY.items()
            if getattr(self, f"{name}_enabled")
        )

        # Use the core Na⁺/K⁺ channel factories from the active preset so
        # that presets with non-default kinetics (e.g. Pospischil, STN) are
        # honoured.  Fall back to HH52 defaults for unknown preset names.
        preset_cfg = patch_sim.NEURON_PRESETS.get(self.active_neuron_type)
        na_factory = (
            preset_cfg.na_channel_factory if preset_cfg else patch_sim.make_na_channel
        )
        k_factory = (
            preset_cfg.k_channel_factory if preset_cfg else patch_sim.make_k_channel
        )

        config = patch_sim.NeuronConfig(
            g_Na=self.g_Na,
            g_K=self.g_K,
            g_L=self.g_L,
            C_m=self.C_m,
            v_rest=self.v_rest,
            Na_out=self.Na_out,
            Na_in=self.Na_in,
            K_out=self.K_out,
            K_in=self.K_in,
            Cl_out=self.Cl_out,
            Cl_in=self.Cl_in,
            Ca_out=self.Ca_out,
            Ca_in=self.Ca_in,
            T=self.T,
            channels=channels,
            na_channel_factory=na_factory,
            k_channel_factory=k_factory,
        )
        return patch_sim.make_neuron(config=config)
