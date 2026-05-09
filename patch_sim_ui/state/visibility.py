"""Trace visibility state for the patch_sim web UI."""

import json
import logging

import reflex as rx

from patch_sim_ui.channels import (
    CHANNELS,
    CURRENT_FIELD_MAP,
    GATING_FIELD_MAP,
)
from patch_sim_ui.plotting import compute_trace_visibility_map
from patch_sim_ui.state._common import _PLOTLY_GD_JS

_VISIBILITY_FIELDS: list[str] = [
    "show_voltage",
    "show_total_current",
    # Per-channel current toggles, derived uniformly from the registry.
    *[ch.current_visibility_field for ch in CHANNELS],
    # Per-channel joint gating toggles for every channel that has gates.
    *[ch.gating_visibility_field for ch in CHANNELS if ch.gating_vars],
]

logger = logging.getLogger(__name__)


def _make_visibility_setter_async(field_name: str):
    """Factory returning an async visibility event handler for ``field_name``.

    The generated handler updates the server-side state var and issues a
    ``Plotly.restyle`` call to toggle the corresponding trace(s) client-side.
    Implemented as async because it reads sweep data from SimulationState to build
    the trace index map.

    Args:
        field_name: Name of the show_* state attribute to update.

    Returns:
        An async event handler method that sets the bool field and yields a
        ``rx.call_script`` event to apply the visibility change in-browser.
    """

    async def setter(self, value: bool):
        """Set the visibility flag and apply a client-side Plotly restyle."""
        from patch_sim_ui.state.simulation import SimulationState

        setattr(self, field_name, value)
        sim_st = await self.get_state(SimulationState)
        trace_map = compute_trace_visibility_map(
            current_sweeps=sim_st._current_sweeps,
            clamp_mode=sim_st._figure_clamp_mode,
            additional_current_field_map=CURRENT_FIELD_MAP,
            additional_gating_field_map=GATING_FIELD_MAP,
            stored_traces=sim_st.stored_traces,
        )
        indices = trace_map.get(field_name, [])
        if indices:
            visible_js = "true" if value else "false"
            js = (
                f"{_PLOTLY_GD_JS}"
                f"if(gd&&gd.data)Plotly.restyle(gd,"
                f"{{visible:{visible_js}}},{json.dumps(indices)})"
            )
            return rx.call_script(js)

    setter.__name__ = f"set_{field_name}"
    setter.__qualname__ = f"VisibilityState.set_{field_name}"
    setter.__doc__ = f"Set {field_name} and apply client-side visibility restyle."
    return setter


class VisibilityState(rx.State):
    """State for trace visibility checkboxes."""

    # ------------------------------------------------------------------ #
    # Trace visibility checkboxes                                        #
    # ------------------------------------------------------------------ #
    show_voltage: bool = True
    show_total_current: bool = True
    show_na_current: bool = True
    show_k_current: bool = True
    show_nal_current: bool = False
    show_kl_current: bool = False
    show_na_gating: bool = True
    show_k_gating: bool = True
    show_ih_current: bool = True
    show_ih_gating: bool = True
    show_ika_current: bool = True
    show_ika_gating: bool = True
    show_ikv31_current: bool = True
    show_ikv31_gating: bool = True
    show_mskv_current: bool = True
    show_mskv_gating: bool = True
    show_inap_current: bool = True
    show_inap_gating: bool = True
    show_inar_current: bool = True
    show_inar_gating: bool = True
    show_im_current: bool = True
    show_im_gating: bool = True
    show_katp_current: bool = True
    show_katp_gating: bool = True
    show_ikir_current: bool = True
    show_ikir_gating: bool = True
    show_ikca_current: bool = True
    show_ikca_gating: bool = True
    show_ical_current: bool = True
    show_ical_gating: bool = True
    show_cav13_current: bool = True
    show_cav13_gating: bool = True
    show_icat_current: bool = True
    show_icat_gating: bool = True
    show_ican_current: bool = True
    show_ican_gating: bool = True
    show_sk_current: bool = True
    show_sk_gating: bool = True

    # ------------------------------------------------------------------ #
    # Visibility setters                                                 #
    # ------------------------------------------------------------------ #
    for _f in _VISIBILITY_FIELDS:
        vars()[f"set_{_f}"] = _make_visibility_setter_async(_f)
