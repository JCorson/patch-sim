"""Trace visibility state for the patch_sim web UI."""

import json
import logging

import reflex as rx

from patch_sim_ui.plotting import compute_trace_visibility_map
from patch_sim_ui.state._common import (
    _ADDITIONAL_CURRENT_FIELD_MAP,
    _ADDITIONAL_GATING_FIELD_MAP,
    _PLOTLY_GD_JS,
    _VISIBILITY_FIELDS,
)

logger = logging.getLogger("patch_sim_ui.state")


def _make_visibility_setter_async(field_name: str):
    """Factory returning an async visibility event handler for ``field_name``.

    The generated handler updates the server-side state var and issues a
    ``Plotly.restyle`` call to toggle the corresponding trace(s) client-side.
    Implemented as async because it reads sweep data from AppState to build
    the trace index map.

    Args:
        field_name: Name of the show_* state attribute to update.

    Returns:
        An async event handler method that sets the bool field and yields a
        ``rx.call_script`` event to apply the visibility change in-browser.
    """

    async def setter(self, value: bool):
        """Set the visibility flag and apply a client-side Plotly restyle."""
        from patch_sim_ui.state.simulation import AppState

        setattr(self, field_name, value)
        app_st = await self.get_state(AppState)
        trace_map = compute_trace_visibility_map(
            current_sweeps=app_st.current_sweeps,
            saved_sweeps=app_st.saved_sweeps,
            clamp_mode=app_st._figure_clamp_mode,
            additional_current_field_map=_ADDITIONAL_CURRENT_FIELD_MAP,
            additional_gating_field_map=_ADDITIONAL_GATING_FIELD_MAP,
            stored_traces=app_st.stored_traces,
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
    show_sodium_current: bool = True
    show_potassium_current: bool = True
    show_leak_current: bool = False
    show_potassium_activation: bool = True
    show_sodium_activation: bool = True
    show_sodium_inactivation: bool = True
    show_ih_current: bool = True
    show_ih_gating: bool = True
    show_ika_current: bool = True
    show_ika_gating: bool = True
    show_ikv31_current: bool = True
    show_ikv31_gating: bool = True
    show_inap_current: bool = True
    show_inap_gating: bool = True
    show_inar_current: bool = True
    show_inar_gating: bool = True
    show_im_current: bool = True
    show_im_gating: bool = True
    show_ikir_current: bool = True
    show_ikir_gating: bool = True
    show_ikca_current: bool = True
    show_ikca_gating: bool = True
    show_ical_current: bool = True
    show_ical_gating: bool = True
    show_icat_current: bool = True
    show_icat_gating: bool = True
    show_ican_current: bool = True
    show_ican_gating: bool = True

    # ------------------------------------------------------------------ #
    # Visibility setters                                                 #
    # ------------------------------------------------------------------ #
    for _f in _VISIBILITY_FIELDS:
        vars()[f"set_{_f}"] = _make_visibility_setter_async(_f)
