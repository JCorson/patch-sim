"""Shared constants and factory functions reused across multiple substate modules.

Only items imported by two or more substate modules live here.  Items used
exclusively within a single substate are defined in that module instead.
"""

# ------------------------------------------------------------------ #
# Additional-channel visibility field maps                           #
# ------------------------------------------------------------------ #
# Maps from additional-channel sweep keys to show_* field names.     #
# Used by VisibilityState and SimulationState.                       #

_ADDITIONAL_CURRENT_FIELD_MAP: dict[str, str] = {
    "Ih": "show_ih_current",
    "IKa": "show_ika_current",
    "IKv31": "show_ikv31_current",
    "INaP": "show_inap_current",
    "INaR": "show_inar_current",
    "IM": "show_im_current",
    "IKir": "show_ikir_current",
    "IKCa": "show_ikca_current",
    "ICaL": "show_ical_current",
    "ICaT": "show_icat_current",
    "ICaN": "show_ican_current",
}

_ADDITIONAL_GATING_FIELD_MAP: dict[str, str] = {
    "r": "show_ih_gating",
    "a": "show_ika_gating",
    "b": "show_ika_gating",
    "nk": "show_ikv31_gating",
    "p": "show_inap_gating",
    "s": "show_inar_gating",
    "hr": "show_inar_gating",
    "w": "show_im_gating",
    "kir": "show_ikir_gating",
    "q": "show_ikca_gating",
    "d": "show_ical_gating",
    "f": "show_ical_gating",
    "dt": "show_icat_gating",
    "ft": "show_icat_gating",
    "dn": "show_ican_gating",
    "fn": "show_ican_gating",
}

# ------------------------------------------------------------------ #
# Shared JS snippets                                                 #
# ------------------------------------------------------------------ #

# Shared JS snippet for targeting the Plotly graph element.
# Used by VisibilityState and SimulationState.
_PLOTLY_GD_JS = "var gd=document.querySelector('.js-plotly-plot');"

# Scroll the log panel viewport to the top so the newest entry (displayed
# first in newest-first order) is always visible after a refresh.
# Used by LogState and SimulationState.
_LOG_SCROLL_JS = (
    "var vp=document.querySelector("
    "'#log-scroll-area [data-radix-scroll-area-viewport]');"
    "if(vp)vp.scrollTop=0;"
)

# ------------------------------------------------------------------ #
# Event handler factories                                            #
# ------------------------------------------------------------------ #


def _make_float_setter(field_name: str, class_name: str = "SimulationState"):
    """Factory returning a float-coercing event handler for ``field_name``.

    Args:
        field_name: Name of the state attribute to update.
        class_name: Owning state class name used in ``__qualname__``.

    Returns:
        An event handler method that accepts ``str | list[float] | float``
        and delegates to ``_set_float``.
    """

    def setter(self, value: "str | list[float] | float") -> None:
        """Set the field from an input or slider event."""
        self._set_float(field_name, value)

    setter.__name__ = f"set_{field_name}"
    setter.__qualname__ = f"{class_name}.set_{field_name}"
    setter.__doc__ = f"Set {field_name} from an input or slider event."
    return setter
