"""Shared constants and factory functions reused across multiple substate modules.

Only items imported by two or more substate modules live here.  Items used
exclusively within a single substate are defined in that module instead.
"""

from patch_sim_ui.channels import ADDITIONAL_CHANNELS

# ------------------------------------------------------------------ #
# Additional-channel visibility field maps                           #
# ------------------------------------------------------------------ #
# Maps from additional-channel sweep keys to show_* field names.     #
# Used by VisibilityState and SimulationState.                       #
# Derived from the channel registry in patch_sim_ui.channels.        #

_ADDITIONAL_CURRENT_FIELD_MAP: dict[str, str] = {
    ch.current_key: ch.current_visibility_field for ch in ADDITIONAL_CHANNELS
}

_ADDITIONAL_GATING_FIELD_MAP: dict[str, str] = {
    gv: ch.gating_visibility_field
    for ch in ADDITIONAL_CHANNELS
    for gv in ch.gating_vars
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


def _set_float(obj: object, field: str, value: "str | list[float] | float") -> None:
    """Coerce value to float and set the named attribute on obj.

    Accepts plain floats, strings (from ``rx.input.on_change``), and
    single-element lists (from ``rx.slider.on_change``).  Silently
    ignores values that cannot be parsed as float.

    Args:
        obj: The state instance on which to set the attribute.
        field: Name of the attribute to update.
        value: Raw value from an input or slider event.
    """
    v = value[0] if isinstance(value, list) else value
    try:
        setattr(obj, field, float(v))
    except (ValueError, TypeError):
        pass


def _make_float_setter(field_name: str, class_name: str):
    """Factory returning a float-coercing event handler for ``field_name``.

    Args:
        field_name: Name of the state attribute to update.
        class_name: Owning state class name used in ``__qualname__``.

    Returns:
        An event handler method that accepts ``str | list[float] | float``
        and delegates to :func:`_set_float`.
    """

    def setter(self, value: "str | list[float] | float") -> None:
        """Set the field from an input or slider event."""
        _set_float(self, field_name, value)

    setter.__name__ = f"set_{field_name}"
    setter.__qualname__ = f"{class_name}.set_{field_name}"
    setter.__doc__ = f"Set {field_name} from an input or slider event."
    return setter
