"""Plotly trace display component: stacked subplots with sweep overlay."""

import reflex as rx

from patch_sim_ui.constants import DARK_AXIS_STYLE
from patch_sim_ui.state import SimulationState

# Layout overrides applied client-side so they track the active colour mode
# without requiring a server round-trip.  These take precedence over any
# server-built layout values.
#
# Design rationale: the server builds figures with ``plotly_white`` which has
# subtle light-grey gridlines.  Switching to ``plotly_dark`` replaces those
# with bright white gridlines that are too prominent on a transparent dark
# surface.  Instead we keep ``plotly_white``'s gridlines and only override
# the things that don't work on a dark background (dark mode only): transparent
# backgrounds and white text/axis colours.  Light mode needs no overrides
# because ``plotly_white`` already supplies the correct opaque white surfaces.
#
# DARK_AXIS_STYLE is defined in patch_sim_ui.constants and also referenced by
# _build_fetch_figure_js in state/simulation.py to keep both sites in sync.
_LAYOUT_LIGHT: dict = {}
_LAYOUT_DARK = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#e8e8e8"},
    "legend": {"bgcolor": "rgba(40,40,40,0.9)", "font": {"color": "#e8e8e8"}},
    "legend2": {"bgcolor": "rgba(40,40,40,0.9)", "font": {"color": "#e8e8e8"}},
    **{
        k: DARK_AXIS_STYLE
        for k in ("xaxis", "yaxis", "xaxis2", "yaxis2", "xaxis3", "yaxis3")
    },
}


def trace_display() -> rx.Component:
    """Main plot area: stacked Plotly subplots with sweep overlays."""
    return rx.box(
        rx.cond(
            SimulationState.has_result,
            rx.plotly(
                data=SimulationState.figure_skeleton,
                layout=rx.color_mode_cond(
                    light=_LAYOUT_LIGHT,
                    dark=_LAYOUT_DARK,
                ),
                use_resize_handler=True,
                height="100%",
                width="100%",
            ),
            rx.center(
                rx.vstack(
                    rx.icon("chart-line", size=48, color="gray"),
                    rx.text(
                        "Run a simulation to see traces",
                        size="3",
                        color="gray",
                    ),
                    spacing="3",
                    align="center",
                ),
                height="400px",
            ),
        ),
        width="100%",
        height="100%",
    )
