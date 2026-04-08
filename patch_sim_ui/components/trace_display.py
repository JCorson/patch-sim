"""Plotly trace display component: stacked subplots with sweep overlay."""

import reflex as rx

from patch_sim_ui.state import SimulationState

# Layout overrides applied client-side so they track the active colour mode
# without requiring a server round-trip.  These take precedence over any
# server-built layout values.  In dark mode the transparent backgrounds let
# the Radix surface colour show through; ``plotly_dark`` applies the matching
# trace/axis colour scheme.  In light mode no overrides are needed because the
# server already builds the figure with ``plotly_white`` and light legend
# colours.
_LAYOUT_LIGHT: dict = {}
_LAYOUT_DARK = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "legend": {"bgcolor": "rgba(40,40,40,0.9)"},
    "legend2": {"bgcolor": "rgba(40,40,40,0.9)"},
}


def trace_display() -> rx.Component:
    """Main plot area: stacked Plotly subplots with sweep overlays."""
    return rx.box(
        rx.cond(
            SimulationState.has_result,
            rx.plotly(
                data=SimulationState.figure_data,
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
