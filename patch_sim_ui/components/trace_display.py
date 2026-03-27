"""Plotly trace display component: stacked subplots with sweep overlay."""

import reflex as rx

from patch_sim_ui.state import AppState

# Legend and background layout overrides applied client-side so they track the
# active colour mode without requiring a server round-trip.  ``rx.plotly``
# merges these into the figure via ``mergician`` before passing to
# ``react-plotly.js``, so they take precedence over any server-built layout
# values.  The transparent backgrounds let the Radix surface colour show
# through in both modes.
_LAYOUT_LIGHT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "legend": {"bgcolor": "rgba(255,255,255,0.7)"},
    "legend2": {"bgcolor": "rgba(255,255,255,0.7)"},
}
_LAYOUT_DARK = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "legend": {"bgcolor": "rgba(40,40,40,0.9)"},
    "legend2": {"bgcolor": "rgba(40,40,40,0.9)"},
}


def trace_display() -> rx.Component:
    """Main plot area: stacked Plotly subplots with sweep overlays."""
    return rx.box(
        rx.cond(
            AppState.has_result,
            rx.plotly(
                data=AppState.figure_data,
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
