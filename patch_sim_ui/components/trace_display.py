"""Plotly trace display component: stacked subplots with sweep overlay."""

import reflex as rx

from patch_sim_ui.state import SimulationState


def trace_display() -> rx.Component:
    """Main plot area: stacked Plotly subplots with sweep overlays."""
    # Both the plot div and the placeholder are always in the DOM.
    # Toggling display (not DOM presence) avoids the first-run problem where
    # Plotly.react is called before the browser has computed the div's layout.
    # rx.cond would add a wrapper element that breaks height:100% propagation.
    return rx.box(
        rx.box(
            id="ps-trace-plot",
            width="100%",
            height="100%",
            display=rx.cond(SimulationState.has_result, "block", "none"),
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
            display=rx.cond(SimulationState.has_result, "none", "flex"),
            height="400px",
        ),
        width="100%",
        height="100%",
    )
