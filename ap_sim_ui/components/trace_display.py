"""Plotly trace display component: stacked subplots with sweep overlay."""

import reflex as rx

from ap_sim_ui.state import AppState


def trace_display() -> rx.Component:
    """Main plot area: stacked Plotly subplots with sweep overlays."""
    return rx.box(
        rx.cond(
            AppState.has_result,
            rx.plotly(
                data=AppState.figure_data,
                use_resize_handler=True,
                height="100%",
                width="100%",
                id="sim-plot",
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
