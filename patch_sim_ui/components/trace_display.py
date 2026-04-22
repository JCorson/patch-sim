"""Plotly trace display component: stacked subplots with sweep overlay."""

import reflex as rx

from patch_sim_ui.state import SimulationState


def trace_display() -> rx.Component:
    """Main plot area: stacked Plotly subplots with sweep overlays."""
    return rx.box(
        rx.cond(
            SimulationState.has_result,
            # Plain div — Plotly is managed entirely by the side-channel JS
            # (_build_fetch_figure_js in state/simulation.py).  Using a plain
            # div prevents react-plotly.js from calling Plotly.react on every
            # React re-render and wiping our injected traces.
            rx.box(
                id="ps-trace-plot",
                width="100%",
                height="100%",
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
