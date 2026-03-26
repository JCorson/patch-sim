"""Plotly trace display component: stacked subplots with sweep overlay."""

import reflex as rx

from patch_sim_ui.state import AppState


def _plotly_chart(figure_var) -> rx.Component:
    """Return a full-size rx.plotly bound to a given figure var.

    Args:
        figure_var: The AppState computed var holding the Plotly figure.

    Returns:
        An ``rx.plotly`` component configured for responsive layout.
    """
    return rx.plotly(
        data=figure_var,
        use_resize_handler=True,
        height="100%",
        width="100%",
    )


def trace_display() -> rx.Component:
    """Main plot area: stacked Plotly subplots with sweep overlays.

    Uses ``rx.color_mode_cond`` to serve a light- or dark-themed figure so
    that the Plotly template matches the active Radix colour mode, including
    when the user selects the system preference.
    """
    return rx.box(
        rx.cond(
            AppState.has_result,
            rx.color_mode_cond(
                light=_plotly_chart(AppState.figure_data_light),
                dark=_plotly_chart(AppState.figure_data_dark),
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
