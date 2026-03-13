"""Main Reflex app entry point and page layout for the ap_sim web UI."""

import reflex as rx

from ap_sim_ui import presets
from ap_sim_ui.components.neuron_panel import neuron_panel
from ap_sim_ui.components.protocol_panel import protocol_panel
from ap_sim_ui.components.sweep_manager import sweep_manager
from ap_sim_ui.components.trace_display import trace_display
from ap_sim_ui.state import AppState


def _header() -> rx.Component:
    """Top navigation bar with preset selector, title, and Run button."""
    return rx.hstack(
        rx.select(
            presets.PRESET_NAMES,
            placeholder="Load preset…",
            on_change=AppState.load_preset,
            width="200px",
            size="2",
        ),
        rx.spacer(),
        rx.heading("AP Simulator", size="4"),
        rx.spacer(),
        rx.cond(
            AppState.is_running,
            rx.hstack(
                rx.spinner(size="2"),
                rx.text("Running…", size="2", color="gray"),
                spacing="2",
                align="center",
            ),
            rx.button(
                rx.icon("play"),
                "Run",
                on_click=AppState.run_simulation,
                color_scheme="blue",
                size="2",
            ),
        ),
        width="100%",
        padding_x="4",
        padding_y="3",
        border_bottom="1px solid var(--gray-4)",
        align="center",
    )


def _error_banner() -> rx.Component:
    """Display an error message banner when simulation fails."""
    return rx.cond(
        AppState.error_message != "",
        rx.callout(
            AppState.error_message,
            icon="circle-x",
            color_scheme="red",
            variant="soft",
            width="100%",
            margin_x="4",
            margin_top="2",
        ),
        rx.fragment(),
    )


def _sidebar() -> rx.Component:
    """Left sidebar containing neuron and protocol parameter panels."""
    return rx.box(
        rx.scroll_area(
            rx.vstack(
                neuron_panel(),
                rx.separator(),
                protocol_panel(),
                spacing="0",
                width="100%",
            ),
            height="calc(100vh - 60px - 2 * var(--space-3))",
        ),
        width="280px",
        min_width="280px",
        border_right="1px solid var(--gray-4)",
        background="var(--gray-1)",
    )


def _main_content() -> rx.Component:
    """Center content area with trace plot and sweep controls."""
    return rx.vstack(
        _error_banner(),
        rx.box(
            trace_display(),
            padding="4",
            width="100%",
            flex="1",
        ),
        rx.box(
            sweep_manager(),
            border_top="1px solid var(--gray-4)",
            width="100%",
            padding_y="2",
        ),
        spacing="0",
        width="100%",
        height="calc(100vh - 60px - 2 * var(--space-3))",
    )


def index() -> rx.Component:
    """Root page component: header + sidebar + main content."""
    return rx.vstack(
        _header(),
        rx.hstack(
            _sidebar(),
            _main_content(),
            spacing="0",
            width="100%",
            align="start",
        ),
        spacing="0",
        width="100%",
        min_height="100vh",
        padding="3",
        overflow="hidden",
    )


app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="blue",
        radius="medium",
    )
)
app.add_page(index, route="/", title="AP Simulator")
