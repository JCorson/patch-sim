"""Main Reflex app entry point and page layout for the patch_sim web UI."""

import reflex as rx

from patch_sim_ui import presets
from patch_sim_ui.components.log_panel import log_panel
from patch_sim_ui.components.neuron_panel import neuron_panel
from patch_sim_ui.components.protocol_panel import protocol_panel
from patch_sim_ui.components.sweep_manager import sweep_manager
from patch_sim_ui.components.trace_display import trace_display
from patch_sim_ui.log_handler import setup_logging
from patch_sim_ui.state import AppState

setup_logging()


def _header() -> rx.Component:
    """Top navigation bar with preset selector, title, and Run/Continuous buttons."""
    return rx.hstack(
        rx.select(
            presets.PRESET_NAMES,
            placeholder="Load preset…",
            on_change=AppState.load_preset,
            width="200px",
            size="2",
        ),
        rx.button(
            rx.icon("rotate-ccw"),
            "Reset",
            on_click=AppState.reset_to_defaults,
            color_scheme="gray",
            variant="soft",
            size="2",
        ),
        rx.spacer(),
        rx.heading("Patch Clamp Simulator", size="4"),
        rx.spacer(),
        rx.hstack(
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
                    disabled=AppState.continuous_loop_running,
                ),
            ),
            rx.cond(
                AppState.continuous_active,
                rx.button(
                    rx.icon("square"),
                    "Stop",
                    on_click=AppState.toggle_continuous_mode,
                    color_scheme="red",
                    variant="soft",
                    size="2",
                ),
                rx.button(
                    rx.icon("repeat"),
                    "Continuous",
                    on_click=AppState.toggle_continuous_mode,
                    color_scheme="green",
                    variant="soft",
                    size="2",
                    disabled=~AppState.can_run_continuous,
                ),
            ),
            spacing="2",
            align="center",
        ),
        rx.menu.root(
            rx.menu.trigger(
                rx.icon_button(
                    rx.icon("sun-moon"),
                    variant="ghost",
                    size="2",
                ),
            ),
            rx.menu.content(
                rx.menu.item(
                    rx.icon("sun", size=14),
                    "Light",
                    on_click=AppState.set_color_mode_and_sync("light"),
                ),
                rx.menu.item(
                    rx.icon("moon", size=14),
                    "Dark",
                    on_click=AppState.set_color_mode_and_sync("dark"),
                ),
                rx.menu.item(
                    rx.icon("monitor", size=14),
                    "System",
                    on_click=AppState.set_color_mode_and_sync("system"),
                ),
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
    """Center content area with trace plot, log panel, and sweep controls."""
    return rx.vstack(
        _error_banner(),
        rx.box(
            trace_display(),
            padding="4",
            width="100%",
            flex="1",
            min_height="0",
            overflow="hidden",
        ),
        log_panel(),
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
        appearance="inherit",
        accent_color="blue",
        radius="medium",
    )
)
app.add_page(
    index,
    route="/",
    title="Patch Clamp Simulator",
    on_load=AppState.sync_initial_color_mode,
)
