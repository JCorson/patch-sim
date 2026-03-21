"""Toggleable log viewer panel for the patch_sim web UI."""

import reflex as rx

from patch_sim_ui.log_handler import UILogRecord
from patch_sim_ui.state import AppState

_LEVEL_COLORS: dict[str, str] = {
    "DEBUG": "gray",
    "INFO": "inherit",
    "WARNING": "orange",
    "ERROR": "red",
    "CRITICAL": "red",
}

_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


def _log_entry(entry: UILogRecord) -> rx.Component:
    """Render a single log entry row.

    Each row shows the timestamp, level badge, and message, colour-coded
    by severity.

    Args:
        entry: The :class:`~patch_sim_ui.log_handler.UILogRecord` to display.

    Returns:
        An hstack row component.
    """
    return rx.hstack(
        rx.text(
            entry.timestamp,
            size="1",
            color="gray",
            white_space="nowrap",
            font_family="monospace",
        ),
        rx.text(
            "[",
            entry.level,
            "]",
            size="1",
            weight="bold",
            color=rx.match(
                entry.level,
                ("DEBUG", _LEVEL_COLORS["DEBUG"]),
                ("INFO", _LEVEL_COLORS["INFO"]),
                ("WARNING", _LEVEL_COLORS["WARNING"]),
                ("ERROR", _LEVEL_COLORS["ERROR"]),
                ("CRITICAL", _LEVEL_COLORS["CRITICAL"]),
                "inherit",
            ),
            white_space="nowrap",
            font_family="monospace",
        ),
        rx.text(
            entry.message,
            size="1",
            font_family="monospace",
            overflow="hidden",
            text_overflow="ellipsis",
            white_space="nowrap",
            flex="1",
        ),
        spacing="2",
        align="center",
        width="100%",
    )


def log_panel() -> rx.Component:
    """Conditionally rendered log viewer panel.

    Renders an empty fragment when ``AppState.log_panel_open`` is False.
    When open, displays a 200 px tall panel with a header bar (level
    filter, Refresh, Clear, Close) and a scrollable list of log entries.

    Returns:
        A conditional component that shows or hides the panel.
    """
    return rx.cond(
        AppState.log_panel_open,
        rx.box(
            # Header bar
            rx.hstack(
                rx.text("Logs", size="2", weight="bold"),
                rx.select(
                    _LOG_LEVELS,
                    value=AppState.log_level_filter,
                    on_change=AppState.set_log_level_filter,
                    size="1",
                    width="90px",
                ),
                rx.spacer(),
                rx.button(
                    rx.icon("refresh-cw", size=12),
                    "Refresh",
                    on_click=AppState.refresh_logs,
                    size="1",
                    variant="ghost",
                ),
                rx.button(
                    rx.icon("trash-2", size=12),
                    "Clear",
                    on_click=AppState.clear_logs,
                    size="1",
                    variant="ghost",
                    color_scheme="red",
                ),
                rx.button(
                    rx.icon("x", size=12),
                    on_click=AppState.toggle_log_panel,
                    size="1",
                    variant="ghost",
                ),
                spacing="2",
                align="center",
                width="100%",
                padding_x="3",
                padding_y="1",
                border_bottom="1px solid var(--gray-4)",
            ),
            # Log entries
            rx.scroll_area(
                rx.vstack(
                    rx.foreach(AppState.filtered_log_entries, _log_entry),
                    spacing="0",
                    width="100%",
                    align="start",
                ),
                id="log-scroll-area",
                height="160px",
                padding_x="3",
                padding_y="1",
            ),
            border_top="1px solid var(--gray-4)",
            width="100%",
            height="200px",
            background="var(--gray-1)",
            font_family="monospace",
        ),
        rx.fragment(),
    )
