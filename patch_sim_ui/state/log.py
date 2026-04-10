"""Log panel state for the patch_sim web UI."""

import logging

import reflex as rx

from patch_sim_ui.log_handler import MAX_LOG_ENTRIES, StateLogHandler, UILogRecord
from patch_sim_ui.state._common import _LOG_SCROLL_JS


class LogState(rx.State):
    """State for the log viewer panel."""

    log_panel_open: bool = False
    log_entries: list[UILogRecord] = []
    log_level_filter: str = "DEBUG"

    @rx.var
    def filtered_log_entries(self) -> list[UILogRecord]:
        """Log entries filtered to the selected minimum level, newest first.

        Returns:
            Entries whose numeric level is >= the selected filter level,
            in reverse chronological order so the most recent entry is
            always visible at the top of the panel without scrolling.
        """
        min_level = logging.getLevelName(self.log_level_filter)
        if not isinstance(min_level, int):
            min_level = logging.DEBUG
        return [
            e
            for e in reversed(self.log_entries)
            if logging.getLevelName(e.level) >= min_level
        ]

    def toggle_log_panel(self):
        """Toggle the log panel open/closed, refreshing logs on open."""
        self.log_panel_open = not self.log_panel_open
        if self.log_panel_open:
            self._refresh_logs()
            return rx.call_script(_LOG_SCROLL_JS)

    def refresh_logs(self):
        """Public event handler: drain buffered records into state."""
        self._refresh_logs()
        return rx.call_script(_LOG_SCROLL_JS)

    def _refresh_logs(self) -> None:
        """Drain buffered log records into state, capping at MAX_LOG_ENTRIES."""
        new_records = StateLogHandler.drain()
        combined = list(self.log_entries) + new_records
        self.log_entries = combined[-MAX_LOG_ENTRIES:]

    def clear_logs(self) -> None:
        """Clear all displayed log entries."""
        self.log_entries = []

    def set_log_level_filter(self, value: str) -> None:
        """Set the minimum display level for log entries.

        Args:
            value: Level name string (e.g. ``"DEBUG"``, ``"INFO"``).
        """
        self.log_level_filter = value
