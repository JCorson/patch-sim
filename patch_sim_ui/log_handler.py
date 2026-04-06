"""Logging infrastructure for the patch_sim web UI.

Provides a custom logging handler that buffers records in a deque so they
can be drained into Reflex session state without the handler needing a
direct reference to any state instance.
"""

import collections
import logging
import threading
from datetime import datetime, timezone

from pydantic import BaseModel

#: Maximum number of log records retained in the handler buffer and in state.
MAX_LOG_ENTRIES = 500


class UILogRecord(BaseModel):
    """Serializable representation of a single log record.

    Attributes:
        timestamp: ISO-8601 formatted UTC timestamp string.
        level: Log level name (e.g. ``"DEBUG"``, ``"INFO"``).
        logger_name: Name of the logger that emitted the record.
        message: Formatted log message.
    """

    timestamp: str
    level: str
    logger_name: str
    message: str


class StateLogHandler(logging.Handler):
    """Logging handler that buffers records in a thread-safe deque.

    Records are accumulated here and drained into Reflex state on demand
    via :meth:`drain`. Using a drain pattern means the handler needs no
    reference to any session-scoped Reflex state object.

    Note:
        ``_buffer`` and ``_lock`` are class-level attributes so that a single
        buffer is shared across all instances.  ``setup_logging`` ensures only
        one instance is ever registered, so this is safe.
    """

    _buffer: collections.deque[UILogRecord] = collections.deque(maxlen=MAX_LOG_ENTRIES)
    _lock: threading.Lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """Format the record and append it to the buffer.

        Args:
            record: The log record to buffer.
        """
        try:
            ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
                "%H:%M:%S.%f"
            )[:-3]
            log_entry = UILogRecord(
                timestamp=ts,
                level=record.levelname,
                logger_name=record.name,
                message=self.format(record),
            )
            with self._lock:
                self._buffer.append(log_entry)
        except Exception:  # noqa: BLE001
            self.handleError(record)

    @classmethod
    def drain(cls) -> list[UILogRecord]:
        """Return and clear all buffered log records.

        Returns:
            A list of buffered :class:`UILogRecord` instances in emission order.
            The internal buffer is empty after this call.
        """
        with cls._lock:
            records = list(cls._buffer)
            cls._buffer.clear()
        return records


def setup_logging() -> None:
    """Configure ``patch_sim_ui`` and ``patch_sim`` with :class:`StateLogHandler`.

    Creates both loggers at DEBUG level and attaches a single shared
    :class:`StateLogHandler` instance to each.  Safe to call multiple times —
    subsequent calls are no-ops if the handler is already registered.
    """
    handler = StateLogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))

    for name in ("patch_sim_ui", "patch_sim"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers if setup_logging is called more than once.
        if not any(isinstance(h, StateLogHandler) for h in logger.handlers):
            logger.addHandler(handler)
