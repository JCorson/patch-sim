"""Tests for patch_sim_ui/log_handler.py.

Covers UILogRecord, StateLogHandler buffering/draining, the MAX_LOG_ENTRIES
cap, thread safety of drain, and the setup_logging idempotency guard.
"""

import logging
import threading

import pytest

from patch_sim_ui.log_handler import (
    MAX_LOG_ENTRIES,
    StateLogHandler,
    UILogRecord,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _emit(handler: StateLogHandler, message: str, level: int = logging.INFO) -> None:
    """Emit a synthetic log record through *handler*.

    Args:
        handler: The handler to emit through.
        message: Log message text.
        level: Numeric log level (default INFO).
    """
    record = logging.LogRecord(
        name="test",
        level=level,
        pathname="",
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    handler.emit(record)


@pytest.fixture(autouse=True)
def clear_buffer():
    """Drain the class-level buffer before and after each test.

    Yields:
        Nothing — used purely for setup/teardown.
    """
    StateLogHandler.drain()
    yield
    StateLogHandler.drain()


# ---------------------------------------------------------------------------
# UILogRecord
# ---------------------------------------------------------------------------


def test_ui_log_record_fields() -> None:
    """UILogRecord stores all four fields correctly."""
    rec = UILogRecord(
        timestamp="12:00:00.000",
        level="INFO",
        logger_name="patch_sim_ui",
        message="hello",
    )
    assert rec.timestamp == "12:00:00.000"
    assert rec.level == "INFO"
    assert rec.logger_name == "patch_sim_ui"
    assert rec.message == "hello"


# ---------------------------------------------------------------------------
# StateLogHandler — basic buffering
# ---------------------------------------------------------------------------


def test_emit_appends_to_buffer() -> None:
    """A single emitted record appears in the buffer after drain."""
    handler = StateLogHandler()
    _emit(handler, "test message")
    records = StateLogHandler.drain()
    assert len(records) == 1
    assert records[0].message == "test message"
    assert records[0].level == "INFO"


def test_drain_clears_buffer() -> None:
    """After drain() the buffer is empty."""
    handler = StateLogHandler()
    _emit(handler, "msg")
    StateLogHandler.drain()
    assert StateLogHandler.drain() == []


def test_drain_returns_records_in_emission_order() -> None:
    """drain() returns records oldest-first."""
    handler = StateLogHandler()
    for i in range(5):
        _emit(handler, f"msg{i}")
    records = StateLogHandler.drain()
    assert [r.message for r in records] == [f"msg{i}" for i in range(5)]


def test_level_name_stored_correctly() -> None:
    """The level field reflects the emitted record's level name."""
    handler = StateLogHandler()
    _emit(handler, "debug msg", level=logging.DEBUG)
    _emit(handler, "warning msg", level=logging.WARNING)
    records = StateLogHandler.drain()
    assert records[0].level == "DEBUG"
    assert records[1].level == "WARNING"


def test_timestamp_format() -> None:
    """Timestamps match HH:MM:SS.mmm format."""
    import re

    handler = StateLogHandler()
    _emit(handler, "ts test")
    records = StateLogHandler.drain()
    assert re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3}$", records[0].timestamp)


# ---------------------------------------------------------------------------
# MAX_LOG_ENTRIES cap
# ---------------------------------------------------------------------------


def test_buffer_respects_max_log_entries_cap() -> None:
    """Emitting more than MAX_LOG_ENTRIES records drops the oldest."""
    handler = StateLogHandler()
    for i in range(MAX_LOG_ENTRIES + 10):
        _emit(handler, f"msg{i}")
    records = StateLogHandler.drain()
    assert len(records) == MAX_LOG_ENTRIES
    # Oldest records were dropped; newest are retained.
    assert records[-1].message == f"msg{MAX_LOG_ENTRIES + 9}"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_drain_is_thread_safe() -> None:
    """Concurrent emits and a drain do not raise or lose records silently."""
    handler = StateLogHandler()
    errors: list[Exception] = []

    def emit_many() -> None:
        """Emit 50 records from a worker thread."""
        try:
            for i in range(50):
                _emit(handler, f"thread-msg-{i}")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=emit_many) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    records = StateLogHandler.drain()
    assert not errors
    # 4 threads × 50 messages = 200, all within MAX_LOG_ENTRIES.
    assert len(records) == 200


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


def test_setup_logging_attaches_handler() -> None:
    """setup_logging registers a StateLogHandler on the patch_sim_ui logger."""
    logger = logging.getLogger("patch_sim_ui")
    # Remove any existing handlers first.
    logger.handlers.clear()

    setup_logging()
    assert any(isinstance(h, StateLogHandler) for h in logger.handlers)


def test_setup_logging_attaches_handler_to_patch_sim():
    """setup_logging registers a StateLogHandler on the patch_sim logger."""
    logger = logging.getLogger("patch_sim")
    logger.handlers.clear()

    setup_logging()
    assert any(isinstance(h, StateLogHandler) for h in logger.handlers)


def test_setup_logging_is_idempotent():
    """Calling setup_logging twice does not add a second handler to either logger."""
    for name in ("patch_sim_ui", "patch_sim"):
        logging.getLogger(name).handlers.clear()

    setup_logging()
    setup_logging()
    for name in ("patch_sim_ui", "patch_sim"):
        handlers = logging.getLogger(name).handlers
        count = sum(1 for h in handlers if isinstance(h, StateLogHandler))
        assert count == 1


def test_patch_sim_child_logger_captured():
    """Log messages from patch_sim.* child loggers appear in the buffer."""
    for name in ("patch_sim_ui", "patch_sim"):
        logging.getLogger(name).handlers.clear()

    setup_logging()

    child_logger = logging.getLogger("patch_sim.hodgkin_huxley")
    child_logger.debug("test core library message")

    records = StateLogHandler.drain()
    assert len(records) == 1
    assert records[0].message == "test core library message"
    assert records[0].level == "DEBUG"
    assert records[0].logger_name == "patch_sim.hodgkin_huxley"
