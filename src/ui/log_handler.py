"""
Log handler for TUI integration with structlog.
"""

import logging
import sys
from typing import TYPE_CHECKING, Any

import structlog
from structlog.typing import EventDict, WrappedLogger

if TYPE_CHECKING:
    from src.ui.tui import TUIDisplay


# Events to display in TUI operations log
IMPORTANT_EVENTS = {
    # Mean Reversion Bot events
    "spike_detected",
    "executing_entry",
    "position_closed",
    "watchlist_refreshed",
    # Arbitrage Bot events
    "arbitrage_opportunity",
    "executing_arbitrage",
    "arbitrage_executed",
    "trade_blocked_by_risk",
    # Bot lifecycle
    "mean_reversion_bot_initialized",
    "arbitrage_bot_initialized",
    "mean_reversion_bot_shutdown",
    "arbitrage_bot_shutdown",
}


class TUILogHandler:
    """Structlog processor that sends logs to TUI."""

    def __init__(self, tui_display: "TUIDisplay | None" = None):
        """
        Initialize TUI log handler.

        Args:
            tui_display: Optional TUI display instance
        """
        self.tui_display = tui_display

    def __call__(
        self, logger: WrappedLogger, method_name: str, event_dict: EventDict
    ) -> EventDict:
        """
        Process log event and send to TUI if available.

        Args:
            logger: Logger instance
            method_name: Log method name (info, debug, etc.)
            event_dict: Event dictionary

        Returns:
            Unmodified event dictionary (pass-through)
        """
        if self.tui_display and self._should_display(event_dict):
            # Extract relevant fields
            level = event_dict.get("level", "info").upper()
            event = event_dict.get("event", "")

            # Filter out context we don't want to display
            context = {
                k: v
                for k, v in event_dict.items()
                if k not in {"event", "level", "timestamp", "logger"}
            }

            # Send to TUI
            try:
                self.tui_display.add_log_entry(level, event, **context)
            except Exception:
                # Don't let TUI errors break logging
                pass

        # Pass through to next processor
        return event_dict

    def _should_display(self, event_dict: EventDict) -> bool:
        """
        Determine if log event should be shown in TUI.

        Args:
            event_dict: Event dictionary

        Returns:
            True if event should be displayed
        """
        level = event_dict.get("level", "info").upper()
        event = event_dict.get("event", "")

        # Always show warnings and errors
        if level in {"WARNING", "ERROR", "CRITICAL"}:
            return True

        # Show specific important events
        if event in IMPORTANT_EVENTS:
            return True

        # Hide debug logs by default
        if level == "DEBUG":
            return False

        return False


class _NullWriter:
    """A file-like object that discards everything written to it."""
    def write(self, *args, **kwargs):
        pass
    def flush(self, *args, **kwargs):
        pass


def configure_tui_logging(
    tui_display: "TUIDisplay",
    level: str = "INFO",
) -> None:
    """
    Configure structlog to route logs to TUI.

    This replaces the standard console renderer with a TUI-aware setup.

    Args:
        tui_display: TUI display instance
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    log_level = getattr(logging, level.upper())

    # Create TUI handler
    tui_handler = TUILogHandler(tui_display)

    # Build processor chain - include a renderer at the end for PrintLogger compatibility
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        tui_handler,  # Intercept logs for TUI
        # Add a minimal renderer for PrintLogger compatibility (output will be discarded)
        structlog.dev.ConsoleRenderer(colors=False),
    ]

    # Configure structlog with a null writer (discards output)
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=_NullWriter()),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging to also use null writer
    logging.basicConfig(
        format="%(message)s",
        stream=_NullWriter(),
        level=log_level,
    )
