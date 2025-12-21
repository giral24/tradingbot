"""
Terminal User Interface (TUI) module for trading bots.
"""

from src.ui.components import (
    create_connection_indicator,
    create_stat_row,
    format_pnl,
    format_percentage,
    format_uptime,
    truncate_address,
)
from src.ui.log_handler import configure_tui_logging
from src.ui.tui import TUIDisplay

__all__ = [
    "TUIDisplay",
    "configure_tui_logging",
    "create_connection_indicator",
    "create_stat_row",
    "format_pnl",
    "format_percentage",
    "format_uptime",
    "truncate_address",
]
