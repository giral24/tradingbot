"""
UI components and helper functions for the TUI.
"""

from typing import Any

from rich.text import Text


def format_pnl(pnl: float) -> Text:
    """
    Format PnL with color based on value.

    Args:
        pnl: PnL value to format

    Returns:
        Colored Text object
    """
    if pnl > 0:
        return Text(f"${pnl:.2f}", style="bold green")
    elif pnl < 0:
        return Text(f"${pnl:.2f}", style="bold red")
    else:
        return Text(f"${pnl:.2f}", style="white")


def format_uptime(seconds: float) -> str:
    """
    Format uptime seconds into human-readable string.

    Args:
        seconds: Uptime in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s", "23m 45s", "45s")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def truncate_address(address: str, max_length: int = 20) -> str:
    """
    Truncate long addresses for display.

    Args:
        address: Address to truncate
        max_length: Maximum length before truncation

    Returns:
        Truncated address (e.g., "0x1234...5678")
    """
    if len(address) <= max_length:
        return address

    # For hex addresses, show first 6 and last 4 characters
    if address.startswith("0x"):
        return f"{address[:8]}...{address[-4:]}"

    # For other strings, show first max_length-3 chars
    return f"{address[:max_length-3]}..."


def create_connection_indicator(connected: bool) -> Text:
    """
    Create colored connection status indicator.

    Args:
        connected: Connection status

    Returns:
        Colored status indicator
    """
    if connected:
        return Text("✓ Connected", style="bold green")
    else:
        return Text("✗ Disconnected", style="bold red")


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format value as percentage.

    Args:
        value: Value to format (0.08 = 8%)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def create_stat_row(label: str, value: Any, value_style: str = "white") -> tuple[str, Text]:
    """
    Create a formatted stat row for tables.

    Args:
        label: Label for the stat
        value: Value to display
        value_style: Rich style for the value

    Returns:
        Tuple of (label, styled_value)
    """
    if isinstance(value, Text):
        return (label, value)
    else:
        return (label, Text(str(value), style=value_style))
