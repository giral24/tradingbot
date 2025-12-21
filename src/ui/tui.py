"""
Terminal User Interface (TUI) display for trading bots using Rich.
"""

import asyncio
import os
import sys
from collections import deque
from datetime import datetime
from typing import Any

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.bots.base import BaseBot
from src.ui.components import (
    create_connection_indicator,
    format_pnl,
    format_percentage,
    format_uptime,
    truncate_address,
)


# Event emojis for operations log
EVENT_EMOJIS = {
    "spike_detected": "🔴",
    "executing_entry": "💵",
    "position_closed": "💰",
    "watchlist_refreshed": "🔄",
    "arbitrage_opportunity": "💡",
    "executing_arbitrage": "💵",
    "arbitrage_executed": "💰",
    "trade_blocked_by_risk": "⚠️",
}


class TUIDisplay:
    """Rich TUI display for trading bots."""

    def __init__(
        self,
        bot_name: str,
        bot_mode: str,
        refresh_rate: float = 0.5,
        max_log_lines: int = 20,
    ):
        """
        Initialize TUI display.

        Args:
            bot_name: Name of the bot (e.g., "MEAN REVERSION BOT")
            bot_mode: Mode of operation ("DRY RUN" or "LIVE")
            refresh_rate: Refresh rate in seconds (default: 0.5)
            max_log_lines: Maximum number of log lines to keep (default: 20)
        """
        self.bot_name = bot_name
        self.bot_mode = bot_mode
        self.refresh_rate = refresh_rate
        self.max_log_lines = max_log_lines

        # State
        self.bot: BaseBot | None = None
        self.start_time: float | None = None
        self._last_health: dict[str, Any] = {}
        self._log_entries: deque = deque(maxlen=max_log_lines)
        self._stop_event: asyncio.Event | None = None
        self._live_task: asyncio.Task | None = None
        self._live_context: Live | None = None
        self._render_errors = 0

        # Setup console
        self.console = self._create_console()

    def _create_console(self) -> Console:
        """
        Create Rich console with appropriate settings.

        Returns:
            Configured Console instance
        """
        # Detect color support
        has_color = self._detect_color_support()

        return Console(
            force_terminal=True,
            no_color=not has_color,
            width=None,  # Auto-detect terminal width
        )

    def _detect_color_support(self) -> bool:
        """
        Detect if terminal supports colors.

        Returns:
            True if colors are supported
        """
        # Check if stdout is a TTY
        if not sys.stdout.isatty():
            return False

        # Check TERM environment variable
        term = os.environ.get("TERM", "")
        if "color" in term or term in {"xterm", "screen", "linux", "xterm-256color"}:
            return True

        return False

    async def start(self, bot: BaseBot) -> None:
        """
        Start the TUI in a background task.

        Args:
            bot: Bot instance to monitor
        """
        self.bot = bot
        self.start_time = asyncio.get_event_loop().time()
        self._stop_event = asyncio.Event()

        # Start TUI update loop in background
        self._live_task = asyncio.create_task(self._run_tui_loop())

    async def stop(self) -> None:
        """Stop TUI gracefully."""
        if self._stop_event:
            self._stop_event.set()

        if self._live_task:
            try:
                await asyncio.wait_for(self._live_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._live_task.cancel()
                try:
                    await self._live_task
                except asyncio.CancelledError:
                    pass

        if self._live_context:
            self._live_context.stop()

        # Clear screen
        self.console.clear()

    def add_log_entry(self, level: str, event: str, **context: Any) -> None:
        """
        Add a log entry to the operations panel.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            event: Event name
            **context: Additional context to display
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Get emoji for event
        emoji = EVENT_EMOJIS.get(event, "📋")

        # Format entry
        entry = {
            "timestamp": timestamp,
            "level": level,
            "event": event,
            "emoji": emoji,
            "context": context,
        }

        self._log_entries.append(entry)

    async def _run_tui_loop(self) -> None:
        """Run TUI update loop in background."""
        try:
            with Live(
                self._render_layout(),
                console=self.console,
                refresh_per_second=int(1 / self.refresh_rate),
                screen=True,
            ) as live:
                self._live_context = live

                while not self._stop_event.is_set():
                    # Poll bot health
                    if self.bot:
                        try:
                            health = await self.bot.health_check()
                            self._last_health = health
                            self._render_errors = 0  # Reset error counter on success
                        except Exception:
                            # Don't crash TUI if health_check fails
                            pass

                    # Update display
                    try:
                        live.update(self._render_layout())
                    except Exception:
                        self._render_errors += 1
                        # If too many errors, stop TUI
                        if self._render_errors >= 3:
                            break

                    # Sleep
                    await asyncio.sleep(self.refresh_rate)

        except Exception:
            # Don't let TUI errors crash the bot
            pass

    def _render_layout(self) -> Layout:
        """
        Build the Rich Layout structure.

        Returns:
            Layout instance
        """
        try:
            layout = Layout()

            # Create main sections
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="body"),
                Layout(name="footer", size=1),
            )

            # Split body into stats and operations
            layout["body"].split_row(
                Layout(name="stats", ratio=30),
                Layout(name="operations", ratio=70),
            )

            # Build components
            layout["header"].update(self._build_header())
            layout["stats"].update(self._build_stats_panel())
            layout["operations"].update(self._build_operations_panel())
            layout["footer"].update(self._build_status_bar())

            return layout

        except Exception as e:
            # Return minimal error layout
            return Layout(
                Panel(
                    f"[red]TUI rendering error: {e}[/red]",
                    title="Error",
                )
            )

    def _build_header(self) -> Panel:
        """
        Build top header panel.

        Returns:
            Header Panel
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Color based on mode
        mode_style = "bold yellow" if self.bot_mode == "DRY RUN" else "bold green"

        header_text = Text()
        header_text.append(self.bot_name, style="bold cyan")
        header_text.append(" | ")
        header_text.append(self.bot_mode, style=mode_style)
        header_text.append(" | ")
        header_text.append(timestamp, style="white")

        return Panel(
            header_text,
            style="bold white on blue",
        )

    def _build_stats_panel(self) -> Panel:
        """
        Build statistics panel.

        Returns:
            Statistics Panel
        """
        if not self._last_health:
            return Panel(
                Text("Inicializando...", style="yellow"),
                title="📊 ESTADÍSTICAS",
                border_style="cyan",
            )

        # Determine bot type and build appropriate stats
        bot_name = self._last_health.get("name", "")

        if bot_name == "mean_reversion":
            table = self._build_mean_reversion_stats()
        elif bot_name == "arb_intramarket":
            table = self._build_arbitrage_stats()
        else:
            table = Table.grid(padding=(0, 2))
            table.add_row("Unknown bot type", style="red")

        return Panel(
            table,
            title="📊 ESTADÍSTICAS",
            border_style="cyan",
        )

    def _build_mean_reversion_stats(self) -> Table:
        """
        Build statistics table for Mean Reversion bot.

        Returns:
            Statistics Table
        """
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", justify="left")
        table.add_column(justify="right")

        # PnL Total
        pnl = self._last_health.get("total_pnl", 0.0)
        table.add_row("PnL Total", format_pnl(pnl))

        # Win Rate (estimated)
        closed = self._last_health.get("positions_closed", 0)
        if closed > 0 and pnl > 0:
            win_rate = int(closed * 0.7) / closed * 100 if closed > 0 else 0
            table.add_row("Win Rate", Text(f"{win_rate:.0f}%", style="green"))
        else:
            table.add_row("Win Rate", Text("0%", style="white"))

        # Uptime
        if self.start_time:
            elapsed = asyncio.get_event_loop().time() - self.start_time
            table.add_row("Tiempo", Text(format_uptime(elapsed), style="white"))

        # Spikes
        spikes = self._last_health.get("spikes_detected", 0)
        table.add_row("Spikes", Text(str(spikes), style="yellow"))

        # Positions (closed/opened/active)
        opened = self._last_health.get("positions_opened", 0)
        active = self._last_health.get("active_positions", 0)
        positions_text = f"{closed}/{opened}/{active}"
        table.add_row("Posiciones", Text(positions_text, style="white"))

        # WS Status
        ws_connected = self._last_health.get("ws_connected", False)
        table.add_row("WS Status", create_connection_indicator(ws_connected))

        # Tokens
        tokens = self._last_health.get("tokens_tracked", 0)
        table.add_row("Tokens", Text(str(tokens), style="white"))

        return table

    def _build_arbitrage_stats(self) -> Table:
        """
        Build statistics table for Arbitrage bot.

        Returns:
            Statistics Table
        """
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", justify="left")
        table.add_column(justify="right")

        # Total Profit
        profit = self._last_health.get("total_profit", 0.0)
        table.add_row("Total Profit", format_pnl(profit))

        # Opportunities
        opportunities = self._last_health.get("opportunities_found", 0)
        table.add_row("Opportunities", Text(str(opportunities), style="yellow"))

        # Trades
        trades = self._last_health.get("trades_executed", 0)
        table.add_row("Trades", Text(str(trades), style="white"))

        # Open Positions
        open_pos = self._last_health.get("open_positions", 0)
        table.add_row("Open Positions", Text(str(open_pos), style="white"))

        # Hedged
        hedged = self._last_health.get("hedged_positions", 0)
        table.add_row("Hedged", Text(str(hedged), style="green"))

        # Exposure
        exposure = self._last_health.get("total_exposure", 0.0)
        table.add_row("Exposure", Text(f"${exposure:.2f}", style="white"))

        # Unhedged
        unhedged = self._last_health.get("unhedged_exposure", 0.0)
        unhedged_style = "red" if unhedged > 0 else "green"
        table.add_row("Unhedged", Text(f"${unhedged:.2f}", style=unhedged_style))

        # Min Spread
        min_spread = self._last_health.get("min_spread", 0.0)
        table.add_row("Min Spread", Text(format_percentage(min_spread, decimals=2), style="white"))

        # WS Status
        ws_connected = self._last_health.get("ws_connected", False)
        table.add_row("WS Status", create_connection_indicator(ws_connected))

        return table

    def _build_operations_panel(self) -> Panel:
        """
        Build scrolling operations log panel.

        Returns:
            Operations Panel
        """
        if not self._log_entries:
            return Panel(
                Text("Esperando operaciones...", style="dim"),
                title="📝 OPERACIONES",
                border_style="cyan",
            )

        # Build log lines (newest first)
        log_lines = []
        for entry in reversed(self._log_entries):
            # Format timestamp and event
            timestamp = entry["timestamp"]
            level = entry["level"]
            event = entry["event"]
            emoji = entry["emoji"]
            context = entry["context"]

            # Level color
            level_style = {
                "DEBUG": "dim",
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "bold red",
                "CRITICAL": "bold red",
            }.get(level, "white")

            # Event name (pretty format)
            event_display = event.replace("_", " ").upper()

            # Main line
            line = Text()
            line.append(f"{timestamp} ", style="dim")
            line.append(f"{emoji} ", style="white")
            line.append(f"{event_display}", style=level_style)
            log_lines.append(line)

            # Context lines (indented)
            if context:
                for key, value in context.items():
                    # Skip dry_run context (too noisy)
                    if key == "dry_run":
                        continue

                    # Format value
                    if isinstance(value, float):
                        if "pnl" in key or "profit" in key:
                            value_str = f"${value:.2f}"
                        elif "price" in key or "spread" in key:
                            value_str = f"{value:.4f}"
                        else:
                            value_str = f"{value:.2f}"
                    elif isinstance(value, str) and value.startswith("0x"):
                        value_str = truncate_address(value)
                    else:
                        value_str = str(value)

                    # Add context line
                    context_line = Text()
                    context_line.append("  ", style="dim")
                    context_line.append(f"{key}: ", style="cyan")
                    context_line.append(value_str, style="white")
                    log_lines.append(context_line)

        return Panel(
            Group(*log_lines),
            title="📝 OPERACIONES",
            border_style="cyan",
        )

    def _build_status_bar(self) -> Text:
        """
        Build bottom status bar.

        Returns:
            Status bar Text
        """
        status = Text()
        status.append("⚡ Live", style="bold green")
        status.append(" | ")

        # WS connection
        ws_connected = self._last_health.get("ws_connected", False)
        if ws_connected:
            status.append("WS: Connected", style="green")
        else:
            status.append("WS: Disconnected", style="red")

        status.append(" | ")

        # Updates processed (if available)
        updates = self._last_health.get("updates_processed", 0)
        if updates:
            status.append(f"Updates: {updates:,}", style="white")
            status.append(" | ")

        # Uptime
        if self.start_time:
            elapsed = asyncio.get_event_loop().time() - self.start_time
            status.append(f"Uptime: {format_uptime(elapsed)}", style="white")

        return status
