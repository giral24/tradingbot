"""
Mean Reversion Bot.

Detects sudden price movements caused by large orders
and trades the opposite direction, expecting price to revert.
"""

import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from src.bots.base import BaseBot
from src.config import settings
from src.clob import ClobApiClient, GammaApiClient
from src.clob.models import OrderSide
from src.ws import WebSocketClient
from src.metrics import MetricsCollector

from .detector import PriceMovementDetector, PriceSpike


@dataclass
class Position:
    """An open position from a mean reversion trade."""

    condition_id: str
    token_id: str
    spike: PriceSpike

    # Entry info
    entries: list[dict] = field(default_factory=list)  # [{price, size, timestamp}]
    total_size: float = 0.0
    avg_entry_price: float = 0.0

    # Targets
    target_price: float = 0.0
    stop_loss_price: float = 0.0
    timeout_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(minutes=30))

    # Status
    entries_made: int = 0
    max_entries: int = 3
    closed: bool = False
    close_reason: str = ""
    pnl: float = 0.0

    def add_entry(self, price: float, size: float) -> None:
        """Record an entry."""
        self.entries.append({
            "price": price,
            "size": size,
            "timestamp": datetime.utcnow(),
        })
        self.entries_made += 1

        # Update totals
        old_total = self.total_size
        self.total_size += size

        # Update average entry price
        if self.total_size > 0:
            self.avg_entry_price = (
                (self.avg_entry_price * old_total + price * size) / self.total_size
            )

    def should_add_entry(self, current_price: float) -> bool:
        """Check if we should add another entry."""
        if self.entries_made >= self.max_entries:
            return False
        if self.closed:
            return False

        # Add entry if price moved further in our favor (cheaper for us)
        if self.spike.direction == "up":
            # We bought the OTHER token - its price should be lower
            # Add entry if price dropped more
            return current_price < self.avg_entry_price * 0.98  # 2% cheaper
        else:
            # We bought THIS token - add if it dropped more
            return current_price < self.avg_entry_price * 0.98

    def check_exit(self, current_price: float) -> tuple[bool, str]:
        """
        Check if position should be closed.

        Returns:
            (should_close, reason)
        """
        if self.closed:
            return False, ""

        # Check timeout
        if datetime.utcnow() >= self.timeout_at:
            return True, "timeout"

        # Check target (50% recovery)
        if self.spike.direction == "up":
            # Price went up, we bought other token
            # Exit when original token price comes back down
            if current_price <= self.target_price:
                return True, "target_reached"
        else:
            # Price went down, we bought this token
            # Exit when price recovers
            if current_price >= self.target_price:
                return True, "target_reached"

        # Check stop loss
        if self.spike.direction == "up":
            if current_price >= self.stop_loss_price:
                return True, "stop_loss"
        else:
            if current_price <= self.stop_loss_price:
                return True, "stop_loss"

        return False, ""


class MeanReversionBot(BaseBot):
    """
    Mean reversion trading bot.

    Strategy:
    - Monitor markets for sudden price spikes (>=8% in <2 min)
    - When detected, trade opposite direction (expect reversion)
    - Scaled entry: 3 buys as price continues moving
    - Exit at 50% recovery, 30 min timeout, or 5% stop-loss
    """

    # Configuration
    DEFAULT_TRADE_SIZE = 10.0  # $10 per entry (x3 = $30 max per trade)
    MIN_LIQUIDITY = 1000  # $1,000 minimum
    MAX_LIQUIDITY = 50000  # $50,000 maximum
    WATCHLIST_REFRESH_INTERVAL = 300  # 5 minutes
    MAX_WATCHLIST_SIZE = 200

    # Detector settings
    PRICE_CHANGE_THRESHOLD = 0.08  # 8% movement
    TIME_WINDOW_SECONDS = 120  # 2 minutes
    RECOVERY_TARGET = 0.50  # 50% recovery
    STOP_LOSS = 0.05  # 5% stop loss
    POSITION_TIMEOUT_MINUTES = 30

    def __init__(
        self,
        trade_size: float = DEFAULT_TRADE_SIZE,
        min_liquidity: float = MIN_LIQUIDITY,
        max_liquidity: float = MAX_LIQUIDITY,
        price_change_threshold: float = PRICE_CHANGE_THRESHOLD,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.trade_size = trade_size
        self.min_liquidity = min_liquidity
        self.max_liquidity = max_liquidity
        self.price_change_threshold = price_change_threshold

        # Clients
        self.clob_client: ClobApiClient | None = None
        self.gamma_client: GammaApiClient | None = None
        self.ws_client: WebSocketClient | None = None
        self.detector: PriceMovementDetector | None = None
        self.metrics: MetricsCollector | None = None

        # State
        self._ws_task: asyncio.Task | None = None
        self._last_watchlist_refresh: datetime | None = None
        self._positions: dict[str, Position] = {}  # token_id -> Position
        self._market_tokens: dict[str, tuple[str, str]] = {}  # condition_id -> (token_a, token_b)

        # Stats
        self._spikes_detected = 0
        self._positions_opened = 0
        self._positions_closed = 0
        self._total_pnl = 0.0

    @property
    def name(self) -> str:
        return "mean_reversion"

    @property
    def description(self) -> str:
        return "Mean reversion: trades against sudden price spikes"

    async def initialize(self) -> None:
        """Initialize bot resources."""
        self.logger.info("initializing_mean_reversion_bot")

        # Initialize clients
        self.clob_client = ClobApiClient(dry_run=self.is_dry_run)
        self.gamma_client = GammaApiClient()

        # Initialize detector
        self.detector = PriceMovementDetector(
            price_change_threshold=self.price_change_threshold,
            time_window_seconds=self.TIME_WINDOW_SECONDS,
            recovery_target=self.RECOVERY_TARGET,
            stop_loss=self.STOP_LOSS,
            on_spike=self._on_spike_detected,
        )

        # Initialize WebSocket
        self.ws_client = WebSocketClient(
            on_orderbook=self._on_orderbook_update,
        )

        # Initialize metrics
        self.metrics = MetricsCollector(export_interval=60)

        # Initial watchlist refresh
        await self._refresh_watchlist()

        self.logger.info(
            "mean_reversion_bot_initialized",
            trade_size=self.trade_size,
            min_liquidity=self.min_liquidity,
            max_liquidity=self.max_liquidity,
            price_change_threshold=f"{self.price_change_threshold:.0%}",
        )

    async def shutdown(self) -> None:
        """Clean up resources."""
        self.logger.info("shutting_down_mean_reversion_bot")

        if self.ws_client:
            await self.ws_client.disconnect()

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self.gamma_client:
            await self.gamma_client.close()

        self.logger.info(
            "mean_reversion_bot_shutdown",
            spikes_detected=self._spikes_detected,
            positions_opened=self._positions_opened,
            positions_closed=self._positions_closed,
            total_pnl=self._total_pnl,
        )

    async def run_slow_loop(self) -> None:
        """Refresh watchlist periodically."""
        now = datetime.utcnow()
        if self._last_watchlist_refresh:
            elapsed = (now - self._last_watchlist_refresh).total_seconds()
            if elapsed < self.WATCHLIST_REFRESH_INTERVAL:
                return

        await self._refresh_watchlist()

    async def _refresh_watchlist(self) -> None:
        """Fetch markets and update subscriptions."""
        if not self.gamma_client or not self.detector or not self.ws_client:
            return

        self.logger.info("refreshing_watchlist")

        try:
            # Fetch markets
            markets = await self.gamma_client.get_all_active_markets(
                max_markets=self.MAX_WATCHLIST_SIZE * 3,
            )

            # Filter by liquidity range and binary
            filtered = [
                m for m in markets
                if m.is_binary
                and m.accepting_orders
                and self.min_liquidity <= m.liquidity <= self.max_liquidity
            ]

            # Sort by liquidity (prefer middle of range)
            target_liquidity = (self.min_liquidity + self.max_liquidity) / 2
            filtered.sort(key=lambda m: abs(m.liquidity - target_liquidity))

            # Take top N
            selected = filtered[:self.MAX_WATCHLIST_SIZE]

            # Get current tokens
            old_tokens = set(self.detector.token_ids)

            # Register markets
            new_tokens = set()
            for m in selected:
                self.detector.register_market(
                    condition_id=m.condition_id,
                    token_a_id=m.token_a_id,
                    token_b_id=m.token_b_id,
                )
                self._market_tokens[m.condition_id] = (m.token_a_id, m.token_b_id)
                new_tokens.add(m.token_a_id)
                new_tokens.add(m.token_b_id)

            # Update WebSocket subscriptions
            to_unsub = old_tokens - new_tokens
            to_sub = new_tokens - old_tokens

            if to_unsub:
                await self.ws_client.unsubscribe(list(to_unsub))
            if to_sub:
                await self.ws_client.subscribe(list(to_sub))

            self._last_watchlist_refresh = datetime.utcnow()

            self.logger.info(
                "watchlist_refreshed",
                total_markets=len(selected),
                tokens=len(new_tokens),
                new_subs=len(to_sub),
            )

        except Exception as e:
            self.logger.error("watchlist_refresh_error", error=str(e))

    async def run_fast_loop(self) -> None:
        """Main loop - WebSocket driven."""
        # Start WebSocket if needed
        if self.ws_client and not self.ws_client.connected:
            if self._ws_task is None or self._ws_task.done():
                self._ws_task = asyncio.create_task(self.ws_client.run())

        # Check positions for exit conditions
        await self._check_positions()

        await asyncio.sleep(0.1)

    def _on_orderbook_update(self, update) -> None:
        """Handle orderbook update from WebSocket."""
        if not self.detector:
            return

        # Extract best ask price (that's what we'd pay to buy)
        token_id = update.asset_id  # asset_id is the token ID
        best_ask = update.best_ask

        if best_ask and best_ask > 0:
            # Update detector (may trigger spike detection)
            self.detector.update_price(token_id, best_ask)

            # Log for debugging (after update)
            is_tracked = token_id in self.detector._trackers
            if is_tracked:
                tracker = self.detector._trackers[token_id]
                self.logger.debug(
                    "price_update_received",
                    token_id=token_id[:20] + "...",
                    price=f"{best_ask:.3f}",
                    baseline=f"{tracker.baseline_price:.3f}" if tracker.baseline_price else "None",
                    history_len=len(tracker.history),
                )

            # Check if we have position to update
            if token_id in self._positions:
                self._positions[token_id].current_price = best_ask

        if self.metrics:
            self.metrics.inc("orderbook_updates")

    def _on_spike_detected(self, spike: PriceSpike) -> None:
        """Handle detected price spike."""
        self._spikes_detected += 1

        self.logger.info(
            "spike_detected",
            condition_id=spike.condition_id[:20] + "...",
            direction=spike.direction,
            change=f"{spike.price_change:.2%}",
            token_to_buy=spike.token_to_buy[:20] + "...",
        )

        if self.metrics:
            self.metrics.inc("spikes_detected")

        # Open position
        asyncio.create_task(self._open_position(spike))

    async def _open_position(self, spike: PriceSpike) -> None:
        """Open a new mean reversion position."""
        # Check if we already have a position for this token
        if spike.token_to_buy in self._positions:
            self.logger.debug("position_already_exists", token=spike.token_to_buy[:20])
            return

        # Create position
        position = Position(
            condition_id=spike.condition_id,
            token_id=spike.token_to_buy,
            spike=spike,
            target_price=spike.target_price,
            stop_loss_price=spike.stop_loss_price,
            timeout_at=datetime.utcnow() + timedelta(minutes=self.POSITION_TIMEOUT_MINUTES),
        )

        self._positions[spike.token_to_buy] = position
        self._positions_opened += 1

        # Execute first entry
        await self._execute_entry(position)

    async def _execute_entry(self, position: Position) -> None:
        """Execute an entry for a position."""
        if not self.clob_client:
            return

        if position.entries_made >= position.max_entries:
            return

        # Calculate entry size (equal splits)
        entry_size = self.trade_size

        # Get current price
        current_price = position.spike.price_after
        if position.entries:
            # Use last known price if available
            current_price = position.entries[-1]["price"]

        self.logger.info(
            "executing_entry",
            condition_id=position.condition_id[:20] + "...",
            token_id=position.token_id[:20] + "...",
            entry_number=position.entries_made + 1,
            size=entry_size,
            price=f"{current_price:.3f}",
            dry_run=self.is_dry_run,
        )

        if self.is_dry_run:
            # Simulate entry
            position.add_entry(current_price, entry_size)
            self.logger.info("dry_run_entry_simulated")
            return

        try:
            # Place buy order
            order = await self.clob_client.place_order(
                token_id=position.token_id,
                side=OrderSide.BUY,
                price=current_price,
                size=entry_size,
            )

            if order:
                position.add_entry(current_price, entry_size)

                if self.metrics:
                    self.metrics.inc("entries_executed")

        except Exception as e:
            self.logger.error("entry_execution_error", error=str(e))

    async def _check_positions(self) -> None:
        """Check all positions for exit conditions or additional entries."""
        if not self.detector:
            return

        positions_to_close = []

        for token_id, position in self._positions.items():
            if position.closed:
                continue

            # Get current price from detector
            tracker = self.detector._trackers.get(token_id)
            if not tracker or not tracker.last_price:
                continue

            current_price = tracker.last_price

            # Check for exit
            should_exit, reason = position.check_exit(current_price)

            if should_exit:
                positions_to_close.append((token_id, position, current_price, reason))
            elif position.should_add_entry(current_price):
                # Add scaled entry
                await self._execute_entry(position)

        # Close positions
        for token_id, position, price, reason in positions_to_close:
            await self._close_position(position, price, reason)

    async def _close_position(
        self,
        position: Position,
        current_price: float,
        reason: str,
    ) -> None:
        """Close a position."""
        if not self.clob_client:
            return

        position.closed = True
        position.close_reason = reason

        # Calculate PnL
        if position.spike.direction == "up":
            # We bought the other token expecting original to come down
            # PnL = (entry_price - current_price) * size (for the other token)
            position.pnl = (position.avg_entry_price - current_price) * position.total_size
        else:
            # We bought this token expecting it to go up
            position.pnl = (current_price - position.avg_entry_price) * position.total_size

        self._total_pnl += position.pnl
        self._positions_closed += 1

        self.logger.info(
            "position_closed",
            condition_id=position.condition_id[:20] + "...",
            token_id=position.token_id[:20] + "...",
            reason=reason,
            entries=position.entries_made,
            avg_entry=f"{position.avg_entry_price:.3f}",
            exit_price=f"{current_price:.3f}",
            pnl=f"${position.pnl:.4f}",
            total_pnl=f"${self._total_pnl:.4f}",
            dry_run=self.is_dry_run,
        )

        if self.metrics:
            self.metrics.inc("positions_closed")
            self.metrics.inc(f"close_reason_{reason}")
            self.metrics.gauge("total_pnl", self._total_pnl)

        if not self.is_dry_run:
            try:
                # Sell position
                await self.clob_client.place_order(
                    token_id=position.token_id,
                    side=OrderSide.SELL,
                    price=current_price,
                    size=position.total_size,
                )
            except Exception as e:
                self.logger.error("close_execution_error", error=str(e))

        # Clear spike from detector
        if self.detector:
            self.detector.clear_spike(position.spike.token_id)

        # Remove from active positions
        if position.token_id in self._positions:
            del self._positions[position.token_id]

    async def health_check(self) -> dict[str, Any]:
        """Return bot health status."""
        detector_stats = self.detector.get_stats() if self.detector else {}
        ws_connected = self.ws_client.connected if self.ws_client else False

        return {
            "name": self.name,
            "is_running": self.is_running,
            "dry_run": self.is_dry_run,
            "trade_size": self.trade_size,
            "ws_connected": ws_connected,
            "tokens_tracked": detector_stats.get("tokens_tracked", 0),
            "active_positions": len([p for p in self._positions.values() if not p.closed]),
            "spikes_detected": self._spikes_detected,
            "positions_opened": self._positions_opened,
            "positions_closed": self._positions_closed,
            "total_pnl": self._total_pnl,
        }
