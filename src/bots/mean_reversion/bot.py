"""
Mean Reversion Bot.

Detects sudden price movements caused by large orders
and trades the opposite direction, expecting price to revert.
"""

import asyncio
from collections import deque
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
from .trade_logger import TradeLogger


@dataclass
class Position:
    """An open position from a mean reversion trade."""

    condition_id: str
    token_id: str
    spike: PriceSpike

    # Entry info
    entries: list[dict] = field(default_factory=list)  # [{price, size_usd, tokens, timestamp}]
    total_size_usd: float = 0.0  # Total USD invested
    total_tokens: float = 0.0  # Total tokens bought
    avg_entry_price: float = 0.0

    # Targets
    target_price: float = 0.0
    stop_loss_price: float = 0.0
    timeout_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(minutes=30))

    # Minimum hold time (realistic execution)
    min_exit_time: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(seconds=10))

    # Status
    entries_made: int = 0
    max_entries: int = 3
    closed: bool = False
    close_reason: str = ""
    pnl: float = 0.0

    def add_entry(self, price: float, size_usd: float) -> None:
        """Record an entry."""
        # Calculate tokens bought: size_usd / price
        tokens = size_usd / price if price > 0 else 0

        self.entries.append({
            "price": price,
            "size_usd": size_usd,
            "tokens": tokens,
            "timestamp": datetime.utcnow(),
        })
        self.entries_made += 1

        # Update totals
        self.total_size_usd += size_usd
        self.total_tokens += tokens

        # Update average entry price (weighted by tokens)
        if self.total_tokens > 0:
            self.avg_entry_price = self.total_size_usd / self.total_tokens

    def should_add_entry(self, current_price: float) -> bool:
        """Check if we should add another entry (scale in)."""
        if self.entries_made >= self.max_entries:
            return False
        if self.closed:
            return False

        # Add entry if price dropped further (better entry price for us)
        # We bought this token expecting it to rise, so lower = better
        return current_price < self.avg_entry_price * 0.98  # 2% cheaper

    def check_exit(self, current_price: float) -> tuple[bool, str]:
        """
        Check if position should be closed.

        We always buy the token that dropped (either directly or the complement),
        so we expect price to rise toward target.

        Returns:
            (should_close, reason)
        """
        if self.closed:
            return False, ""

        now = datetime.utcnow()

        # Check timeout (always applies)
        if now >= self.timeout_at:
            return True, "timeout"

        # Minimum hold time - don't exit before 60 seconds (realistic execution)
        # This prevents reacting to orderbook noise
        if now < self.min_exit_time:
            return False, ""

        # Check target - price recovered to target (goes UP)
        # We bought low, sell when price rises to target
        if current_price >= self.target_price:
            return True, "target_reached"

        # Check stop loss - price dropped further (goes DOWN more)
        # Exit if price falls below stop loss
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

    # Realistic execution settings
    SLIPPAGE = 0.01  # 1% slippage on entry/exit
    MIN_HOLD_SECONDS = 10  # Minimum 10 seconds before exit
    MARKET_COOLDOWN_SECONDS = 300  # 5 min cooldown per market after closing
    SPIKE_CONFIRMATION_SECONDS = 3  # Wait 3 seconds to confirm spike is real

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
        self.trade_logger: TradeLogger | None = None

        # State
        self._ws_task: asyncio.Task | None = None
        self._last_watchlist_refresh: datetime | None = None
        self._positions: dict[str, Position] = {}  # token_id -> Position (open positions)
        self._closed_positions: deque[Position] = deque(maxlen=50)  # Keep last 50 closed positions
        self._market_tokens: dict[str, tuple[str, str]] = {}  # condition_id -> (token_a, token_b)
        self._market_cooldowns: dict[str, datetime] = {}  # condition_id -> cooldown_until
        self._pending_spikes: dict[str, tuple[PriceSpike, datetime, float]] = {}  # token_id -> (spike, detect_time, initial_price)

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

        # Initialize trade logger for verification
        self.trade_logger = TradeLogger(log_dir="data/trade_logs")

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

        # Check pending spikes for confirmation (3 second delay)
        await self._check_pending_spikes()

        # Check positions for exit conditions
        await self._check_positions()

        # Log estado cada 30 segundos
        now = datetime.utcnow()
        if not hasattr(self, '_last_status_log'):
            self._last_status_log = now

        if (now - self._last_status_log).total_seconds() >= 30:
            active_positions = len([p for p in self._positions.values() if not p.closed])
            self.logger.info(
                "bot_status",
                spikes=self._spikes_detected,
                opened=self._positions_opened,
                closed=self._positions_closed,
                active=active_positions,
                total_pnl=f"${self._total_pnl:.2f}",
            )
            self._last_status_log = now

        await asyncio.sleep(0.1)

    async def _check_pending_spikes(self) -> None:
        """Check pending spikes and confirm if price is still at spike level after 3 seconds."""
        if not self.detector:
            return

        now = datetime.utcnow()
        to_remove = []
        to_confirm = []

        for token_id, (spike, detect_time, initial_price) in self._pending_spikes.items():
            elapsed = (now - detect_time).total_seconds()

            # Not ready yet
            if elapsed < self.SPIKE_CONFIRMATION_SECONDS:
                continue

            # Get current price
            tracker = self.detector._trackers.get(token_id)
            if not tracker or not tracker.last_price:
                to_remove.append(token_id)
                continue

            current_price = tracker.last_price
            baseline = tracker.baseline_price or spike.price_before

            # Check if spike is still valid (price still significantly different from baseline)
            # The price should still be at least halfway to the spike level
            if spike.direction == "down":
                # For down spike: price dropped, we want to buy expecting recovery
                # Confirm if price is still below baseline by at least half the original spike
                min_threshold = baseline * (1 - abs(spike.price_change) / 2)
                is_still_valid = current_price <= min_threshold
            else:
                # For up spike: price rose, we want to short expecting drop
                # Confirm if price is still above baseline by at least half the original spike
                min_threshold = baseline * (1 + abs(spike.price_change) / 2)
                is_still_valid = current_price >= min_threshold

            if is_still_valid:
                self.logger.info(
                    "spike_confirmed",
                    token_id=token_id[:20] + "...",
                    direction=spike.direction,
                    initial_price=f"{initial_price:.3f}",
                    current_price=f"{current_price:.3f}",
                    baseline=f"{baseline:.3f}",
                    elapsed_seconds=f"{elapsed:.1f}",
                )
                to_confirm.append((token_id, spike))
            else:
                self.logger.info(
                    "spike_rejected_price_reverted",
                    token_id=token_id[:20] + "...",
                    direction=spike.direction,
                    initial_price=f"{initial_price:.3f}",
                    current_price=f"{current_price:.3f}",
                    baseline=f"{baseline:.3f}",
                    elapsed_seconds=f"{elapsed:.1f}",
                )
                to_remove.append(token_id)

        # Remove rejected spikes
        for token_id in to_remove:
            del self._pending_spikes[token_id]

        # Confirm valid spikes and open positions
        for token_id, spike in to_confirm:
            del self._pending_spikes[token_id]
            await self._open_position(spike)

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

            # Log price for tokens with open positions (for verification)
            if self.trade_logger and token_id in self._positions:
                self.trade_logger.add_price(token_id, best_ask)

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

            # Note: position prices are checked in _check_positions using tracker.last_price

        if self.metrics:
            self.metrics.inc("orderbook_updates")

    def _on_spike_detected(self, spike: PriceSpike) -> None:
        """Handle detected price spike - add to pending for confirmation."""
        self._spikes_detected += 1

        # Skip if already pending or has position
        if spike.token_to_buy in self._pending_spikes:
            return
        if spike.token_to_buy in self._positions:
            return

        self.logger.info(
            "spike_detected_pending_confirmation",
            condition_id=spike.condition_id[:20] + "...",
            direction=spike.direction,
            change=f"{spike.price_change:.2%}",
            token_to_buy=spike.token_to_buy[:20] + "...",
            confirmation_seconds=self.SPIKE_CONFIRMATION_SECONDS,
        )

        if self.metrics:
            self.metrics.inc("spikes_detected")

        # Add to pending spikes - will be confirmed after SPIKE_CONFIRMATION_SECONDS
        self._pending_spikes[spike.token_to_buy] = (
            spike,
            datetime.utcnow(),
            spike.price_after,  # Initial price at detection
        )

    async def _open_position(self, spike: PriceSpike) -> None:
        """Open a new mean reversion position."""
        # Check if we already have a position for this token
        if spike.token_to_buy in self._positions:
            self.logger.debug("position_already_exists", token=spike.token_to_buy[:20])
            return

        # Check market cooldown (avoid rapid re-entry after closing)
        now = datetime.utcnow()
        cooldown_until = self._market_cooldowns.get(spike.condition_id)
        if cooldown_until and now < cooldown_until:
            self.logger.debug(
                "market_in_cooldown",
                condition_id=spike.condition_id[:20] + "...",
                seconds_left=int((cooldown_until - now).total_seconds()),
            )
            return

        # Create position
        now = datetime.utcnow()
        position = Position(
            condition_id=spike.condition_id,
            token_id=spike.token_to_buy,
            spike=spike,
            target_price=spike.target_price,
            stop_loss_price=spike.stop_loss_price,
            timeout_at=now + timedelta(minutes=self.POSITION_TIMEOUT_MINUTES),
            min_exit_time=now + timedelta(seconds=self.MIN_HOLD_SECONDS),
        )

        self._positions[spike.token_to_buy] = position
        self._positions_opened += 1

        # Start trade logging for verification
        if self.trade_logger and self.detector:
            tracker = self.detector._trackers.get(spike.token_to_buy)
            price_history = []
            if tracker:
                # Get recent price history from detector
                price_history = [(p.timestamp, p.price) for p in tracker.history]

            self.trade_logger.start_trade(
                token_id=spike.token_to_buy,
                condition_id=spike.condition_id,
                direction=spike.direction,
                entry_price=spike.price_after,
                baseline_price=spike.price_before,
                spike_change=spike.price_change,
                price_history=price_history,
                market_question="",  # Could fetch from gamma API
            )

        # Execute first entry
        await self._execute_entry(position)

    async def _execute_entry(self, position: Position) -> None:
        """Execute an entry for a position."""
        if not self.clob_client or not self.detector:
            return

        if position.entries_made >= position.max_entries:
            return

        # Calculate entry size (equal splits)
        entry_size = self.trade_size

        # Get CURRENT price from detector (not spike price which is stale)
        tracker = self.detector._trackers.get(position.token_id)
        if tracker and tracker.last_price:
            current_price = tracker.last_price
        else:
            # Fallback to spike price for first entry
            current_price = position.spike.price_after

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
            # Simulate entry WITH slippage (we pay more than the displayed price)
            entry_price_with_slippage = current_price * (1 + self.SLIPPAGE)
            position.add_entry(entry_price_with_slippage, entry_size)
            self.logger.info(
                "dry_run_entry_simulated",
                orderbook_price=f"{current_price:.3f}",
                entry_price_with_slippage=f"{entry_price_with_slippage:.3f}",
            )
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

            # Sanity check: reject prices that differ too much from entry
            # This protects against bad data during initialization
            if position.avg_entry_price > 0:
                price_diff_ratio = abs(current_price - position.avg_entry_price) / position.avg_entry_price
                if price_diff_ratio > 0.5:  # More than 50% difference is suspicious
                    self.logger.warning(
                        "rejecting_suspicious_price",
                        token_id=token_id,  # Full ID for debugging
                        avg_entry=f"{position.avg_entry_price:.3f}",
                        current_price=f"{current_price:.3f}",
                        diff_ratio=f"{price_diff_ratio:.2%}",
                    )
                    # Skip this price update - wait for more reasonable data
                    continue

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

        # Apply slippage on exit (we sell for less than displayed price)
        exit_price_with_slippage = current_price * (1 - self.SLIPPAGE)

        # Calculate PnL correctly with slippage:
        # PnL = (exit_price_with_slippage * tokens) - total_usd_invested
        exit_value = exit_price_with_slippage * position.total_tokens
        position.pnl = exit_value - position.total_size_usd

        self._total_pnl += position.pnl
        self._positions_closed += 1

        # Set market cooldown
        self._market_cooldowns[position.condition_id] = (
            datetime.utcnow() + timedelta(seconds=self.MARKET_COOLDOWN_SECONDS)
        )

        # Save trade log for verification
        csv_path = None
        if self.trade_logger:
            csv_path = self.trade_logger.close_trade(
                token_id=position.token_id,
                exit_price=exit_price_with_slippage,
                exit_reason=reason,
                pnl=position.pnl,
            )

        # Log detallado del cierre
        profit_emoji = "✅" if position.pnl > 0 else "❌" if position.pnl < 0 else "➖"
        self.logger.info(
            "position_closed",
            result=profit_emoji,
            condition_id=position.condition_id[:20] + "...",
            token_id=position.token_id[:20] + "...",
            reason=reason,
            entries=position.entries_made,
            avg_entry=f"{position.avg_entry_price:.3f}",
            orderbook_price=f"{current_price:.3f}",
            exit_price_with_slippage=f"{exit_price_with_slippage:.3f}",
            pnl=f"${position.pnl:.4f}",
            total_pnl=f"${self._total_pnl:.4f}",
            dry_run=self.is_dry_run,
            log_file=csv_path or "N/A",
        )

        if self.metrics:
            self.metrics.inc("positions_closed")
            self.metrics.inc(f"close_reason_{reason}")
            self.metrics.gauge("total_pnl", self._total_pnl)

        if not self.is_dry_run:
            try:
                # Sell position (sell all tokens)
                await self.clob_client.place_order(
                    token_id=position.token_id,
                    side=OrderSide.SELL,
                    price=current_price,
                    size=position.total_tokens,  # Sell tokens, not USD
                )
            except Exception as e:
                self.logger.error("close_execution_error", error=str(e))

        # Clear spike from detector
        if self.detector:
            self.detector.clear_spike(position.spike.token_id)

        # Move from active to closed positions
        if position.token_id in self._positions:
            # Add to closed positions history
            self._closed_positions.append(position)
            # Remove from active positions
            del self._positions[position.token_id]

    def _serialize_position(self, position: Position) -> dict[str, Any]:
        """Serialize a position for TUI display."""
        return {
            "condition_id": position.condition_id,
            "token_id": position.token_id,
            "closed": position.closed,
            "close_reason": position.close_reason,
            "pnl": position.pnl,
            "total_size_usd": position.total_size_usd,
            "total_tokens": position.total_tokens,
            "avg_entry_price": position.avg_entry_price,
            "target_price": position.target_price,
            "stop_loss_price": position.stop_loss_price,
            "entries_made": position.entries_made,
            "max_entries": position.max_entries,
            "timeout_at": position.timeout_at.isoformat(),
            "spike_direction": position.spike.direction,  # Already a string
            "spike_magnitude": position.spike.price_change,  # Use price_change, not magnitude
            "entries": position.entries,
        }

    async def health_check(self) -> dict[str, Any]:
        """Return bot health status."""
        detector_stats = self.detector.get_stats() if self.detector else {}
        ws_connected = self.ws_client.connected if self.ws_client else False

        # Serialize open positions (from _positions dict)
        open_positions = [self._serialize_position(p) for p in self._positions.values()]

        # Serialize closed positions (from _closed_positions deque, newest first)
        closed_positions = [
            self._serialize_position(p)
            for p in reversed(self._closed_positions)
        ]

        return {
            "name": self.name,
            "is_running": self.is_running,
            "dry_run": self.is_dry_run,
            "trade_size": self.trade_size,
            "ws_connected": ws_connected,
            "tokens_tracked": detector_stats.get("tokens_tracked", 0),
            "active_positions": len(open_positions),
            "spikes_detected": self._spikes_detected,
            "positions_opened": self._positions_opened,
            "positions_closed": self._positions_closed,
            "total_pnl": self._total_pnl,
            "open_positions": open_positions,
            "closed_positions": closed_positions,
        }
