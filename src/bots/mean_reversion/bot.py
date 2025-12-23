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
from src.clob.models import OrderSide, OrderType
from src.ws import WebSocketClient
from src.metrics import MetricsCollector

from .detector import PriceMovementDetector, PriceSpike
from .trade_logger import TradeLogger

# Sports keywords to identify live sports markets (mean reversion doesn't work on these)
# Timestamps from Polymarket API are in UTC (ISO 8601 with 'Z' suffix)
SPORTS_KEYWORDS = [
    # === ESPORTS ===
    "dota", "dota2", "lol", "league-of-legends", "csgo", "cs2", "valorant",
    "esports", "esport", "overwatch", "mobile-legends", "honor-of-kings",
    "rainbow-six", "call-of-duty", "rocket-league",

    # === AMERICAN SPORTS ===
    # Football
    "nfl", "cfb", "college-football", "super-bowl", "touchdown",
    # Basketball
    "nba", "wnba", "ncaa", "cbb", "cwbb", "euroleague", "basketball",
    # Baseball
    "mlb", "kbo", "baseball",
    # Hockey
    "nhl", "shl", "ahl", "khl", "del", "extraliga", "snl", "hockey",

    # === SOCCER/FOOTBALL ===
    "soccer", "football", "epl", "premier-league", "la-liga", "laliga",
    "bundesliga", "serie-a", "ligue-1", "mls", "ucl", "uel",
    "champions-league", "europa-league", "eredivisie", "liga-mx",
    "super-lig", "primeira-liga", "a-league", "fa-cup", "efl",
    "fifa", "wc-qualifiers", "world-cup",

    # === COMBAT SPORTS ===
    "ufc", "boxing", "mma", "fight", "bellator", "pfl",

    # === INDIVIDUAL SPORTS ===
    "tennis", "atp", "wta", "wimbledon", "us-open", "french-open", "australian-open",
    "golf", "pga", "lpga", "masters",
    "f1", "formula-1", "formula1", "nascar", "motogp", "racing",

    # === OTHER SPORTS ===
    "cricket", "t20", "odi", "test-match", "ipl",
    "chess",
    "sailing", "sailgp",
    "olympics",

    # === COMMON PATTERNS FOR LIVE EVENTS ===
    "-vs-", "-game", "-match", "-winner", "game-1", "game-2", "game-3",
    "game-4", "game-5", "game-6", "game-7", "round-", "set-",
]


@dataclass
class Position:
    """An open position from a mean reversion trade."""

    condition_id: str
    token_id: str
    spike: PriceSpike
    market_name: str = ""  # Human-readable market name

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
    - Monitor markets for sudden price DROPS (>=6% in <2 min)
    - When detected, BUY the dropped token (expect recovery)
    - Scaled entry: up to 3 buys as price continues dropping
    - Exit at 60% recovery, 10 min timeout, or 10% stop-loss
    - Uses FAK orders in real mode for fast execution
    """

    # Configuration
    DEFAULT_TRADE_SIZE = 10.0  # $10 per entry (x3 = $30 max per trade)
    MIN_LIQUIDITY = 1000  # $1,000 minimum
    MAX_LIQUIDITY = 100000  # $100,000 maximum
    WATCHLIST_REFRESH_INTERVAL = 300  # 5 minutes
    MAX_WATCHLIST_SIZE = 200

    # Detector settings
    PRICE_CHANGE_THRESHOLD = 0.06  # 6% movement
    TIME_WINDOW_SECONDS = 120  # 2 minutes
    RECOVERY_TARGET = 0.60  # 60% recovery
    STOP_LOSS = 0.10  # 10% stop loss
    POSITION_TIMEOUT_MINUTES = 10

    # Realistic execution settings
    SLIPPAGE = 0.01  # 1% slippage on entry/exit
    MIN_HOLD_SECONDS = 5  # Minimum 5 seconds before exit
    MARKET_COOLDOWN_SECONDS_LOSS = 300  # 5 min cooldown after loss
    MARKET_COOLDOWN_SECONDS_WIN = 0  # No cooldown after win
    SPIKE_CONFIRMATION_SECONDS = 0  # No delay - buy immediately on spike detection

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
        self._markets: dict[str, str] = {}  # condition_id -> market_name (question)
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
            # Fetch markets ordered by 24h volume (most active first)
            markets = await self.gamma_client.get_all_active_markets(
                max_markets=self.MAX_WATCHLIST_SIZE * 3,
                order_by="volume24hr",  # Most active markets first
            )

            # Filter by liquidity range, binary, and exclude live sports
            filtered = [
                m for m in markets
                if m.is_binary
                and m.accepting_orders
                and self.min_liquidity <= m.liquidity <= self.max_liquidity
                and not self._is_live_sports_market(m)
            ]

            # Sort by liquidity (prefer more liquid markets)
            filtered.sort(key=lambda m: m.liquidity, reverse=True)

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
                self._markets[m.condition_id] = m.question  # Store market name
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

            # Log stats about selected markets
            avg_volume = sum(m.volume_24h for m in selected) / len(selected) if selected else 0
            avg_liquidity = sum(m.liquidity for m in selected) / len(selected) if selected else 0

            self.logger.info(
                "watchlist_refreshed",
                total_markets=len(selected),
                tokens=len(new_tokens),
                new_subs=len(to_sub),
                avg_volume_24h=f"${avg_volume:,.0f}",
                avg_liquidity=f"${avg_liquidity:,.0f}",
            )

        except Exception as e:
            self.logger.error("watchlist_refresh_error", error=str(e))

    def _is_live_sports_market(self, market) -> bool:
        """
        Check if a market is a LIVE sports event (already started).

        Mean reversion doesn't work on live sports because price changes
        reflect actual game events (goals, points), not temporary order imbalances.

        Returns:
            True if this is a LIVE sports market (should be excluded)
            False if it's not sports OR sports but hasn't started yet
        """
        slug = market.market_slug.lower()
        question = market.question.lower()

        # Check for sports keywords in slug or question
        is_sports = any(kw in slug or kw in question for kw in SPORTS_KEYWORDS)

        if not is_sports:
            return False

        # It's a sports market - check if it has started
        now = datetime.utcnow()

        # Priority 1: Use game_start_time if available (exact time)
        if market.game_start_time:
            try:
                # Parse ISO format: "2025-12-22T19:30:00Z"
                start_time_str = market.game_start_time.replace("Z", "+00:00")
                start_time = datetime.fromisoformat(start_time_str).replace(tzinfo=None)

                if now >= start_time:
                    self.logger.debug(
                        "excluding_live_sports_market",
                        slug=slug[:40],
                        game_start_time=market.game_start_time,
                        status="started",
                    )
                    return True
                else:
                    # Game hasn't started yet - allow trading
                    return False
            except (ValueError, TypeError) as e:
                self.logger.debug(
                    "sports_market_bad_game_time",
                    slug=slug[:40],
                    error=str(e),
                )
                # Fall through to start_date_iso check

        # Priority 2: Use start_date_iso (only date, no time)
        if market.start_date_iso:
            try:
                from datetime import date
                start_date = date.fromisoformat(market.start_date_iso)
                today = date.today()

                if today >= start_date:
                    # Today or past - could be live, exclude to be safe
                    self.logger.debug(
                        "excluding_sports_market_today_or_past",
                        slug=slug[:40],
                        start_date=market.start_date_iso,
                    )
                    return True
                else:
                    # Future date - allow trading
                    return False
            except (ValueError, TypeError) as e:
                self.logger.debug(
                    "sports_market_bad_date",
                    slug=slug[:40],
                    error=str(e),
                )

        # No date info available - exclude to be safe
        self.logger.debug(
            "excluding_sports_market_no_date",
            slug=slug[:40],
        )
        return True

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

            # Not ready yet (only if confirmation delay > 0)
            if self.SPIKE_CONFIRMATION_SECONDS > 0 and elapsed < self.SPIKE_CONFIRMATION_SECONDS:
                continue

            # If no confirmation delay, buy immediately without validation
            if self.SPIKE_CONFIRMATION_SECONDS == 0:
                to_confirm.append((token_id, spike))
                continue

            # Get current price for validation (only when delay > 0)
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

        # Extract prices from orderbook
        token_id = update.asset_id  # asset_id is the token ID
        best_ask = update.best_ask  # Price to BUY
        best_bid = update.best_bid  # Price to SELL

        if best_ask and best_ask > 0:
            # Update detector with both ask and bid (may trigger spike detection)
            self.detector.update_price(token_id, best_ask, bid=best_bid)

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
        """Handle detected price spike."""
        self._spikes_detected += 1

        # Skip if already has position for this token
        if spike.token_to_buy in self._positions:
            return

        # Log spike detection
        self.logger.info(
            "spike_detected",
            condition_id=spike.condition_id[:20] + "...",
            direction=spike.direction,
            change=f"{spike.price_change:.2%}",
        )

        if self.metrics:
            self.metrics.inc("spikes_detected")

        # No confirmation delay - buy immediately
        if self.SPIKE_CONFIRMATION_SECONDS == 0:
            asyncio.create_task(self._open_position(spike))
        else:
            # Add to pending spikes for confirmation after delay
            if spike.token_to_buy not in self._pending_spikes:
                self._pending_spikes[spike.token_to_buy] = (
                    spike,
                    datetime.utcnow(),
                    spike.price_after,
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
        market_name = self._markets.get(spike.condition_id, "Unknown Market")
        position = Position(
            condition_id=spike.condition_id,
            token_id=spike.token_to_buy,
            spike=spike,
            market_name=market_name,
            target_price=spike.target_price,
            stop_loss_price=spike.stop_loss_price,
            timeout_at=now + timedelta(minutes=self.POSITION_TIMEOUT_MINUTES),
            # In dry-run: wait MIN_HOLD_SECONDS to avoid orderbook noise
            # In real mode: no wait - market filters naturally via limit orders
            min_exit_time=now + timedelta(seconds=self.MIN_HOLD_SECONDS if self.is_dry_run else 0),
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

            # Recalculate target and stop_loss based on ACTUAL entry price
            self._recalculate_exit_prices(position)

            self.logger.info(
                "dry_run_entry_simulated",
                orderbook_price=f"{current_price:.3f}",
                entry_price_with_slippage=f"{entry_price_with_slippage:.3f}",
                new_target=f"{position.target_price:.3f}",
                new_stop_loss=f"{position.stop_loss_price:.3f}",
            )
            return

        try:
            # Use FAK (Fill-And-Kill) with price padding for better fill
            # Add 0.5% to buy price to increase fill probability
            padded_price = min(current_price * 1.005, 0.99)  # Cap at 0.99

            order = await self.clob_client.place_order(
                token_id=position.token_id,
                side=OrderSide.BUY,
                price=padded_price,
                size=entry_size,
                order_type=OrderType.FAK,  # Partial fills OK
            )

            if order and order.order_id:
                # Wait briefly for order to process
                await asyncio.sleep(0.5)

                # Verify actual fill
                filled_order = await self.clob_client.get_order(order.order_id)

                if filled_order and filled_order.filled_size > 0:
                    # Use actual filled data
                    actual_filled_usd = filled_order.filled_size * filled_order.price
                    position.add_entry(filled_order.price, actual_filled_usd)

                    # Recalculate target and stop_loss based on ACTUAL entry price
                    self._recalculate_exit_prices(position)

                    self.logger.info(
                        "real_entry_verified",
                        order_id=order.order_id,
                        requested_size=entry_size,
                        filled_size=filled_order.filled_size,
                        fill_price=f"{filled_order.price:.4f}",
                        fill_pct=f"{(filled_order.filled_size / (entry_size / padded_price)) * 100:.1f}%",
                    )

                    if self.metrics:
                        self.metrics.inc("entries_executed")
                else:
                    self.logger.warning(
                        "entry_order_no_fill",
                        order_id=order.order_id,
                        status=filled_order.status if filled_order else "unknown",
                    )
            else:
                self.logger.warning(
                    "entry_order_rejected",
                    token_id=position.token_id[:20] + "...",
                    price=f"{padded_price:.4f}",
                )

        except Exception as e:
            self.logger.error("entry_execution_error", error=str(e))

    def _recalculate_exit_prices(self, position: Position) -> None:
        """
        Recalculate target and stop_loss based on ACTUAL entry price.

        This fixes the bug where targets were calculated from spike detection price
        but entry happened at a different (often higher) price.

        Logic:
        - baseline = the "normal" price before the spike (spike.price_before)
        - We want price to recover toward baseline from our actual entry
        - target = entry + (baseline - entry) * recovery_target
        - stop_loss = entry * (1 - stop_loss_pct)
        """
        baseline = position.spike.price_before
        entry = position.avg_entry_price

        # Target: recover X% of the way from entry toward baseline
        # If entry < baseline (normal case): target > entry (we profit when price rises)
        # If entry > baseline (entered too high): target could be < entry (we'd lose)
        position.target_price = entry + (baseline - entry) * self.RECOVERY_TARGET

        # Stop loss: exit if price drops X% below our entry
        position.stop_loss_price = entry * (1 - self.STOP_LOSS)

        # Safety check: target must be above entry for profit
        # If we entered above baseline, the math gives us a losing target
        # In that case, set a minimum profit target of 2%
        if position.target_price <= entry:
            self.logger.warning(
                "target_below_entry_adjusting",
                entry=f"{entry:.3f}",
                baseline=f"{baseline:.3f}",
                old_target=f"{position.target_price:.3f}",
            )
            position.target_price = entry * 1.02  # Minimum 2% profit target

    async def _check_positions(self) -> None:
        """Check all positions for exit conditions or additional entries."""
        if not self.detector:
            return

        positions_to_close = []

        for token_id, position in self._positions.items():
            if position.closed:
                continue

            # Get current prices from detector
            tracker = self.detector._trackers.get(token_id)
            if not tracker or not tracker.last_price:
                continue

            current_ask = tracker.last_price  # best_ask (for buying/logging)
            current_bid = tracker.last_bid    # best_bid (for selling and target check)

            # Need bid price to check exit - skip if not available
            if not current_bid or current_bid <= 0:
                continue

            # Sanity check: reject prices that differ too much from entry
            # This protects against bad data during initialization
            # ONLY applies to dry-run - in real mode, market filters naturally
            if self.is_dry_run and position.avg_entry_price > 0:
                price_diff_ratio = abs(current_bid - position.avg_entry_price) / position.avg_entry_price
                if price_diff_ratio > 0.5:  # More than 50% difference is suspicious
                    self.logger.warning(
                        "rejecting_suspicious_price",
                        token_id=token_id,  # Full ID for debugging
                        avg_entry=f"{position.avg_entry_price:.3f}",
                        current_bid=f"{current_bid:.3f}",
                        diff_ratio=f"{price_diff_ratio:.2%}",
                    )
                    # Skip this price update - wait for more reasonable data
                    continue

            # Check for exit using BID price (what we actually get when selling)
            should_exit, reason = position.check_exit(current_bid)

            if should_exit:
                # Pass both ask and bid prices for realistic exit
                positions_to_close.append((token_id, position, current_ask, current_bid, reason))
            elif position.should_add_entry(current_ask):
                # Add scaled entry (use ask because we're BUYING)
                await self._execute_entry(position)

        # Close positions
        for token_id, position, ask_price, bid_price, reason in positions_to_close:
            await self._close_position(position, ask_price, bid_price, reason)

    async def _close_position(
        self,
        position: Position,
        ask_price: float,
        bid_price: float | None,
        reason: str,
    ) -> None:
        """
        Close a position.

        Args:
            position: The position to close
            ask_price: Current best_ask (used for real orders and logging)
            bid_price: Current best_bid (used for realistic dry-run exit price)
            reason: Why we're closing (target_reached, stop_loss, timeout)
        """
        if not self.clob_client:
            return

        position.closed = True
        position.close_reason = reason

        # For dry-run: use actual best_bid for realistic exit simulation
        # For real: we'll place a sell order at market
        if self.is_dry_run and bid_price is not None and bid_price > 0:
            # Use actual best_bid - this is what you'd really get when selling
            exit_price_with_slippage = bid_price * (1 - self.SLIPPAGE)
        else:
            # Fallback: estimate bid from ask with slippage
            exit_price_with_slippage = ask_price * (1 - self.SLIPPAGE)

        # Calculate PnL correctly with slippage:
        # PnL = (exit_price_with_slippage * tokens) - total_usd_invested
        exit_value = exit_price_with_slippage * position.total_tokens
        position.pnl = exit_value - position.total_size_usd

        self._total_pnl += position.pnl
        self._positions_closed += 1

        # Set market cooldown: no cooldown if profit, 5 min if loss
        if position.pnl >= 0:
            cooldown_seconds = self.MARKET_COOLDOWN_SECONDS_WIN
        else:
            cooldown_seconds = self.MARKET_COOLDOWN_SECONDS_LOSS

        if cooldown_seconds > 0:
            self._market_cooldowns[position.condition_id] = (
                datetime.utcnow() + timedelta(seconds=cooldown_seconds)
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
            best_ask=f"{ask_price:.3f}",
            best_bid=f"{bid_price:.3f}" if bid_price else "N/A",
            exit_price=f"{exit_price_with_slippage:.3f}",
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
                # Use FAK with price padding for faster fill
                # Subtract 0.5% from sell price to increase fill probability
                base_price = bid_price if bid_price and bid_price > 0 else ask_price
                padded_sell_price = max(base_price * 0.995, 0.01)  # Floor at 0.01

                order = await self.clob_client.place_order(
                    token_id=position.token_id,
                    side=OrderSide.SELL,
                    price=padded_sell_price,
                    size=position.total_tokens,
                    order_type=OrderType.FAK,  # Partial fills OK
                )

                if order and order.order_id:
                    # Wait briefly for order to process
                    await asyncio.sleep(0.5)

                    # Verify actual fill
                    filled_order = await self.clob_client.get_order(order.order_id)

                    if filled_order and filled_order.filled_size > 0:
                        self.logger.info(
                            "real_exit_verified",
                            order_id=order.order_id,
                            requested_tokens=position.total_tokens,
                            filled_tokens=filled_order.filled_size,
                            fill_price=f"{filled_order.price:.4f}",
                            fill_pct=f"{(filled_order.filled_size / position.total_tokens) * 100:.1f}%",
                        )

                        # If partial fill, log warning about remaining tokens
                        if filled_order.filled_size < position.total_tokens * 0.95:
                            remaining = position.total_tokens - filled_order.filled_size
                            self.logger.warning(
                                "partial_exit_fill",
                                remaining_tokens=remaining,
                                note="Some tokens may still be in wallet",
                            )
                    else:
                        self.logger.warning(
                            "exit_order_no_fill",
                            order_id=order.order_id,
                            status=filled_order.status if filled_order else "unknown",
                        )
                else:
                    self.logger.warning(
                        "exit_order_rejected",
                        token_id=position.token_id[:20] + "...",
                        price=f"{padded_sell_price:.4f}",
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
        # Get entry time from first entry
        entry_time = None
        if position.entries:
            first_entry = position.entries[0]
            if isinstance(first_entry.get("timestamp"), datetime):
                entry_time = first_entry["timestamp"].strftime("%H:%M:%S")

        return {
            "condition_id": position.condition_id,
            "token_id": position.token_id,
            "market_name": position.market_name,
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
            "entry_time": entry_time,  # First entry time for display
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

        # Calculate total capital invested (sum of open positions)
        total_invested = sum(p.total_size_usd for p in self._positions.values())

        return {
            "name": self.name,
            "is_running": self.is_running,
            "dry_run": self.is_dry_run,
            "trade_size": self.trade_size,
            "ws_connected": ws_connected,
            "tokens_tracked": detector_stats.get("tokens_tracked", 0),
            "tokens_warmed_up": detector_stats.get("tokens_warmed_up", 0),
            "tokens_active": detector_stats.get("tokens_active", 0),
            "active_positions": len(open_positions),
            "spikes_detected": self._spikes_detected,
            "positions_opened": self._positions_opened,
            "positions_closed": self._positions_closed,
            "total_pnl": self._total_pnl,
            "total_invested": total_invested,
            "open_positions": open_positions,
            "closed_positions": closed_positions,
        }
