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

    # Trailing stop loss
    highest_price_seen: float = 0.0  # Track highest bid price since entry
    trailing_stop_active: bool = False  # True when profit >= TRAILING_STOP_ACTIVATION
    trailing_stop_price: float = 0.0  # Dynamic stop loss that trails price

    def add_entry(self, price: float, size_usd: float) -> None:
        """Record an entry."""
        # Calculate tokens bought: size_usd / price
        # Round to 2 decimals (Polymarket requirement)
        tokens = round(size_usd / price, 2) if price > 0 else 0

        self.entries.append({
            "price": round(price, 2),
            "size_usd": round(size_usd, 2),
            "tokens": tokens,
            "timestamp": datetime.utcnow(),
        })
        self.entries_made += 1

        # Update totals (keep rounded)
        self.total_size_usd = round(self.total_size_usd + size_usd, 2)
        self.total_tokens = round(self.total_tokens + tokens, 2)

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
        return current_price < self.avg_entry_price * 0.94  # 6% cheaper (avoid triggering on normal volatility)

    def update_trailing_stop(
        self,
        current_price: float,
        activation_threshold: float = 0.03,
        trail_distance: float = 0.05,
    ) -> None:
        """
        Update trailing stop loss based on current price.

        Args:
            current_price: Current bid price
            activation_threshold: Profit % to activate trailing stop (e.g., 0.03 = 3%)
            trail_distance: Distance to trail below highest price (e.g., 0.05 = 5%)
        """
        if self.avg_entry_price <= 0:
            return

        # Calculate current profit percentage
        profit_pct = (current_price - self.avg_entry_price) / self.avg_entry_price

        # Activate trailing stop once we're in profit by activation_threshold
        if not self.trailing_stop_active and profit_pct >= activation_threshold:
            self.trailing_stop_active = True
            self.highest_price_seen = current_price
            self.trailing_stop_price = current_price * (1 - trail_distance)

        # Update highest price and trailing stop if active
        if self.trailing_stop_active and current_price > self.highest_price_seen:
            self.highest_price_seen = current_price
            self.trailing_stop_price = current_price * (1 - trail_distance)

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

        # Check trailing stop (if active, takes precedence over fixed stop loss)
        if self.trailing_stop_active and current_price <= self.trailing_stop_price:
            return True, "trailing_stop"

        # Check fixed stop loss - price dropped further (goes DOWN more)
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
    # Polymarket minimums: 5 tokens AND $1 USD - we always use the minimum
    MIN_TOKENS = 5
    MIN_USD = 1.0
    MAX_TOTAL_EXPOSURE = 10.0  # Maximum $10 invested at any time (across all positions)
    MIN_LIQUIDITY = 1000  # $1,000 minimum
    MAX_LIQUIDITY = 100000  # $100,000 maximum
    WATCHLIST_REFRESH_INTERVAL = 300  # 5 minutes
    MAX_WATCHLIST_SIZE = 200  # 200 markets = 400 tokens max

    # Detector settings
    PRICE_CHANGE_THRESHOLD = 0.08  # 8% movement (spikes más claros)
    TIME_WINDOW_SECONDS = 120  # 2 minutes
    RECOVERY_TARGET = 0.50  # 50% recovery (target más realista)
    STOP_LOSS = 0.07  # 7% stop loss (cortar pérdidas antes)
    POSITION_TIMEOUT_MINUTES = 7  # 7 min timeout (no esperar tanto)

    # Realistic execution settings
    SLIPPAGE = 0.01  # 1% slippage on entry/exit (realistic for liquid Polymarket markets)
    MIN_HOLD_SECONDS = 5  # Minimum 5 seconds before exit
    MARKET_COOLDOWN_SECONDS_LOSS = 300  # 5 min cooldown after loss
    MARKET_COOLDOWN_SECONDS_WIN = 0  # No cooldown after win
    SPIKE_CONFIRMATION_SECONDS = 1  # Dry run only: wait 1 second to confirm spike

    # Spike validation settings
    MIN_PRICE_FOR_TRADE = 0.05  # Don't trade tokens priced below $0.05 (illiquid)
    MAX_PRICE_FOR_TRADE = 0.95  # Don't trade tokens priced above $0.95 (illiquid)
    MIN_SPREAD_BID_ASK = 0.02  # Require at least 2% spread visibility
    MIN_BASELINE_HISTORY = 10  # Need at least 10 price points for valid baseline
    MAX_ENTRY_DEVIATION = 0.15  # 15% max deviation from spike price at entry (reject bad data)

    # Live trading settings
    ORDER_FILL_TIMEOUT = 3.0  # Max seconds to wait for order fill
    ORDER_FILL_POLL_INTERVAL = 0.3  # Poll every 300ms
    ORDER_RETRY_ATTEMPTS = 2  # Retry failed orders up to 2 times
    ORDER_RETRY_DELAY = 0.5  # Wait 500ms between retries
    BALANCE_CHECK_INTERVAL = 60  # Check balance every 60 seconds
    MIN_USDC_BALANCE = 1.0  # Minimum USDC to keep trading

    # Trailing stop loss settings
    TRAILING_STOP_ACTIVATION = 0.03  # Activate trailing stop after 3% profit
    TRAILING_STOP_DISTANCE = 0.05  # Trail 5% below highest price

    def __init__(
        self,
        max_exposure: float = MAX_TOTAL_EXPOSURE,
        min_liquidity: float = MIN_LIQUIDITY,
        max_liquidity: float = MAX_LIQUIDITY,
        price_change_threshold: float = PRICE_CHANGE_THRESHOLD,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_exposure = max_exposure  # Maximum total $ invested at any time
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
        self._pending_spikes: dict[str, tuple[PriceSpike, datetime, float]] = {}  # Dry run only: token_id -> (spike, detect_time, initial_price)

        # Stats
        self._spikes_detected = 0
        self._positions_opened = 0
        self._positions_closed = 0
        self._total_pnl = 0.0
        self._wins = 0  # Positions closed with profit
        self._losses = 0  # Positions closed with loss

        # Live trading: cached USDC balance
        self._cached_balance: float | None = None
        self._balance_last_checked: datetime | None = None

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
            min_order=f"{self.MIN_TOKENS} tokens / ${self.MIN_USD}",
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

            # Filter by liquidity range, binary, and exclude problematic market types
            filtered = [
                m for m in markets
                if m.is_binary
                and m.accepting_orders
                and not m.closed  # Exclude closed markets (API sometimes returns them)
                and not m.neg_risk  # Exclude negative risk markets (have special token relationships that cause data mixing)
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
            # IMPORTANT: Never unsubscribe tokens with open positions
            tokens_with_positions = set(self._positions.keys())
            to_unsub = old_tokens - new_tokens - tokens_with_positions
            to_sub = new_tokens - old_tokens

            # Log if we're protecting tokens with positions
            protected_tokens = (old_tokens - new_tokens) & tokens_with_positions
            if protected_tokens:
                self.logger.info(
                    "protecting_tokens_with_positions",
                    count=len(protected_tokens),
                    note="Keeping subscriptions until positions close",
                )

            if to_unsub:
                await self.ws_client.unsubscribe(list(to_unsub))
                # CRITICAL: Also remove old tokens from detector to prevent memory growth
                self.detector.unregister_tokens(to_unsub)
                # Clean up market data for unsubscribed tokens
                for token_id in to_unsub:
                    # Find and remove condition_id from _market_tokens
                    for cid, (ta, tb) in list(self._market_tokens.items()):
                        if ta == token_id or tb == token_id:
                            del self._market_tokens[cid]
                            if cid in self._markets:
                                del self._markets[cid]
                            break

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

        # Check pending spikes for confirmation (dry run only)
        if self.is_dry_run:
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
        """Check pending spikes and confirm if price is still at spike level (dry run only)."""
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

            # Get current price for validation
            tracker = self.detector._trackers.get(token_id)
            if not tracker or not tracker.last_price:
                to_remove.append(token_id)
                continue

            current_price = tracker.last_price
            current_bid = tracker.last_bid
            baseline = tracker.baseline_price or spike.price_before

            # === VALIDATION ===

            # 1. Must have received updates during confirmation period
            if tracker.last_update_time:
                time_since_update = (now - tracker.last_update_time).total_seconds()
                if time_since_update > 2.0:
                    self.logger.info(
                        "spike_rejected_stale_data",
                        token_id=token_id[:20] + "...",
                        seconds_since_update=f"{time_since_update:.1f}",
                    )
                    to_remove.append(token_id)
                    continue

            # 2. Must still have valid bid
            if not current_bid or current_bid <= 0:
                self.logger.info(
                    "spike_rejected_no_bid",
                    token_id=token_id[:20] + "...",
                )
                to_remove.append(token_id)
                continue

            # 3. Check bid-ask spread is reasonable (< 15%)
            spread = (current_price - current_bid) / current_price if current_price > 0 else 1.0
            if spread > 0.15:
                self.logger.info(
                    "spike_rejected_wide_spread",
                    token_id=token_id[:20] + "...",
                    ask=f"{current_price:.3f}",
                    bid=f"{current_bid:.3f}",
                    spread=f"{spread:.1%}",
                )
                to_remove.append(token_id)
                continue

            # 4. Check if spike is still valid (price still below threshold)
            min_threshold = baseline * (1 - abs(spike.price_change) / 2)
            is_still_valid = current_price <= min_threshold

            # 5. Price must still be in tradeable range
            if current_price < self.MIN_PRICE_FOR_TRADE or current_price > self.MAX_PRICE_FOR_TRADE:
                self.logger.info(
                    "spike_rejected_price_out_of_range",
                    token_id=token_id[:20] + "...",
                    current_price=f"{current_price:.3f}",
                )
                to_remove.append(token_id)
                continue

            if is_still_valid:
                self.logger.info(
                    "spike_confirmed",
                    token_id=token_id[:20] + "...",
                    direction=spike.direction,
                    initial_price=f"{initial_price:.3f}",
                    current_price=f"{current_price:.3f}",
                    baseline=f"{baseline:.3f}",
                )
                to_confirm.append((token_id, spike))
            else:
                self.logger.info(
                    "spike_rejected_price_reverted",
                    token_id=token_id[:20] + "...",
                    initial_price=f"{initial_price:.3f}",
                    current_price=f"{current_price:.3f}",
                    baseline=f"{baseline:.3f}",
                )
                to_remove.append(token_id)

        # Remove rejected spikes
        for token_id in to_remove:
            if token_id in self._pending_spikes:
                del self._pending_spikes[token_id]

        # Confirm valid spikes and open positions
        for token_id, spike in to_confirm:
            if token_id in self._pending_spikes:
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

        # CRITICAL: Only process tokens we explicitly registered
        # The WebSocket sends updates for related tokens we didn't subscribe to
        if token_id not in self.detector._trackers:
            return  # Ignore unregistered tokens completely


        if best_ask and best_ask > 0:
            # Update detector with ask price
            # Note: bid may be None for price_change events (no real orderbook)
            # The detector will only trigger spikes when bid is available
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

        # Live mode: buy immediately
        # Dry run: add to pending spikes for confirmation
        if not self.is_dry_run:
            asyncio.create_task(self._open_position(spike))
        else:
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

        # Check exposure limit (max $10 invested at any time)
        current_exposure = self._get_current_exposure()
        available_budget = self.max_exposure + self._total_pnl - current_exposure

        if available_budget <= 0:
            self.logger.warning(
                "max_exposure_reached",
                current_exposure=f"${current_exposure:.2f}",
                max_exposure=f"${self.max_exposure:.2f}",
                total_pnl=f"${self._total_pnl:.2f}",
                available=f"${available_budget:.2f}",
            )
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

        # LIVE TRADING: Check balance before opening position
        if not self.is_dry_run:
            has_balance = await self._check_balance()
            if not has_balance:
                self.logger.warning(
                    "insufficient_balance",
                    min_required=f"${self.MIN_USDC_BALANCE:.2f}",
                )
                return

        # Create position object (but don't add to active positions yet for live trading)
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

        # LIVE TRADING: Execute entry FIRST, only add position if fill succeeds
        if not self.is_dry_run:
            success = await self._execute_entry_live(position)
            if not success:
                # Entry failed - don't create position
                self.logger.warning(
                    "position_not_opened_entry_failed",
                    token_id=spike.token_to_buy[:20] + "...",
                )
                if self.detector:
                    self.detector.clear_spike(spike.token_id)
                return

        # Add position to active positions (after successful entry for live)
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

        # DRY RUN: Execute entry after adding position
        if self.is_dry_run:
            await self._execute_entry(position)

    async def _execute_entry(self, position: Position) -> None:
        """Execute an entry for a position."""
        if not self.clob_client or not self.detector:
            return

        if position.entries_made >= position.max_entries:
            return

        # Get CURRENT price from detector (not spike price which is stale)
        tracker = self.detector._trackers.get(position.token_id)
        if tracker and tracker.last_price:
            current_price = tracker.last_price
        else:
            # Fallback to spike price for first entry
            current_price = position.spike.price_after

        # Calculate tokens to buy - Polymarket minimums: 5 tokens AND $1 USD
        min_tokens_for_usd = int(self.MIN_USD / current_price) + 1 if current_price > 0 else self.MIN_TOKENS
        tokens_to_buy = max(self.MIN_TOKENS, min_tokens_for_usd)
        size_usd = tokens_to_buy * current_price

        # VALIDATION: For first entry in DRY RUN, check price hasn't deviated too much
        # In LIVE mode, skip this check - we want to enter fast before price moves more
        if self.is_dry_run and position.entries_made == 0:
            spike_price = position.spike.price_after
            deviation = abs(current_price - spike_price) / spike_price if spike_price > 0 else 1.0

            if deviation > self.MAX_ENTRY_DEVIATION:
                self.logger.warning(
                    "entry_cancelled_price_deviation",
                    token_id=position.token_id[:20] + "...",
                    spike_price=f"{spike_price:.3f}",
                    current_price=f"{current_price:.3f}",
                    deviation=f"{deviation:.1%}",
                    max_allowed=f"{self.MAX_ENTRY_DEVIATION:.0%}",
                )
                # Close position without trading
                position.closed = True
                position.close_reason = "price_deviation"
                if position.token_id in self._positions:
                    del self._positions[position.token_id]
                # Clear spike from detector
                self.detector.clear_spike(position.spike.token_id)
                return

        self.logger.info(
            "executing_entry",
            condition_id=position.condition_id[:20] + "...",
            token_id=position.token_id[:20] + "...",
            entry_number=position.entries_made + 1,
            tokens=f"{tokens_to_buy}",
            usd=f"${size_usd:.2f}",
            price=f"{current_price:.3f}",
            dry_run=self.is_dry_run,
        )

        if self.is_dry_run:
            # Simulate entry WITH slippage (we pay more than the displayed price)
            entry_price_with_slippage = current_price * (1 + self.SLIPPAGE)
            # add_entry expects (price, size_usd)
            position.add_entry(entry_price_with_slippage, size_usd)

            # Recalculate target and stop_loss based on ACTUAL entry price
            self._recalculate_exit_prices(position)

            self.logger.info(
                "dry_run_entry_simulated",
                tokens=f"{tokens_to_buy}",
                orderbook_price=f"{current_price:.3f}",
                entry_price_with_slippage=f"{entry_price_with_slippage:.3f}",
                entry_usd=f"${size_usd:.2f}",
                new_target=f"{position.target_price:.3f}",
                new_stop_loss=f"{position.stop_loss_price:.3f}",
            )
            return

        try:
            # Use GTC (Good Till Cancel) with manual timeout for better fill rates
            # Add 1% to buy price to increase fill probability (more aggressive)
            # Round to 2 decimals (Polymarket requirement)
            padded_price = round(min(current_price * 1.01, 0.99), 2)

            # Polymarket minimums: 5 tokens AND $1 USD - use whichever is higher
            min_tokens_for_usd = int(self.MIN_USD / padded_price) + 1 if padded_price > 0 else self.MIN_TOKENS
            tokens_to_buy = max(self.MIN_TOKENS, min_tokens_for_usd)

            order = await self.clob_client.place_order(
                token_id=position.token_id,
                side=OrderSide.BUY,
                price=padded_price,
                size=tokens_to_buy,
                order_type=OrderType.GTC,  # GTC - will cancel manually if not filled
            )

            if order and order.order_id:
                # GTC order placed - poll for fill with timeout
                # Wait up to 3 seconds, checking every 0.5s
                max_wait = 3.0
                check_interval = 0.5
                elapsed = 0.0
                filled_order = None

                while elapsed < max_wait:
                    await asyncio.sleep(check_interval)
                    elapsed += check_interval

                    filled_order = await self.clob_client.get_order(order.order_id)
                    if filled_order and filled_order.filled_size > 0:
                        break  # Order filled!

                    # Check if order is still open (not cancelled/expired)
                    if filled_order and filled_order.status not in ["LIVE", "OPEN", "live", "open"]:
                        break  # Order no longer active

                # If not filled after timeout, cancel the order
                if not filled_order or filled_order.filled_size == 0:
                    try:
                        await self.clob_client.cancel_order(order.order_id)
                        self.logger.info(
                            "gtc_order_cancelled_timeout",
                            order_id=order.order_id,
                            waited_seconds=elapsed,
                        )
                    except Exception as cancel_err:
                        self.logger.warning(
                            "gtc_order_cancel_failed",
                            order_id=order.order_id,
                            error=str(cancel_err),
                        )

                if filled_order and filled_order.filled_size > 0:
                    # Use actual filled data
                    # filled_size is in tokens, price is per token
                    actual_filled_usd = filled_order.filled_size * filled_order.price
                    position.add_entry(filled_order.price, actual_filled_usd)

                    # Recalculate target and stop_loss based on ACTUAL entry price
                    self._recalculate_exit_prices(position)

                    self.logger.info(
                        "real_entry_verified",
                        order_id=order.order_id,
                        requested_tokens=tokens_to_buy,
                        filled_tokens=filled_order.filled_size,
                        filled_usd=f"${actual_filled_usd:.2f}",
                        fill_price=f"{filled_order.price:.4f}",
                    )

                    if self.metrics:
                        self.metrics.inc("entries_executed")
                else:
                    # FIXED: No fill - close position immediately if this is the first entry
                    self.logger.warning(
                        "entry_order_no_fill",
                        order_id=order.order_id,
                        status=filled_order.status if filled_order else "unknown",
                        entry_number=position.entries_made + 1,
                    )

                    # If this is the first entry and no fill, close the position
                    if position.entries_made == 0:
                        self.logger.warning(
                            "closing_position_no_entry_fill",
                            token_id=position.token_id[:20] + "...",
                        )
                        position.closed = True
                        position.close_reason = "no_fill"
                        position.pnl = 0.0
                        if position.token_id in self._positions:
                            del self._positions[position.token_id]
                        if self.detector:
                            self.detector.clear_spike(position.spike.token_id)
            else:
                # Order rejected
                self.logger.warning(
                    "entry_order_rejected",
                    token_id=position.token_id[:20] + "...",
                    price=f"{padded_price:.4f}",
                )

                # If this is the first entry and rejected, close the position
                if position.entries_made == 0:
                    self.logger.warning(
                        "closing_position_order_rejected",
                        token_id=position.token_id[:20] + "...",
                    )
                    position.closed = True
                    position.close_reason = "order_rejected"
                    position.pnl = 0.0
                    if position.token_id in self._positions:
                        del self._positions[position.token_id]
                    if self.detector:
                        self.detector.clear_spike(position.spike.token_id)

        except Exception as e:
            self.logger.error("entry_execution_error", error=str(e))

            # If this is the first entry and error, close the position
            if position.entries_made == 0:
                self.logger.warning(
                    "closing_position_entry_error",
                    token_id=position.token_id[:20] + "...",
                )
                position.closed = True
                position.close_reason = "entry_error"
                position.pnl = 0.0
                if position.token_id in self._positions:
                    del self._positions[position.token_id]
                if self.detector:
                    self.detector.clear_spike(position.spike.token_id)

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

    async def _check_balance(self) -> bool:
        """
        Check if we have enough USDC balance to trade.
        Caches result for BALANCE_CHECK_INTERVAL seconds.

        Returns:
            True if balance >= MIN_USDC_BALANCE
        """
        if not self.clob_client:
            return False

        now = datetime.utcnow()

        # Use cached balance if recent
        if (
            self._cached_balance is not None
            and self._balance_last_checked is not None
            and (now - self._balance_last_checked).total_seconds() < self.BALANCE_CHECK_INTERVAL
        ):
            return self._cached_balance >= self.MIN_USDC_BALANCE

        try:
            balance = await self.clob_client.get_balance()
            self._cached_balance = balance
            self._balance_last_checked = now

            self.logger.debug(
                "balance_checked",
                balance=f"${balance:.2f}",
                min_required=f"${self.MIN_USDC_BALANCE:.2f}",
            )

            return balance >= self.MIN_USDC_BALANCE

        except Exception as e:
            self.logger.error("balance_check_error", error=str(e))
            # If we can't check balance, assume we have enough (fail open)
            return True

    async def _wait_for_fill(self, order_id: str) -> tuple[float, float] | None:
        """
        Poll for order fill with timeout.

        Args:
            order_id: The order ID to check

        Returns:
            (filled_size, fill_price) or None if no fill
        """
        if not self.clob_client:
            return None

        start_time = asyncio.get_event_loop().time()
        timeout = self.ORDER_FILL_TIMEOUT
        poll_interval = self.ORDER_FILL_POLL_INTERVAL

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                self.logger.warning(
                    "order_fill_timeout",
                    order_id=order_id,
                    timeout_seconds=timeout,
                )
                return None

            try:
                filled_order = await self.clob_client.get_order(order_id)

                if filled_order and filled_order.filled_size > 0:
                    return (filled_order.filled_size, filled_order.price)

                # Check if order was cancelled or rejected
                if filled_order and filled_order.status in ("CANCELLED", "REJECTED"):
                    self.logger.warning(
                        "order_cancelled_or_rejected",
                        order_id=order_id,
                        status=filled_order.status,
                    )
                    return None

            except Exception as e:
                self.logger.error("get_order_error", order_id=order_id, error=str(e))

            await asyncio.sleep(poll_interval)

    async def _execute_entry_live(self, position: Position) -> bool:
        """
        Execute entry for live trading with retry logic.
        Must be called BEFORE adding position to active positions.

        Args:
            position: The position to execute entry for

        Returns:
            True if entry was successful, False otherwise
        """
        if not self.clob_client or not self.detector:
            return False

        # Get current price
        tracker = self.detector._trackers.get(position.token_id)
        if tracker and tracker.last_price:
            current_price = tracker.last_price
        else:
            current_price = position.spike.price_after

        # Validate price deviation - reject if price moved too much from spike
        spike_price = position.spike.price_after
        deviation = abs(current_price - spike_price) / spike_price if spike_price > 0 else 1.0

        if deviation > self.MAX_ENTRY_DEVIATION:
            self.logger.warning(
                "entry_cancelled_price_deviation",
                token_id=position.token_id[:20] + "...",
                spike_price=f"{spike_price:.3f}",
                current_price=f"{current_price:.3f}",
                deviation=f"{deviation:.1%}",
                max_allowed=f"{self.MAX_ENTRY_DEVIATION:.0%}",
            )
            return False

        # Calculate tokens from USD
        # Round price to 2 decimals (Polymarket requirement)
        padded_price = round(min(current_price * 1.005, 0.99), 2)

        # Polymarket minimums: 5 tokens AND $1 USD - use whichever is higher
        min_tokens_for_usd = int(self.MIN_USD / padded_price) + 1 if padded_price > 0 else self.MIN_TOKENS
        tokens_to_buy = max(self.MIN_TOKENS, min_tokens_for_usd)

        self.logger.info(
            "executing_live_entry",
            token_id=position.token_id[:20] + "...",
            tokens=f"{tokens_to_buy}",
            usd=f"${tokens_to_buy * padded_price:.2f}",
            price=f"{padded_price:.4f}",
        )

        # Single attempt - no retry to avoid duplicate orders
        try:
            order = await self.clob_client.place_order(
                token_id=position.token_id,
                side=OrderSide.BUY,
                price=padded_price,
                size=tokens_to_buy,
                order_type=OrderType.GTC,  # GTC - will cancel manually if not filled
            )

            if not order or not order.order_id:
                self.logger.warning(
                    "entry_order_rejected",
                    token_id=position.token_id[:20] + "...",
                )
                return False

            # Wait for fill with polling
            fill_result = await self._wait_for_fill(order.order_id)

            if fill_result:
                filled_size, fill_price = fill_result
                actual_filled_usd = filled_size * fill_price
                position.add_entry(fill_price, actual_filled_usd)

                # Recalculate targets based on actual entry
                self._recalculate_exit_prices(position)

                self.logger.info(
                    "live_entry_success",
                    order_id=order.order_id,
                    filled_tokens=filled_size,
                    fill_price=f"{fill_price:.4f}",
                    filled_usd=f"${actual_filled_usd:.2f}",
                )

                if self.metrics:
                    self.metrics.inc("entries_executed")

                return True
            else:
                # No fill - cancel the order
                self.logger.warning(
                    "entry_no_fill_cancelling",
                    order_id=order.order_id,
                )
                await self.clob_client.cancel_order(order.order_id)
                return False

        except Exception as e:
            self.logger.error(
                "entry_execution_error",
                error=str(e),
            )
            return False

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

            # Update trailing stop loss (live trading only)
            if not self.is_dry_run:
                position.update_trailing_stop(
                    current_bid,
                    activation_threshold=self.TRAILING_STOP_ACTIVATION,
                    trail_distance=self.TRAILING_STOP_DISTANCE,
                )

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

        # For dry-run: determine exit price based on reason
        if self.is_dry_run:
            if reason == "stop_loss":
                # For stop loss, use the stop loss price (not current bid which might be garbage data)
                # In reality you'd have a stop order that triggers at this price
                exit_price_with_slippage = position.stop_loss_price * (1 - self.SLIPPAGE)
            elif reason == "target_reached":
                # For target, use the target price (conservative estimate)
                exit_price_with_slippage = position.target_price * (1 - self.SLIPPAGE)
            elif bid_price is not None and bid_price > 0:
                # For timeout or other reasons, use actual bid
                exit_price_with_slippage = bid_price * (1 - self.SLIPPAGE)
            else:
                # Fallback: estimate from ask
                exit_price_with_slippage = ask_price * (1 - self.SLIPPAGE)

            # Calculate PnL for dry run
            exit_value = exit_price_with_slippage * position.total_tokens
            position.pnl = exit_value - position.total_size_usd

            self._total_pnl += position.pnl
            self._positions_closed += 1

            # Track wins/losses
            if position.pnl > 0:
                self._wins += 1
            elif position.pnl < 0:
                self._losses += 1

            # Set market cooldown
            self._set_market_cooldown(position)

            # Save trade log for verification
            csv_path = self._save_trade_log(position, exit_price_with_slippage, reason)

            # Log detallado del cierre
            self._log_position_closed(position, ask_price, bid_price, exit_price_with_slippage, reason, csv_path)

            if self.metrics:
                self.metrics.inc("positions_closed")
                self.metrics.inc(f"close_reason_{reason}")
                self.metrics.gauge("total_pnl", self._total_pnl)

        else:
            # REAL TRADING: Execute sell order and calculate P&L from actual fill
            actual_exit_price = None
            actual_filled_tokens = 0.0

            try:
                # Check if we have tokens to sell
                if position.total_tokens <= 0:
                    self.logger.warning(
                        "no_tokens_to_sell",
                        token_id=position.token_id[:20] + "...",
                        total_tokens=position.total_tokens,
                    )
                    # Still close the position but with 0 P&L
                    position.pnl = 0.0
                    self._positions_closed += 1
                    self._set_market_cooldown(position)
                    self._log_position_closed(position, ask_price, bid_price, 0.0, reason, None)
                    return

                # Use GTC with price padding and timeout for better fill
                # Subtract 1% from sell price to increase fill probability
                base_price = bid_price if bid_price and bid_price > 0 else ask_price
                padded_sell_price = max(base_price * 0.99, 0.01)  # Floor at 0.01

                # py-clob-client expects size in TOKENS for SELL orders
                order = await self.clob_client.place_order(
                    token_id=position.token_id,
                    side=OrderSide.SELL,
                    price=padded_sell_price,
                    size=position.total_tokens,  # Token count to sell
                    order_type=OrderType.GTC,  # GTC - will cancel manually if not filled
                )

                if order and order.order_id:
                    # GTC order placed - poll for fill with timeout
                    # Wait up to 5 seconds for sell orders (more time to find liquidity)
                    max_wait = 5.0
                    check_interval = 0.5
                    elapsed = 0.0
                    filled_order = None

                    while elapsed < max_wait:
                        await asyncio.sleep(check_interval)
                        elapsed += check_interval

                        filled_order = await self.clob_client.get_order(order.order_id)
                        if filled_order and filled_order.filled_size > 0:
                            break  # Order filled!

                        # Check if order is still open
                        if filled_order and filled_order.status not in ["LIVE", "OPEN", "live", "open"]:
                            break  # Order no longer active

                    # If not filled after timeout, cancel the order
                    if not filled_order or filled_order.filled_size == 0:
                        try:
                            await self.clob_client.cancel_order(order.order_id)
                            self.logger.info(
                                "gtc_sell_order_cancelled_timeout",
                                order_id=order.order_id,
                                waited_seconds=elapsed,
                            )
                        except Exception as cancel_err:
                            self.logger.warning(
                                "gtc_sell_order_cancel_failed",
                                order_id=order.order_id,
                                error=str(cancel_err),
                            )

                    if filled_order and filled_order.filled_size > 0:
                        actual_filled_tokens = filled_order.filled_size
                        actual_exit_price = filled_order.price

                        self.logger.info(
                            "real_exit_verified",
                            order_id=order.order_id,
                            requested_tokens=position.total_tokens,
                            filled_tokens=actual_filled_tokens,
                            fill_price=f"{actual_exit_price:.4f}",
                            fill_pct=f"{(actual_filled_tokens / position.total_tokens) * 100:.1f}%",
                        )

                        # Handle partial fills
                        if actual_filled_tokens < position.total_tokens * 0.95:
                            remaining = position.total_tokens - actual_filled_tokens
                            self.logger.warning(
                                "partial_exit_fill",
                                remaining_tokens=remaining,
                                note="Attempting to sell remaining tokens...",
                            )
                            # Try to sell remaining tokens with more aggressive pricing
                            await self._sell_remaining_tokens(
                                position.token_id,
                                remaining,
                                base_price * 0.98,  # More aggressive price
                            )
                    else:
                        self.logger.warning(
                            "exit_order_no_fill",
                            order_id=order.order_id,
                            status=filled_order.status if filled_order else "unknown",
                        )
                        # Fallback: estimate exit price
                        actual_exit_price = base_price * 0.99
                        actual_filled_tokens = 0  # Assume no fill
                else:
                    self.logger.warning(
                        "exit_order_rejected",
                        token_id=position.token_id[:20] + "...",
                        price=f"{padded_sell_price:.4f}",
                    )
                    # Fallback: estimate exit price for P&L calculation
                    actual_exit_price = base_price * 0.99
                    actual_filled_tokens = 0

            except Exception as e:
                self.logger.error("close_execution_error", error=str(e))
                # Fallback: estimate exit price
                actual_exit_price = (bid_price or ask_price) * 0.99
                actual_filled_tokens = 0

            # FIXED: Calculate P&L from ACTUAL fill data
            if actual_exit_price and actual_filled_tokens > 0:
                # Real P&L based on actual fill
                exit_value = actual_exit_price * actual_filled_tokens
                # Adjust for partial fill: only count the USD we actually recovered
                position.pnl = exit_value - (position.total_size_usd * (actual_filled_tokens / position.total_tokens))
            else:
                # No fill - estimate loss (we still own the tokens but couldn't sell)
                estimated_exit = (bid_price or ask_price or 0) * 0.95
                exit_value = estimated_exit * position.total_tokens
                position.pnl = exit_value - position.total_size_usd
                self.logger.warning(
                    "pnl_estimated_no_fill",
                    estimated_pnl=f"${position.pnl:.4f}",
                    note="Could not sell - tokens may still be in wallet",
                )

            self._total_pnl += position.pnl
            self._positions_closed += 1

            # Track wins/losses
            if position.pnl > 0:
                self._wins += 1
            elif position.pnl < 0:
                self._losses += 1

            # Set market cooldown
            self._set_market_cooldown(position)

            # Save trade log
            csv_path = self._save_trade_log(position, actual_exit_price or 0, reason)

            # Log position closed with ACTUAL data
            self._log_position_closed(
                position, ask_price, bid_price,
                actual_exit_price or 0, reason, csv_path,
                actual_filled_tokens=actual_filled_tokens,
            )

            if self.metrics:
                self.metrics.inc("positions_closed")
                self.metrics.inc(f"close_reason_{reason}")
                self.metrics.gauge("total_pnl", self._total_pnl)

        # Clear spike from detector
        if self.detector:
            self.detector.clear_spike(position.spike.token_id)

        # Move from active to closed positions
        if position.token_id in self._positions:
            # Add to closed positions history
            self._closed_positions.append(position)
            # Remove from active positions
            del self._positions[position.token_id]

    def _get_current_exposure(self) -> float:
        """Get total USD currently invested in open positions."""
        return sum(p.total_size_usd for p in self._positions.values())

    def _set_market_cooldown(self, position: Position) -> None:
        """Set market cooldown based on P&L."""
        if position.pnl >= 0:
            cooldown_seconds = self.MARKET_COOLDOWN_SECONDS_WIN
        else:
            cooldown_seconds = self.MARKET_COOLDOWN_SECONDS_LOSS

        if cooldown_seconds > 0:
            self._market_cooldowns[position.condition_id] = (
                datetime.utcnow() + timedelta(seconds=cooldown_seconds)
            )

    def _save_trade_log(self, position: Position, exit_price: float, reason: str) -> str | None:
        """Save trade log for verification."""
        csv_path = None
        if self.trade_logger:
            csv_path = self.trade_logger.close_trade(
                token_id=position.token_id,
                exit_price=exit_price,
                exit_reason=reason,
                pnl=position.pnl,
            )
        return csv_path

    def _log_position_closed(
        self,
        position: Position,
        ask_price: float,
        bid_price: float | None,
        exit_price: float,
        reason: str,
        csv_path: str | None,
        actual_filled_tokens: float | None = None,
    ) -> None:
        """Log position closed details."""
        profit_emoji = "✅" if position.pnl > 0 else "❌" if position.pnl < 0 else "➖"

        log_data = {
            "result": profit_emoji,
            "condition_id": position.condition_id[:20] + "...",
            "token_id": position.token_id[:20] + "...",
            "reason": reason,
            "entries": position.entries_made,
            "avg_entry": f"{position.avg_entry_price:.3f}",
            "best_ask": f"{ask_price:.3f}",
            "best_bid": f"{bid_price:.3f}" if bid_price else "N/A",
            "exit_price": f"{exit_price:.3f}",
            "pnl": f"${position.pnl:.4f}",
            "total_pnl": f"${self._total_pnl:.4f}",
            "dry_run": self.is_dry_run,
            "log_file": csv_path or "N/A",
        }

        if actual_filled_tokens is not None:
            log_data["filled_tokens"] = f"{actual_filled_tokens:.2f}"
            log_data["requested_tokens"] = f"{position.total_tokens:.2f}"

        self.logger.info("position_closed", **log_data)

    async def _sell_remaining_tokens(
        self,
        token_id: str,
        remaining_tokens: float,
        aggressive_price: float,
    ) -> None:
        """Attempt to sell remaining tokens after partial fill."""
        if not self.clob_client or remaining_tokens <= 0:
            return

        try:
            # py-clob-client expects size in TOKENS for SELL orders
            order = await self.clob_client.place_order(
                token_id=token_id,
                side=OrderSide.SELL,
                price=max(aggressive_price, 0.01),
                size=remaining_tokens,  # Token count to sell
                order_type=OrderType.GTC,  # GTC - will cancel manually if not filled
            )

            if order and order.order_id:
                # GTC order placed - poll for fill with timeout (5 seconds)
                max_wait = 5.0
                check_interval = 0.5
                elapsed = 0.0
                filled = None

                while elapsed < max_wait:
                    await asyncio.sleep(check_interval)
                    elapsed += check_interval

                    filled = await self.clob_client.get_order(order.order_id)
                    if filled and filled.filled_size > 0:
                        break

                    if filled and filled.status not in ["LIVE", "OPEN", "live", "open"]:
                        break

                # Cancel if not filled
                if not filled or filled.filled_size == 0:
                    try:
                        await self.clob_client.cancel_order(order.order_id)
                    except Exception:
                        pass

                if filled and filled.filled_size > 0:
                    self.logger.info(
                        "remaining_tokens_sold",
                        filled=filled.filled_size,
                        price=f"{filled.price:.4f}",
                    )
                else:
                    self.logger.warning(
                        "remaining_tokens_unsold",
                        remaining=remaining_tokens,
                        note="Manual intervention may be needed",
                    )
        except Exception as e:
            self.logger.error("sell_remaining_error", error=str(e))

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

        # Calculate real win rate
        total_trades = self._wins + self._losses
        win_rate = (self._wins / total_trades * 100) if total_trades > 0 else 0.0

        return {
            "name": self.name,
            "is_running": self.is_running,
            "dry_run": self.is_dry_run,
            "min_tokens": self.MIN_TOKENS,
            "min_usd": self.MIN_USD,
            "ws_connected": ws_connected,
            "tokens_tracked": detector_stats.get("tokens_tracked", 0),
            "tokens_warmed_up": detector_stats.get("tokens_warmed_up", 0),
            "tokens_active": detector_stats.get("tokens_active", 0),
            "active_positions": len(open_positions),
            "spikes_detected": self._spikes_detected,
            "positions_opened": self._positions_opened,
            "positions_closed": self._positions_closed,
            "wins": self._wins,
            "losses": self._losses,
            "win_rate": win_rate,
            "total_pnl": self._total_pnl,
            "total_invested": total_invested,
            "open_positions": open_positions,
            "closed_positions": closed_positions,
        }
