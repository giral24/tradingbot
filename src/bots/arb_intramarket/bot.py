"""
Intra-market Arbitrage Bot.

Detects arbitrage opportunities when ask_A + ask_B < 1.0
and executes both sides to lock in risk-free profit.
"""

import asyncio
from pathlib import Path
from typing import Any
from datetime import datetime

from src.bots.base import BaseBot
from src.runner.registry import BotRegistry
from src.config import settings
from src.clob import ClobApiClient, GammaApiClient
from src.clob.models import OrderSide
from src.watchlist import WatchlistManager
from src.ws import WebSocketClient, LocalOrderbookManager, ArbitrageOpportunity
from src.risk import RiskManager
from src.metrics import MetricsCollector


@BotRegistry.register_with_name("arb_intramarket")
class ArbIntramarketBot(BaseBot):
    """
    Intra-market arbitrage bot.

    Strategy:
    - Monitor orderbooks for ALL markets in watchlist via WebSocket
    - When ask_A + ask_B < 1.0, there's arbitrage opportunity
    - Buy both tokens to lock in guaranteed profit

    Example:
        ask_A = 0.48, ask_B = 0.51
        Total cost = 0.99
        Guaranteed payout = 1.00
        Profit = 0.01 (1%)
    """

    # Configuration
    DEFAULT_TRADE_SIZE = 1.0  # $1 per trade
    MIN_SPREAD = 0.001  # 0.1% minimum spread to trade
    WATCHLIST_REFRESH_INTERVAL = 300  # 5 minutes
    MAX_WATCHLIST_SIZE = 300  # Monitor up to 300 markets

    def __init__(
        self,
        trade_size: float = DEFAULT_TRADE_SIZE,
        min_spread: float = MIN_SPREAD,
        watchlist_path: Path | None = None,
        **kwargs,
    ):
        """
        Initialize arbitrage bot.

        Args:
            trade_size: Size per trade in dollars
            min_spread: Minimum spread (profit) to execute
            watchlist_path: Path to persist watchlist
        """
        super().__init__(**kwargs)

        self.trade_size = trade_size
        self.min_spread = min_spread
        self.watchlist_path = watchlist_path or Path("data/watchlist.json")

        # Clients (initialized in initialize())
        self.clob_client: ClobApiClient | None = None
        self.gamma_client: GammaApiClient | None = None
        self.watchlist: WatchlistManager | None = None
        self.ws_client: WebSocketClient | None = None
        self.orderbook_manager: LocalOrderbookManager | None = None
        self.risk_manager: RiskManager | None = None
        self.metrics: MetricsCollector | None = None

        # State
        self._ws_task: asyncio.Task | None = None
        self._last_watchlist_refresh: datetime | None = None
        self._opportunities_found = 0
        self._trades_executed = 0
        self._total_profit = 0.0

    @property
    def name(self) -> str:
        return "arb_intramarket"

    @property
    def description(self) -> str:
        return "Intra-market arbitrage: exploits ask_A + ask_B < 1.0"

    async def initialize(self) -> None:
        """Initialize bot resources."""
        self.logger.info("initializing_arb_bot")

        # Initialize clients
        self.clob_client = ClobApiClient(dry_run=self.is_dry_run)
        self.gamma_client = GammaApiClient()

        # Initialize watchlist manager
        self.watchlist = WatchlistManager(
            gamma_client=self.gamma_client,
            max_size=self.MAX_WATCHLIST_SIZE,
            entry_threshold=0.1,  # Low threshold - include many markets
            exit_threshold=0.05,
            watchlist_path=self.watchlist_path,
        )

        # Initialize orderbook manager with arbitrage callback
        self.orderbook_manager = LocalOrderbookManager(
            on_arbitrage=self._on_arbitrage,
            min_spread=self.min_spread,
            min_size=self.trade_size,
        )

        # Initialize WebSocket client
        self.ws_client = WebSocketClient(
            on_orderbook=self._on_orderbook_update,
        )

        # Initialize risk manager
        self.risk_manager = RiskManager()

        # Initialize metrics
        self.metrics = MetricsCollector(export_interval=60)

        # Initial watchlist refresh
        await self._refresh_watchlist()

        self.logger.info(
            "arb_bot_initialized",
            trade_size=self.trade_size,
            min_spread=self.min_spread,
            watchlist_size=len(self.watchlist.entries) if self.watchlist else 0,
        )

    async def shutdown(self) -> None:
        """Clean up resources."""
        self.logger.info("shutting_down_arb_bot")

        # Stop WebSocket
        if self.ws_client:
            await self.ws_client.disconnect()

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        # Close clients
        if self.gamma_client:
            await self.gamma_client.close()

        # Log final stats
        self.logger.info(
            "arb_bot_shutdown_complete",
            opportunities_found=self._opportunities_found,
            trades_executed=self._trades_executed,
            total_profit=self._total_profit,
        )

    async def run_slow_loop(self) -> None:
        """
        Update the watchlist periodically.

        - Fetch markets from Gamma API
        - Score and select top markets
        - Update WebSocket subscriptions
        """
        # Check if refresh needed
        now = datetime.utcnow()
        if self._last_watchlist_refresh:
            elapsed = (now - self._last_watchlist_refresh).total_seconds()
            if elapsed < self.WATCHLIST_REFRESH_INTERVAL:
                return

        await self._refresh_watchlist()

    async def _refresh_watchlist(self) -> None:
        """Refresh the watchlist and update subscriptions."""
        if not self.watchlist or not self.orderbook_manager or not self.ws_client:
            return

        self.logger.info("slow_loop_refreshing_watchlist")

        try:
            # Refresh watchlist
            entries = await self.watchlist.refresh()

            # Get current tokens
            old_tokens = set(self.orderbook_manager.token_ids)

            # Register new markets
            new_tokens = set()
            for entry in entries:
                self.orderbook_manager.register_market(
                    condition_id=entry.condition_id,
                    token_a_id=entry.token_a_id,
                    token_b_id=entry.token_b_id,
                )
                new_tokens.add(entry.token_a_id)
                new_tokens.add(entry.token_b_id)

            # Update subscriptions
            to_subscribe = new_tokens - old_tokens
            to_unsubscribe = old_tokens - new_tokens

            if to_unsubscribe:
                await self.ws_client.unsubscribe(list(to_unsubscribe))

            if to_subscribe:
                await self.ws_client.subscribe(list(to_subscribe))

            self._last_watchlist_refresh = datetime.utcnow()

            self.logger.info(
                "slow_loop_complete",
                watchlist_size=len(entries),
                tokens_subscribed=len(new_tokens),
                new_subscriptions=len(to_subscribe),
            )

        except Exception as e:
            self.logger.error("watchlist_refresh_error", error=str(e))

    async def run_fast_loop(self) -> None:
        """
        Main trading loop - WebSocket driven.

        The real work happens in callbacks:
        - _on_orderbook_update: Update local state
        - _on_arbitrage: Execute trades when opportunity found
        """
        # Start WebSocket if not running
        if self.ws_client and not self.ws_client.connected:
            if self._ws_task is None or self._ws_task.done():
                self._ws_task = asyncio.create_task(self.ws_client.run())
                self.logger.info("fast_loop_ws_started")

        # Let WebSocket run - callbacks handle the rest
        await asyncio.sleep(0.1)

    def _on_orderbook_update(self, update) -> None:
        """Handle orderbook update from WebSocket."""
        if self.orderbook_manager:
            # This triggers arbitrage check internally
            self.orderbook_manager.handle_update(update)

        # Record metric
        if self.metrics:
            self.metrics.inc("orderbook_updates")

    def _on_arbitrage(self, opportunity: ArbitrageOpportunity) -> None:
        """
        Handle detected arbitrage opportunity.

        This is called synchronously from the orderbook manager.
        Schedule the async trade execution.
        """
        self._opportunities_found += 1

        # Record metric
        if self.metrics:
            self.metrics.inc("arbitrage_opportunities")
            self.metrics.gauge("last_opportunity_spread", opportunity.spread)

        self.logger.info(
            "arbitrage_opportunity",
            condition_id=opportunity.condition_id[:20] + "...",
            ask_a=opportunity.ask_a,
            ask_b=opportunity.ask_b,
            spread=f"{opportunity.spread:.4f}",
            max_profit=f"${opportunity.max_profit:.4f}",
        )

        # Schedule trade execution
        asyncio.create_task(self._execute_arbitrage(opportunity))

    async def _execute_arbitrage(self, opp: ArbitrageOpportunity) -> None:
        """
        Execute arbitrage trade.

        Buy both tokens at their ask prices.
        """
        if not self.clob_client or not self.risk_manager:
            return

        # Get market name from watchlist
        market_name = ""
        if self.watchlist:
            entry = self.watchlist.get_entry(opp.condition_id)
            if entry:
                market_name = entry.question

        # Calculate sizes
        size = min(self.trade_size, opp.max_size)

        # Check risk limits
        can_trade, reason = self.risk_manager.can_trade(opp.condition_id, size)
        if not can_trade:
            self.logger.warning(
                "trade_blocked_by_risk",
                condition_id=opp.condition_id[:20] + "...",
                reason=reason,
            )
            return

        self.logger.info(
            "executing_arbitrage",
            condition_id=opp.condition_id[:20] + "...",
            size=size,
            cost=opp.cost_per_pair * size,
            expected_profit=opp.profit_per_pair * size,
            dry_run=self.is_dry_run,
        )

        if self.is_dry_run:
            self.logger.info("dry_run_trade_skipped")
            # Record fills for risk tracking even in dry run
            self.risk_manager.record_fill(
                opp.condition_id, opp.token_a_id, "A", size, opp.ask_a, market_name
            )
            self.risk_manager.record_fill(
                opp.condition_id, opp.token_b_id, "B", size, opp.ask_b, market_name
            )
            self._trades_executed += 1
            self._total_profit += opp.profit_per_pair * size
            return

        try:
            # Place both orders
            # Order 1: Buy token A at ask_a
            order_a = await self.clob_client.place_order(
                token_id=opp.token_a_id,
                side=OrderSide.BUY,
                price=opp.ask_a,
                size=size,
            )

            # Record fill for token A
            if order_a:
                self.risk_manager.record_fill(
                    opp.condition_id, opp.token_a_id, "A", size, opp.ask_a, market_name
                )

            # Order 2: Buy token B at ask_b
            order_b = await self.clob_client.place_order(
                token_id=opp.token_b_id,
                side=OrderSide.BUY,
                price=opp.ask_b,
                size=size,
            )

            # Record fill for token B
            if order_b:
                self.risk_manager.record_fill(
                    opp.condition_id, opp.token_b_id, "B", size, opp.ask_b, market_name
                )

            self._trades_executed += 1
            self._total_profit += opp.profit_per_pair * size

            # Record metrics
            if self.metrics:
                self.metrics.inc("trades_executed")
                self.metrics.inc("total_profit", opp.profit_per_pair * size)
                self.metrics.gauge("total_profit_cumulative", self._total_profit)

            self.logger.info(
                "arbitrage_executed",
                order_a=order_a.order_id if order_a else "failed",
                order_b=order_b.order_id if order_b else "failed",
                profit=opp.profit_per_pair * size,
            )

        except Exception as e:
            if self.metrics:
                self.metrics.inc("trade_errors")
            self.logger.error(
                "arbitrage_execution_error",
                error=str(e),
                condition_id=opp.condition_id,
            )

    def _serialize_market_position(self, market_pos: Any) -> dict[str, Any]:
        """Serialize a market position for TUI display."""
        # Use realized PnL if closed, otherwise expected profit
        pnl = market_pos.realized_pnl if market_pos.closed else market_pos.expected_profit

        result = {
            "condition_id": market_pos.condition_id,
            "token_id": market_pos.condition_id[:42],  # Use condition_id as token_id for display
            "market_name": market_pos.market_name,
            "closed": market_pos.closed,
            "close_reason": market_pos.close_reason,
            "is_hedged": market_pos.is_hedged,
            "total_size_usd": market_pos.total_cost,
            "expected_profit": market_pos.expected_profit,
            "unhedged_size": market_pos.unhedged_size,
            "pnl": pnl,
            "entries_made": 2 if market_pos.position_a and market_pos.position_b else 1,
            "avg_entry_price": 0.0,  # Not applicable for arbitrage
            "target_price": 0.0,  # Not applicable
            "stop_loss_price": 0.0,  # Not applicable
        }

        if market_pos.position_a:
            result["position_a"] = {
                "token_id": market_pos.position_a.token_id,
                "side": market_pos.position_a.side,
                "size": market_pos.position_a.size,
                "entry_price": market_pos.position_a.entry_price,
                "cost": market_pos.position_a.cost,
                "entry_time": market_pos.position_a.entry_time.isoformat(),
            }

        if market_pos.position_b:
            result["position_b"] = {
                "token_id": market_pos.position_b.token_id,
                "side": market_pos.position_b.side,
                "size": market_pos.position_b.size,
                "entry_price": market_pos.position_b.entry_price,
                "cost": market_pos.position_b.cost,
                "entry_time": market_pos.position_b.entry_time.isoformat(),
            }

        return result

    async def health_check(self) -> dict[str, Any]:
        """Return bot health status."""
        ws_connected = self.ws_client.connected if self.ws_client else False
        orderbook_stats = self.orderbook_manager.get_stats() if self.orderbook_manager else {}
        risk_stats = self.risk_manager.get_stats() if self.risk_manager else {}

        # Serialize positions from risk manager
        open_positions = []
        closed_positions = []
        total_invested = 0.0

        if self.risk_manager:
            # Serialize open positions
            for market_pos in self.risk_manager._positions.values():
                open_positions.append(self._serialize_market_position(market_pos))
                total_invested += market_pos.total_cost

            # Serialize closed positions (newest first)
            for market_pos in reversed(self.risk_manager._closed_positions):
                closed_positions.append(self._serialize_market_position(market_pos))

        return {
            "name": self.name,
            "is_running": self.is_running,
            "dry_run": self.is_dry_run,
            "trade_size": self.trade_size,
            "min_spread": self.min_spread,
            "watchlist_size": len(self.watchlist.entries) if self.watchlist else 0,
            "ws_connected": ws_connected,
            "tokens_subscribed": orderbook_stats.get("tokens_tracked", 0),
            "updates_processed": orderbook_stats.get("updates_processed", 0),
            "opportunities_found": self._opportunities_found,
            "trades_executed": self._trades_executed,
            "total_profit": self._total_profit,
            "total_invested": total_invested,
            # Risk stats
            "active_positions": risk_stats.get("open_positions", 0),
            "hedged_positions": risk_stats.get("hedged_positions", 0),
            "total_exposure": risk_stats.get("total_exposure", 0),
            "unhedged_exposure": risk_stats.get("unhedged_exposure", 0),
            "blocked_by_risk": risk_stats.get("blocked_by_risk", 0),
            # Position lists for TUI
            "open_positions": open_positions,
            "closed_positions": closed_positions,
        }
