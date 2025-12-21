"""
Local Orderbook State Manager.

Maintains local orderbook state from WebSocket updates.
Detects arbitrage opportunities when ask_A + ask_B < 1.0
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Any

import structlog

from src.ws.client import OrderBookUpdate


@dataclass
class TokenOrderbook:
    """Local orderbook state for a single token."""

    token_id: str
    condition_id: str  # Market condition ID

    # Best prices
    best_bid: float | None = None
    best_ask: float | None = None

    # Sizes at best prices
    bid_size: float = 0.0
    ask_size: float = 0.0

    # Full book (price, size) - sorted
    bids: list[tuple[float, float]] = field(default_factory=list)
    asks: list[tuple[float, float]] = field(default_factory=list)

    # Metadata
    last_update: datetime | None = None
    update_count: int = 0

    def update(self, ws_update: OrderBookUpdate) -> None:
        """Update from WebSocket message."""
        self.bids = ws_update.bids
        self.asks = ws_update.asks
        self.best_bid = ws_update.best_bid
        self.best_ask = ws_update.best_ask
        self.bid_size = ws_update.best_bid_size or 0.0
        self.ask_size = ws_update.best_ask_size or 0.0
        self.last_update = datetime.utcnow()
        self.update_count += 1


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""

    condition_id: str
    token_a_id: str
    token_b_id: str

    # Prices
    ask_a: float  # Best ask for token A
    ask_b: float  # Best ask for token B
    spread: float  # 1.0 - (ask_a + ask_b) = profit per $1

    # Sizes (max executable)
    size_a: float  # Available at ask_a
    size_b: float  # Available at ask_b
    max_size: float  # min(size_a, size_b)

    # Profit calculation
    cost_per_pair: float  # ask_a + ask_b
    profit_per_pair: float  # 1.0 - cost_per_pair
    max_profit: float  # profit_per_pair * max_size

    # Timing
    detected_at: datetime = field(default_factory=datetime.utcnow)


# Callback for arbitrage detection
ArbitrageCallback = Callable[[ArbitrageOpportunity], None]


class LocalOrderbookManager:
    """
    Manages local orderbook state for all subscribed markets.

    Key responsibilities:
    1. Store orderbook state per token
    2. Track token pairs (A, B) per market
    3. Detect arbitrage: ask_A + ask_B < 1.0
    """

    def __init__(
        self,
        on_arbitrage: ArbitrageCallback | None = None,
        min_spread: float = 0.0,  # Minimum spread to report (0 = any profit)
        min_size: float = 1.0,  # Minimum size to consider
    ):
        """
        Initialize orderbook manager.

        Args:
            on_arbitrage: Callback when arbitrage detected
            min_spread: Minimum spread (profit) to trigger callback
            min_size: Minimum executable size
        """
        self.on_arbitrage = on_arbitrage
        self.min_spread = min_spread
        self.min_size = min_size

        self.logger = structlog.get_logger(__name__)

        # Token ID -> TokenOrderbook
        self._orderbooks: dict[str, TokenOrderbook] = {}

        # Condition ID -> (token_a_id, token_b_id)
        self._market_pairs: dict[str, tuple[str, str]] = {}

        # Token ID -> Condition ID (reverse lookup)
        self._token_to_market: dict[str, str] = {}

        # Stats
        self._updates_processed = 0
        self._arbitrage_detected = 0

    def register_market(
        self,
        condition_id: str,
        token_a_id: str,
        token_b_id: str,
    ) -> None:
        """
        Register a binary market for monitoring.

        Args:
            condition_id: Market condition ID
            token_a_id: First token ID
            token_b_id: Second token ID
        """
        self._market_pairs[condition_id] = (token_a_id, token_b_id)
        self._token_to_market[token_a_id] = condition_id
        self._token_to_market[token_b_id] = condition_id

        # Initialize orderbooks
        if token_a_id not in self._orderbooks:
            self._orderbooks[token_a_id] = TokenOrderbook(
                token_id=token_a_id,
                condition_id=condition_id,
            )
        if token_b_id not in self._orderbooks:
            self._orderbooks[token_b_id] = TokenOrderbook(
                token_id=token_b_id,
                condition_id=condition_id,
            )

        self.logger.debug(
            "market_registered",
            condition_id=condition_id[:16] + "...",
            token_a=token_a_id[:16] + "...",
            token_b=token_b_id[:16] + "...",
        )

    def unregister_market(self, condition_id: str) -> None:
        """Remove a market from monitoring."""
        if condition_id not in self._market_pairs:
            return

        token_a, token_b = self._market_pairs[condition_id]

        del self._market_pairs[condition_id]
        self._token_to_market.pop(token_a, None)
        self._token_to_market.pop(token_b, None)
        self._orderbooks.pop(token_a, None)
        self._orderbooks.pop(token_b, None)

    def handle_update(self, update: OrderBookUpdate) -> ArbitrageOpportunity | None:
        """
        Handle orderbook update from WebSocket.

        Args:
            update: OrderBook update

        Returns:
            ArbitrageOpportunity if detected, None otherwise
        """
        self._updates_processed += 1
        token_id = update.asset_id

        # Update local state
        if token_id in self._orderbooks:
            self._orderbooks[token_id].update(update)
        else:
            # Token not registered - might be from a different subscription
            return None

        # Check for arbitrage
        condition_id = self._token_to_market.get(token_id)
        if not condition_id:
            return None

        return self._check_arbitrage(condition_id)

    def _check_arbitrage(self, condition_id: str) -> ArbitrageOpportunity | None:
        """
        Check if arbitrage exists for a market.

        Args:
            condition_id: Market to check

        Returns:
            ArbitrageOpportunity if found
        """
        if condition_id not in self._market_pairs:
            return None

        token_a_id, token_b_id = self._market_pairs[condition_id]

        book_a = self._orderbooks.get(token_a_id)
        book_b = self._orderbooks.get(token_b_id)

        if not book_a or not book_b:
            return None

        # Need asks on both sides
        if book_a.best_ask is None or book_b.best_ask is None:
            return None

        ask_a = book_a.best_ask
        ask_b = book_b.best_ask

        # Check for arbitrage: ask_A + ask_B < 1.0
        total_cost = ask_a + ask_b

        if total_cost >= 1.0:
            return None

        spread = 1.0 - total_cost

        # Check minimum spread
        if spread < self.min_spread:
            return None

        # Check executable size
        size_a = book_a.ask_size
        size_b = book_b.ask_size
        max_size = min(size_a, size_b)

        if max_size < self.min_size:
            return None

        # Create opportunity
        opportunity = ArbitrageOpportunity(
            condition_id=condition_id,
            token_a_id=token_a_id,
            token_b_id=token_b_id,
            ask_a=ask_a,
            ask_b=ask_b,
            spread=spread,
            size_a=size_a,
            size_b=size_b,
            max_size=max_size,
            cost_per_pair=total_cost,
            profit_per_pair=spread,
            max_profit=spread * max_size,
        )

        self._arbitrage_detected += 1

        self.logger.info(
            "arbitrage_detected",
            condition_id=condition_id[:16] + "...",
            ask_a=ask_a,
            ask_b=ask_b,
            spread=f"{spread:.4f}",
            max_size=max_size,
            max_profit=f"${opportunity.max_profit:.4f}",
        )

        # Call callback
        if self.on_arbitrage:
            self.on_arbitrage(opportunity)

        return opportunity

    def check_all_markets(self) -> list[ArbitrageOpportunity]:
        """
        Check all registered markets for arbitrage.

        Returns:
            List of opportunities found
        """
        opportunities = []

        for condition_id in self._market_pairs:
            opp = self._check_arbitrage(condition_id)
            if opp:
                opportunities.append(opp)

        return opportunities

    def get_orderbook(self, token_id: str) -> TokenOrderbook | None:
        """Get orderbook for a token."""
        return self._orderbooks.get(token_id)

    def get_market_state(self, condition_id: str) -> dict[str, Any] | None:
        """Get current state for a market."""
        if condition_id not in self._market_pairs:
            return None

        token_a_id, token_b_id = self._market_pairs[condition_id]
        book_a = self._orderbooks.get(token_a_id)
        book_b = self._orderbooks.get(token_b_id)

        if not book_a or not book_b:
            return None

        return {
            "condition_id": condition_id,
            "token_a": {
                "id": token_a_id,
                "best_bid": book_a.best_bid,
                "best_ask": book_a.best_ask,
                "bid_size": book_a.bid_size,
                "ask_size": book_a.ask_size,
                "updates": book_a.update_count,
            },
            "token_b": {
                "id": token_b_id,
                "best_bid": book_b.best_bid,
                "best_ask": book_b.best_ask,
                "bid_size": book_b.bid_size,
                "ask_size": book_b.ask_size,
                "updates": book_b.update_count,
            },
            "total_ask": (book_a.best_ask or 0) + (book_b.best_ask or 0),
            "arbitrage": (book_a.best_ask or 1) + (book_b.best_ask or 1) < 1.0,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        return {
            "markets_registered": len(self._market_pairs),
            "tokens_tracked": len(self._orderbooks),
            "updates_processed": self._updates_processed,
            "arbitrage_detected": self._arbitrage_detected,
        }

    @property
    def token_ids(self) -> list[str]:
        """Get all token IDs being tracked."""
        return list(self._orderbooks.keys())
