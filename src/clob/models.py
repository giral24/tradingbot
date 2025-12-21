"""
Data models for CLOB API responses.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class OrderSide(str, Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""
    GTC = "GTC"  # Good-Til-Cancelled
    GTD = "GTD"  # Good-Til-Date
    FOK = "FOK"  # Fill-Or-Kill
    FAK = "FAK"  # Fill-And-Kill (partial fill allowed)


class TokenOutcome(str, Enum):
    """Token outcome type."""
    YES = "YES"
    NO = "NO"


@dataclass
class Token:
    """Token in a market."""
    token_id: str
    outcome: str  # "Yes" or "No"
    price: float | None = None
    winner: bool = False


@dataclass
class Market:
    """Polymarket market."""
    condition_id: str
    question_id: str
    tokens: list[Token]

    # Metadata
    question: str = ""
    description: str = ""
    end_date_iso: str | None = None
    game_start_time: str | None = None

    # Status
    active: bool = True
    closed: bool = False
    archived: bool = False
    accepting_orders: bool = True

    # Metrics (populated by watchlist)
    volume_24h: float = 0.0
    liquidity: float = 0.0
    spread: float | None = None

    # Score for watchlist
    score: float = 0.0

    @property
    def yes_token_id(self) -> str | None:
        """Get YES token ID."""
        for token in self.tokens:
            if token.outcome.upper() == "YES":
                return token.token_id
        return None

    @property
    def no_token_id(self) -> str | None:
        """Get NO token ID."""
        for token in self.tokens:
            if token.outcome.upper() == "NO":
                return token.token_id
        return None


@dataclass
class OrderBookLevel:
    """Single level in orderbook."""
    price: float
    size: float


@dataclass
class OrderBook:
    """Orderbook for a token."""
    token_id: str
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def best_bid(self) -> float | None:
        """Get best (highest) bid price."""
        if not self.bids:
            return None
        return max(level.price for level in self.bids)

    @property
    def best_ask(self) -> float | None:
        """Get best (lowest) ask price."""
        if not self.asks:
            return None
        return min(level.price for level in self.asks)

    @property
    def best_bid_size(self) -> float | None:
        """Get size at best bid."""
        if not self.bids:
            return None
        best = self.best_bid
        for level in self.bids:
            if level.price == best:
                return level.size
        return None

    @property
    def best_ask_size(self) -> float | None:
        """Get size at best ask."""
        if not self.asks:
            return None
        best = self.best_ask
        for level in self.asks:
            if level.price == best:
                return level.size
        return None

    @property
    def spread(self) -> float | None:
        """Get bid-ask spread."""
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid


@dataclass
class Order:
    """Order representation."""
    order_id: str
    market_id: str
    token_id: str
    side: OrderSide
    price: float
    size: float
    order_type: OrderType = OrderType.GTC

    # Status
    filled_size: float = 0.0
    status: str = "live"  # live, matched, cancelled
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def remaining_size(self) -> float:
        """Get unfilled size."""
        return self.size - self.filled_size

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.filled_size >= self.size

    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status == "live" and not self.is_filled


@dataclass
class Trade:
    """Executed trade."""
    trade_id: str
    order_id: str
    market_id: str
    token_id: str
    side: OrderSide
    price: float
    size: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""
    market: Market
    ask_yes: float
    ask_no: float
    size_yes: float  # Available size at ask_yes
    size_no: float   # Available size at ask_no

    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_cost(self) -> float:
        """Total cost to buy both sides."""
        return self.ask_yes + self.ask_no

    @property
    def edge(self) -> float:
        """Profit margin (1.0 - total_cost)."""
        return 1.0 - self.total_cost

    @property
    def max_size(self) -> float:
        """Maximum size executable (min of both sides)."""
        return min(self.size_yes, self.size_no)

    @property
    def expected_profit(self) -> float:
        """Expected profit at max size."""
        return self.edge * self.max_size

    def is_valid(self, min_edge: float = 0.0, min_size: float = 0.0) -> bool:
        """Check if opportunity meets thresholds."""
        return (
            self.edge > min_edge
            and self.max_size >= min_size
            and self.ask_yes > 0
            and self.ask_no > 0
        )
