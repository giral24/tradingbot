"""
Risk Management for Arbitrage Trading.

Handles:
- Position tracking
- Exposure limits
- Partial fill management
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from src.config import settings


@dataclass
class Position:
    """A position in a market."""

    condition_id: str
    token_id: str
    side: str  # "A" or "B"
    size: float
    entry_price: float
    entry_time: datetime = field(default_factory=datetime.utcnow)

    @property
    def cost(self) -> float:
        """Total cost of position."""
        return self.size * self.entry_price


@dataclass
class MarketPosition:
    """Combined position for both sides of a market."""

    condition_id: str
    position_a: Position | None = None
    position_b: Position | None = None

    @property
    def is_hedged(self) -> bool:
        """Check if position is fully hedged (both sides equal)."""
        if not self.position_a or not self.position_b:
            return False
        return abs(self.position_a.size - self.position_b.size) < 0.001

    @property
    def unhedged_size(self) -> float:
        """Get unhedged exposure."""
        size_a = self.position_a.size if self.position_a else 0
        size_b = self.position_b.size if self.position_b else 0
        return abs(size_a - size_b)

    @property
    def total_cost(self) -> float:
        """Total cost of both positions."""
        cost_a = self.position_a.cost if self.position_a else 0
        cost_b = self.position_b.cost if self.position_b else 0
        return cost_a + cost_b

    @property
    def expected_profit(self) -> float:
        """Expected profit if both sides resolve."""
        if not self.is_hedged:
            return 0.0
        # Hedged position guarantees $1 per share
        hedged_size = min(
            self.position_a.size if self.position_a else 0,
            self.position_b.size if self.position_b else 0,
        )
        return hedged_size - self.total_cost


class RiskManager:
    """
    Manages risk and position tracking for arbitrage bot.

    Responsibilities:
    - Track positions per market
    - Enforce exposure limits
    - Handle partial fills
    """

    def __init__(
        self,
        max_position_per_market: float | None = None,
        max_total_exposure: float | None = None,
        max_unhedged_exposure: float | None = None,
    ):
        """
        Initialize risk manager.

        Args:
            max_position_per_market: Max position size per market
            max_total_exposure: Max total capital at risk
            max_unhedged_exposure: Max unhedged (partial fill) exposure
        """
        self.max_position_per_market = max_position_per_market or settings.arb_max_position
        self.max_total_exposure = max_total_exposure or settings.risk_max_exposure
        self.max_unhedged_exposure = max_unhedged_exposure or (self.max_total_exposure * 0.2)

        self.logger = structlog.get_logger(__name__)

        # Position tracking
        self._positions: dict[str, MarketPosition] = {}

        # Stats
        self._total_trades = 0
        self._partial_fills = 0
        self._blocked_by_risk = 0

    def can_trade(
        self,
        condition_id: str,
        size: float,
    ) -> tuple[bool, str]:
        """
        Check if a trade is allowed by risk limits.

        Args:
            condition_id: Market to trade
            size: Trade size

        Returns:
            (allowed, reason)
        """
        # Check market position limit
        market_pos = self._positions.get(condition_id)
        if market_pos:
            current_size = max(
                market_pos.position_a.size if market_pos.position_a else 0,
                market_pos.position_b.size if market_pos.position_b else 0,
            )
            if current_size + size > self.max_position_per_market:
                self._blocked_by_risk += 1
                return False, f"Would exceed max position ({self.max_position_per_market})"

        # Check total exposure
        total_exposure = self.total_exposure
        if total_exposure + (size * 2) > self.max_total_exposure:  # *2 for both sides
            self._blocked_by_risk += 1
            return False, f"Would exceed max exposure ({self.max_total_exposure})"

        # Check unhedged exposure
        if self.unhedged_exposure > self.max_unhedged_exposure:
            self._blocked_by_risk += 1
            return False, f"Too much unhedged exposure ({self.unhedged_exposure:.2f})"

        return True, "OK"

    def record_fill(
        self,
        condition_id: str,
        token_id: str,
        side: str,  # "A" or "B"
        size: float,
        price: float,
    ) -> None:
        """
        Record a filled order.

        Args:
            condition_id: Market condition ID
            token_id: Token ID
            side: "A" or "B"
            size: Fill size
            price: Fill price
        """
        self._total_trades += 1

        # Get or create market position
        if condition_id not in self._positions:
            self._positions[condition_id] = MarketPosition(condition_id=condition_id)

        market_pos = self._positions[condition_id]

        # Create position
        position = Position(
            condition_id=condition_id,
            token_id=token_id,
            side=side,
            size=size,
            entry_price=price,
        )

        # Update market position
        if side == "A":
            if market_pos.position_a:
                # Add to existing
                old = market_pos.position_a
                new_size = old.size + size
                avg_price = (old.cost + position.cost) / new_size
                market_pos.position_a = Position(
                    condition_id=condition_id,
                    token_id=token_id,
                    side="A",
                    size=new_size,
                    entry_price=avg_price,
                    entry_time=old.entry_time,
                )
            else:
                market_pos.position_a = position
        else:
            if market_pos.position_b:
                old = market_pos.position_b
                new_size = old.size + size
                avg_price = (old.cost + position.cost) / new_size
                market_pos.position_b = Position(
                    condition_id=condition_id,
                    token_id=token_id,
                    side="B",
                    size=new_size,
                    entry_price=avg_price,
                    entry_time=old.entry_time,
                )
            else:
                market_pos.position_b = position

        # Check for partial fill warning
        if not market_pos.is_hedged and market_pos.unhedged_size > 0:
            self._partial_fills += 1
            self.logger.warning(
                "partial_fill_detected",
                condition_id=condition_id[:20] + "...",
                unhedged_size=market_pos.unhedged_size,
            )

        self.logger.debug(
            "fill_recorded",
            condition_id=condition_id[:20] + "...",
            side=side,
            size=size,
            price=price,
            is_hedged=market_pos.is_hedged,
        )

    def record_resolution(
        self,
        condition_id: str,
        winning_side: str,  # "A" or "B"
    ) -> float:
        """
        Record market resolution and calculate profit.

        Args:
            condition_id: Market that resolved
            winning_side: Which side won ("A" or "B")

        Returns:
            Realized profit (or loss)
        """
        market_pos = self._positions.get(condition_id)
        if not market_pos:
            return 0.0

        # Winning side pays $1 per share
        winning_pos = market_pos.position_a if winning_side == "A" else market_pos.position_b
        losing_pos = market_pos.position_b if winning_side == "A" else market_pos.position_a

        payout = winning_pos.size if winning_pos else 0
        cost = market_pos.total_cost

        profit = payout - cost

        self.logger.info(
            "market_resolved",
            condition_id=condition_id[:20] + "...",
            winning_side=winning_side,
            payout=payout,
            cost=cost,
            profit=profit,
        )

        # Remove position
        del self._positions[condition_id]

        return profit

    def get_unhedged_positions(self) -> list[MarketPosition]:
        """Get all positions with unhedged exposure."""
        return [
            pos for pos in self._positions.values()
            if not pos.is_hedged and pos.unhedged_size > 0
        ]

    @property
    def total_exposure(self) -> float:
        """Get total capital at risk."""
        return sum(pos.total_cost for pos in self._positions.values())

    @property
    def unhedged_exposure(self) -> float:
        """Get total unhedged exposure."""
        return sum(pos.unhedged_size for pos in self._positions.values())

    @property
    def hedged_positions(self) -> int:
        """Count of fully hedged positions."""
        return sum(1 for pos in self._positions.values() if pos.is_hedged)

    @property
    def expected_profit(self) -> float:
        """Expected profit from hedged positions."""
        return sum(
            pos.expected_profit
            for pos in self._positions.values()
            if pos.is_hedged
        )

    def get_stats(self) -> dict[str, Any]:
        """Get risk manager statistics."""
        return {
            "total_trades": self._total_trades,
            "partial_fills": self._partial_fills,
            "blocked_by_risk": self._blocked_by_risk,
            "open_positions": len(self._positions),
            "hedged_positions": self.hedged_positions,
            "total_exposure": self.total_exposure,
            "unhedged_exposure": self.unhedged_exposure,
            "expected_profit": self.expected_profit,
            "max_position_per_market": self.max_position_per_market,
            "max_total_exposure": self.max_total_exposure,
        }

    def get_position(self, condition_id: str) -> MarketPosition | None:
        """Get position for a specific market."""
        return self._positions.get(condition_id)
