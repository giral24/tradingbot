"""
Price movement detector for mean reversion strategy.

Detects sudden price spikes caused by large orders,
which are likely to revert to the mean.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from typing import Callable

import structlog


@dataclass
class PricePoint:
    """A single price observation."""
    price: float
    timestamp: datetime


@dataclass
class PriceSpike:
    """Detected price spike that may revert."""

    condition_id: str
    token_id: str

    # Price info
    price_before: float  # Price before spike
    price_after: float   # Current price after spike
    price_change: float  # Percentage change (e.g., 0.10 = 10%)

    # Direction
    direction: str  # "up" or "down"

    # Trade opportunity
    # If price went UP, we buy the OTHER token (betting it comes down)
    # If price went DOWN, we buy THIS token (betting it comes up)
    token_to_buy: str
    target_price: float  # 50% recovery target
    stop_loss_price: float  # -5% stop loss

    # Metadata
    detected_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def potential_profit(self) -> float:
        """Potential profit percentage if price reverts 50%."""
        return abs(self.price_change) / 2


@dataclass
class TokenPriceTracker:
    """Tracks price history for a single token."""

    token_id: str
    condition_id: str
    other_token_id: str  # The complementary token

    # Price history (last 5 minutes)
    history: deque = field(default_factory=lambda: deque(maxlen=300))

    # Current state
    last_price: float | None = None
    baseline_price: float | None = None  # Rolling average

    def add_price(self, price: float, timestamp: datetime | None = None) -> None:
        """Add a new price observation."""
        ts = timestamp or datetime.utcnow()
        self.history.append(PricePoint(price=price, timestamp=ts))
        self.last_price = price

        # Update baseline (average of last 2 minutes, excluding last 10 seconds)
        self._update_baseline()

    def _update_baseline(self) -> None:
        """Calculate baseline price (rolling average excluding recent spikes)."""
        if len(self.history) < 10:
            return

        now = datetime.utcnow()
        cutoff_recent = now - timedelta(seconds=10)  # Exclude last 10 seconds
        cutoff_old = now - timedelta(minutes=2)  # Use last 2 minutes

        prices = [
            p.price for p in self.history
            if cutoff_old <= p.timestamp <= cutoff_recent
        ]

        if prices:
            self.baseline_price = sum(prices) / len(prices)


# Callback type for when a spike is detected
SpikeCallback = Callable[[PriceSpike], None]


class PriceMovementDetector:
    """
    Detects sudden price movements that may revert.

    Strategy:
    - Track price history for each token
    - Detect when price moves >= threshold in < time_window
    - Signal opportunity to buy the opposite direction
    """

    # Configuration
    PRICE_CHANGE_THRESHOLD = 0.08  # 8% minimum movement
    TIME_WINDOW_SECONDS = 120  # Movement must happen within 2 minutes
    RECOVERY_TARGET = 0.50  # Exit when price recovers 50%
    STOP_LOSS = 0.05  # Exit if price moves 5% against us

    def __init__(
        self,
        price_change_threshold: float = PRICE_CHANGE_THRESHOLD,
        time_window_seconds: int = TIME_WINDOW_SECONDS,
        recovery_target: float = RECOVERY_TARGET,
        stop_loss: float = STOP_LOSS,
        on_spike: SpikeCallback | None = None,
    ):
        self.price_change_threshold = price_change_threshold
        self.time_window_seconds = time_window_seconds
        self.recovery_target = recovery_target
        self.stop_loss = stop_loss
        self.on_spike = on_spike

        self.logger = structlog.get_logger(__name__)

        # Token trackers: token_id -> TokenPriceTracker
        self._trackers: dict[str, TokenPriceTracker] = {}

        # Active spikes (to avoid duplicate signals)
        self._active_spikes: dict[str, PriceSpike] = {}

        # Stats
        self._spikes_detected = 0

    def register_market(
        self,
        condition_id: str,
        token_a_id: str,
        token_b_id: str,
    ) -> None:
        """Register a market to track."""
        if token_a_id not in self._trackers:
            self._trackers[token_a_id] = TokenPriceTracker(
                token_id=token_a_id,
                condition_id=condition_id,
                other_token_id=token_b_id,
            )
        if token_b_id not in self._trackers:
            self._trackers[token_b_id] = TokenPriceTracker(
                token_id=token_b_id,
                condition_id=condition_id,
                other_token_id=token_a_id,
            )

    def update_price(self, token_id: str, price: float) -> None:
        """
        Update price for a token and check for spikes.

        Called on each orderbook update.
        """
        tracker = self._trackers.get(token_id)
        if not tracker:
            return

        # Store previous baseline before updating
        old_baseline = tracker.baseline_price

        # Add new price
        tracker.add_price(price)

        # Check for spike
        if old_baseline is not None:
            self._check_for_spike(tracker, old_baseline, price)

    def _check_for_spike(
        self,
        tracker: TokenPriceTracker,
        baseline: float,
        current: float,
    ) -> None:
        """Check if current price represents a spike from baseline."""

        # Calculate change
        if baseline == 0:
            return

        change = (current - baseline) / baseline

        # Check if change exceeds threshold
        if abs(change) < self.price_change_threshold:
            return

        # Check if we already have an active spike for this token
        if tracker.token_id in self._active_spikes:
            return

        # Determine direction and what to buy
        if change > 0:
            # Price went UP - buy the OTHER token (it went down)
            direction = "up"
            token_to_buy = tracker.other_token_id
            # Target: price goes back down 50% of the move
            target_price = current - (current - baseline) * self.recovery_target
            # Stop loss: price goes up another 5%
            stop_loss_price = current * (1 + self.stop_loss)
        else:
            # Price went DOWN - buy THIS token (it's cheap)
            direction = "down"
            token_to_buy = tracker.token_id
            # Target: price recovers 50% of the drop
            target_price = current + (baseline - current) * self.recovery_target
            # Stop loss: price drops another 5%
            stop_loss_price = current * (1 - self.stop_loss)

        # Create spike signal
        spike = PriceSpike(
            condition_id=tracker.condition_id,
            token_id=tracker.token_id,
            price_before=baseline,
            price_after=current,
            price_change=change,
            direction=direction,
            token_to_buy=token_to_buy,
            target_price=target_price,
            stop_loss_price=stop_loss_price,
        )

        # Record and signal
        self._active_spikes[tracker.token_id] = spike
        self._spikes_detected += 1

        self.logger.info(
            "price_spike_detected",
            condition_id=tracker.condition_id[:20] + "...",
            token_id=tracker.token_id[:20] + "...",
            direction=direction,
            change=f"{change:.2%}",
            price_before=f"{baseline:.3f}",
            price_after=f"{current:.3f}",
            token_to_buy=token_to_buy[:20] + "...",
        )

        # Trigger callback
        if self.on_spike:
            self.on_spike(spike)

    def clear_spike(self, token_id: str) -> None:
        """Clear an active spike (after trade is closed)."""
        if token_id in self._active_spikes:
            del self._active_spikes[token_id]

    def get_active_spike(self, token_id: str) -> PriceSpike | None:
        """Get active spike for a token."""
        return self._active_spikes.get(token_id)

    @property
    def token_ids(self) -> list[str]:
        """Get all tracked token IDs."""
        return list(self._trackers.keys())

    def get_stats(self) -> dict:
        """Get detector statistics."""
        return {
            "tokens_tracked": len(self._trackers),
            "active_spikes": len(self._active_spikes),
            "spikes_detected": self._spikes_detected,
        }
