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

    # Warmup tracking
    first_price_time: datetime | None = None
    is_warmed_up: bool = False

    def add_price(self, price: float, timestamp: datetime | None = None) -> None:
        """Add a new price observation."""
        ts = timestamp or datetime.utcnow()
        self.history.append(PricePoint(price=price, timestamp=ts))
        self.last_price = price

        # Track first price time for warmup
        if self.first_price_time is None:
            self.first_price_time = ts

        # Update baseline (average of last 2 minutes, excluding last 10 seconds)
        self._update_baseline()

    def _update_baseline(self) -> None:
        """Calculate baseline price (rolling average excluding recent spikes)."""
        if len(self.history) < 10:
            return

        now = datetime.utcnow()

        # Warmup: need at least 90 seconds of data before detecting spikes
        # This ensures we have a stable baseline before detecting movements
        if self.first_price_time:
            time_since_start = (now - self.first_price_time).total_seconds()
            if time_since_start < 90:
                # Still warming up - don't set baseline yet, wait for stable data
                return
            elif not self.is_warmed_up:
                # Just finished warmup - reset baseline using only recent data (last 30 sec)
                # This avoids using stale prices from WebSocket initialization
                self.is_warmed_up = True
                cutoff = now - timedelta(seconds=30)
                recent_prices = [p.price for p in self.history if p.timestamp >= cutoff]
                if recent_prices:
                    self.baseline_price = sum(recent_prices) / len(recent_prices)
                return

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

        # Track warmup state before update
        was_warmed_up = tracker.is_warmed_up

        # Store previous baseline before updating
        old_baseline = tracker.baseline_price

        # Add new price
        tracker.add_price(price)

        # Log warmup completion
        if not was_warmed_up and tracker.is_warmed_up:
            warmed_up_count = sum(1 for t in self._trackers.values() if t.is_warmed_up)
            total_count = len(self._trackers)
            self.logger.info(
                "warmup_progress",
                warmed_up=warmed_up_count,
                total=total_count,
                percent=f"{warmed_up_count/total_count*100:.0f}%",
            )
            # Log when all tokens are warmed up
            if warmed_up_count == total_count:
                self.logger.info(
                    "warmup_complete",
                    tokens=total_count,
                    message="All tokens ready - spike detection active",
                )

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

        # Skip if not warmed up (need 30 seconds of data)
        if not tracker.is_warmed_up:
            return

        # Calculate change
        if baseline == 0:
            return

        change = (current - baseline) / baseline

        # Log significant movements (>0.5%) for debugging
        if abs(change) >= 0.005:
            self.logger.debug(
                "price_movement",
                token_id=tracker.token_id[:20] + "...",
                change=f"{change:.2%}",
                baseline=f"{baseline:.3f}",
                current=f"{current:.3f}",
                threshold=f"{self.price_change_threshold:.0%}",
            )

        # Check if change exceeds threshold
        if abs(change) < self.price_change_threshold:
            return

        # Reject unrealistic spikes (>30% change is almost certainly bad data)
        # Real market movements in prediction markets rarely exceed 30% in 2 minutes
        if abs(change) > 0.30:
            self.logger.debug(
                "rejecting_unrealistic_spike",
                token_id=tracker.token_id[:20] + "...",
                change=f"{change:.2%}",
                baseline=f"{baseline:.3f}",
                current=f"{current:.3f}",
            )
            return

        # Check if we already have an active spike for this token OR its complement
        # (to avoid double-detecting the same market movement)
        if tracker.token_id in self._active_spikes:
            return
        if tracker.other_token_id in self._active_spikes:
            return

        # Determine direction and what to buy
        # Key insight: we always buy the token that DROPPED, expecting it to recover
        if change > 0:
            # Token A went UP → Token B went DOWN → Buy Token B
            direction = "up"
            token_to_buy = tracker.other_token_id

            # Get ACTUAL price of token B from its tracker (not approximation)
            other_tracker = self._trackers.get(tracker.other_token_id)
            if not other_tracker or not other_tracker.last_price or not other_tracker.baseline_price:
                # Can't trade without real price data for the token we want to buy
                self.logger.debug(
                    "skip_up_spike_no_other_price",
                    token_id=tracker.token_id[:20] + "...",
                    other_token_id=tracker.other_token_id[:20] + "...",
                )
                return

            other_current = other_tracker.last_price
            other_baseline = other_tracker.baseline_price

            # Verify token B actually dropped significantly
            other_change = (other_current - other_baseline) / other_baseline if other_baseline > 0 else 0
            if other_change >= -0.03:  # Token B didn't drop enough (at least 3%)
                return

            # Target: token B recovers 50% of its drop
            target_price = other_current + (other_baseline - other_current) * self.recovery_target

            # Stop loss: token B drops another 5%
            stop_loss_price = other_current * (1 - self.stop_loss)

            # Store prices for the token we're buying
            price_before_for_spike = other_baseline
            price_after_for_spike = other_current
        else:
            # Token A went DOWN → Buy Token A (it's cheap)
            direction = "down"
            token_to_buy = tracker.token_id

            # Target: price recovers 50% of the drop
            target_price = current + (baseline - current) * self.recovery_target

            # Stop loss: price drops another 5% from current
            stop_loss_price = current * (1 - self.stop_loss)

            # Store prices for the token we're buying (same token)
            price_before_for_spike = baseline
            price_after_for_spike = current

        # Create spike signal
        spike = PriceSpike(
            condition_id=tracker.condition_id,
            token_id=tracker.token_id,  # Token that triggered detection
            price_before=price_before_for_spike,  # Baseline of token we're buying
            price_after=price_after_for_spike,  # Current price of token we're buying
            price_change=change,
            direction=direction,
            token_to_buy=token_to_buy,
            target_price=target_price,
            stop_loss_price=stop_loss_price,
        )

        # Record spike for BOTH tokens to prevent double-detection
        self._active_spikes[tracker.token_id] = spike
        self._active_spikes[tracker.other_token_id] = spike
        self._spikes_detected += 1

        # For logging, show the price of the token we're buying
        if direction == "up":
            log_price_before = other_baseline
            log_price_after = other_current
        else:
            log_price_before = baseline
            log_price_after = current

        self.logger.info(
            "price_spike_detected",
            condition_id=tracker.condition_id,  # Full ID
            token_id=tracker.token_id,  # Full ID
            direction=direction,
            change=f"{change:.2%}",
            buy_token_price_before=f"{log_price_before:.3f}",
            buy_token_price_after=f"{log_price_after:.3f}",
            target=f"{target_price:.3f}",
            stop_loss=f"{stop_loss_price:.3f}",
            token_to_buy=token_to_buy,  # Full ID
        )

        # Trigger callback
        if self.on_spike:
            self.on_spike(spike)

    def clear_spike(self, token_id: str) -> None:
        """Clear an active spike (after trade is closed)."""
        # Get the spike to find the other token in the pair
        spike = self._active_spikes.get(token_id)
        if spike:
            # Clear the spike for all related tokens
            # spike.token_id = the token that triggered detection
            # spike.token_to_buy = the token we bought
            # We also need to find the "other" token in the market
            tracker = self._trackers.get(spike.token_id)
            tokens_to_clear = {spike.token_id, spike.token_to_buy}
            if tracker:
                tokens_to_clear.add(tracker.other_token_id)

            for t in tokens_to_clear:
                if t in self._active_spikes:
                    del self._active_spikes[t]

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
