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
    last_price: float | None = None  # Last best_ask (for buying)
    last_bid: float | None = None    # Last best_bid (for selling)
    baseline_price: float | None = None  # Rolling average

    # Warmup tracking
    first_price_time: datetime | None = None
    is_warmed_up: bool = False

    # Spike validation - track price stability
    last_update_time: datetime | None = None
    consecutive_stable_updates: int = 0  # Count updates where price is stable

    def add_price(self, price: float, bid: float | None = None, timestamp: datetime | None = None) -> None:
        """Add a new price observation."""
        ts = timestamp or datetime.utcnow()

        # Track price stability (for spike validation)
        if self.last_price is not None:
            price_change = abs(price - self.last_price) / self.last_price if self.last_price > 0 else 0
            if price_change < 0.02:  # Less than 2% change = stable
                self.consecutive_stable_updates += 1
            else:
                self.consecutive_stable_updates = 0

        self.history.append(PricePoint(price=price, timestamp=ts))
        self.last_price = price
        self.last_update_time = ts
        if bid is not None:
            self.last_bid = bid

        # Track first price time for warmup
        if self.first_price_time is None:
            self.first_price_time = ts

        # Update baseline (average of last 2 minutes, excluding last 10 seconds)
        self._update_baseline()

    def _update_baseline(self) -> None:
        """Calculate baseline price (median excluding recent spikes)."""
        if len(self.history) < 5:
            return

        now = datetime.utcnow()

        # Warmup: need at least 90 seconds of data before detecting spikes
        if self.first_price_time:
            time_since_start = (now - self.first_price_time).total_seconds()
            if time_since_start < 90:
                # Still warming up - don't set baseline yet
                return
            elif not self.is_warmed_up:
                # Just finished warmup - set baseline using median of recent data
                self.is_warmed_up = True
                cutoff = now - timedelta(seconds=60)
                recent_prices = [p.price for p in self.history if p.timestamp >= cutoff]
                if recent_prices:
                    recent_prices.sort()
                    mid = len(recent_prices) // 2
                    self.baseline_price = recent_prices[mid]
                return

        cutoff_recent = now - timedelta(seconds=5)  # Exclude last 5 seconds
        cutoff_old = now - timedelta(minutes=2)  # Use last 2 minutes

        prices = [
            p.price for p in self.history
            if cutoff_old <= p.timestamp <= cutoff_recent
        ]

        if prices:
            # Use median instead of average (more robust to outliers)
            prices.sort()
            mid = len(prices) // 2
            self.baseline_price = prices[mid]


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

    # Configuration (defaults - overridden by bot.py)
    PRICE_CHANGE_THRESHOLD = 0.08  # 8% minimum movement
    TIME_WINDOW_SECONDS = 120  # Movement must happen within 2 minutes
    RECOVERY_TARGET = 0.50  # Exit when price recovers 50%
    STOP_LOSS = 0.07  # Exit if price moves 7% against us

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

    def unregister_tokens(self, token_ids: set[str]) -> None:
        """Remove tokens from tracking (when they leave watchlist)."""
        for token_id in token_ids:
            # Clear from trackers
            if token_id in self._trackers:
                del self._trackers[token_id]
            # Clear any active spikes for this token
            if token_id in self._active_spikes:
                del self._active_spikes[token_id]

    def update_price(self, token_id: str, price: float, bid: float | None = None) -> None:
        """
        Update price for a token and check for spikes.

        Called on each orderbook update.

        Args:
            token_id: The token to update
            price: The best_ask price (what you pay to buy)
            bid: The best_bid price (what you get when selling)
        """
        tracker = self._trackers.get(token_id)
        if not tracker:
            return

        # Track warmup state before update
        was_warmed_up = tracker.is_warmed_up

        # Store previous baseline before updating
        old_baseline = tracker.baseline_price

        # Add new price (with bid for selling)
        tracker.add_price(price, bid=bid)

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
        """Check if current price represents a DROP from baseline."""

        # Skip if not warmed up
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

        # ONLY detect DROPS (negative change)
        # We buy tokens that dropped, expecting them to recover
        if change >= 0:
            return

        # Check if drop exceeds threshold
        if abs(change) < self.price_change_threshold:
            return

        # === NEW VALIDATIONS TO REJECT FALSE SPIKES ===

        # 1. Reject prices outside tradeable range (illiquid tokens)
        if current < 0.05 or current > 0.95:
            self.logger.debug(
                "rejecting_illiquid_price",
                token_id=tracker.token_id[:20] + "...",
                current=f"{current:.3f}",
                reason="price outside 0.05-0.95 range",
            )
            return

        # 2. Reject if baseline is also in illiquid range
        if baseline < 0.05 or baseline > 0.95:
            self.logger.debug(
                "rejecting_illiquid_baseline",
                token_id=tracker.token_id[:20] + "...",
                baseline=f"{baseline:.3f}",
            )
            return

        # 3. Reject unrealistic drops (>40% is almost certainly bad data or multi-token confusion)
        if abs(change) > 0.40:
            self.logger.debug(
                "rejecting_unrealistic_drop",
                token_id=tracker.token_id[:20] + "...",
                change=f"{change:.2%}",
                baseline=f"{baseline:.3f}",
                current=f"{current:.3f}",
            )
            return

        # 4. Reject if we don't have enough history (baseline might be unreliable)
        if len(tracker.history) < 15:
            self.logger.debug(
                "rejecting_insufficient_history",
                token_id=tracker.token_id[:20] + "...",
                history_len=len(tracker.history),
            )
            return

        # 5. Reject if price was already volatile (not a sudden spike, just noisy data)
        # We want spikes from STABLE prices, not already-moving prices
        if tracker.consecutive_stable_updates < 3:
            self.logger.debug(
                "rejecting_already_volatile",
                token_id=tracker.token_id[:20] + "...",
                stable_updates=tracker.consecutive_stable_updates,
            )
            return

        # 6. Require bid price to exist and be reasonable (confirms real liquidity)
        if tracker.last_bid is None or tracker.last_bid <= 0:
            self.logger.debug(
                "rejecting_no_bid",
                token_id=tracker.token_id[:20] + "...",
            )
            return

        # 7. Check bid-ask spread isn't too wide (>20% spread = illiquid)
        spread = (current - tracker.last_bid) / current if current > 0 else 1.0
        if spread > 0.20:
            self.logger.debug(
                "rejecting_wide_spread",
                token_id=tracker.token_id[:20] + "...",
                ask=f"{current:.3f}",
                bid=f"{tracker.last_bid:.3f}",
                spread=f"{spread:.1%}",
            )
            return

        # === END NEW VALIDATIONS ===

        # Check if we already have an active spike for this token OR its complement
        if tracker.token_id in self._active_spikes:
            return
        if tracker.other_token_id in self._active_spikes:
            return

        # Price dropped → Buy this token (it's cheap, expect recovery)
        token_to_buy = tracker.token_id

        # Target: price recovers X% of the drop
        target_price = current + (baseline - current) * self.recovery_target

        # Stop loss: price drops another X% from current
        stop_loss_price = current * (1 - self.stop_loss)

        # Create spike signal
        spike = PriceSpike(
            condition_id=tracker.condition_id,
            token_id=tracker.token_id,
            price_before=baseline,
            price_after=current,
            price_change=change,
            direction="down",
            token_to_buy=token_to_buy,
            target_price=target_price,
            stop_loss_price=stop_loss_price,
        )

        # Record spike for BOTH tokens to prevent double-detection
        self._active_spikes[tracker.token_id] = spike
        self._active_spikes[tracker.other_token_id] = spike
        self._spikes_detected += 1

        self.logger.info(
            "price_drop_detected",
            condition_id=tracker.condition_id[:20] + "...",
            token_id=tracker.token_id[:20] + "...",
            change=f"{change:.2%}",
            baseline=f"{baseline:.3f}",
            current=f"{current:.3f}",
            target=f"{target_price:.3f}",
            stop_loss=f"{stop_loss_price:.3f}",
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
        warmed_up = sum(1 for t in self._trackers.values() if t.is_warmed_up)
        # "Active" = received price update in last 60 seconds
        now = datetime.utcnow()
        active = sum(
            1 for t in self._trackers.values()
            if t.history and (now - t.history[-1].timestamp).total_seconds() < 60
        )
        return {
            "tokens_tracked": len(self._trackers),
            "tokens_warmed_up": warmed_up,
            "tokens_active": active,
            "active_spikes": len(self._active_spikes),
            "spikes_detected": self._spikes_detected,
        }
