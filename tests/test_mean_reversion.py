"""
Tests for Mean Reversion Bot components.

Run with: python -m pytest tests/test_mean_reversion.py -v
"""

import pytest
from datetime import datetime, timedelta

from src.bots.mean_reversion.detector import (
    PriceMovementDetector,
    PriceSpike,
    TokenPriceTracker,
)
from src.bots.mean_reversion.bot import Position


class TestTokenPriceTracker:
    """Test price tracking functionality."""

    def test_add_price_updates_last_price(self):
        """Test that adding price updates last_price."""
        tracker = TokenPriceTracker(
            token_id="token_a",
            condition_id="condition_1",
            other_token_id="token_b",
        )

        tracker.add_price(0.50)
        assert tracker.last_price == 0.50

        tracker.add_price(0.55)
        assert tracker.last_price == 0.55

    def test_baseline_calculation(self):
        """Test that baseline is calculated from history."""
        tracker = TokenPriceTracker(
            token_id="token_a",
            condition_id="condition_1",
            other_token_id="token_b",
        )

        # Add many prices to build history
        base_time = datetime.utcnow() - timedelta(minutes=1)
        for i in range(20):
            ts = base_time + timedelta(seconds=i * 3)
            tracker.add_price(0.50, ts)

        # Baseline should be around 0.50
        assert tracker.baseline_price is not None
        assert 0.49 <= tracker.baseline_price <= 0.51

    def test_history_limit(self):
        """Test that history doesn't grow infinitely."""
        tracker = TokenPriceTracker(
            token_id="token_a",
            condition_id="condition_1",
            other_token_id="token_b",
        )

        # Add more than maxlen prices
        for i in range(500):
            tracker.add_price(0.50 + i * 0.001)

        # Should be limited to 300 (maxlen)
        assert len(tracker.history) == 300


class TestPriceMovementDetector:
    """Test spike detection functionality."""

    def test_register_market(self):
        """Test market registration."""
        detector = PriceMovementDetector()

        detector.register_market(
            condition_id="cond_1",
            token_a_id="token_a",
            token_b_id="token_b",
        )

        assert "token_a" in detector.token_ids
        assert "token_b" in detector.token_ids
        assert len(detector.token_ids) == 2

    def test_no_spike_for_small_movement(self):
        """Test that small movements don't trigger spike."""
        spikes_detected = []

        detector = PriceMovementDetector(
            price_change_threshold=0.08,  # 8%
            on_spike=lambda s: spikes_detected.append(s),
        )

        detector.register_market("cond_1", "token_a", "token_b")

        # Build baseline
        base_time = datetime.utcnow() - timedelta(minutes=1)
        for i in range(20):
            detector._trackers["token_a"].add_price(0.50, base_time + timedelta(seconds=i * 3))

        # Small movement (5% - below threshold)
        detector.update_price("token_a", 0.525)

        assert len(spikes_detected) == 0

    def test_spike_detected_for_large_movement(self):
        """Test that large movements trigger spike."""
        spikes_detected = []

        detector = PriceMovementDetector(
            price_change_threshold=0.08,  # 8%
            on_spike=lambda s: spikes_detected.append(s),
        )

        detector.register_market("cond_1", "token_a", "token_b")

        # Build baseline at 0.50
        base_time = datetime.utcnow() - timedelta(minutes=1)
        for i in range(20):
            detector._trackers["token_a"].add_price(0.50, base_time + timedelta(seconds=i * 3))

        # Force baseline
        detector._trackers["token_a"].baseline_price = 0.50

        # Large movement (10% - above threshold)
        detector.update_price("token_a", 0.55)

        assert len(spikes_detected) == 1
        spike = spikes_detected[0]
        assert spike.direction == "up"
        assert spike.token_to_buy == "token_b"  # Buy opposite when price goes up

    def test_spike_direction_down(self):
        """Test spike detection for downward movement."""
        spikes_detected = []

        detector = PriceMovementDetector(
            price_change_threshold=0.08,
            on_spike=lambda s: spikes_detected.append(s),
        )

        detector.register_market("cond_1", "token_a", "token_b")

        # Build baseline at 0.50
        detector._trackers["token_a"].baseline_price = 0.50
        detector._trackers["token_a"].last_price = 0.50

        # Large downward movement
        detector.update_price("token_a", 0.45)  # -10%

        assert len(spikes_detected) == 1
        spike = spikes_detected[0]
        assert spike.direction == "down"
        assert spike.token_to_buy == "token_a"  # Buy same token when price drops

    def test_no_duplicate_spikes(self):
        """Test that same token doesn't trigger multiple spikes."""
        spikes_detected = []

        detector = PriceMovementDetector(
            price_change_threshold=0.08,
            on_spike=lambda s: spikes_detected.append(s),
        )

        detector.register_market("cond_1", "token_a", "token_b")
        detector._trackers["token_a"].baseline_price = 0.50

        # First spike
        detector.update_price("token_a", 0.55)
        assert len(spikes_detected) == 1

        # Another movement - should not trigger (already active)
        detector.update_price("token_a", 0.58)
        assert len(spikes_detected) == 1  # Still 1

    def test_clear_spike_allows_new_detection(self):
        """Test that clearing spike allows new detection."""
        spikes_detected = []

        detector = PriceMovementDetector(
            price_change_threshold=0.08,
            on_spike=lambda s: spikes_detected.append(s),
        )

        detector.register_market("cond_1", "token_a", "token_b")
        detector._trackers["token_a"].baseline_price = 0.50

        # First spike
        detector.update_price("token_a", 0.55)
        assert len(spikes_detected) == 1

        # Clear spike
        detector.clear_spike("token_a")

        # Reset baseline and trigger new spike
        detector._trackers["token_a"].baseline_price = 0.55
        detector.update_price("token_a", 0.60)
        assert len(spikes_detected) == 2


class TestPosition:
    """Test position management."""

    def test_add_entry_updates_totals(self):
        """Test that adding entry updates position totals."""
        spike = PriceSpike(
            condition_id="cond_1",
            token_id="token_a",
            price_before=0.50,
            price_after=0.55,
            price_change=0.10,
            direction="up",
            token_to_buy="token_b",
            target_price=0.525,
            stop_loss_price=0.5775,
        )

        position = Position(
            condition_id="cond_1",
            token_id="token_b",
            spike=spike,
            target_price=0.525,
            stop_loss_price=0.5775,
        )

        position.add_entry(0.45, 10.0)

        assert position.entries_made == 1
        assert position.total_size == 10.0
        assert position.avg_entry_price == 0.45

    def test_average_entry_price_calculation(self):
        """Test average entry price with multiple entries."""
        spike = PriceSpike(
            condition_id="cond_1",
            token_id="token_a",
            price_before=0.50,
            price_after=0.55,
            price_change=0.10,
            direction="up",
            token_to_buy="token_b",
            target_price=0.525,
            stop_loss_price=0.5775,
        )

        position = Position(
            condition_id="cond_1",
            token_id="token_b",
            spike=spike,
            target_price=0.525,
            stop_loss_price=0.5775,
        )

        # Entry 1: 10 @ 0.45
        position.add_entry(0.45, 10.0)
        # Entry 2: 10 @ 0.43
        position.add_entry(0.43, 10.0)
        # Entry 3: 10 @ 0.41
        position.add_entry(0.41, 10.0)

        assert position.entries_made == 3
        assert position.total_size == 30.0
        # Average: (0.45*10 + 0.43*10 + 0.41*10) / 30 = 0.43
        assert abs(position.avg_entry_price - 0.43) < 0.001

    def test_max_entries_limit(self):
        """Test that position respects max entries."""
        spike = PriceSpike(
            condition_id="cond_1",
            token_id="token_a",
            price_before=0.50,
            price_after=0.55,
            price_change=0.10,
            direction="up",
            token_to_buy="token_b",
            target_price=0.525,
            stop_loss_price=0.5775,
        )

        position = Position(
            condition_id="cond_1",
            token_id="token_b",
            spike=spike,
            target_price=0.525,
            stop_loss_price=0.5775,
            max_entries=3,
        )

        position.add_entry(0.45, 10.0)
        position.add_entry(0.43, 10.0)
        position.add_entry(0.41, 10.0)

        # Should not allow 4th entry
        assert position.should_add_entry(0.39) == False

    def test_exit_on_target_reached_up(self):
        """Test exit when target reached (price went up)."""
        spike = PriceSpike(
            condition_id="cond_1",
            token_id="token_a",
            price_before=0.50,
            price_after=0.55,
            price_change=0.10,
            direction="up",
            token_to_buy="token_b",
            target_price=0.525,  # 50% recovery target
            stop_loss_price=0.5775,
        )

        position = Position(
            condition_id="cond_1",
            token_id="token_b",
            spike=spike,
            target_price=0.525,
            stop_loss_price=0.5775,
        )

        position.add_entry(0.45, 10.0)

        # Price hasn't reached target yet
        should_exit, reason = position.check_exit(0.54)
        assert should_exit == False

        # Price reached target (came back down to 0.525)
        should_exit, reason = position.check_exit(0.52)
        assert should_exit == True
        assert reason == "target_reached"

    def test_exit_on_stop_loss(self):
        """Test exit on stop loss."""
        spike = PriceSpike(
            condition_id="cond_1",
            token_id="token_a",
            price_before=0.50,
            price_after=0.55,
            price_change=0.10,
            direction="up",
            token_to_buy="token_b",
            target_price=0.525,
            stop_loss_price=0.5775,  # 5% stop loss
        )

        position = Position(
            condition_id="cond_1",
            token_id="token_b",
            spike=spike,
            target_price=0.525,
            stop_loss_price=0.5775,
        )

        position.add_entry(0.45, 10.0)

        # Price moved against us (went up more)
        should_exit, reason = position.check_exit(0.58)
        assert should_exit == True
        assert reason == "stop_loss"

    def test_exit_on_timeout(self):
        """Test exit on timeout."""
        spike = PriceSpike(
            condition_id="cond_1",
            token_id="token_a",
            price_before=0.50,
            price_after=0.55,
            price_change=0.10,
            direction="up",
            token_to_buy="token_b",
            target_price=0.525,
            stop_loss_price=0.5775,
        )

        position = Position(
            condition_id="cond_1",
            token_id="token_b",
            spike=spike,
            target_price=0.525,
            stop_loss_price=0.5775,
            timeout_at=datetime.utcnow() - timedelta(minutes=1),  # Already expired
        )

        position.add_entry(0.45, 10.0)

        should_exit, reason = position.check_exit(0.53)
        assert should_exit == True
        assert reason == "timeout"


class TestIntegration:
    """Integration tests."""

    def test_full_flow_spike_to_position(self):
        """Test full flow from spike detection to position creation."""
        spikes = []

        detector = PriceMovementDetector(
            price_change_threshold=0.08,
            on_spike=lambda s: spikes.append(s),
        )

        detector.register_market("cond_1", "token_a", "token_b")

        # Build baseline
        detector._trackers["token_a"].baseline_price = 0.50

        # Trigger spike
        detector.update_price("token_a", 0.56)  # +12%

        assert len(spikes) == 1
        spike = spikes[0]

        # Create position from spike
        position = Position(
            condition_id=spike.condition_id,
            token_id=spike.token_to_buy,
            spike=spike,
            target_price=spike.target_price,
            stop_loss_price=spike.stop_loss_price,
        )

        # Add entries
        position.add_entry(0.44, 10.0)
        position.add_entry(0.42, 10.0)

        assert position.entries_made == 2
        assert position.total_size == 20.0

        # Check exit conditions
        should_exit, reason = position.check_exit(spike.target_price - 0.01)
        assert should_exit == True
        assert reason == "target_reached"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
