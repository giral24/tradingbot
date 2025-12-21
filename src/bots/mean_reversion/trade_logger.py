"""
Trade logger for verifying price movements.

Saves detailed price history for each trade to CSV files
so you can verify the bot's detections against real market data.
"""

import csv
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class TradeLog:
    """Log of a single trade with price history."""

    trade_id: str
    condition_id: str
    token_id: str
    market_question: str
    direction: str  # "up" or "down"

    # Entry info
    entry_time: datetime
    entry_price: float
    spike_change: float  # % change that triggered
    baseline_price: float

    # Price history: [(timestamp, price), ...]
    price_history: list = field(default_factory=list)

    # Exit info (filled when closed)
    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl: float | None = None


class TradeLogger:
    """
    Logs detailed price history for each trade.

    Creates a CSV file per trade in data/trade_logs/
    """

    def __init__(self, log_dir: str = "data/trade_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Active trades being logged
        self._active_logs: dict[str, TradeLog] = {}  # token_id -> TradeLog
        self._trade_counter = 0

    def start_trade(
        self,
        token_id: str,
        condition_id: str,
        direction: str,
        entry_price: float,
        baseline_price: float,
        spike_change: float,
        price_history: list,  # Recent price history from detector
        market_question: str = "",
    ) -> str:
        """
        Start logging a new trade.

        Args:
            token_id: Token being traded
            condition_id: Market condition ID
            direction: "up" or "down"
            entry_price: Price at entry
            baseline_price: Baseline price before spike
            spike_change: Percentage change that triggered
            price_history: List of (timestamp, price) from detector
            market_question: Market question for reference

        Returns:
            trade_id for reference
        """
        self._trade_counter += 1
        trade_id = f"trade_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{self._trade_counter}"

        log = TradeLog(
            trade_id=trade_id,
            condition_id=condition_id,
            token_id=token_id,
            market_question=market_question,
            direction=direction,
            entry_time=datetime.utcnow(),
            entry_price=entry_price,
            spike_change=spike_change,
            baseline_price=baseline_price,
            price_history=list(price_history),  # Copy
        )

        self._active_logs[token_id] = log
        return trade_id

    def add_price(self, token_id: str, price: float, timestamp: datetime | None = None) -> None:
        """Add a price point to an active trade log."""
        if token_id not in self._active_logs:
            return

        ts = timestamp or datetime.utcnow()
        self._active_logs[token_id].price_history.append((ts, price))

    def close_trade(
        self,
        token_id: str,
        exit_price: float,
        exit_reason: str,
        pnl: float,
    ) -> str | None:
        """
        Close a trade and save the CSV file.

        Returns:
            Path to saved CSV file, or None if not found
        """
        if token_id not in self._active_logs:
            return None

        log = self._active_logs.pop(token_id)
        log.exit_time = datetime.utcnow()
        log.exit_price = exit_price
        log.exit_reason = exit_reason
        log.pnl = pnl

        # Save to CSV
        return self._save_csv(log)

    def _save_csv(self, log: TradeLog) -> str:
        """Save trade log to CSV file."""
        filename = f"{log.trade_id}.csv"
        filepath = self.log_dir / filename

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header info
            writer.writerow(["# TRADE LOG"])
            writer.writerow(["trade_id", log.trade_id])
            writer.writerow(["condition_id", log.condition_id])
            writer.writerow(["token_id", log.token_id])
            writer.writerow(["market", log.market_question[:100] if log.market_question else ""])
            writer.writerow(["direction", log.direction])
            writer.writerow([])

            # Trade summary
            writer.writerow(["# TRADE SUMMARY"])
            writer.writerow(["entry_time", log.entry_time.isoformat()])
            writer.writerow(["exit_time", log.exit_time.isoformat() if log.exit_time else ""])
            writer.writerow(["baseline_price", f"{log.baseline_price:.4f}"])
            writer.writerow(["entry_price", f"{log.entry_price:.4f}"])
            writer.writerow(["exit_price", f"{log.exit_price:.4f}" if log.exit_price else ""])
            writer.writerow(["spike_change", f"{log.spike_change:.2%}"])
            writer.writerow(["exit_reason", log.exit_reason or ""])
            writer.writerow(["pnl", f"${log.pnl:.4f}" if log.pnl is not None else ""])
            writer.writerow([])

            # Price history
            writer.writerow(["# PRICE HISTORY (to verify against Polymarket data)"])
            writer.writerow(["timestamp_utc", "price"])
            for ts, price in log.price_history:
                if isinstance(ts, datetime):
                    ts_str = ts.isoformat()
                else:
                    ts_str = str(ts)
                writer.writerow([ts_str, f"{price:.4f}"])

        return str(filepath)
