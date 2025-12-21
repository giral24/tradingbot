"""
Watchlist manager with hysteresis.

Maintains a stable list of markets to monitor, avoiding
frequent add/remove cycles for borderline markets.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from src.config import settings
from src.clob.gamma_client import GammaMarket, GammaApiClient
from src.watchlist.scorer import MarketScorer, MarketScore


@dataclass
class WatchlistEntry:
    """Entry in the watchlist."""

    condition_id: str
    token_a_id: str
    token_b_id: str
    question: str
    outcomes: list[str]

    # Scoring
    score: float
    volume_24h: float
    liquidity: float

    # Tracking
    added_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "condition_id": self.condition_id,
            "token_a_id": self.token_a_id,
            "token_b_id": self.token_b_id,
            "question": self.question,
            "outcomes": self.outcomes,
            "score": self.score,
            "volume_24h": self.volume_24h,
            "liquidity": self.liquidity,
            "added_at": self.added_at.isoformat(),
            "last_seen": self.last_seen.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WatchlistEntry":
        """Create from dictionary."""
        return cls(
            condition_id=data["condition_id"],
            token_a_id=data["token_a_id"],
            token_b_id=data["token_b_id"],
            question=data["question"],
            outcomes=data["outcomes"],
            score=data["score"],
            volume_24h=data["volume_24h"],
            liquidity=data["liquidity"],
            added_at=datetime.fromisoformat(data["added_at"]),
            last_seen=datetime.fromisoformat(data["last_seen"]),
        )

    @classmethod
    def from_market_score(cls, ms: MarketScore) -> "WatchlistEntry":
        """Create from MarketScore."""
        return cls(
            condition_id=ms.market.condition_id,
            token_a_id=ms.market.token_a_id or "",
            token_b_id=ms.market.token_b_id or "",
            question=ms.market.question,
            outcomes=ms.market.outcomes,
            score=ms.total_score,
            volume_24h=ms.market.volume_24h,
            liquidity=ms.market.liquidity,
        )


class WatchlistManager:
    """
    Manages the watchlist with hysteresis.

    Hysteresis prevents frequent add/remove cycles:
    - To ENTER watchlist: score must be >= entry_threshold
    - To EXIT watchlist: score must be < exit_threshold
    - exit_threshold < entry_threshold creates a stable zone
    """

    def __init__(
        self,
        gamma_client: GammaApiClient,
        scorer: MarketScorer | None = None,
        max_size: int | None = None,
        entry_threshold: float | None = None,
        exit_threshold: float | None = None,
        watchlist_path: Path | None = None,
    ):
        self.gamma = gamma_client
        self.scorer = scorer or MarketScorer()
        self.max_size = max_size or settings.watchlist_max_size
        self.entry_threshold = entry_threshold or settings.watchlist_entry_threshold
        self.exit_threshold = exit_threshold or settings.watchlist_exit_threshold
        self.watchlist_path = watchlist_path or settings.watchlist_path

        self.logger = structlog.get_logger(__name__)

        # Current watchlist: condition_id -> WatchlistEntry
        self._watchlist: dict[str, WatchlistEntry] = {}

        # Load persisted watchlist
        self._load()

    def _load(self) -> None:
        """Load watchlist from disk."""
        if not self.watchlist_path.exists():
            self.logger.info("no_existing_watchlist")
            return

        try:
            with open(self.watchlist_path) as f:
                data = json.load(f)

            for entry_data in data.get("entries", []):
                entry = WatchlistEntry.from_dict(entry_data)
                self._watchlist[entry.condition_id] = entry

            self.logger.info(
                "watchlist_loaded",
                count=len(self._watchlist),
            )
        except Exception as e:
            self.logger.error("watchlist_load_error", error=str(e))

    def _save(self) -> None:
        """Save watchlist to disk."""
        self.watchlist_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "updated_at": datetime.utcnow().isoformat(),
            "count": len(self._watchlist),
            "entries": [e.to_dict() for e in self._watchlist.values()],
        }

        with open(self.watchlist_path, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.debug("watchlist_saved", count=len(self._watchlist))

    async def refresh(self) -> list[WatchlistEntry]:
        """
        Refresh the watchlist from Gamma API.

        Returns:
            Current watchlist entries
        """
        self.logger.info("refreshing_watchlist")

        # Fetch active markets
        markets = await self.gamma.get_all_active_markets(
            max_markets=self.max_size * 3,  # Fetch more than needed for scoring
        )

        # Filter to binary markets accepting orders
        binary_markets = [
            m for m in markets
            if m.is_binary and m.accepting_orders
        ]

        self.logger.debug(
            "markets_fetched",
            total=len(markets),
            binary=len(binary_markets),
        )

        # Score all markets
        scores = self.scorer.score_markets(binary_markets)

        # Build new watchlist with hysteresis
        now = datetime.utcnow()
        new_watchlist: dict[str, WatchlistEntry] = {}
        added = 0
        removed = 0
        kept = 0

        for ms in scores:
            cid = ms.market.condition_id

            # Already in watchlist?
            if cid in self._watchlist:
                # Check exit threshold
                if ms.total_score >= self.exit_threshold:
                    # Keep in watchlist, update score
                    entry = self._watchlist[cid]
                    entry.score = ms.total_score
                    entry.volume_24h = ms.market.volume_24h
                    entry.liquidity = ms.market.liquidity
                    entry.last_seen = now
                    new_watchlist[cid] = entry
                    kept += 1
                else:
                    # Score too low, remove
                    removed += 1
                    self.logger.debug(
                        "market_removed",
                        question=ms.market.question[:40],
                        score=ms.total_score,
                    )
            else:
                # Not in watchlist, check entry threshold
                if (
                    ms.total_score >= self.entry_threshold
                    and len(new_watchlist) < self.max_size
                ):
                    # Add to watchlist
                    new_watchlist[cid] = WatchlistEntry.from_market_score(ms)
                    added += 1
                    self.logger.debug(
                        "market_added",
                        question=ms.market.question[:40],
                        score=ms.total_score,
                    )

            # Stop if we have enough
            if len(new_watchlist) >= self.max_size:
                break

        # Update watchlist
        self._watchlist = new_watchlist
        self._save()

        self.logger.info(
            "watchlist_refreshed",
            total=len(self._watchlist),
            added=added,
            removed=removed,
            kept=kept,
        )

        return list(self._watchlist.values())

    @property
    def entries(self) -> list[WatchlistEntry]:
        """Get current watchlist entries."""
        return list(self._watchlist.values())

    @property
    def token_ids(self) -> list[str]:
        """Get all token IDs in watchlist (for WebSocket subscriptions)."""
        ids = []
        for entry in self._watchlist.values():
            if entry.token_a_id:
                ids.append(entry.token_a_id)
            if entry.token_b_id:
                ids.append(entry.token_b_id)
        return ids

    def get_entry(self, condition_id: str) -> WatchlistEntry | None:
        """Get entry by condition ID."""
        return self._watchlist.get(condition_id)

    def get_by_token(self, token_id: str) -> WatchlistEntry | None:
        """Get entry by token ID."""
        for entry in self._watchlist.values():
            if entry.token_a_id == token_id or entry.token_b_id == token_id:
                return entry
        return None
