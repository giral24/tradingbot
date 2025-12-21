"""
Gamma API Client for market discovery.

The Gamma API provides access to active markets with metadata.
Use this to find markets, then use CLOB API for orderbooks and trading.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from src.config import settings


def parse_json_field(value: Any) -> list:
    """Parse a field that might be a JSON string or already a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


@dataclass
class GammaMarket:
    """Market from Gamma API."""

    # Core identifiers
    condition_id: str
    question_id: str

    # Token IDs and outcomes (binary markets have 2 outcomes)
    # outcomes: ["Yes", "No"] or ["Trump", "Biden"] etc.
    # token_ids: corresponding CLOB token IDs
    outcomes: list[str]
    token_ids: list[str]

    # Metadata
    question: str
    description: str
    market_slug: str

    # Status
    active: bool
    closed: bool
    accepting_orders: bool

    # Metrics
    volume: float  # Total volume
    volume_24h: float  # 24h volume
    liquidity: float

    # Prices from Gamma API
    outcome_prices: list[float]  # Current prices for each outcome
    best_bid: float | None
    best_ask: float | None

    # Timing
    end_date_iso: str | None

    @property
    def is_binary(self) -> bool:
        """Check if this is a binary (2-outcome) market."""
        return len(self.token_ids) == 2

    @property
    def token_a_id(self) -> str | None:
        """Get first token ID (typically YES or first outcome)."""
        return self.token_ids[0] if self.token_ids else None

    @property
    def token_b_id(self) -> str | None:
        """Get second token ID (typically NO or second outcome)."""
        return self.token_ids[1] if len(self.token_ids) > 1 else None

    @property
    def outcome_a(self) -> str | None:
        """Get first outcome name."""
        return self.outcomes[0] if self.outcomes else None

    @property
    def outcome_b(self) -> str | None:
        """Get second outcome name."""
        return self.outcomes[1] if len(self.outcomes) > 1 else None


class GammaApiClient:
    """
    Client for Polymarket Gamma API.

    Used for discovering active markets with liquidity.
    """

    def __init__(self, api_url: str | None = None):
        self.api_url = api_url or settings.gamma_api_url
        self.logger = structlog.get_logger(__name__)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.api_url,
                timeout=30.0,
                http2=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_active_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "createdAt",  # Default: newest first
    ) -> list[GammaMarket]:
        """
        Get active markets from Gamma API.

        Args:
            limit: Max markets per page
            offset: Pagination offset
            order_by: Sort field (createdAt, volume24hr, liquidity)

        Returns:
            List of active markets
        """
        client = await self._get_client()

        params = {
            "closed": "false",  # Only active markets
            "order": order_by,  # Order by creation date (newest first)
            "ascending": "false",
            "limit": limit,
            "offset": offset,
        }

        try:
            response = await client.get("/markets", params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            self.logger.error("gamma_api_error", error=str(e))
            raise

        markets = []
        for m in data:
            # Parse JSON string fields
            outcomes = parse_json_field(m.get("outcomes"))
            token_ids = parse_json_field(m.get("clobTokenIds"))
            outcome_prices_raw = parse_json_field(m.get("outcomePrices"))

            # Parse outcome prices to floats
            outcome_prices = []
            for p in outcome_prices_raw:
                try:
                    outcome_prices.append(float(p) if p else 0.0)
                except (ValueError, TypeError):
                    outcome_prices.append(0.0)

            # Parse best bid/ask
            best_bid = None
            best_ask = None
            try:
                if m.get("bestBid"):
                    best_bid = float(m.get("bestBid"))
                if m.get("bestAsk"):
                    best_ask = float(m.get("bestAsk"))
            except (ValueError, TypeError):
                pass

            markets.append(GammaMarket(
                condition_id=m.get("conditionId", ""),
                question_id=m.get("questionID", ""),
                outcomes=outcomes,
                token_ids=token_ids,
                question=m.get("question", ""),
                description=m.get("description", ""),
                market_slug=m.get("slug", ""),
                active=m.get("active", False),
                closed=m.get("closed", False),
                accepting_orders=m.get("acceptingOrders", False),
                volume=float(m.get("volume", 0) or 0),
                volume_24h=float(m.get("volume24hr", 0) or 0),
                liquidity=float(m.get("liquidity", 0) or 0),
                outcome_prices=outcome_prices,
                best_bid=best_bid,
                best_ask=best_ask,
                end_date_iso=m.get("endDateIso"),
            ))

        self.logger.debug(
            "fetched_gamma_markets",
            count=len(markets),
            offset=offset,
        )

        return markets

    async def get_all_active_markets(
        self,
        max_markets: int = 500,
        order_by: str = "createdAt",
    ) -> list[GammaMarket]:
        """
        Get all active markets with pagination.

        Args:
            max_markets: Maximum total markets to fetch
            order_by: Sort field (createdAt for new markets)

        Returns:
            List of all active markets
        """
        all_markets = []
        offset = 0
        limit = 100

        while len(all_markets) < max_markets:
            markets = await self.get_active_markets(
                limit=limit,
                offset=offset,
                order_by=order_by,
            )

            if not markets:
                break

            all_markets.extend(markets)
            offset += limit

            # Small delay to avoid rate limits
            await asyncio.sleep(0.1)

        self.logger.info(
            "fetched_all_gamma_markets",
            total=len(all_markets),
        )

        return all_markets[:max_markets]

    async def get_events(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Get active events (groups of related markets).

        Args:
            limit: Max events per page
            offset: Pagination offset

        Returns:
            List of events with their markets
        """
        client = await self._get_client()

        params = {
            "closed": "false",
            "order": "id",
            "ascending": "false",
            "limit": limit,
            "offset": offset,
        }

        try:
            response = await client.get("/events", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error("gamma_events_error", error=str(e))
            raise
