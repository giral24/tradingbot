"""
CLOB REST API Client.

Wraps py-clob-client with:
- Retry with exponential backoff
- Rate limit handling
- DRY_RUN mode
- Structured logging
"""

import asyncio
from typing import Any

import structlog
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType as ClobOrderType

from src.config import settings
from src.clob.models import (
    Market,
    Token,
    OrderBook,
    OrderBookLevel,
    Order,
    OrderSide,
    OrderType,
)


class ClobApiError(Exception):
    """CLOB API error."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ClobApiClient:
    """
    Polymarket CLOB API client.

    Provides methods for:
    - Reading market data (no auth required)
    - Placing/canceling orders (auth required)

    All methods support retry with exponential backoff.
    """

    # Retry configuration
    MAX_RETRIES = 3
    BASE_DELAY = 1.0  # seconds
    MAX_DELAY = 30.0  # seconds

    # Rate limit tracking
    RATE_LIMIT_STATUS = 429

    def __init__(
        self,
        api_url: str | None = None,
        private_key: str | None = None,
        chain_id: int | None = None,
        dry_run: bool | None = None,
    ):
        """
        Initialize the CLOB client.

        Args:
            api_url: CLOB API URL (defaults to config)
            private_key: Wallet private key for signing (defaults to config)
            chain_id: Chain ID (defaults to config, 137 for Polygon)
            dry_run: If True, don't execute real orders (defaults to config)
        """
        self.api_url = api_url or settings.clob_api_url
        self.private_key = private_key or settings.private_key
        self.chain_id = chain_id or settings.chain_id
        self.dry_run = dry_run if dry_run is not None else settings.dry_run

        self.logger = structlog.get_logger(__name__)

        # Initialize client
        self._client: ClobClient | None = None
        self._authenticated = False

    def _get_client(self) -> ClobClient:
        """Get or create the CLOB client."""
        if self._client is None:
            if self.private_key:
                self._client = ClobClient(
                    self.api_url,
                    key=self.private_key,
                    chain_id=self.chain_id,
                )
                # Set API credentials for authenticated requests
                try:
                    creds = self._client.create_or_derive_api_creds()
                    self._client.set_api_creds(creds)
                    self._authenticated = True
                    self.logger.info("clob_client_authenticated")
                except Exception as e:
                    self.logger.warning(
                        "clob_auth_failed",
                        error=str(e),
                    )
            else:
                # Read-only client
                self._client = ClobClient(self.api_url)
                self.logger.info("clob_client_readonly")

        return self._client

    async def _retry_async(
        self,
        func: callable,
        *args,
        no_retry_on_404: bool = False,
        **kwargs,
    ) -> Any:
        """
        Execute function with retry and exponential backoff.

        Args:
            func: Sync function to call (will be run in executor)
            *args: Positional arguments
            no_retry_on_404: If True, don't retry on 404 errors
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            ClobApiError: If all retries fail
        """
        last_error = None
        delay = self.BASE_DELAY

        for attempt in range(self.MAX_RETRIES):
            try:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: func(*args, **kwargs),
                )
                return result

            except Exception as e:
                last_error = e
                error_str = str(e)

                # Don't retry on 404 if specified
                if no_retry_on_404 and ("404" in error_str or "No orderbook" in error_str):
                    raise ClobApiError(f"Not found: {error_str}", status_code=404)

                # Check for rate limit
                if "429" in error_str or "rate" in error_str.lower():
                    self.logger.warning(
                        "rate_limited",
                        attempt=attempt + 1,
                        delay=delay,
                    )
                else:
                    self.logger.warning(
                        "api_error_retrying",
                        attempt=attempt + 1,
                        error=error_str,
                        delay=delay,
                    )

                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self.MAX_DELAY)

        raise ClobApiError(f"Failed after {self.MAX_RETRIES} retries: {last_error}")

    # =========================================================================
    # Read Methods (No Auth Required)
    # =========================================================================

    async def get_markets(
        self,
        next_cursor: str | None = None,
    ) -> tuple[list[Market], str | None]:
        """
        Get paginated list of markets.

        Args:
            next_cursor: Pagination cursor

        Returns:
            Tuple of (markets, next_cursor)
        """
        client = self._get_client()

        def fetch():
            if next_cursor:
                return client.get_markets(next_cursor=next_cursor)
            return client.get_markets()

        result = await self._retry_async(fetch)

        markets = []
        for m in result.get("data", []):
            tokens = []
            for t in m.get("tokens", []):
                tokens.append(Token(
                    token_id=t.get("token_id", ""),
                    outcome=t.get("outcome", ""),
                    price=float(t.get("price", 0)) if t.get("price") else None,
                    winner=t.get("winner", False),
                ))

            markets.append(Market(
                condition_id=m.get("condition_id", ""),
                question_id=m.get("question_id", ""),
                tokens=tokens,
                question=m.get("question", ""),
                description=m.get("description", ""),
                end_date_iso=m.get("end_date_iso"),
                game_start_time=m.get("game_start_time"),
                active=m.get("active", True),
                closed=m.get("closed", False),
                archived=m.get("archived", False),
                accepting_orders=m.get("accepting_orders", True),
            ))

        return markets, result.get("next_cursor")

    async def get_all_markets(self) -> list[Market]:
        """
        Get all markets (handles pagination).

        Returns:
            List of all markets
        """
        all_markets = []
        cursor = None

        while True:
            markets, cursor = await self.get_markets(next_cursor=cursor)
            all_markets.extend(markets)

            self.logger.debug(
                "fetched_markets_page",
                count=len(markets),
                total=len(all_markets),
            )

            if not cursor:
                break

        return all_markets

    async def get_orderbook(self, token_id: str) -> OrderBook | None:
        """
        Get orderbook for a token.

        Args:
            token_id: The token ID

        Returns:
            OrderBook with bids and asks, or None if not found
        """
        client = self._get_client()

        try:
            result = await self._retry_async(
                client.get_order_book,
                token_id,
                no_retry_on_404=True,
            )
        except ClobApiError as e:
            # 404 means no orderbook exists
            if e.status_code == 404 or "404" in str(e) or "No orderbook" in str(e):
                return None
            raise

        # py-clob-client returns OrderBookSummary dataclass
        bids = [
            OrderBookLevel(
                price=float(level.price),
                size=float(level.size),
            )
            for level in (result.bids or [])
        ]

        asks = [
            OrderBookLevel(
                price=float(level.price),
                size=float(level.size),
            )
            for level in (result.asks or [])
        ]

        return OrderBook(
            token_id=token_id,
            bids=bids,
            asks=asks,
        )

    async def get_price(
        self,
        token_id: str,
        side: OrderSide = OrderSide.BUY,
    ) -> float | None:
        """
        Get current price for a token.

        Args:
            token_id: The token ID
            side: BUY or SELL

        Returns:
            Price or None if no liquidity
        """
        client = self._get_client()

        result = await self._retry_async(
            client.get_price,
            token_id,
            side.value.lower(),
        )

        price = result.get("price")
        return float(price) if price else None

    async def get_midpoint(self, token_id: str) -> float | None:
        """
        Get midpoint price for a token.

        Args:
            token_id: The token ID

        Returns:
            Midpoint price or None
        """
        client = self._get_client()

        result = await self._retry_async(
            client.get_midpoint,
            token_id,
        )

        mid = result.get("mid")
        return float(mid) if mid else None

    # =========================================================================
    # Write Methods (Auth Required)
    # =========================================================================

    def _check_auth(self) -> None:
        """Check if client is authenticated."""
        if not self._authenticated:
            raise ClobApiError("Client not authenticated. Set PRIVATE_KEY in config.")

    async def place_order(
        self,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
        order_type: OrderType = OrderType.GTC,
    ) -> Order | None:
        """
        Place a limit order.

        Args:
            token_id: Token to trade
            side: BUY or SELL
            price: Limit price (0.01 to 0.99)
            size: Size in shares
            order_type: GTC, GTD, FOK, or FAK

        Returns:
            Order object or None if DRY_RUN
        """
        self._check_auth()

        self.logger.info(
            "placing_order",
            token_id=token_id,
            side=side.value,
            price=price,
            size=size,
            order_type=order_type.value,
            dry_run=self.dry_run,
        )

        if self.dry_run:
            self.logger.info("dry_run_order_skipped")
            return None

        client = self._get_client()

        # Map order type
        clob_order_type = {
            OrderType.GTC: ClobOrderType.GTC,
            OrderType.GTD: ClobOrderType.GTD,
            OrderType.FOK: ClobOrderType.FOK,
            OrderType.FAK: ClobOrderType.FOK,  # FAK not in py-clob-client, use FOK
        }.get(order_type, ClobOrderType.GTC)

        # Create order
        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=side.value,
        )

        signed_order = client.create_order(order_args)
        result = await self._retry_async(
            client.post_order,
            signed_order,
            clob_order_type,
        )

        self.logger.info(
            "order_placed",
            order_id=result.get("orderID"),
            status=result.get("status"),
        )

        return Order(
            order_id=result.get("orderID", ""),
            market_id="",  # Not returned by API
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            order_type=order_type,
            status=result.get("status", "live"),
        )

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled
        """
        self._check_auth()

        self.logger.info(
            "cancelling_order",
            order_id=order_id,
            dry_run=self.dry_run,
        )

        if self.dry_run:
            self.logger.info("dry_run_cancel_skipped")
            return True

        client = self._get_client()

        result = await self._retry_async(
            client.cancel,
            order_id,
        )

        success = result.get("canceled", [])
        return order_id in success or len(success) > 0

    async def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.

        Returns:
            Number of orders cancelled
        """
        self._check_auth()

        self.logger.info(
            "cancelling_all_orders",
            dry_run=self.dry_run,
        )

        if self.dry_run:
            self.logger.info("dry_run_cancel_all_skipped")
            return 0

        client = self._get_client()

        result = await self._retry_async(
            client.cancel_all,
        )

        cancelled = result.get("canceled", [])
        self.logger.info("orders_cancelled", count=len(cancelled))

        return len(cancelled)

    async def get_open_orders(self) -> list[Order]:
        """
        Get all open orders.

        Returns:
            List of open orders
        """
        self._check_auth()
        client = self._get_client()

        result = await self._retry_async(
            client.get_orders,
        )

        orders = []
        for o in result:
            orders.append(Order(
                order_id=o.get("id", ""),
                market_id=o.get("market", ""),
                token_id=o.get("asset_id", ""),
                side=OrderSide(o.get("side", "BUY").upper()),
                price=float(o.get("price", 0)),
                size=float(o.get("original_size", 0)),
                filled_size=float(o.get("size_matched", 0)),
                status=o.get("status", "live"),
            ))

        return orders

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> dict[str, Any]:
        """
        Check API connectivity.

        Returns:
            Health status dict
        """
        client = self._get_client()

        try:
            result = await self._retry_async(client.get_ok)
            server_time = await self._retry_async(client.get_server_time)

            return {
                "ok": result == "OK",
                "server_time": server_time,
                "authenticated": self._authenticated,
                "dry_run": self.dry_run,
            }
        except Exception as e:
            return {
                "ok": False,
                "error": str(e),
                "authenticated": self._authenticated,
                "dry_run": self.dry_run,
            }
