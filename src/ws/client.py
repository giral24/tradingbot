"""
WebSocket Client for Polymarket CLOB.

Connects to wss://ws-subscriptions-clob.polymarket.com
Subscribes to market channel for real-time orderbook updates.
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Callable, Any
from datetime import datetime

import structlog
import websockets
from websockets.client import WebSocketClientProtocol

from src.config import settings


@dataclass
class OrderBookUpdate:
    """Parsed orderbook update from WebSocket."""

    asset_id: str  # Token ID
    market: str  # Condition ID
    timestamp: int  # Unix ms
    bids: list[tuple[float, float]]  # [(price, size), ...]
    asks: list[tuple[float, float]]  # [(price, size), ...]

    @property
    def best_bid(self) -> float | None:
        """Best bid price."""
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> float | None:
        """Best ask price."""
        return self.asks[0][0] if self.asks else None

    @property
    def best_bid_size(self) -> float | None:
        """Size at best bid."""
        return self.bids[0][1] if self.bids else None

    @property
    def best_ask_size(self) -> float | None:
        """Size at best ask."""
        return self.asks[0][1] if self.asks else None


# Callback type for orderbook updates
OrderBookCallback = Callable[[OrderBookUpdate], None]


class WebSocketClient:
    """
    WebSocket client for Polymarket market channel.

    Subscribes to token IDs and receives real-time orderbook updates.
    """

    WSS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    PING_INTERVAL = 10  # seconds
    RECONNECT_DELAY = 5  # seconds
    MAX_RECONNECT_DELAY = 60  # seconds

    def __init__(
        self,
        on_orderbook: OrderBookCallback | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        api_passphrase: str | None = None,
    ):
        """
        Initialize WebSocket client.

        Args:
            on_orderbook: Callback for orderbook updates
            api_key: API key (optional, for authenticated features)
            api_secret: API secret
            api_passphrase: API passphrase
        """
        self.on_orderbook = on_orderbook
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase

        self.logger = structlog.get_logger(__name__)

        # Connection state
        self._ws: WebSocketClientProtocol | None = None
        self._connected = False
        self._running = False
        self._reconnect_delay = self.RECONNECT_DELAY

        # Subscriptions
        self._subscribed_tokens: set[str] = set()
        self._pending_subscriptions: set[str] = set()

        # Tasks
        self._receive_task: asyncio.Task | None = None
        self._ping_task: asyncio.Task | None = None

        # Stats
        self._messages_received = 0
        self._last_message_time: datetime | None = None

    @property
    def connected(self) -> bool:
        """Check if connected."""
        return self._connected and self._ws is not None

    @property
    def subscribed_tokens(self) -> set[str]:
        """Get set of subscribed token IDs."""
        return self._subscribed_tokens.copy()

    async def connect(self) -> bool:
        """
        Connect to WebSocket server.

        Returns:
            True if connected successfully
        """
        if self._connected:
            return True

        try:
            self.logger.info("ws_connecting", url=self.WSS_URL)

            self._ws = await websockets.connect(
                self.WSS_URL,
                ping_interval=None,  # We handle pings manually
                ping_timeout=30,
                close_timeout=10,
            )

            self._connected = True
            self._reconnect_delay = self.RECONNECT_DELAY

            self.logger.info("ws_connected")
            return True

        except Exception as e:
            self.logger.error("ws_connect_failed", error=str(e))
            return False

    async def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        self._running = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            await self._ws.close()
            self._ws = None

        self._connected = False
        self._subscribed_tokens.clear()

        self.logger.info("ws_disconnected")

    async def subscribe(self, token_ids: list[str]) -> bool:
        """
        Subscribe to orderbook updates for tokens.

        Args:
            token_ids: List of token IDs to subscribe to

        Returns:
            True if subscription sent successfully
        """
        if not token_ids:
            return True

        if not self._connected or not self._ws:
            # Queue for when we connect
            self._pending_subscriptions.update(token_ids)
            self.logger.debug("ws_subscription_queued", count=len(token_ids))
            return False

        try:
            # Send subscription message
            message = {
                "assets_ids": token_ids,
                "type": "market",
            }

            await self._ws.send(json.dumps(message))

            self._subscribed_tokens.update(token_ids)
            self.logger.info(
                "ws_subscribed",
                count=len(token_ids),
                total=len(self._subscribed_tokens),
            )

            return True

        except Exception as e:
            self.logger.error("ws_subscribe_failed", error=str(e))
            return False

    async def unsubscribe(self, token_ids: list[str]) -> bool:
        """
        Unsubscribe from token updates.

        Note: Polymarket CLOB WebSocket does NOT support unsubscribe.
        We just remove from our tracking set and ignore future updates.
        The tokens will be fully removed on next reconnection.

        Args:
            token_ids: Token IDs to unsubscribe from

        Returns:
            True if removed from tracking
        """
        if not token_ids:
            return False

        # Just remove from our tracking - we can't actually unsubscribe
        # The WebSocket will still send updates but we'll ignore them
        self._subscribed_tokens -= set(token_ids)
        self.logger.info(
            "ws_unsubscribed_local",
            count=len(token_ids),
            remaining=len(self._subscribed_tokens),
            note="Polymarket does not support unsubscribe, will fully remove on reconnect",
        )

        return True

    async def run(self) -> None:
        """
        Run the WebSocket client with auto-reconnect.

        This method blocks until disconnect() is called.
        """
        self._running = True

        while self._running:
            try:
                # Connect
                if not await self.connect():
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(
                        self._reconnect_delay * 2,
                        self.MAX_RECONNECT_DELAY,
                    )
                    continue

                # Subscribe to pending tokens
                if self._pending_subscriptions:
                    await self.subscribe(list(self._pending_subscriptions))
                    self._pending_subscriptions.clear()

                # Start ping task
                self._ping_task = asyncio.create_task(self._ping_loop())

                # Receive messages
                await self._receive_loop()

            except websockets.ConnectionClosed as e:
                self.logger.warning("ws_connection_closed", code=e.code, reason=e.reason)
                self._connected = False

            except Exception as e:
                self.logger.error("ws_error", error=str(e))
                self._connected = False

            # Reconnect delay
            if self._running:
                self.logger.info("ws_reconnecting", delay=self._reconnect_delay)
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self.MAX_RECONNECT_DELAY,
                )

    async def _receive_loop(self) -> None:
        """Receive and process messages."""
        if not self._ws:
            return

        async for message in self._ws:
            if not self._running:
                break

            try:
                self._messages_received += 1
                self._last_message_time = datetime.utcnow()

                data = json.loads(message)
                await self._handle_message(data)

            except json.JSONDecodeError:
                self.logger.warning("ws_invalid_json", message=message[:100])
            except Exception as e:
                self.logger.error("ws_message_error", error=str(e))

    async def _handle_message(self, data: dict[str, Any] | list) -> None:
        """
        Handle incoming WebSocket message.

        Args:
            data: Parsed JSON message (can be dict or list of dicts)
        """
        # Handle list of messages (batch update)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    await self._handle_single_message(item)
            return

        await self._handle_single_message(data)

    async def _handle_single_message(self, data: dict[str, Any]) -> None:
        """Handle a single message dict."""
        event_type = data.get("event_type")

        if event_type == "book":
            await self._handle_orderbook(data)
        elif event_type == "price_change":
            await self._handle_price_change(data)
        elif event_type == "tick_size_change":
            pass
        else:
            self.logger.debug("ws_unknown_event", event_type=event_type, data_keys=list(data.keys())[:5])

    async def _handle_orderbook(self, data: dict[str, Any]) -> None:
        """
        Handle orderbook update message.

        Args:
            data: Orderbook update data
        """
        try:
            # Parse bids
            bids = []
            for b in data.get("bids", []):
                price = float(b.get("price", 0))
                size = float(b.get("size", 0))
                if price > 0 and size > 0:
                    bids.append((price, size))

            # Sort bids descending (best first)
            bids.sort(key=lambda x: x[0], reverse=True)

            # Parse asks
            asks = []
            for a in data.get("asks", []):
                price = float(a.get("price", 0))
                size = float(a.get("size", 0))
                if price > 0 and size > 0:
                    asks.append((price, size))

            # Sort asks ascending (best first)
            asks.sort(key=lambda x: x[0])

            # Create update object
            timestamp = data.get("timestamp")
            timestamp_int = int(timestamp) if timestamp is not None else 0

            update = OrderBookUpdate(
                asset_id=data.get("asset_id", ""),
                market=data.get("market", ""),
                timestamp=timestamp_int,
                bids=bids,
                asks=asks,
            )

            # Call callback
            if self.on_orderbook:
                self.on_orderbook(update)

        except Exception as e:
            self.logger.error("ws_orderbook_parse_error", error=str(e), data=data)

    async def _handle_price_change(self, data: dict[str, Any]) -> None:
        """
        Handle price change event.

        These events have price but no real bid/ask spread.
        We pass them with bid=None so the bot can filter them if needed.
        """
        try:
            price_changes = data.get("price_changes", [])
            market = data.get("market", "")
            timestamp = data.get("timestamp")
            timestamp_int = int(timestamp) if timestamp is not None else 0

            for change in price_changes:
                asset_id = change.get("asset_id", "")
                price = change.get("price")

                if not asset_id or price is None:
                    continue

                price_float = float(price)

                # Send with asks only (no bids) - bot will know this is price_change data
                update = OrderBookUpdate(
                    asset_id=asset_id,
                    market=market,
                    timestamp=timestamp_int,
                    bids=[],  # Empty bids = no bid data
                    asks=[(price_float, 0)] if price_float > 0 else [],
                )

                if self.on_orderbook:
                    self.on_orderbook(update)

        except Exception as e:
            self.logger.error("ws_price_change_error", error=str(e))

    async def _ping_loop(self) -> None:
        """Send periodic pings to keep connection alive."""
        while self._running and self._connected:
            try:
                await asyncio.sleep(self.PING_INTERVAL)

                if self._ws and self._connected:
                    await self._ws.ping()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning("ws_ping_failed", error=str(e))

    def get_stats(self) -> dict[str, Any]:
        """Get connection statistics."""
        return {
            "connected": self._connected,
            "subscribed_tokens": len(self._subscribed_tokens),
            "messages_received": self._messages_received,
            "last_message": self._last_message_time.isoformat() if self._last_message_time else None,
        }
