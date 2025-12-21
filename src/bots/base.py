"""
Abstract base class for all trading bots.
All bot implementations must inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Any

import structlog

from src.config import settings, Settings


class BaseBot(ABC):
    """
    Abstract base class for trading bots.

    Each bot implementation must:
    1. Implement the abstract methods
    2. Handle its own slow loop (watchlist management)
    3. Handle its own fast loop (signal detection + execution)
    4. Respect DRY_RUN mode
    """

    def __init__(
        self,
        config: Settings | None = None,
        dry_run: bool | None = None,
    ) -> None:
        """
        Initialize the bot.

        Args:
            config: Settings instance (uses global settings if not provided)
            dry_run: Override dry_run setting (uses config value if not provided)
        """
        self.config = config or settings
        self._dry_run_override = dry_run
        self.logger = structlog.get_logger(self.__class__.__name__)
        self._running = False
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the bot's unique identifier."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of what this bot does."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the bot's resources.

        This is called once before the bot starts running.
        Use this to:
        - Connect to APIs
        - Load persisted state
        - Initialize data structures
        """
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Clean up the bot's resources.

        This is called when the bot stops.
        Use this to:
        - Close connections
        - Persist state
        - Cancel pending orders (if not DRY_RUN)
        """
        ...

    @abstractmethod
    async def run_slow_loop(self) -> None:
        """
        Execute the slow loop once.

        This handles:
        - Fetching market data
        - Updating the watchlist
        - Calculating scores

        Called periodically (e.g., every 5 minutes).
        """
        ...

    @abstractmethod
    async def run_fast_loop(self) -> None:
        """
        Execute the fast loop.

        This handles:
        - WebSocket subscriptions
        - Real-time orderbook updates
        - Signal detection
        - Order execution

        This is the main event loop that runs continuously.
        """
        ...

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Return the bot's health status.

        Returns:
            Dictionary with health metrics like:
            - is_running: bool
            - last_slow_loop: datetime
            - open_orders: int
            - etc.
        """
        ...

    async def start(self) -> None:
        """Start the bot."""
        if self._running:
            self.logger.warning("bot_already_running")
            return

        self.logger.info(
            "bot_starting",
            bot=self.name,
            dry_run=self.config.dry_run,
        )

        if not self._initialized:
            await self.initialize()
            self._initialized = True

        self._running = True
        self.logger.info("bot_started", bot=self.name)

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        if not self._running:
            self.logger.warning("bot_not_running")
            return

        self.logger.info("bot_stopping", bot=self.name)
        self._running = False
        await self.shutdown()
        self.logger.info("bot_stopped", bot=self.name)

    @property
    def is_running(self) -> bool:
        """Check if the bot is currently running."""
        return self._running

    @property
    def is_dry_run(self) -> bool:
        """Check if the bot is in dry run mode."""
        if self._dry_run_override is not None:
            return self._dry_run_override
        return self.config.dry_run
