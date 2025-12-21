"""
Main runner for the trading bot.
Handles bot selection, lifecycle, and signal handling.
"""

import asyncio
import signal
from typing import NoReturn

import structlog

from src.config import settings, setup_logging
from src.bots.base import BaseBot
from src.runner.registry import BotRegistry

# Import bots to register them
# This import has side effects (registration)
import src.bots.arb_intramarket  # noqa: F401


logger = structlog.get_logger(__name__)


class BotRunner:
    """
    Main runner that manages bot lifecycle.

    Handles:
    - Bot selection
    - Graceful shutdown on signals
    - Slow loop scheduling
    - Fast loop execution
    """

    def __init__(self, bot_name: str | None = None) -> None:
        """
        Initialize the runner.

        Args:
            bot_name: Name of the bot to run (uses config if not provided)
        """
        self.bot_name = bot_name or settings.bot_name
        self.bot: BaseBot | None = None
        self._shutdown_event = asyncio.Event()
        self._slow_loop_task: asyncio.Task | None = None
        self._fast_loop_task: asyncio.Task | None = None

    def _setup_signal_handlers(self) -> None:
        """Setup handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._handle_signal(s)),
            )

    async def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signal."""
        logger.info("shutdown_signal_received", signal=sig.name)
        self._shutdown_event.set()

    async def _slow_loop_scheduler(self) -> None:
        """Schedule the slow loop to run periodically."""
        interval = settings.watchlist_refresh_interval

        while not self._shutdown_event.is_set():
            try:
                logger.debug("slow_loop_starting")
                await self.bot.run_slow_loop()
                logger.debug("slow_loop_completed")
            except Exception as e:
                logger.error("slow_loop_error", error=str(e), exc_info=True)

            # Wait for interval or shutdown
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=interval,
                )
                # If we get here, shutdown was requested
                break
            except asyncio.TimeoutError:
                # Normal timeout, continue loop
                pass

    async def _fast_loop_runner(self) -> None:
        """Run the fast loop continuously."""
        while not self._shutdown_event.is_set():
            try:
                await self.bot.run_fast_loop()
            except asyncio.CancelledError:
                logger.info("fast_loop_cancelled")
                break
            except Exception as e:
                logger.error("fast_loop_error", error=str(e), exc_info=True)
                # Brief pause before retry
                await asyncio.sleep(1)

    async def run(self) -> None:
        """Run the selected bot."""
        # Setup logging
        setup_logging(
            level=settings.log_level,
            format=settings.log_format,
        )

        logger.info(
            "runner_starting",
            bot=self.bot_name,
            dry_run=settings.dry_run,
        )

        # Get the bot class
        bot_class = BotRegistry.get(self.bot_name)
        if bot_class is None:
            available = BotRegistry.list_bots()
            logger.error(
                "bot_not_found",
                requested=self.bot_name,
                available=available,
            )
            raise ValueError(
                f"Bot '{self.bot_name}' not found. Available: {available}"
            )

        # Create and initialize bot
        self.bot = bot_class()
        await self.bot.start()

        # Setup signal handlers
        self._setup_signal_handlers()

        try:
            # Run initial slow loop
            logger.info("running_initial_slow_loop")
            await self.bot.run_slow_loop()

            # Start both loops
            self._slow_loop_task = asyncio.create_task(
                self._slow_loop_scheduler(),
                name="slow_loop",
            )
            self._fast_loop_task = asyncio.create_task(
                self._fast_loop_runner(),
                name="fast_loop",
            )

            # Wait for shutdown signal
            await self._shutdown_event.wait()

        finally:
            # Cancel tasks
            logger.info("cancelling_tasks")

            if self._fast_loop_task:
                self._fast_loop_task.cancel()
                try:
                    await self._fast_loop_task
                except asyncio.CancelledError:
                    pass

            if self._slow_loop_task:
                self._slow_loop_task.cancel()
                try:
                    await self._slow_loop_task
                except asyncio.CancelledError:
                    pass

            # Stop bot
            if self.bot:
                await self.bot.stop()

            logger.info("runner_stopped")


def main() -> NoReturn:
    """Entry point for the bot."""
    runner = BotRunner()
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
