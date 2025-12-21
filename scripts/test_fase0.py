#!/usr/bin/env python3
"""
Smoke test for FASE 0.
Verifies that the project structure, config, logging, and runner work.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings, setup_logging, get_logger
from src.runner.registry import BotRegistry

# Import to register bots
import src.bots.arb_intramarket  # noqa: F401


def test_config():
    """Test configuration loading."""
    print("=" * 50)
    print("Testing Configuration")
    print("=" * 50)

    print(f"  Bot name: {settings.bot_name}")
    print(f"  DRY_RUN: {settings.dry_run}")
    print(f"  CLOB API URL: {settings.clob_api_url}")
    print(f"  Log level: {settings.log_level}")
    print(f"  Has credentials: {settings.has_credentials()}")
    print("  [OK] Configuration loaded successfully")


def test_logging():
    """Test structured logging."""
    print("\n" + "=" * 50)
    print("Testing Logging")
    print("=" * 50)

    setup_logging(level="DEBUG", format="console")
    logger = get_logger("test")

    logger.info("test_message", key="value", number=42)
    logger.debug("debug_message", nested={"a": 1, "b": 2})
    print("  [OK] Logging works correctly")


def test_registry():
    """Test bot registry."""
    print("\n" + "=" * 50)
    print("Testing Bot Registry")
    print("=" * 50)

    available = BotRegistry.list_bots()
    print(f"  Available bots: {available}")

    bot_class = BotRegistry.get("arb_intramarket")
    if bot_class is None:
        print("  [FAIL] arb_intramarket not registered!")
        return False

    print(f"  Got bot class: {bot_class.__name__}")
    print("  [OK] Bot registry works correctly")
    return True


async def test_bot_lifecycle():
    """Test bot initialization and shutdown."""
    print("\n" + "=" * 50)
    print("Testing Bot Lifecycle")
    print("=" * 50)

    bot_class = BotRegistry.get("arb_intramarket")
    bot = bot_class()

    print(f"  Bot name: {bot.name}")
    print(f"  Bot description: {bot.description}")
    print(f"  Is running: {bot.is_running}")
    print(f"  Is dry run: {bot.is_dry_run}")

    # Test start
    await bot.start()
    print(f"  After start - is_running: {bot.is_running}")

    # Test health check
    health = await bot.health_check()
    print(f"  Health check: {health}")

    # Test stop
    await bot.stop()
    print(f"  After stop - is_running: {bot.is_running}")

    print("  [OK] Bot lifecycle works correctly")


def main():
    """Run all tests."""
    print("\n" + "#" * 50)
    print("# FASE 0 - Smoke Test")
    print("#" * 50)

    test_config()
    test_logging()

    if not test_registry():
        sys.exit(1)

    asyncio.run(test_bot_lifecycle())

    print("\n" + "#" * 50)
    print("# All FASE 0 tests passed!")
    print("#" * 50 + "\n")


if __name__ == "__main__":
    main()
