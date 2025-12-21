#!/usr/bin/env python3
"""
Smoke test for FASE 4 - Full Arbitrage Bot.

Runs the complete bot in DRY_RUN mode for 60 seconds.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import setup_logging
from src.bots.arb_intramarket import ArbIntramarketBot


async def main():
    """Run the full arbitrage bot test."""
    setup_logging(level="INFO", format="console")

    print("\n" + "#" * 50)
    print("# FASE 4 - Full Arbitrage Bot Test")
    print("#" * 50)
    print("\n   Running in DRY_RUN mode for 60 seconds...")
    print("   Watching for arbitrage opportunities...\n")

    # Create bot in dry run mode
    bot = ArbIntramarketBot(
        trade_size=1.0,  # $1 trades
        min_spread=0.0,  # Report any profit (for testing)
        watchlist_path=Path("data/test_watchlist.json"),
        dry_run=True,
    )

    try:
        # Initialize
        print("=" * 50)
        print("1. Initializing Bot")
        print("=" * 50)

        await bot.initialize()

        health = await bot.health_check()
        print(f"   Watchlist: {health['watchlist_size']} markets")
        print(f"   Dry run: {health['dry_run']}")
        print("   [OK] Bot initialized")

        # Run for 60 seconds
        print("\n" + "=" * 50)
        print("2. Running Bot (60 seconds)")
        print("=" * 50)
        print("   Monitoring orderbooks via WebSocket...")
        print("   Any arbitrage opportunities will be logged.\n")

        # Start bot loops
        bot._is_running = True

        # Run slow loop once
        await bot.run_slow_loop()

        # Run fast loop for 60 seconds
        end_time = asyncio.get_event_loop().time() + 60

        while asyncio.get_event_loop().time() < end_time:
            await bot.run_fast_loop()

            # Print status every 10 seconds
            elapsed = 60 - (end_time - asyncio.get_event_loop().time())
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                health = await bot.health_check()
                print(f"   [{int(elapsed)}s] Updates: {health['updates_processed']} | "
                      f"Opportunities: {health['opportunities_found']} | "
                      f"WS: {'connected' if health['ws_connected'] else 'disconnected'}")

        # Final stats
        print("\n" + "=" * 50)
        print("3. Final Statistics")
        print("=" * 50)

        health = await bot.health_check()
        print(f"   Markets monitored: {health['watchlist_size']}")
        print(f"   Tokens subscribed: {health['tokens_subscribed']}")
        print(f"   Updates processed: {health['updates_processed']}")
        print(f"   Opportunities found: {health['opportunities_found']}")
        print(f"   Trades executed (dry): {health['trades_executed']}")
        print(f"   Total profit (simulated): ${health['total_profit']:.4f}")

        # Summary
        print("\n" + "#" * 50)
        print("# FASE 4 Test Complete!")
        print("#" * 50)

        if health['updates_processed'] > 0:
            print("\n   [OK] Bot running successfully!")
            if health['opportunities_found'] == 0:
                print("   (No arbitrage found - markets are efficient right now)")
        else:
            print("\n   [WARN] No updates received - check WebSocket")

    finally:
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
