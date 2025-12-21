#!/usr/bin/env python3
"""
Run the Mean Reversion Bot.

Usage:
    python scripts/run_mean_reversion.py [--dry-run] [--trade-size 10.0] [--duration 3600]

Examples:
    # Dry run mode (no real trades)
    python scripts/run_mean_reversion.py --dry-run

    # Real trading with $10 per entry
    python scripts/run_mean_reversion.py --trade-size 10.0

    # Run for 1 hour
    python scripts/run_mean_reversion.py --dry-run --duration 3600
"""

import argparse
import asyncio
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import setup_logging
from src.bots.mean_reversion import MeanReversionBot


async def main(args):
    """Run the mean reversion bot."""

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level, format="console")

    print("\n" + "=" * 60)
    print("   MEAN REVERSION BOT")
    print("=" * 60)
    print(f"\n   Mode: {'DRY RUN (no real trades)' if args.dry_run else 'LIVE TRADING'}")
    print(f"   Trade size: ${args.trade_size} per entry (max 3 entries)")
    print(f"   Price threshold: {args.threshold * 100:.1f}% movement")
    print(f"   Liquidity range: ${args.min_liquidity:,} - ${args.max_liquidity:,}")
    print(f"   Duration: {args.duration}s" if args.duration else "   Duration: unlimited")
    print("\n" + "=" * 60 + "\n")

    # Create bot
    bot = MeanReversionBot(
        trade_size=args.trade_size,
        min_liquidity=args.min_liquidity,
        max_liquidity=args.max_liquidity,
        price_change_threshold=args.threshold,
        dry_run=args.dry_run,
    )

    # Handle shutdown gracefully
    shutdown_event = asyncio.Event()

    def handle_signal(sig):
        print(f"\n\n   Received {sig.name}, shutting down...")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))

    try:
        # Initialize bot
        print("   Initializing bot...")
        await bot.initialize()

        health = await bot.health_check()
        print(f"   Tokens tracked: {health['tokens_tracked']}")
        print("\n   Bot running. Press Ctrl+C to stop.\n")

        # Start bot
        bot._running = True

        # Main loop
        start_time = asyncio.get_event_loop().time()

        while not shutdown_event.is_set():
            # Check duration limit
            if args.duration:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= args.duration:
                    print(f"\n   Duration limit ({args.duration}s) reached.")
                    break

            # Run loops
            await bot.run_slow_loop()
            await bot.run_fast_loop()

            # Export metrics periodically
            if bot.metrics and bot.metrics.should_export():
                bot.metrics.export_to_log()
                bot.metrics.export_to_file("data/metrics_mean_reversion.json")

            # Small wait to check shutdown flag
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=0.1)
                break
            except asyncio.TimeoutError:
                pass

    except Exception as e:
        print(f"\n   ERROR: {e}")
        raise

    finally:
        # Shutdown
        print("\n   Shutting down...")
        await bot.shutdown()

        # Final stats
        health = await bot.health_check()
        print("\n" + "=" * 60)
        print("   FINAL STATISTICS")
        print("=" * 60)
        print(f"   Spikes detected: {health['spikes_detected']}")
        print(f"   Positions opened: {health['positions_opened']}")
        print(f"   Positions closed: {health['positions_closed']}")
        print(f"   Total PnL: ${health['total_pnl']:.4f}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mean Reversion Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run without executing real trades (default: True)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (overrides --dry-run)",
    )
    parser.add_argument(
        "--trade-size",
        type=float,
        default=10.0,
        help="Trade size in dollars per entry (default: 10.0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.08,
        help="Price change threshold to trigger (default: 0.08 = 8%%)",
    )
    parser.add_argument(
        "--min-liquidity",
        type=float,
        default=1000,
        help="Minimum market liquidity (default: 1000)",
    )
    parser.add_argument(
        "--max-liquidity",
        type=float,
        default=50000,
        help="Maximum market liquidity (default: 50000)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Run duration in seconds (default: unlimited)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Live flag overrides dry-run
    if args.live:
        args.dry_run = False

    asyncio.run(main(args))
