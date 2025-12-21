#!/usr/bin/env python3
"""
Run the Polymarket Arbitrage Bot.

Usage:
    python scripts/run_bot.py [--dry-run] [--trade-size 1.0] [--duration 3600]

Examples:
    # Dry run mode (no real trades)
    python scripts/run_bot.py --dry-run

    # Real trading with $1 trades
    python scripts/run_bot.py --trade-size 1.0

    # Run for 1 hour
    python scripts/run_bot.py --dry-run --duration 3600
"""

import argparse
import asyncio
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import setup_logging, settings
from src.bots.arb_intramarket import ArbIntramarketBot

# Try to import TUI components (graceful fallback if not available)
try:
    from src.ui import TUIDisplay, configure_tui_logging
    TUI_AVAILABLE = True
except ImportError:
    TUI_AVAILABLE = False


async def main(args):
    """Run the arbitrage bot."""

    # Determine if TUI should be enabled (enabled by default unless --no-tui is used)
    enable_tui = TUI_AVAILABLE and not args.no_tui

    # Setup logging and TUI
    log_level = "DEBUG" if args.verbose else "INFO"
    tui_display = None

    if enable_tui:
        try:
            tui_display = TUIDisplay(
                bot_name="ARBITRAGE BOT",
                bot_mode="DRY RUN" if args.dry_run else "LIVE",
                refresh_rate=0.5,
                max_log_lines=20,
            )

            # Configure logging to route to TUI
            configure_tui_logging(tui_display, level=log_level)

        except Exception as e:
            print(f"Failed to initialize TUI: {e}")
            print("Falling back to console logging...")
            setup_logging(level=log_level, format="console")
            tui_display = None
    else:
        # Use standard console logging
        setup_logging(level=log_level, format="console")

    # Show banner only if not using TUI
    if not tui_display:
        print("\n" + "=" * 60)
        print("   POLYMARKET ARBITRAGE BOT")
        print("=" * 60)
        print(f"\n   Mode: {'DRY RUN (no real trades)' if args.dry_run else 'LIVE TRADING'}")
        print(f"   Trade size: ${args.trade_size}")
        print(f"   Min spread: {args.min_spread * 100:.2f}%")
        print(f"   Duration: {args.duration}s" if args.duration else "   Duration: unlimited")
        print("\n" + "=" * 60 + "\n")

        if not args.dry_run:
            print("   ⚠️  WARNING: Live trading mode!")
            print("   Make sure you have configured your PRIVATE_KEY in .env")
            print()
            if not settings.private_key:
                print("   ERROR: PRIVATE_KEY not set. Please configure .env")
                return

    # Create bot
    bot = ArbIntramarketBot(
        trade_size=args.trade_size,
        min_spread=args.min_spread,
        watchlist_path=Path("data/watchlist.json"),
        dry_run=args.dry_run,
    )

    # Handle shutdown gracefully
    shutdown_event = asyncio.Event()

    def handle_signal(sig):
        print(f"\n\n   Received {sig.name}, shutting down...")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))

    try:
        # Initialize bot
        if not tui_display:
            print("   Initializing bot...")
        await bot.initialize()

        # Start TUI if enabled
        if tui_display:
            await tui_display.start(bot)
        else:
            health = await bot.health_check()
            print(f"   Watchlist: {health['watchlist_size']} markets")
            print(f"   Tokens: {health['tokens_subscribed']}")
            print("\n   Bot running. Press Ctrl+C to stop.\n")

        # Start bot
        bot._is_running = True

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
                bot.metrics.export_to_file("data/metrics.json")

            # Small wait to check shutdown flag
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=0.1)
                break
            except asyncio.TimeoutError:
                pass

    except (KeyboardInterrupt, asyncio.CancelledError):
        if not tui_display:
            print("\n\n   Received KeyboardInterrupt, shutting down...")
    except Exception as e:
        if not tui_display:
            print(f"\n   ERROR: {e}")
        raise

    finally:
        # Stop TUI first
        if tui_display:
            await tui_display.stop()

        # Shutdown
        if not tui_display:
            print("\n   Shutting down...")
        await bot.shutdown()

        # Final stats (show after TUI stops)
        health = await bot.health_check()
        print("\n" + "=" * 60)
        print("   FINAL STATISTICS")
        print("=" * 60)
        print(f"   Updates processed: {health['updates_processed']}")
        print(f"   Opportunities found: {health['opportunities_found']}")
        print(f"   Trades executed: {health['trades_executed']}")
        print(f"   Total profit: ${health['total_profit']:.4f}")
        print(f"   Open positions: {health['open_positions']}")
        print(f"   Blocked by risk: {health['blocked_by_risk']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Polymarket Arbitrage Bot",
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
        default=1.0,
        help="Trade size in dollars (default: 1.0)",
    )
    parser.add_argument(
        "--min-spread",
        type=float,
        default=0.001,
        help="Minimum spread to trade (default: 0.001 = 0.1%%)",
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
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Disable TUI and use standard console logging",
    )

    args = parser.parse_args()

    # Live flag overrides dry-run
    if args.live:
        args.dry_run = False

    asyncio.run(main(args))
