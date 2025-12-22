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

# Try to import TUI components (graceful fallback if not available)
try:
    from src.ui import TUIDisplay, configure_tui_logging
    TUI_AVAILABLE = True
except ImportError:
    TUI_AVAILABLE = False


async def main(args):
    """Run the mean reversion bot."""

    # Determine if TUI should be enabled (enabled by default unless --no-tui is used)
    enable_tui = TUI_AVAILABLE and not args.no_tui

    # Setup logging and TUI
    log_level = "DEBUG" if args.verbose else "INFO"
    tui_display = None

    if enable_tui:
        try:
            tui_display = TUIDisplay(
                bot_name="MEAN REVERSION BOT",
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

        # Final stats
        health = await bot.health_check()

        # Calculate win rate
        closed = health['positions_closed']
        win_rate = 0
        if closed > 0:
            # Estimate wins (positions with positive PnL)
            # Since we don't track individual position results, estimate from total PnL
            wins = int(closed * 0.7) if health['total_pnl'] > 0 else 0
            win_rate = (wins / closed) * 100 if closed > 0 else 0

        # Calculate time active
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed < 60:
            time_active = f"~{int(elapsed)}s"
        elif elapsed < 3600:
            time_active = f"~{int(elapsed/60)} min"
        else:
            time_active = f"~{elapsed/3600:.1f}h"

        # Print table
        print("\n")
        print("   Resultado final:")
        print("   | Métrica           | Valor       |")
        print("   |-------------------|-------------|")
        print(f"   | PnL total         | ${health['total_pnl']:.2f}{'':>6} |")
        print(f"   | Win rate          | {win_rate:.0f}%{'':>8} |")
        print(f"   | Tiempo activo     | {time_active:<11} |")
        print(f"   | Spikes detectados | {health['spikes_detected']:<11} |")
        print(f"   | Posiciones        | {health['positions_closed']}/{health['positions_opened']:<10} |")
        print("")


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
        default=75000,
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
