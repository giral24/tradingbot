#!/usr/bin/env python3
"""
Smoke test for FASE 3 - WebSocket and Local Orderbook.

Tests:
1. WebSocket connection to Polymarket
2. Subscribe to market channel
3. Receive orderbook updates
4. Local orderbook state management
5. Arbitrage detection
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import setup_logging
from src.clob import GammaApiClient
from src.watchlist import WatchlistManager
from src.ws import WebSocketClient, LocalOrderbookManager, ArbitrageOpportunity


# Globals for callbacks
updates_received = 0
arbitrage_opportunities: list[ArbitrageOpportunity] = []


def on_arbitrage(opp: ArbitrageOpportunity) -> None:
    """Callback when arbitrage detected."""
    global arbitrage_opportunities
    arbitrage_opportunities.append(opp)
    print(f"\n   !!! ARBITRAGE DETECTED !!!")
    print(f"   Market: {opp.condition_id[:20]}...")
    print(f"   Ask A: {opp.ask_a:.4f} + Ask B: {opp.ask_b:.4f} = {opp.cost_per_pair:.4f}")
    print(f"   Spread: {opp.spread:.4f} ({opp.spread*100:.2f}%)")
    print(f"   Max size: {opp.max_size:.2f} | Max profit: ${opp.max_profit:.4f}")


async def main():
    """Run WebSocket smoke tests."""
    global updates_received

    setup_logging(level="INFO", format="console")

    print("\n" + "#" * 50)
    print("# FASE 3 - WebSocket & Local Orderbook Test")
    print("#" * 50)

    gamma = GammaApiClient()

    # Load watchlist
    manager = WatchlistManager(
        gamma_client=gamma,
        max_size=20,  # More markets for testing
        entry_threshold=0.1,
        exit_threshold=0.05,
        watchlist_path=Path("data/test_watchlist.json"),
    )

    try:
        # Test 1: Refresh watchlist to get markets
        print("\n" + "=" * 50)
        print("1. Loading Watchlist")
        print("=" * 50)

        entries = await manager.refresh()
        print(f"   Loaded {len(entries)} markets from watchlist")

        if not entries:
            print("   [FAIL] No markets in watchlist")
            return

        # Test 2: Setup local orderbook manager
        print("\n" + "=" * 50)
        print("2. Setting Up Local Orderbook Manager")
        print("=" * 50)

        orderbook_manager = LocalOrderbookManager(
            on_arbitrage=on_arbitrage,
            min_spread=0.0,  # Report any profit
            min_size=0.1,  # Low threshold for testing
        )

        # Register markets
        for entry in entries:
            orderbook_manager.register_market(
                condition_id=entry.condition_id,
                token_a_id=entry.token_a_id,
                token_b_id=entry.token_b_id,
            )

        print(f"   Registered {len(entries)} markets")
        print(f"   Tracking {len(orderbook_manager.token_ids)} tokens")
        print("   [OK] Orderbook manager ready")

        # Test 3: Connect WebSocket
        print("\n" + "=" * 50)
        print("3. Connecting to WebSocket")
        print("=" * 50)

        def on_orderbook(update):
            global updates_received
            updates_received += 1
            orderbook_manager.handle_update(update)

            # Print progress every 10 updates
            if updates_received % 10 == 0:
                print(f"   Received {updates_received} updates...")

        ws = WebSocketClient(on_orderbook=on_orderbook)

        # Connect
        connected = await ws.connect()
        if not connected:
            print("   [FAIL] Could not connect to WebSocket")
            return

        print("   [OK] Connected to WebSocket")

        # Test 4: Subscribe to tokens
        print("\n" + "=" * 50)
        print("4. Subscribing to Market Channel")
        print("=" * 50)

        token_ids = orderbook_manager.token_ids
        print(f"   Subscribing to {len(token_ids)} tokens...")

        # Subscribe in batches
        batch_size = 50
        for i in range(0, len(token_ids), batch_size):
            batch = token_ids[i:i + batch_size]
            await ws.subscribe(batch)
            print(f"   Batch {i//batch_size + 1}: {len(batch)} tokens")

        print("   [OK] Subscribed to all tokens")

        # Test 5: Listen for updates
        print("\n" + "=" * 50)
        print("5. Listening for Orderbook Updates (30 seconds)")
        print("=" * 50)
        print("   Waiting for updates...")
        print("   (Arbitrage opportunities will be shown if found)")

        # Run receiver in background
        receive_task = asyncio.create_task(ws.run())

        # Wait 30 seconds for updates
        try:
            await asyncio.wait_for(receive_task, timeout=30.0)
        except asyncio.TimeoutError:
            pass

        # Stop WebSocket
        await ws.disconnect()

        print(f"\n   Total updates received: {updates_received}")
        print(f"   Arbitrage opportunities: {len(arbitrage_opportunities)}")

        if updates_received > 0:
            print("   [OK] WebSocket receiving updates")
        else:
            print("   [WARN] No updates received (market might be slow)")

        # Test 6: Check orderbook state
        print("\n" + "=" * 50)
        print("6. Checking Orderbook State")
        print("=" * 50)

        stats = orderbook_manager.get_stats()
        print(f"   Markets: {stats['markets_registered']}")
        print(f"   Tokens: {stats['tokens_tracked']}")
        print(f"   Updates processed: {stats['updates_processed']}")
        print(f"   Arbitrage detected: {stats['arbitrage_detected']}")

        # Show sample market state
        if entries:
            state = orderbook_manager.get_market_state(entries[0].condition_id)
            if state:
                print(f"\n   Sample market: {entries[0].question[:40]}...")
                print(f"   Token A - Ask: {state['token_a']['best_ask']} | Bid: {state['token_a']['best_bid']}")
                print(f"   Token B - Ask: {state['token_b']['best_ask']} | Bid: {state['token_b']['best_bid']}")
                print(f"   Total Ask: {state['total_ask']:.4f}")
                print(f"   Arbitrage: {state['arbitrage']}")

        print("   [OK] Orderbook state check passed")

        # Test 7: Manual arbitrage scan
        print("\n" + "=" * 50)
        print("7. Manual Arbitrage Scan")
        print("=" * 50)

        opportunities = orderbook_manager.check_all_markets()
        print(f"   Found {len(opportunities)} arbitrage opportunities")

        if opportunities:
            for i, opp in enumerate(opportunities[:5]):
                print(f"\n   Opportunity {i+1}:")
                print(f"   Ask A + Ask B = {opp.ask_a:.4f} + {opp.ask_b:.4f} = {opp.cost_per_pair:.4f}")
                print(f"   Profit: {opp.spread*100:.2f}% (${opp.max_profit:.4f} max)")

        # Summary
        print("\n" + "#" * 50)
        print("# FASE 3 Test Complete!")
        print("#" * 50)
        print(f"\n   Updates: {updates_received}")
        print(f"   Arbitrage found: {len(arbitrage_opportunities)}")

        if updates_received > 0:
            print("\n   [OK] All tests passed!")
        else:
            print("\n   [WARN] No updates - verify market is active")

    finally:
        await gamma.close()


if __name__ == "__main__":
    asyncio.run(main())
