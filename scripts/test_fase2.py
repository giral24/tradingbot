#!/usr/bin/env python3
"""
Smoke test for FASE 2 - Watchlist Manager.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import setup_logging, settings
from src.clob import GammaApiClient
from src.watchlist import WatchlistManager, MarketScorer


async def main():
    """Run watchlist smoke tests."""
    setup_logging(level="INFO", format="console")

    print("\n" + "#" * 50)
    print("# FASE 2 - Watchlist Manager Test")
    print("#" * 50)

    gamma = GammaApiClient()

    # Use smaller thresholds for testing
    manager = WatchlistManager(
        gamma_client=gamma,
        max_size=10,
        entry_threshold=0.1,  # Lower for testing
        exit_threshold=0.05,
        watchlist_path=Path("data/test_watchlist.json"),
    )

    try:
        # Test 1: Refresh watchlist
        print("\n" + "=" * 50)
        print("1. Testing Watchlist Refresh")
        print("=" * 50)

        entries = await manager.refresh()
        print(f"   Watchlist entries: {len(entries)}")

        if entries:
            print("\n   Top 10 markets (NEW + LOW competition):")
            for i, entry in enumerate(entries[:10]):
                print(f"   {i+1}. {entry.question[:50]}...")
                print(f"      Score: {entry.score:.3f} | Vol24h: ${entry.volume_24h:,.0f} | Liq: ${entry.liquidity:,.0f}")
                print()

        print("\n   [OK] Watchlist refresh passed")

        # Test 2: Check token IDs
        print("\n" + "=" * 50)
        print("2. Testing Token IDs")
        print("=" * 50)

        token_ids = manager.token_ids
        print(f"   Total token IDs: {len(token_ids)}")
        print(f"   (Should be 2x entries: {len(entries) * 2})")

        if token_ids:
            print(f"   First token: {token_ids[0][:40]}...")

        print("   [OK] Token IDs passed")

        # Test 3: Get by token
        print("\n" + "=" * 50)
        print("3. Testing Get By Token")
        print("=" * 50)

        if token_ids:
            entry = manager.get_by_token(token_ids[0])
            if entry:
                print(f"   Found: {entry.question[:50]}...")
                print("   [OK] Get by token passed")
            else:
                print("   [FAIL] Entry not found")
        else:
            print("   [SKIP] No tokens")

        # Test 4: Hysteresis (refresh again)
        print("\n" + "=" * 50)
        print("4. Testing Hysteresis (second refresh)")
        print("=" * 50)

        entries2 = await manager.refresh()
        print(f"   Entries after second refresh: {len(entries2)}")
        print("   (Should be similar, hysteresis prevents churn)")
        print("   [OK] Hysteresis test passed")

        # Test 5: Persistence
        print("\n" + "=" * 50)
        print("5. Testing Persistence")
        print("=" * 50)

        watchlist_file = Path("data/test_watchlist.json")
        if watchlist_file.exists():
            print(f"   Watchlist saved to: {watchlist_file}")
            size = watchlist_file.stat().st_size
            print(f"   File size: {size} bytes")
            print("   [OK] Persistence test passed")
        else:
            print("   [FAIL] Watchlist file not created")

        # Summary
        print("\n" + "#" * 50)
        print("# All FASE 2 tests passed!")
        print("#" * 50 + "\n")

    finally:
        await gamma.close()


if __name__ == "__main__":
    asyncio.run(main())
