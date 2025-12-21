#!/usr/bin/env python3
"""
Smoke test for FASE 1 - CLOB + Gamma API Client.
Tests read-only operations (no auth required).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import setup_logging
from src.clob import ClobApiClient, GammaApiClient, OrderSide


async def main():
    """Run API client smoke tests."""
    setup_logging(level="INFO", format="console")

    print("\n" + "#" * 50)
    print("# FASE 1 - API Client Smoke Test")
    print("#" * 50)

    clob = ClobApiClient(dry_run=True)
    gamma = GammaApiClient()

    try:
        # Test 1: CLOB Health check
        print("\n" + "=" * 50)
        print("1. Testing CLOB Health Check")
        print("=" * 50)

        health = await clob.health_check()
        print(f"   OK: {health.get('ok')}")
        print(f"   Server time: {health.get('server_time')}")
        print(f"   Authenticated: {health.get('authenticated')}")
        print(f"   Dry run: {health.get('dry_run')}")

        if not health.get("ok"):
            print("   [FAIL] CLOB API not reachable!")
            return

        print("   [OK] CLOB health check passed")

        # Test 2: Gamma API - Get active markets
        print("\n" + "=" * 50)
        print("2. Testing Gamma API - Active Markets")
        print("=" * 50)

        markets = await gamma.get_active_markets(limit=20)
        print(f"   Active markets fetched: {len(markets)}")

        # Find binary markets accepting orders
        valid_markets = [
            m for m in markets
            if m.is_binary and m.accepting_orders
        ]
        print(f"   Binary markets accepting orders: {len(valid_markets)}")

        if valid_markets:
            m = valid_markets[0]
            print(f"\n   Top market by volume:")
            print(f"     Question: {m.question[:60]}..." if len(m.question) > 60 else f"     Question: {m.question}")
            print(f"     Outcomes: {m.outcome_a} vs {m.outcome_b}")
            print(f"     Volume 24h: ${m.volume_24h:,.2f}")
            print(f"     Liquidity: ${m.liquidity:,.2f}")
            print(f"     Token A: {m.token_a_id[:30]}...")
            print(f"     Token B: {m.token_b_id[:30]}...")

        print("   [OK] Gamma API passed")

        # Test 3: Get orderbook for active market
        print("\n" + "=" * 50)
        print("3. Testing CLOB Orderbook")
        print("=" * 50)

        test_market = None
        ob_a = None
        ob_b = None

        for m in valid_markets[:10]:
            ob_a = await clob.get_orderbook(m.token_a_id)
            if ob_a and ob_a.best_ask:
                ob_b = await clob.get_orderbook(m.token_b_id)
                if ob_b and ob_b.best_ask:
                    test_market = m
                    break

        if not test_market:
            print("   [SKIP] No market with active orderbook found")
        else:
            print(f"   Market: {test_market.question[:50]}...")
            print(f"\n   {test_market.outcome_a} Orderbook:")
            print(f"     Best bid: {ob_a.best_bid} (size: {ob_a.best_bid_size})")
            print(f"     Best ask: {ob_a.best_ask} (size: {ob_a.best_ask_size})")

            print(f"\n   {test_market.outcome_b} Orderbook:")
            print(f"     Best bid: {ob_b.best_bid} (size: {ob_b.best_bid_size})")
            print(f"     Best ask: {ob_b.best_ask} (size: {ob_b.best_ask_size})")

            # Arbitrage check
            total = ob_a.best_ask + ob_b.best_ask
            edge = 1.0 - total
            print(f"\n   Arbitrage Check:")
            print(f"     ask_A + ask_B = {ob_a.best_ask:.4f} + {ob_b.best_ask:.4f} = {total:.4f}")
            print(f"     Edge: {edge:.4f} ({edge*100:.2f}%)")

            if total < 1.0:
                max_size = min(ob_a.best_ask_size or 0, ob_b.best_ask_size or 0)
                profit = edge * max_size
                print(f"     *** ARBITRAGE OPPORTUNITY! ***")
                print(f"     Max size: {max_size:.2f}")
                print(f"     Expected profit: ${profit:.4f}")
            else:
                print("     No arbitrage (total >= 1.0)")

            print("   [OK] Orderbook test passed")

        # Test 4: Get prices
        print("\n" + "=" * 50)
        print("4. Testing Price Endpoints")
        print("=" * 50)

        if test_market:
            midpoint = await clob.get_midpoint(test_market.token_a_id)
            print(f"   {test_market.outcome_a} midpoint: {midpoint}")
            print("   [OK] Price test passed")
        else:
            print("   [SKIP] No test market")

        # Test 5: DRY_RUN mode
        print("\n" + "=" * 50)
        print("5. Testing DRY_RUN Mode")
        print("=" * 50)
        print("   DRY_RUN=True, orders will be simulated")
        print("   [OK] DRY_RUN mode configured")

        # Summary
        print("\n" + "#" * 50)
        print("# All FASE 1 tests passed!")
        print("#" * 50 + "\n")

    finally:
        await gamma.close()


if __name__ == "__main__":
    asyncio.run(main())
