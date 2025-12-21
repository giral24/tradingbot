#!/usr/bin/env python3
"""Debug script to understand market API responses."""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from py_clob_client.client import ClobClient

async def main():
    client = ClobClient("https://clob.polymarket.com")

    # Try get_simplified_markets
    print("=== Simplified Markets ===")
    try:
        markets = client.get_simplified_markets()
        print(f"Got {len(markets)} simplified markets")
        if markets:
            m = markets[0]
            print(f"First market keys: {m.keys() if isinstance(m, dict) else dir(m)}")
            print(f"First market: {m}")
    except Exception as e:
        print(f"Error: {e}")

    # Try get_markets
    print("\n=== Regular Markets ===")
    try:
        result = client.get_markets()
        data = result.get("data", [])
        print(f"Got {len(data)} regular markets")

        # Find one with tokens
        for m in data[:20]:
            tokens = m.get("tokens", [])
            if tokens and len(tokens) >= 2:
                print(f"\nMarket with tokens:")
                print(f"  Question: {m.get('question', '')[:60]}")
                print(f"  Condition ID: {m.get('condition_id')}")
                print(f"  Active: {m.get('active')}")
                print(f"  Accepting orders: {m.get('accepting_orders')}")
                print(f"  Tokens: {tokens}")

                # Try to get orderbook
                for t in tokens:
                    token_id = t.get("token_id")
                    if token_id:
                        try:
                            ob = client.get_order_book(token_id)
                            print(f"  Orderbook for {t.get('outcome')}: bids={len(ob.get('bids', []))}, asks={len(ob.get('asks', []))}")
                        except Exception as e:
                            print(f"  Orderbook error for {t.get('outcome')}: {e}")
                break
    except Exception as e:
        print(f"Error: {e}")

    # Try sampling markets
    print("\n=== Sampling Markets ===")
    try:
        result = client.get_sampling_markets()
        print(f"Got sampling result: {type(result)}")
        if isinstance(result, dict):
            print(f"Keys: {result.keys()}")
        elif isinstance(result, list) and result:
            print(f"First item: {result[0]}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
