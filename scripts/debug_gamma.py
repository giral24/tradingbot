#!/usr/bin/env python3
"""Debug Gamma API response structure."""

import asyncio
import httpx
import json

async def main():
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://gamma-api.polymarket.com/markets",
            params={
                "closed": "false",
                "order": "volume24hr",
                "ascending": "false",
                "limit": 3,
            }
        )
        data = response.json()

        print("=== First 3 markets from Gamma API ===\n")
        for i, m in enumerate(data):
            print(f"Market {i+1}:")
            print(f"  question: {m.get('question', '')[:60]}")
            print(f"  conditionId: {m.get('conditionId')}")
            print(f"  active: {m.get('active')}")
            print(f"  closed: {m.get('closed')}")
            print(f"  acceptingOrders: {m.get('acceptingOrders')}")
            print(f"  outcomes: {m.get('outcomes')}")
            print(f"  clobTokenIds: {m.get('clobTokenIds')}")
            print(f"  volume24hr: {m.get('volume24hr')}")
            print(f"  liquidity: {m.get('liquidity')}")
            print()

            # Print all keys
            print(f"  All keys: {list(m.keys())}")
            print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
