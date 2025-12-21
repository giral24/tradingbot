#!/usr/bin/env python3
"""Look up market details from condition_id or token_id."""
import sys
import httpx

def lookup_market(condition_id: str):
    """Get market info from condition_id."""
    url = f"https://clob.polymarket.com/markets/{condition_id}"
    try:
        resp = httpx.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"\n{'='*60}")
            print(f"Mercado: {data.get('question', 'N/A')}")
            print(f"Slug: {data.get('market_slug', 'N/A')}")
            print(f"URL: https://polymarket.com/event/{data.get('market_slug', '')}")
            print(f"Activo: {data.get('active', 'N/A')}")
            print(f"Cerrado: {data.get('closed', 'N/A')}")
            print(f"{'='*60}\n")
            return data
        else:
            print(f"Error: {resp.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python lookup_market.py <condition_id>")
        print("Ejemplo: python lookup_market.py 0x87a1615285594252cf...")
        sys.exit(1)

    condition_id = sys.argv[1]
    lookup_market(condition_id)
