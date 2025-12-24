#!/usr/bin/env python3
"""Test Polymarket authentication."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from py_clob_client.client import ClobClient
from src.config import settings

print("=" * 50)
print("Testing Polymarket Authentication")
print("=" * 50)

print(f"\nPrivate Key: {settings.private_key[:10]}...{settings.private_key[-6:]}")
print(f"Chain ID: {settings.chain_id}")
print(f"API URL: {settings.clob_api_url}")

# First, get the wallet address from private key
from eth_account import Account
account = Account.from_key(settings.private_key)
wallet_address = account.address
print(f"Wallet Address: {wallet_address}")

# For Magic.Link / email login, we need signature_type=1 and funder
# The funder is the proxy address that holds your funds
print("\nTrying signature_type=1 (Magic/Email login)...")

try:
    client = ClobClient(
        settings.clob_api_url,
        key=settings.private_key,
        chain_id=settings.chain_id,
        signature_type=1,  # 1 = Magic/Email wallet
        funder=wallet_address,  # The proxy address
    )
    print("\n[OK] ClobClient created with signature_type=1")

    # Try to derive API credentials
    print("\nDeriving API credentials...")
    creds = client.create_or_derive_api_creds()
    print(f"[OK] API Key: {creds.api_key[:20]}...")

    # Set credentials
    client.set_api_creds(creds)
    print("[OK] Credentials set")

    # Test if we can create and sign an order (proves auth works)
    print("\nTesting order signing...")
    from py_clob_client.clob_types import OrderArgs

    test_order_args = OrderArgs(
        token_id="21742633143463906290569050155826241533067272736897614950488156847949938836455",
        price=0.50,
        size=1.0,
        side="BUY",
    )

    try:
        signed_order = client.create_order(test_order_args)
        # SignedOrder is an object, not a dict
        print(f"[OK] Order signing works!")
        print(f"    Signature: {signed_order.signature[:30]}...")

        print("\n" + "=" * 50)
        print("SUCCESS - Authentication working!")
        print("You can place orders.")
        print("=" * 50)
        print(f"\nWallet: {wallet_address}")
        print("Check balance at: https://polygonscan.com/address/" + wallet_address)
    except Exception as e:
        print(f"[ERROR] Order creation failed: {e}")

except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
    print("\nPossible causes:")
    print("1. Private key is not from Polymarket wallet")
    print("2. Wallet not registered/activated on Polymarket")
    print("3. Network issues")
    print("\nMake sure you exported the key from:")
    print("Polymarket.com > Wallet > ⋮ > Export Private Key")
