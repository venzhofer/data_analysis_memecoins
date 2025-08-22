#!/usr/bin/env python3
"""
Get Token Addresses
Fetches all token addresses from the first webhook
"""

import requests
import json

def get_token_addresses():
    """Get all token addresses from the first webhook"""
    
    tokens_webhook = "https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4"
    
    print("🔍 Fetching all token addresses...")
    print("=" * 50)
    
    try:
        response = requests.get(tokens_webhook, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Handle nested structure
        if isinstance(data, list) and len(data) > 0 and 'data' in data[0]:
            tokens = data[0]['data']
            print(f"Found nested structure with {len(tokens)} tokens")
        elif isinstance(data, list):
            tokens = data
        elif isinstance(data, dict) and 'data' in data:
            tokens = data['data']
        else:
            tokens = [data] if isinstance(data, dict) else []
        
        print(f"Total tokens found: {len(tokens)}")
        print("=" * 50)
        
        # Display all tokens with their addresses
        for i, token in enumerate(tokens):
            if isinstance(token, dict):
                # Handle both 'address' and 'adress' fields
                address = token.get('address') or token.get('adress')
                token_name = token.get('name', f'Token_{i}')
                
                if address:
                    print(f"{i+1:2d}. {token_name}")
                    print(f"     Address: {address}")
                    print(f"     Start FDV: {token.get('start_fdv', 'N/A')}")
                    print(f"     Created: {token.get('created_at', 'N/A')}")
                    print()
                else:
                    print(f"{i+1:2d}. {token_name} - NO ADDRESS FOUND")
                    print(f"     Available fields: {list(token.keys())}")
                    print()
        
        # Save all addresses to a file
        addresses = []
        for token in tokens:
            if isinstance(token, dict):
                address = token.get('address') or token.get('adress')
                token_name = token.get('name', 'Unknown')
                if address:
                    addresses.append({
                        'name': token_name,
                        'address': address,
                        'start_fdv': token.get('start_fdv', 0),
                        'created_at': token.get('created_at', '')
                    })
        
        # Save to file
        with open('all_token_addresses.json', 'w') as f:
            json.dump(addresses, f, indent=2, default=str)
        print(f"💾 Saved {len(addresses)} token addresses to: all_token_addresses.json")
        
        # Display just the addresses for easy copying
        print("\n" + "=" * 50)
        print("📋 TOKEN ADDRESSES (for easy copying):")
        print("=" * 50)
        for addr in addresses:
            print(f"{addr['address']}")
        
        print(f"\nTotal addresses: {len(addresses)}")
        
    except Exception as e:
        print(f"❌ Error fetching tokens: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    get_token_addresses()
