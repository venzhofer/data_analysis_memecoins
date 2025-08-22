#!/usr/bin/env python3
"""
Debug Webhook Structure
Examines the actual structure of webhook data to understand how to extract price timeseries
"""

import requests
import json
from pathlib import Path

def debug_webhook_structure():
    """Debug the structure of webhook data"""
    
    # Webhook endpoints
    tokens_webhook = "https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4"
    trading_webhook = "https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0"
    
    print("🔍 Debugging Webhook Data Structure...")
    print("=" * 50)
    
    # 1. Check tokens webhook
    print("\n📊 TOKENS WEBHOOK STRUCTURE:")
    print("-" * 30)
    try:
        response = requests.get(tokens_webhook, timeout=30)
        response.raise_for_status()
        tokens_data = response.json()
        
        print(f"Response type: {type(tokens_data)}")
        print(f"Response length: {len(tokens_data) if isinstance(tokens_data, list) else 'N/A'}")
        
        if isinstance(tokens_data, list) and len(tokens_data) > 0:
            print(f"First item type: {type(tokens_data[0])}")
            print(f"First item keys: {list(tokens_data[0].keys()) if isinstance(tokens_data[0], dict) else 'N/A'}")
            
            # Show first token structure
            if isinstance(tokens_data[0], dict):
                print("\nFirst token structure:")
                for key, value in tokens_data[0].items():
                    print(f"  {key}: {type(value)} = {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
        
        # Save sample for inspection
        with open("debug_tokens_structure.json", "w") as f:
            json.dump(tokens_data, f, indent=2, default=str)
        print(f"\n✅ Tokens data saved to: debug_tokens_structure.json")
        
    except Exception as e:
        print(f"❌ Error fetching tokens: {e}")
        return
    
    # 2. Check trading webhook for a sample token
    print("\n📈 TRADING WEBHOOK STRUCTURE:")
    print("-" * 30)
    
    try:
        # Get first token address from the nested structure
        if isinstance(tokens_data, list) and len(tokens_data) > 0:
            first_item = tokens_data[0]
            if isinstance(first_item, dict) and 'data' in first_item:
                tokens_list = first_item['data']
                if isinstance(tokens_list, list) and len(tokens_list) > 0:
                    first_token = tokens_list[0]
                    if isinstance(first_token, dict):
                        # Try different possible address fields
                        address = first_token.get('address') or first_token.get('adress')
                        token_name = first_token.get('name', 'Unknown')
                        
                        if address:
                            print(f"Testing with token: {token_name} ({address[:20]}...)")
                            
                            # Fetch trading data
                            trading_url = f"{trading_webhook}?token={address}"
                            print(f"Trading URL: {trading_url}")
                            
                            trading_response = requests.get(trading_url, timeout=30)
                            trading_response.raise_for_status()
                            
                            if trading_response.text.strip():
                                trading_data = trading_response.json()
                                
                                print(f"Trading response type: {type(trading_data)}")
                                print(f"Trading response length: {len(trading_data) if isinstance(trading_data, list) else 'N/A'}")
                                
                                if isinstance(trading_data, list) and len(trading_data) > 0:
                                    print(f"First trading item type: {type(trading_data[0])}")
                                    print(f"First trading item keys: {list(trading_data[0].keys()) if isinstance(trading_data[0], dict) else 'N/A'}")
                                    
                                    # Show first trading data structure
                                    if isinstance(trading_data[0], dict):
                                        print("\nFirst trading data structure:")
                                        for key, value in trading_data[0].items():
                                            print(f"  {key}: {type(value)} = {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
                                
                                # Save sample for inspection
                                with open("debug_trading_structure.json", "w") as f:
                                    json.dump(trading_data, f, indent=2, default=str)
                                print(f"\n✅ Trading data saved to: debug_trading_structure.json")
                                
                                # Try to find price and timestamp fields
                                print("\n🔍 SEARCHING FOR PRICE AND TIMESTAMP FIELDS:")
                                print("-" * 40)
                                
                                if isinstance(trading_data, list):
                                    for i, item in enumerate(trading_data[:5]):  # Check first 5 items
                                        if isinstance(item, dict):
                                            print(f"\nItem {i+1}:")
                                            price_fields = []
                                            time_fields = []
                                            
                                            for key, value in item.items():
                                                if any(price_word in key.lower() for price_word in ['price', 'amount', 'value', 'fdv', 'market_cap']):
                                                    price_fields.append(f"{key}: {value}")
                                                if any(time_word in key.lower() for time_word in ['time', 'date', 'timestamp', 'created']):
                                                    time_fields.append(f"{key}: {value}")
                                            
                                            if price_fields:
                                                print(f"  Price fields: {price_fields}")
                                            if time_fields:
                                                print(f"  Time fields: {time_fields}")
                                            
                                            if not price_fields and not time_fields:
                                                print(f"  No obvious price/time fields found")
                                                print(f"  All fields: {list(item.keys())}")
                        else:
                            print("❌ No address found in first token")
                            print(f"Available fields: {list(first_token.keys())}")
                    else:
                        print("❌ First token is not a dictionary")
                else:
                    print("❌ No tokens in data list")
            else:
                print("❌ No data field found in first item")
        else:
            print("❌ No tokens data available")
            
    except Exception as e:
        print(f"❌ Error fetching trading data: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🔍 Debug complete! Check the generated JSON files for detailed structure.")

if __name__ == "__main__":
    debug_webhook_structure()
