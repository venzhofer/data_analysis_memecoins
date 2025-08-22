#!/usr/bin/env python3
"""
Test Trading Webhook with Real Token Address
Tests the exact URL format with a real token address
"""

import requests
import json
from datetime import datetime

def test_with_real_address():
    """Test the trading webhook with a real token address"""
    
    base_url = "https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0"
    test_token = "9pag2RpRugnhiJMdzwRSGP2xTaLr1szCTh6UsVm5sCM9"  # deadly weapon
    
    url = f"{base_url}?token={test_token}"
    
    print("🔍 Testing Trading Webhook with Real Token Address")
    print("=" * 60)
    print(f"URL: {url}")
    print(f"Token: deadly weapon ({test_token})")
    print("=" * 60)
    
    try:
        print("📡 Making request...")
        response = requests.get(url, timeout=120)
        
        print(f"✅ Status Code: {response.status_code}")
        print(f"📊 Response Size: {len(response.content)} bytes")
        print(f"📅 Response Time: {response.elapsed.total_seconds():.2f}s")
        
        # Check headers
        print("\n📋 Response Headers:")
        for key, value in response.headers.items():
            print(f"   {key}: {value}")
        
        # Try to parse JSON
        print("\n🔍 Attempting to parse response...")
        try:
            data = response.json()
            
            if isinstance(data, list):
                print(f"📊 Data is a LIST with {len(data)} items")
                if len(data) > 0:
                    print(f"   First item type: {type(data[0])}")
                    print(f"   First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
                    
                    # Show first few items
                    print(f"\n📝 First 3 items:")
                    for i, item in enumerate(data[:3]):
                        print(f"   Item {i+1}: {json.dumps(item, indent=2, default=str)}")
                        
            elif isinstance(data, dict):
                print(f"📊 Data is a DICT with keys: {list(data.keys())}")
                print(f"📝 Full response: {json.dumps(data, indent=2, default=str)}")
            else:
                print(f"📊 Data type: {type(data)}")
                print(f"📝 Data: {data}")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            print(f"📝 Raw response (first 500 chars): {response.text[:500]}")
            
            # Try to find patterns in the response
            content = response.text
            if len(content) > 100:
                print(f"\n🔍 Response analysis:")
                print(f"   Contains '[': {'[' in content}")
                print(f"   Contains ']': {']' in content}")
                print(f"   Contains 'created_at': {'created_at' in content}")
                print(f"   Contains 'price': {'price' in content}")
                print(f"   Contains 'address': {'address' in content}")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_real_address()
