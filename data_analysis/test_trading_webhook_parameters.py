#!/usr/bin/env python3
"""
Test Trading Webhook Parameters
Tests different parameters to see if we can get historical price data
"""

import requests
import json
from datetime import datetime, timedelta

def test_webhook_parameters():
    """Test different parameters to get historical data"""
    
    base_url = "https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0"
    test_token = "9pag2RpRugnhiJMdzwRSGP2xTaLr1szCTh6UsVm5sCM9"  # deadly weapon - fixed address
    
    print("🔍 Testing Trading Webhook Parameters...")
    print("=" * 50)
    
    # Test different parameter combinations
    test_params = [
        {"token": test_token},
        {"token": test_token, "start": "2025-08-14T00:00:00Z"},
        {"token": test_token, "end": "2025-08-14T23:59:59Z"},
        {"token": test_token, "start": "2025-08-14T00:00:00Z", "end": "2025-08-14T23:59:59Z"},
        {"token": test_token, "limit": "100"},
        {"token": test_token, "period": "24h"},
        {"token": test_token, "interval": "5m"},
        {"token": test_token, "history": "true"},
        {"token": test_token, "timeseries": "true"},
        {"token": test_token, "data": "full"},
        {"token": test_token, "format": "json"},
        {"token": test_token, "type": "price"},
        {"token": test_token, "mode": "detailed"},
        {"token": test_token, "include": "history"},
        {"token": test_token, "since": "2025-08-14T00:00:00Z"},
        {"token": test_token, "from": "2025-08-14T00:00:00Z"},
        {"token": test_token, "to": "2025-08-14T23:59:59Z"},
        {"token": test_token, "range": "24h"},
        {"token": test_token, "window": "1d"},
        {"token": test_token, "granularity": "5m"}
    ]
    
    for i, params in enumerate(test_params):
        try:
            print(f"\n🧪 Test {i+1}: {params}")
            
            response = requests.get(base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ Status: {response.status_code}")
                print(f"📊 Response type: {type(data)}")
                
                if isinstance(data, list):
                    print(f"📈 Data points: {len(data)}")
                    if len(data) > 0:
                        print(f"🔍 First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'N/A'}")
                elif isinstance(data, dict):
                    print(f"🔍 Keys: {list(data.keys())}")
                    if 'data' in data and isinstance(data['data'], list):
                        print(f"📈 Nested data points: {len(data['data'])}")
                
                # Check if we got multiple data points
                if isinstance(data, list) and len(data) > 1:
                    print(f"🎯 SUCCESS! Got {len(data)} data points!")
                    print(f"📅 First timestamp: {data[0].get('created_at', 'N/A')}")
                    print(f"📅 Last timestamp: {data[-1].get('created_at', 'N/A')}")
                    print(f"💰 First price: {data[0].get('price', 'N/A')}")
                    print(f"💰 Last price: {data[-1].get('price', 'N/A')}")
                    
                    # Save this successful response
                    with open(f"successful_webhook_test_{i+1}.json", "w") as f:
                        json.dump(data, f, indent=2, default=str)
                    print(f"💾 Saved successful response to: successful_webhook_test_{i+1}.json")
                    
                elif isinstance(data, dict) and 'data' in data and isinstance(data['data'], list) and len(data['data']) > 1:
                    print(f"🎯 SUCCESS! Got {len(data['data'])} nested data points!")
                    print(f"📅 First timestamp: {data['data'][0].get('created_at', 'N/A')}")
                    print(f"📅 Last timestamp: {data['data'][-1].get('created_at', 'N/A')}")
                    print(f"💰 First price: {data['data'][0].get('price', 'N/A')}")
                    print(f"💰 Last price: {data['data'][-1].get('price', 'N/A')}")
                    
                    # Save this successful response
                    with open(f"successful_webhook_test_{i+1}.json", "w") as f:
                        json.dump(data, f, indent=2, default=str)
                    print(f"💾 Saved successful response to: successful_webhook_test_{i+1}.json")
                    
                else:
                    print(f"❌ Only got single data point or no data")
                    
            else:
                print(f"❌ Status: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 30)
    
    print("\n" + "=" * 50)
    print("🔍 Webhook parameter testing complete!")

if __name__ == "__main__":
    test_webhook_parameters()
