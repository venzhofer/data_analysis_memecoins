#!/usr/bin/env python3
"""
Test Exact URL Format
Tests the exact URL format: https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0?token=token_address
"""

import requests
import json
from datetime import datetime

def test_exact_url_format():
    """Test the exact URL format the user specified"""
    
    base_url = "https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0"
    test_token = "9pag2RpRugnhiJMdzwRSGP2xTaLr1szCTh6UsVm5sCM9"  # deadly weapon
    
    print("🔍 Testing Exact URL Format...")
    print("=" * 50)
    print(f"URL: {base_url}?token={test_token}")
    print("=" * 50)
    
    try:
        # Test the exact format you specified
        url = f"{base_url}?token={test_token}"
        print(f"Calling: {url}")
        
        response = requests.get(url, timeout=120)  # Increased timeout for large data
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Size: {len(response.content)} bytes")
        print(f"Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
        
        if response.status_code == 200:
            if response.text.strip():
                print(f"Response Text Length: {len(response.text)} characters")
                print(f"First 500 chars: {response.text[:500]}")
                print(f"Last 500 chars: {response.text[-500:]}")
                
                try:
                    data = response.json()
                    print(f"JSON Data Type: {type(data)}")
                    
                    if isinstance(data, list):
                        print(f"🎯 SUCCESS! Got {len(data)} rows!")
                        if len(data) > 0:
                            print(f"First Item Keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'N/A'}")
                            print(f"Sample First Item: {json.dumps(data[0], indent=2)[:500]}...")
                            
                            # Save the response if it's large
                            if len(data) > 100:
                                filename = f"large_webhook_response_{len(data)}_rows.json"
                                with open(filename, "w") as f:
                                    json.dump(data, f, indent=2, default=str)
                                print(f"💾 Saved large response to: {filename}")
                                
                    elif isinstance(data, dict):
                        print(f"Dict Keys: {list(data.keys())}")
                        if 'data' in data and isinstance(data['data'], list):
                            print(f"🎯 SUCCESS! Got {len(data['data'])} nested rows!")
                            print(f"Sample First Item: {json.dumps(data['data'][0], indent=2)[:500]}...")
                            
                            # Save the response if it's large
                            if len(data['data']) > 100:
                                filename = f"large_nested_webhook_response_{len(data['data'])}_rows.json"
                                with open(filename, "w") as f:
                                    json.dump(data, f, indent=2, default=str)
                                print(f"💾 Saved large nested response to: {filename}")
                        else:
                            print(f"❌ Only got single data point: {data}")
                    else:
                        print(f"❌ Unexpected data type: {type(data)}")
                        print(f"Data: {data}")
                        
                except json.JSONDecodeError as e:
                    print(f"JSON Decode Error: {e}")
                    print(f"Raw Response (first 1000 chars): {response.text[:1000]}")
                    
            else:
                print("❌ Empty response")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🔍 Test complete!")

if __name__ == "__main__":
    test_exact_url_format()
