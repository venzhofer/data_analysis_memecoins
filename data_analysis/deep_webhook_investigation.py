#!/usr/bin/env python3
"""
Deep Webhook Investigation
Investigates why we're not getting the 170,000+ rows that should be available
"""

import requests
import json
from datetime import datetime, timedelta

def deep_webhook_investigation():
    """Deep investigation of the webhook to find the missing data"""
    
    base_url = "https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0"
    test_token = "9pag2RpRugnhiJMdzwRSGP2xTaLr1szCTh6UsVm5sCM9"  # deadly weapon
    
    print("🔍 DEEP WEBHOOK INVESTIGATION - Finding the Missing 170,000+ Rows...")
    print("=" * 70)
    
    # Test 1: Basic call with different response handling
    print("\n🧪 TEST 1: Basic call with detailed response analysis")
    print("-" * 50)
    
    try:
        response = requests.get(f"{base_url}?token={test_token}", timeout=60)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Size: {len(response.content)} bytes")
        print(f"Response Text (first 500 chars): {response.text[:500]}")
        
        if response.status_code == 200 and response.text.strip():
            try:
                data = response.json()
                print(f"JSON Data Type: {type(data)}")
                if isinstance(data, list):
                    print(f"List Length: {len(data)}")
                    if len(data) > 0:
                        print(f"First Item Type: {type(data[0])}")
                        print(f"First Item Keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'N/A'}")
                elif isinstance(data, dict):
                    print(f"Dict Keys: {list(data.keys())}")
                    if 'data' in data and isinstance(data['data'], list):
                        print(f"Nested Data Length: {len(data['data'])}")
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                print(f"Raw Response: {response.text}")
        else:
            print("Empty or invalid response")
            
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Try without token parameter
    print("\n🧪 TEST 2: Call without token parameter")
    print("-" * 50)
    
    try:
        response = requests.get(base_url, timeout=60)
        print(f"Status Code: {response.status_code}")
        print(f"Response Size: {len(response.content)} bytes")
        print(f"Response Text (first 500 chars): {response.text[:500]}")
        
        if response.status_code == 200 and response.text.strip():
            try:
                data = response.json()
                print(f"JSON Data Type: {type(data)}")
                if isinstance(data, list):
                    print(f"List Length: {len(data)}")
                elif isinstance(data, dict):
                    print(f"Dict Keys: {list(data.keys())}")
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
        else:
            print("Empty or invalid response")
            
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Try with different HTTP methods
    print("\n🧪 TEST 3: Try POST method")
    print("-" * 50)
    
    try:
        response = requests.post(base_url, json={"token": test_token}, timeout=60)
        print(f"Status Code: {response.status_code}")
        print(f"Response Size: {len(response.content)} bytes")
        print(f"Response Text (first 500 chars): {response.text[:500]}")
        
        if response.status_code == 200 and response.text.strip():
            try:
                data = response.json()
                print(f"JSON Data Type: {type(data)}")
                if isinstance(data, list):
                    print(f"List Length: {len(data)}")
                elif isinstance(data, dict):
                    print(f"Dict Keys: {list(data.keys())}")
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
        else:
            print("Empty or invalid response")
            
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Try with different token format
    print("\n🧪 TEST 4: Try with different token format")
    print("-" * 50)
    
    try:
        # Try with token in body instead of query
        response = requests.get(base_url, params={"token": test_token, "format": "full"}, timeout=60)
        print(f"Status Code: {response.status_code}")
        print(f"Response Size: {len(response.content)} bytes")
        print(f"Response Text (first 500 chars): {response.text[:500]}")
        
        if response.status_code == 200 and response.text.strip():
            try:
                data = response.json()
                print(f"JSON Data Type: {type(data)}")
                if isinstance(data, list):
                    print(f"List Length: {len(data)}")
                elif isinstance(data, dict):
                    print(f"Dict Keys: {list(data.keys())}")
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
        else:
            print("Empty or invalid response")
            
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: Check if there are different endpoints
    print("\n🧪 TEST 5: Check for different endpoints")
    print("-" * 50)
    
    # Try some common variations
    endpoints_to_try = [
        f"{base_url}/history",
        f"{base_url}/timeseries",
        f"{base_url}/data",
        f"{base_url}/prices",
        f"{base_url}/trades"
    ]
    
    for endpoint in endpoints_to_try:
        try:
            response = requests.get(endpoint, params={"token": test_token}, timeout=30)
            print(f"Endpoint: {endpoint}")
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                print(f"Response Size: {len(response.content)} bytes")
                try:
                    data = response.json()
                    if isinstance(data, list):
                        print(f"List Length: {len(data)}")
                    elif isinstance(data, dict):
                        print(f"Dict Keys: {list(data.keys())}")
                except:
                    print("Not JSON response")
            print("-" * 30)
        except Exception as e:
            print(f"Endpoint: {endpoint} - Error: {e}")
    
    print("\n" + "=" * 70)
    print("🔍 Deep investigation complete!")
    print("If you're getting 170,000+ rows, please tell me:")
    print("1. What exact URL you're calling")
    print("2. What parameters you're using")
    print("3. What headers you're sending")
    print("4. What the response looks like")

if __name__ == "__main__":
    deep_webhook_investigation()
