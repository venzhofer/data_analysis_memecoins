import requests
import json

def debug_webhook_detailed():
    """Debug the webhook response format in detail"""
    url = "https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4"
    
    try:
        print("Fetching webhook data...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'Unknown')}")
        print(f"Response length: {len(response.text)} characters")
        
        # Try to parse as JSON
        try:
            data = response.json()
            print(f"JSON parsing successful!")
            print(f"Data type: {type(data)}")
            
            if isinstance(data, list):
                print(f"List length: {len(data)}")
                print(f"First item type: {type(data[0]) if data else 'Empty list'}")
                
                if data and len(data) > 0:
                    first_item = data[0]
                    print(f"First item keys: {list(first_item.keys()) if isinstance(first_item, dict) else 'Not a dict'}")
                    
                    # Check if it's a nested structure
                    if isinstance(first_item, dict) and 'data' in first_item:
                        print(f"Found 'data' field in first item")
                        nested_data = first_item['data']
                        print(f"Nested data type: {type(nested_data)}")
                        print(f"Nested data length: {len(nested_data) if isinstance(nested_data, list) else 'N/A'}")
                        
                        if isinstance(nested_data, list) and len(nested_data) > 0:
                            first_token = nested_data[0]
                            print(f"First token keys: {list(first_token.keys()) if isinstance(first_token, dict) else 'Not a dict'}")
                            print(f"First token name: {first_token.get('name', 'Unknown') if isinstance(first_token, dict) else 'N/A'}")
                            print(f"First token address: {first_token.get('adress', 'Unknown') if isinstance(first_token, dict) else 'N/A'}")
                    
                    # Show first few items
                    print(f"\nFirst 3 items structure:")
                    for i, item in enumerate(data[:3]):
                        print(f"  Item {i}: {type(item)} - {str(item)[:200]}")
                        
            elif isinstance(data, dict):
                print(f"Dictionary keys: {list(data.keys())}")
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Raw response (first 1000 chars):")
            print(response.text[:1000])
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_webhook_detailed()
