import requests
import json

def debug_webhook():
    """Debug the webhook response format"""
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
            print(f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            
            # Print first few lines of the response
            print(f"\nFirst 500 characters of response:")
            print(response.text[:500])
            
            # If it's a dict, show the structure
            if isinstance(data, dict):
                print(f"\nDictionary structure:")
                for key, value in data.items():
                    print(f"  {key}: {type(value)} - {str(value)[:100]}")
                    
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Raw response (first 1000 chars):")
            print(response.text[:1000])
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_webhook()
