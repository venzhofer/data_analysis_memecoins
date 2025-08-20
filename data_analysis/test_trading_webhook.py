import requests
import json

def test_trading_webhook():
    """Test the trading data webhook with a real token address"""
    
    # First get a token address from the tokens webhook
    tokens_url = "https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4"
    trading_base_url = "https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0"
    
    try:
        print("1. Fetching token info...")
        response = requests.get(tokens_url, timeout=30)
        response.raise_for_status()
        
        token_data = response.json()
        print(f"Token: {token_data.get('name', 'Unknown')}")
        
        # Get the address (try both field names)
        address = token_data.get('address') or token_data.get('adress')
        if not address:
            print("No address found in token data")
            return
        
        print(f"Address: {address}")
        
        # Test trading data webhook
        print(f"\n2. Testing trading data webhook...")
        trading_url = f"{trading_base_url}?token={address}"
        print(f"URL: {trading_url}")
        
        trading_response = requests.get(trading_url, timeout=30)
        print(f"Status: {trading_response.status_code}")
        
        if trading_response.status_code == 200:
            try:
                trading_data = trading_response.json()
                print(f"Trading data type: {type(trading_data)}")
                print(f"Trading data length: {len(trading_data) if isinstance(trading_data, list) else 'N/A'}")
                
                if isinstance(trading_data, list) and len(trading_data) > 0:
                    print(f"First trading data point: {trading_data[0]}")
                elif isinstance(trading_data, dict):
                    print(f"Trading data keys: {list(trading_data.keys())}")
                    print(f"Trading data: {trading_data}")
                else:
                    print(f"Unexpected trading data format: {trading_data}")
                    
            except json.JSONDecodeError as e:
                print(f"Trading data JSON parsing failed: {e}")
                print(f"Raw response (first 500 chars):")
                print(trading_response.text[:500])
        else:
            print(f"Trading webhook failed with status {trading_response.status_code}")
            print(f"Response: {trading_response.text[:500]}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_trading_webhook()
