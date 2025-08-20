import requests
import json

def test_tokens_webhook():
    """Test the tokens information webhook"""
    url = "https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4"
    
    try:
        print("Testing tokens webhook...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Tokens webhook working!")
        print(f"   Status: {response.status_code}")
        print(f"   Response type: {type(data)}")
        print(f"   Data length: {len(data) if isinstance(data, list) else 'N/A'}")
        
        if isinstance(data, list) and len(data) > 0:
            print(f"   First token sample: {data[0] if isinstance(data[0], dict) else 'Not a dict'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokens webhook failed: {e}")
        return False

def test_trading_webhook():
    """Test the trading data webhook with a sample token"""
    base_url = "https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0"
    
    # Test with a dummy token address (this will likely fail, but we can see the response format)
    test_token = "0x1234567890123456789012345678901234567890"
    url = f"{base_url}?token={test_token}"
    
    try:
        print("\nTesting trading data webhook...")
        response = requests.get(url, timeout=30)
        
        print(f"   Status: {response.status_code}")
        print(f"   URL: {url}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"✅ Trading webhook accessible!")
                print(f"   Response type: {type(data)}")
                print(f"   Data length: {len(data) if isinstance(data, list) else 'N/A'}")
                
                if isinstance(data, list) and len(data) > 0:
                    print(f"   First data point sample: {data[0] if isinstance(data, list) else 'Not a list'}")
                
            except json.JSONDecodeError:
                print(f"   Response is not JSON: {response.text[:200]}...")
                
        else:
            print(f"   Response: {response.text[:200]}...")
            
        return True
        
    except Exception as e:
        print(f"❌ Trading webhook failed: {e}")
        return False

def main():
    """Run webhook tests"""
    print("🔍 Testing Webhook Endpoints")
    print("=" * 40)
    
    tokens_ok = test_tokens_webhook()
    trading_ok = test_trading_webhook()
    
    print("\n" + "=" * 40)
    if tokens_ok and trading_ok:
        print("✅ Both webhooks are accessible!")
        print("   You can now run the main analysis with: python token_analyzer.py")
    else:
        print("❌ Some webhooks failed. Check the errors above.")
        print("   You may need to verify the webhook URLs or check your internet connection.")

if __name__ == "__main__":
    main()
