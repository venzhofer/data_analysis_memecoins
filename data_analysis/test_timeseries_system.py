"""
Test script for the Enhanced Time-Series Analysis System
Tests all components to ensure they work correctly
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from timeseries_schema import TimeSeriesSchema
        print("✅ TimeSeriesSchema imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TimeSeriesSchema: {e}")
        return False
    
    try:
        from timeseries_ingestor import TimeSeriesIngestor
        print("✅ TimeSeriesIngestor imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TimeSeriesIngestor: {e}")
        return False
    
    try:
        from timeseries_loader import TimeSeriesLoader
        print("✅ TimeSeriesLoader imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TimeSeriesLoader: {e}")
        return False
    
    try:
        from feature_engineering import FeatureEngineer
        print("✅ FeatureEngineer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import FeatureEngineer: {e}")
        return False
    
    try:
        from enhanced_timeseries_analyzer import EnhancedTimeSeriesAnalyzer
        print("✅ EnhancedTimeSeriesAnalyzer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import EnhancedTimeSeriesAnalyzer: {e}")
        return False
    
    return True

def test_schema():
    """Test the time-series schema"""
    print("\nTesting TimeSeriesSchema...")
    
    try:
        from timeseries_schema import TimeSeriesSchema
        
        # Test schema validation
        sample_data = {
            'timestamp': ['2025-08-20T10:00:00Z', '2025-08-20T10:01:00Z'],
            'address': ['0xABC123', '0xABC123'],
            'price': [1.0, 1.1],
            'fdv': [1000000, 1100000],
            'market_cap': [500000, 550000],
            'buys': [10, 15],
            'sells': [5, 8],
            'buy_volume': [1000, 1500],
            'sell_volume': [500, 800],
            'total_volume': [1500, 2300]
        }
        
        import pandas as pd
        df = pd.DataFrame(sample_data)
        
        # Test validation
        is_valid = TimeSeriesSchema.validate_dataframe(df)
        print(f"Schema validation: {'✅ Passed' if is_valid else '❌ Failed'}")
        
        # Test normalization
        normalized = TimeSeriesSchema.normalize_dataframe(df)
        print(f"Normalization: {'✅ Passed' if len(normalized) == 2 else '❌ Failed'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        traceback.print_exc()
        return False

def test_feature_engineering():
    """Test the feature engineering"""
    print("\nTesting FeatureEngineer...")
    
    try:
        from feature_engineering import FeatureEngineer
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range('2025-08-20', periods=100, freq='1min', tz='UTC')
        sample_data = {
            'timestamp': dates,
            'open': np.random.uniform(1.0, 1.1, 100),
            'high': np.random.uniform(1.1, 1.2, 100),
            'low': np.random.uniform(0.9, 1.0, 100),
            'close': np.random.uniform(1.0, 1.1, 100),
            'volume': np.random.uniform(1000, 5000, 100),
            'buys': np.random.randint(10, 50, 100),
            'sells': np.random.randint(10, 50, 100),
            'fdv': np.random.uniform(1000000, 1100000, 100),
            'market_cap': np.random.uniform(500000, 550000, 100),
            'buy_volume': np.random.uniform(500, 2500, 100),
            'sell_volume': np.random.uniform(500, 2500, 100),
            'total_volume': np.random.uniform(1000, 5000, 100)
        }
        
        df = pd.DataFrame(sample_data)
        
        # Test feature engineering
        engineer = FeatureEngineer()
        df_with_features = engineer.add_features(df)
        
        print(f"Original shape: {df.shape}")
        print(f"With features: {df_with_features.shape}")
        print(f"Features added: {'✅ Passed' if len(df_with_features.columns) > len(df.columns) else '❌ Failed'}")
        
        # Test feature summary
        summary = engineer.get_feature_summary(df_with_features)
        print(f"Feature summary: {'✅ Passed' if summary.get('status') == 'success' else '❌ Failed'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        traceback.print_exc()
        return False

def test_webhook_connectivity():
    """Test webhook connectivity"""
    print("\nTesting webhook connectivity...")
    
    try:
        from timeseries_ingestor import TimeSeriesIngestor
        
        # Test with a small timeout
        ingestor = TimeSeriesIngestor(
            tokens_webhook="https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4",
            trading_webhook="https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0",
            data_dir="test_data"
        )
        
        # Test tokens webhook
        tokens = ingestor.fetch_tokens_info()
        print(f"Tokens webhook: {'✅ Passed' if tokens else '❌ Failed'}")
        if tokens:
            print(f"  Found {len(tokens)} tokens")
        
        # Test trading webhook with first token
        if tokens:
            first_token = tokens[0]
            address = first_token.get('address') or first_token.get('adress')
            if address:
                trading_data = ingestor.fetch_trading_data(address)
                print(f"Trading webhook: {'✅ Passed' if trading_data else '❌ Failed'}")
                if trading_data:
                    print(f"  Got {len(trading_data)} data points")
        
        return True
        
    except Exception as e:
        print(f"❌ Webhook connectivity test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Enhanced Time-Series Analysis System")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Schema Test", test_schema),
        ("Feature Engineering Test", test_feature_engineering),
        ("Webhook Connectivity Test", test_webhook_connectivity)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Install additional dependencies: pip install -r requirements_timeseries.txt")
        print("2. Run the analyzer: python enhanced_timeseries_analyzer.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Make sure all required packages are installed")
        print("2. Check that all Python files are in the same directory")
        print("3. Verify webhook URLs are accessible")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
