#!/usr/bin/env python3
"""
Simple test script for the live trading system
"""

from live_trading_system import LiveTradingPatternDetector

def test_system():
    print("🧪 Testing Live Trading Pattern Detection System")
    print("=" * 50)
    
    # Create detector
    detector = LiveTradingPatternDetector()
    
    # Test with just 2 tokens to verify functionality
    print("\n🚀 Running test analysis on 2 tokens...")
    
    try:
        results = detector.run_live_analysis(max_tokens=2)
        
        if results:
            print(f"\n✅ Test successful! Analyzed {len(results)} tokens")
            print(f"🎯 Detected {len(detector.detected_patterns)} patterns")
        else:
            print("\n⚠️ No results returned - check webhook connectivity")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("This might be due to webhook connectivity issues")

if __name__ == "__main__":
    test_system()
