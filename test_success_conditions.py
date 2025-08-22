#!/usr/bin/env python3
"""
Test Success/Failure Condition Analysis
Demonstrates the concept of finding predictive conditions
"""

import numpy as np
import pandas as pd

def create_sample_data():
    """Create sample data to demonstrate the analysis"""
    
    # Sample transaction metrics
    data = []
    for i in range(100):
        # Simulate different market conditions
        if i < 70:  # First 70: Success conditions (70% success rate)
            h1_buy_ratio = np.random.uniform(0.6, 0.8)  # High buy ratio
            h1_buy_dominance = np.random.uniform(100, 500)  # High buy dominance
            volume_h1 = np.random.uniform(10000, 50000)  # High volume
            success = True
        else:  # Last 30: Failure conditions (30% failure rate)
            h1_buy_ratio = np.random.uniform(0.3, 0.5)  # Low buy ratio
            h1_buy_dominance = np.random.uniform(-200, 100)  # Low buy dominance
            volume_h1 = np.random.uniform(1000, 8000)  # Low volume
            success = False
        
        data.append({
            'pattern': 'golden_cross',
            'success': success,
            'h1_buy_ratio': h1_buy_ratio,
            'h1_buy_dominance': h1_buy_dominance,
            'volume_h1': volume_h1,
            'h6_buy_ratio': h1_buy_ratio + np.random.uniform(-0.1, 0.1),
            'h6_buy_dominance': h1_buy_dominance + np.random.uniform(-50, 50),
            'volume_h6': volume_h1 + np.random.uniform(-5000, 5000),
            'h24_buy_ratio': h1_buy_ratio + np.random.uniform(-0.15, 0.15),
            'h24_buy_dominance': h1_buy_dominance + np.random.uniform(-100, 100),
            'volume_h24': volume_h1 + np.random.uniform(-10000, 10000)
        })
    
    return pd.DataFrame(data)

def analyze_success_conditions(df):
    """Analyze success/failure conditions"""
    
    print("🔍 Analyzing Success/Failure Conditions...")
    print("=" * 60)
    
    # Split data by success
    success_data = df[df['success'] == True]
    failure_data = df[df['success'] == False]
    
    print(f"✅ Successful patterns: {len(success_data)}")
    print(f"❌ Failed patterns: {len(failure_data)}")
    print(f"📊 Overall success rate: {len(success_data)/len(df)*100:.1f}%")
    
    # Analyze each metric
    metrics = ['h1_buy_ratio', 'h1_buy_dominance', 'volume_h1', 
               'h6_buy_ratio', 'h6_buy_dominance', 'volume_h6',
               'h24_buy_ratio', 'h24_buy_dominance', 'volume_h24']
    
    print(f"\n📈 SUCCESS/FAILURE CONDITION ANALYSIS:")
    print("=" * 60)
    
    for metric in metrics:
        success_avg = success_data[metric].mean()
        failure_avg = failure_data[metric].mean()
        difference = success_avg - failure_avg
        
        print(f"\n📊 {metric}:")
        print(f"   ✅ Success Average: {success_avg:.3f}")
        print(f"   ❌ Failure Average: {failure_avg:.3f}")
        print(f"   📈 Difference: {difference:.3f}")
        print(f"   🎯 Success Threshold: {success_avg:.3f}")
        
        # Interpret the condition
        if 'buy_ratio' in metric:
            if difference > 0:
                print(f"   💡 Higher buy ratio = Better success")
                print(f"   🎯 Target: > {success_avg:.3f}")
            else:
                print(f"   💡 Lower buy ratio = Better success")
                print(f"   🎯 Target: < {success_avg:.3f}")
        elif 'buy_dominance' in metric:
            if difference > 0:
                print(f"   💡 More buy dominance = Better success")
                print(f"   🎯 Target: > {success_avg:.0f}")
            else:
                print(f"   💡 Less buy dominance = Better success")
                print(f"   🎯 Target: < {success_avg:.0f}")
        elif 'volume' in metric:
            if difference > 0:
                print(f"   💡 Higher volume = Better success")
                print(f"   🎯 Target: > {success_avg:.0f}")
            else:
                print(f"   💡 Lower volume = Better success")
                print(f"   🎯 Target: < {success_avg:.0f}")

def create_trading_rules(df):
    """Create trading rules based on the analysis"""
    
    print(f"\n🎯 TRADING RULES BASED ON SUCCESS CONDITIONS:")
    print("=" * 60)
    
    success_data = df[df['success'] == True]
    
    # Calculate thresholds
    h1_buy_ratio_threshold = success_data['h1_buy_ratio'].mean()
    h1_buy_dominance_threshold = success_data['h1_buy_dominance'].mean()
    volume_h1_threshold = success_data['volume_h1'].mean()
    
    print(f"\n📋 GOLDEN CROSS ENTRY CONDITIONS:")
    print(f"   🎯 H1 Buy Ratio: > {h1_buy_ratio_threshold:.3f}")
    print(f"   🎯 H1 Buy Dominance: > {h1_buy_dominance_threshold:.0f}")
    print(f"   🎯 H1 Volume: > {volume_h1_threshold:.0f}")
    
    print(f"\n💡 INTERPRETATION:")
    print(f"   • Buy when 1-hour buy ratio > {h1_buy_ratio_threshold:.1%}")
    print(f"   • Buy when 1-hour buy dominance > {h1_buy_dominance_threshold:.0f} transactions")
    print(f"   • Buy when 1-hour volume > {volume_h1_threshold:.0f}")
    
    print(f"\n⚠️  RISK MANAGEMENT:")
    print(f"   • Avoid entries when buy ratio < {h1_buy_ratio_threshold:.1%}")
    print(f"   • Avoid entries when sell dominance > {abs(h1_buy_dominance_threshold):.0f} transactions")
    print(f"   • Avoid entries when volume < {volume_h1_threshold:.0f}")

def main():
    print("🚀 SUCCESS/FAILURE CONDITION ANALYSIS DEMO")
    print("=" * 60)
    
    # Create sample data
    print("📊 Creating sample data...")
    df = create_sample_data()
    
    # Analyze conditions
    analyze_success_conditions(df)
    
    # Create trading rules
    create_trading_rules(df)
    
    print(f"\n✅ Analysis complete!")
    print(f"\n💡 This demonstrates how to find predictive conditions:")
    print(f"   • Transaction balance (buy vs sell ratios)")
    print(f"   • Buy dominance (buy - sell transaction counts)")
    print(f"   • Volume thresholds")
    print(f"   • Success thresholds for each metric")

if __name__ == "__main__":
    main()
