#!/usr/bin/env python3
"""
Success/Failure Condition Analyzer - Simplified Version
Finds predictive conditions that determine pattern success vs failure
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time

def fetch_tokens(tokens_webhook):
    """Fetch all tokens from the first webhook"""
    try:
        response = requests.get(tokens_webhook, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if isinstance(data, list) and len(data) > 0 and 'data' in data[0]:
            tokens = data[0]['data']
        else:
            tokens = data if isinstance(data, list) else []
            
        return tokens
    except Exception as e:
        print(f"❌ Error fetching tokens: {e}")
        return []

def fetch_token_history(trading_webhook, token_address, token_name):
    """Fetch historical data for a specific token"""
    url = f"{trading_webhook}?token={token_address}"
    
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'data' in data:
            return data['data']
        else:
            return []
            
    except Exception as e:
        print(f"   ❌ Error fetching {token_name}: {e}")
        return []

def extract_transaction_metrics(data_point):
    """Extract transaction balance and volume metrics"""
    metrics = {}
    
    # Extract transaction counts
    if 'transactions' in data_point:
        transactions = data_point['transactions']
        
        # H1 transactions (1 hour)
        if 'h1' in transactions:
            h1 = transactions['h1']
            metrics['h1_buys'] = h1.get('buys', 0)
            metrics['h1_sells'] = h1.get('sells', 0)
            metrics['h1_total'] = metrics['h1_buys'] + metrics['h1_sells']
            metrics['h1_buy_ratio'] = metrics['h1_buys'] / max(metrics['h1_total'], 1)
            metrics['h1_buy_dominance'] = metrics['h1_buys'] - metrics['h1_sells']
        
        # H6 transactions (6 hours)
        if 'h6' in transactions:
            h6 = transactions['h6']
            metrics['h6_buys'] = h6.get('buys', 0)
            metrics['h6_sells'] = h6.get('sells', 0)
            metrics['h6_total'] = metrics['h6_buys'] + metrics['h6_sells']
            metrics['h6_buy_ratio'] = metrics['h6_buys'] / max(metrics['h6_total'], 1)
            metrics['h6_buy_dominance'] = metrics['h6_buys'] - metrics['h6_sells']
        
        # H24 transactions (24 hours)
        if 'h24' in transactions:
            h24 = transactions['h24']
            metrics['h24_buys'] = h24.get('buys', 0)
            metrics['h24_sells'] = h24.get('sells', 0)
            metrics['h24_total'] = metrics['h24_buys'] + metrics['h24_sells']
            metrics['h24_buy_ratio'] = metrics['h24_buys'] / max(metrics['h24_total'], 1)
            metrics['h24_buy_dominance'] = metrics['h24_buys'] - metrics['h24_sells']
    
    # Extract volume metrics
    if 'volume' in data_point:
        volume = data_point['volume']
        metrics['volume_h1'] = volume.get('h1', 0)
        metrics['volume_h6'] = volume.get('h6', 0)
        metrics['volume_h24'] = volume.get('h24', 0)
    
    # Extract price and market data
    metrics['price'] = data_point.get('price', 0)
    metrics['timestamp'] = data_point.get('created_at', '')
    
    return metrics

def analyze_success_conditions(df_results):
    """Find the conditions that predict success vs failure"""
    print("🔍 Analyzing Success/Failure Conditions...")
    
    success_conditions = {}
    
    # Analyze each pattern type
    for pattern in df_results['pattern'].unique():
        pattern_data = df_results[df_results['pattern'] == pattern]
        
        print(f"\n📊 Analyzing {pattern} patterns...")
        print(f"Total patterns: {len(pattern_data)}")
        print(f"Success rate: {pattern_data['success'].mean()*100:.1f}%")
        
        # Find conditions that predict success
        success_data = pattern_data[pattern_data['success'] == True]
        failure_data = pattern_data[pattern_data['success'] == False]
        
        if len(success_data) > 0 and len(failure_data) > 0:
            conditions = {}
            
            # Analyze transaction balance conditions
            for metric in ['h1_buy_ratio', 'h6_buy_ratio', 'h24_buy_ratio']:
                if metric in success_data.columns:
                    success_avg = success_data[metric].mean()
                    failure_avg = failure_data[metric].mean()
                    
                    conditions[metric] = {
                        'success_avg': success_avg,
                        'failure_avg': failure_avg,
                        'difference': success_avg - failure_avg,
                        'success_threshold': success_avg
                    }
            
            # Analyze buy dominance conditions
            for metric in ['h1_buy_dominance', 'h6_buy_dominance', 'h24_buy_dominance']:
                if metric in success_data.columns:
                    success_avg = success_data[metric].mean()
                    failure_avg = failure_data[metric].mean()
                    
                    conditions[metric] = {
                        'success_avg': success_avg,
                        'failure_avg': failure_avg,
                        'difference': success_avg - failure_avg,
                        'success_threshold': success_avg
                    }
            
            # Analyze volume conditions
            for metric in ['volume_h1', 'volume_h6', 'volume_h24']:
                if metric in success_data.columns:
                    success_avg = success_data[metric].mean()
                    failure_avg = failure_data[metric].mean()
                    
                    conditions[metric] = {
                        'success_avg': success_avg,
                        'failure_avg': failure_avg,
                        'difference': success_avg - failure_avg,
                        'success_threshold': success_avg
                    }
            
            success_conditions[pattern] = conditions
    
    return success_conditions

def create_success_condition_report(success_conditions):
    """Create a comprehensive report of success conditions"""
    print("\n" + "=" * 80)
    print("🎯 SUCCESS/FAILURE CONDITION ANALYSIS REPORT")
    print("=" * 80)
    
    for pattern, conditions in success_conditions.items():
        print(f"\n🏆 {pattern.upper()} SUCCESS CONDITIONS:")
        print("-" * 50)
        
        # Sort conditions by difference (most predictive first)
        sorted_conditions = sorted(conditions.items(), 
                                 key=lambda x: abs(x[1]['difference']), 
                                 reverse=True)
        
        for metric, data in sorted_conditions:
            if abs(data['difference']) > 0.01:  # Only show meaningful differences
                print(f"📊 {metric}:")
                print(f"   ✅ Success Average: {data['success_avg']:.3f}")
                print(f"   ❌ Failure Average: {data['failure_avg']:.3f}")
                print(f"   📈 Difference: {data['difference']:.3f}")
                print(f"   🎯 Success Threshold: {data['success_threshold']:.3f}")
                
                # Interpret the condition
                if 'buy_ratio' in metric:
                    if data['difference'] > 0:
                        print(f"   💡 Higher buy ratio = Better success")
                    else:
                        print(f"   💡 Lower buy ratio = Better success")
                elif 'buy_dominance' in metric:
                    if data['difference'] > 0:
                        print(f"   💡 More buy dominance = Better success")
                    else:
                        print(f"   💡 Less buy dominance = Better success")
                elif 'volume' in metric:
                    if data['difference'] > 0:
                        print(f"   💡 Higher volume = Better success")
                    else:
                        print(f"   💡 Lower volume = Better success")
                
                print()

def main():
    print("🚀 Starting Success/Failure Condition Analysis...")
    print("=" * 60)
    
    tokens_webhook = "https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4"
    trading_webhook = "https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0"
    
    # Fetch tokens
    tokens = fetch_tokens(tokens_webhook)
    if not tokens:
        print("❌ No tokens found")
        return
    
    print(f"📋 Found {len(tokens)} tokens")
    
    all_results = []
    
    # Analyze each token
    for i, token in enumerate(tokens[:3]):  # Start with first 3 tokens for testing
        if isinstance(token, dict):
            address = token.get('address') or token.get('adress')
            name = token.get('name', f'Token_{i}')
            
            if not address:
                continue
            
            print(f"\n🔍 Analyzing {i+1}/3: {name}")
            
            # Fetch and process data
            history = fetch_token_history(trading_webhook, address, name)
            if history:
                print(f"   📊 Processing {len(history)} data points...")
                
                # Analyze transaction metrics for each data point
                for j, data_point in enumerate(history[:100]):  # Analyze first 100 points
                    if j % 20 == 0:
                        print(f"   📈 Processed {j}/{min(100, len(history))} data points...")
                    
                    # Extract transaction metrics
                    tx_metrics = extract_transaction_metrics(data_point)
                    
                    # Simple pattern detection (Golden Cross)
                    if 'price' in tx_metrics and tx_metrics['price'] > 0:
                        # Create a simple pattern result
                        result = {
                            'timestamp': tx_metrics.get('timestamp', ''),
                            'pattern': 'golden_cross',  # Simplified for testing
                            'signal': 'bullish',
                            'entry_price': tx_metrics['price'],
                            'success': np.random.choice([True, False], p=[0.3, 0.7]),  # Simulated for testing
                            **tx_metrics
                        }
                        all_results.append(result)
                
                print(f"   ✅ Analysis complete")
            else:
                print(f"   ❌ No history data")
            
            time.sleep(0.5)  # Be respectful to API
    
    if not all_results:
        print("❌ No results to analyze")
        return
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Find success conditions
    success_conditions = analyze_success_conditions(df_results)
    
    # Create comprehensive report
    create_success_condition_report(success_conditions)
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
