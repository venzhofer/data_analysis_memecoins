#!/usr/bin/env python3
"""
COMPREHENSIVE Success/Failure Condition Analyzer
Analyzes ALL 26 tokens to find complete predictive conditions
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

def fetch_tokens(tokens_webhook):
    """Fetch all tokens from the first webhook"""
    try:
        print("🔍 Fetching tokens...")
        response = requests.get(tokens_webhook, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if isinstance(data, list) and len(data) > 0 and 'data' in data[0]:
            tokens = data[0]['data']
        else:
            tokens = data if isinstance(data, list) else []
            
        print(f"✅ Found {len(tokens)} tokens")
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

def identify_patterns_with_conditions(history):
    """Identify patterns and extract success/failure conditions from real data"""
    patterns = []
    
    if len(history) < 50:
        return patterns
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['created_at'])
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.sort_values('timestamp')
    df = df.dropna(subset=['price'])
    
    if len(df) < 50:
        return patterns
    
    # Add basic technical indicators
    df['price_change'] = df['price'].pct_change()
    df['sma_5'] = df['price'].rolling(window=5).mean()
    df['sma_20'] = df['price'].rolling(window=20).mean()
    df['volatility'] = df['price_change'].rolling(window=20).std()
    
    # Look for patterns in the data with more flexible criteria
    for i in range(25, len(df)):
        current = df.iloc[i]
        
        if pd.isna(current['price']) or current['price'] <= 0:
            continue
            
        # Extract transaction metrics for this pattern
        tx_metrics = extract_transaction_metrics(current)
        
        # More flexible pattern detection
        if i > 0:
            prev_price = df.iloc[i-1]['price']
            price_change = (current['price'] / prev_price - 1) * 100
            
            # Bullish patterns (various thresholds)
            if price_change > 2:  # 2% increase
                pattern = {
                    'timestamp': current['timestamp'],
                    'price': current['price'],
                    'pattern': 'bullish_breakout',
                    'signal': 'bullish',
                    'price_change': price_change,
                    'conditions': tx_metrics
                }
                patterns.append(pattern)
            
            elif price_change > 1:  # 1% increase
                pattern = {
                    'timestamp': current['timestamp'],
                    'price': current['price'],
                    'pattern': 'bullish_momentum',
                    'signal': 'bullish',
                    'price_change': price_change,
                    'conditions': tx_metrics
                }
                patterns.append(pattern)
            
            # Bearish patterns
            elif price_change < -2:  # 2% decrease
                pattern = {
                    'timestamp': current['timestamp'],
                    'price': current['price'],
                    'pattern': 'bearish_breakdown',
                    'signal': 'bearish',
                    'price_change': price_change,
                    'conditions': tx_metrics
                }
                patterns.append(pattern)
            
            elif price_change < -1:  # 1% decrease
                pattern = {
                    'timestamp': current['timestamp'],
                    'price': current['price'],
                    'pattern': 'bearish_momentum',
                    'signal': 'bearish',
                    'price_change': price_change,
                    'conditions': tx_metrics
                }
                patterns.append(pattern)
            
            # Sideways patterns (low volatility)
            elif abs(price_change) < 0.5 and current['volatility'] < 0.02:
                pattern = {
                    'timestamp': current['timestamp'],
                    'price': current['price'],
                    'pattern': 'sideways_consolidation',
                    'signal': 'neutral',
                    'price_change': price_change,
                    'conditions': tx_metrics
                }
                patterns.append(pattern)
    
    return patterns

def analyze_future_performance(history, patterns, lookforward_hours=6):
    """Analyze how patterns with different conditions perform in the future"""
    results = []
    
    if len(history) < 10:
        return results
    
    # Convert to DataFrame
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['created_at'])
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.sort_values('timestamp')
    df = df.dropna(subset=['price'])
    
    for pattern in patterns:
        pattern_time = pattern['timestamp']
        future_time = pattern_time + timedelta(hours=lookforward_hours)
        
        # Find future price
        future_data = df[df['timestamp'] >= future_time]
        if len(future_data) > 0:
            future_price = future_data.iloc[0]['price']
            current_price = pattern['price']
            
            if future_price > 0 and current_price > 0:
                future_return = (future_price / current_price - 1) * 100
                
                # Determine success based on signal type
                if pattern['signal'] == 'bullish':
                    success = future_return > 0
                elif pattern['signal'] == 'bearish':
                    success = future_return < 0
                else:  # neutral
                    success = abs(future_return) < 1  # Small movement
                
                result = {
                    'timestamp': pattern_time,
                    'pattern': pattern['pattern'],
                    'signal': pattern['signal'],
                    'entry_price': current_price,
                    'future_price': future_price,
                    'return_6h': future_return,
                    'success': success,
                    **pattern['conditions']  # Include all condition metrics
                }
                results.append(result)
    
    return results

def find_success_conditions(df_results):
    """Find the conditions that predict success vs failure"""
    print("🔍 Analyzing Success/Failure Conditions...")
    
    if len(df_results) == 0:
        print("❌ No results to analyze")
        return {}
    
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
                if metric in success_data.columns and metric in failure_data.columns:
                    success_avg = success_data[metric].mean()
                    failure_avg = failure_data[metric].mean()
                    
                    if not pd.isna(success_avg) and not pd.isna(failure_avg):
                        conditions[metric] = {
                            'success_avg': success_avg,
                            'failure_avg': failure_avg,
                            'difference': success_avg - failure_avg,
                            'success_threshold': success_avg
                        }
            
            # Analyze buy dominance conditions
            for metric in ['h1_buy_dominance', 'h6_buy_dominance', 'h24_buy_dominance']:
                if metric in success_data.columns and metric in failure_data.columns:
                    success_avg = success_data[metric].mean()
                    failure_avg = failure_data[metric].mean()
                    
                    if not pd.isna(success_avg) and not pd.isna(failure_avg):
                        conditions[metric] = {
                            'success_avg': success_avg,
                            'failure_avg': failure_avg,
                            'difference': success_avg - failure_avg,
                            'success_threshold': success_avg
                        }
            
            # Analyze volume conditions
            for metric in ['volume_h1', 'volume_h6', 'volume_h24']:
                if metric in success_data.columns and metric in failure_data.columns:
                    success_avg = success_data[metric].mean()
                    failure_avg = failure_data[metric].mean()
                    
                    if not pd.isna(success_avg) and not pd.isna(failure_avg):
                        conditions[metric] = {
                            'success_avg': success_avg,
                            'failure_avg': failure_avg,
                            'difference': success_avg - failure_avg,
                            'success_threshold': success_avg
                        }
            
            success_conditions[pattern] = conditions
    
    return success_conditions

def create_comprehensive_report(success_conditions, df_results):
    """Create a comprehensive report of success conditions across ALL tokens"""
    print("\n" + "=" * 100)
    print("🎯 COMPREHENSIVE SUCCESS/FAILURE CONDITION ANALYSIS - ALL 26 TOKENS")
    print("=" * 100)
    
    if not success_conditions:
        print("❌ No success conditions found")
        return
    
    # Overall statistics
    total_patterns = len(df_results)
    total_tokens = df_results['token'].nunique()
    overall_success_rate = df_results['success'].mean() * 100
    
    print(f"\n📊 OVERALL STATISTICS ACROSS ALL TOKENS:")
    print(f"   • Total patterns analyzed: {total_patterns:,}")
    print(f"   • Total tokens analyzed: {total_tokens}")
    print(f"   • Overall success rate: {overall_success_rate:.1f}%")
    
    # Token-by-token breakdown
    print(f"\n🔍 TOKEN-BY-TOKEN BREAKDOWN:")
    print("-" * 60)
    
    token_stats = df_results.groupby('token').agg({
        'success': ['count', 'mean']
    }).round(3)
    
    for token in df_results['token'].unique():
        token_data = df_results[df_results['token'] == token]
        pattern_count = len(token_data)
        success_rate = token_data['success'].mean() * 100
        
        print(f"   • {token}: {pattern_count:,} patterns, {success_rate:.1f}% success")
    
    # Pattern analysis
    print(f"\n📈 PATTERN ANALYSIS:")
    print("-" * 60)
    
    for pattern, conditions in success_conditions.items():
        pattern_data = df_results[df_results['pattern'] == pattern]
        pattern_count = len(pattern_data)
        pattern_success_rate = pattern_data['success'].mean() * 100
        
        print(f"\n🏆 {pattern.upper()} SUCCESS CONDITIONS:")
        print(f"   📊 Total patterns: {pattern_count:,}")
        print(f"   📊 Success rate: {pattern_success_rate:.1f}%")
        print("-" * 50)
        
        # Sort conditions by difference (most predictive first)
        sorted_conditions = sorted(conditions.items(), 
                                 key=lambda x: abs(x[1]['difference']), 
                                 reverse=True)
        
        for metric, data in sorted_conditions:
            if abs(data['difference']) > 0.01:  # Only show meaningful differences
                print(f"   📊 {metric}:")
                print(f"      ✅ Success Average: {data['success_avg']:.3f}")
                print(f"      ❌ Failure Average: {data['failure_avg']:.3f}")
                print(f"      📈 Difference: {data['difference']:.3f}")
                print(f"      🎯 Success Threshold: {data['success_threshold']:.3f}")
                
                # Interpret the condition
                if 'buy_ratio' in metric:
                    if data['difference'] > 0:
                        print(f"      💡 Higher buy ratio = Better success")
                        print(f"      🎯 Target: > {data['success_threshold']:.3f}")
                    else:
                        print(f"      💡 Lower buy ratio = Better success")
                        print(f"      🎯 Target: < {data['success_threshold']:.3f}")
                elif 'buy_dominance' in metric:
                    if data['difference'] > 0:
                        print(f"      💡 More buy dominance = Better success")
                        print(f"      🎯 Target: > {data['success_threshold']:.0f}")
                    else:
                        print(f"      💡 Less buy dominance = Better success")
                        print(f"      🎯 Target: < {data['success_threshold']:.0f}")
                elif 'volume' in metric:
                    if data['difference'] > 0:
                        print(f"      💡 Higher volume = Better success")
                        print(f"      🎯 Target: > {data['success_threshold']:.0f}")
                    else:
                        print(f"      💡 Lower volume = Better success")
                        print(f"      🎯 Target: < {data['success_threshold']:.0f}")
                
                print()

def main():
    print("🚀 COMPREHENSIVE SUCCESS/FAILURE CONDITION ANALYSIS - ALL 26 TOKENS")
    print("=" * 80)
    
    tokens_webhook = "https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4"
    trading_webhook = "https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0"
    
    # Fetch tokens
    tokens = fetch_tokens(tokens_webhook)
    if not tokens:
        print("❌ No tokens found")
        return
    
    all_results = []
    
    # Analyze ALL tokens
    for i, token in enumerate(tokens):
        if isinstance(token, dict):
            address = token.get('address') or token.get('adress')
            name = token.get('name', f'Token_{i}')
            
            if not address:
                continue
            
            print(f"\n🔍 Analyzing {i+1}/{len(tokens)}: {name}")
            
            # Fetch and process data
            history = fetch_token_history(trading_webhook, address, name)
            if history:
                print(f"   📊 Processing {len(history):,} data points...")
                
                # Identify patterns with conditions
                patterns = identify_patterns_with_conditions(history)
                print(f"   🎯 Found {len(patterns):,} patterns with conditions")
                
                # Analyze future performance
                results = analyze_future_performance(history, patterns)
                print(f"   📈 Analyzed {len(results):,} pattern outcomes")
                
                # Store results
                for result in results:
                    result['token'] = name
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
    success_conditions = find_success_conditions(df_results)
    
    # Create comprehensive report
    create_comprehensive_report(success_conditions, df_results)
    
    print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"\n💾 Final Results Summary:")
    print(f"   • Total patterns analyzed: {len(all_results):,}")
    print(f"   • Total tokens analyzed: {df_results['token'].nunique()}")
    print(f"   • Pattern types: {df_results['pattern'].nunique()}")
    print(f"   • Overall success rate: {df_results['success'].mean()*100:.1f}%")
    
    # Show pattern distribution
    print(f"\n📊 Overall Pattern Distribution:")
    pattern_counts = df_results['pattern'].value_counts()
    for pattern, count in pattern_counts.items():
        success_rate = df_results[df_results['pattern'] == pattern]['success'].mean() * 100
        print(f"   • {pattern}: {count:,} patterns, {success_rate:.1f}% success")

if __name__ == "__main__":
    main()
