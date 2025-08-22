#!/usr/bin/env python3
"""
FIXED Comprehensive Success/Failure Condition Analyzer
Analyzes ALL 26 tokens to find universal trading patterns
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

def safe_parse_timestamp(timestamp_str):
    """Safely parse timestamp with multiple format support"""
    if not timestamp_str:
        return None
    
    try:
        # Try ISO format first
        return pd.to_datetime(timestamp_str, format='ISO8601')
    except:
        try:
            # Try mixed format
            return pd.to_datetime(timestamp_str, format='mixed')
        except:
            try:
                # Try inferring format
                return pd.to_datetime(timestamp_str, infer_datetime_format=True)
            except:
                print(f"      ⚠️ Could not parse timestamp: {timestamp_str[:50]}...")
                return None

def identify_patterns_with_conditions(history):
    """Identify patterns and extract success/failure conditions from real data"""
    patterns = []
    
    if len(history) < 50:
        return patterns
    
    try:
        # Convert to DataFrame for analysis
        df = pd.DataFrame(history)
        
        # Safe timestamp parsing
        df['timestamp'] = df['created_at'].apply(safe_parse_timestamp)
        df = df.dropna(subset=['timestamp'])
        
        if len(df) < 50:
            return patterns
        
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price'])
        df = df.sort_values('timestamp')
        
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
        
    except Exception as e:
        print(f"      ❌ Error in pattern identification: {e}")
        return []

def analyze_future_performance(history, patterns, lookforward_hours=6):
    """Analyze how patterns with different conditions perform in the future"""
    results = []
    
    if len(history) < 10:
        return results
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(history)
        df['timestamp'] = df['created_at'].apply(safe_parse_timestamp)
        df = df.dropna(subset=['timestamp'])
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price'])
        df = df.sort_values('timestamp')
        
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
        
    except Exception as e:
        print(f"      ❌ Error in future performance analysis: {e}")
        return []

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

def create_universal_trading_report(success_conditions, df_results):
    """Create a comprehensive report focused on universal trading patterns"""
    print("\n" + "=" * 120)
    print("🎯 UNIVERSAL TRADING PATTERNS ANALYSIS - ALL 26 TOKENS")
    print("=" * 120)
    
    if not success_conditions:
        print("❌ No success conditions found")
        return
    
    # Overall statistics
    total_patterns = len(df_results)
    total_tokens = df_results['token'].nunique()
    overall_success_rate = df_results['success'].mean() * 100
    
    print(f"\n📊 UNIVERSAL PATTERN STATISTICS:")
    print(f"   • Total patterns analyzed: {total_patterns:,}")
    print(f"   • Total tokens analyzed: {total_tokens}")
    print(f"   • Overall success rate: {overall_success_rate:.1f}%")
    
    # Token-by-token breakdown
    print(f"\n🔍 TOKEN-BY-TOKEN SUCCESS RATES:")
    print("-" * 80)
    
    token_stats = []
    for token in df_results['token'].unique():
        token_data = df_results[df_results['token'] == token]
        pattern_count = len(token_data)
        success_rate = token_data['success'].mean() * 100
        token_stats.append((token, success_rate, pattern_count))
    
    # Sort by success rate (highest first)
    token_stats.sort(key=lambda x: x[1], reverse=True)
    
    for token, success_rate, pattern_count in token_stats:
        print(f"   • {token}: {success_rate:.1f}% success ({pattern_count:,} patterns)")
    
    # Universal pattern analysis
    print(f"\n🏆 UNIVERSAL TRADING PATTERNS:")
    print("-" * 80)
    
    for pattern, conditions in success_conditions.items():
        pattern_data = df_results[df_results['pattern'] == pattern]
        pattern_count = len(pattern_data)
        pattern_success_rate = pattern_data['success'].mean() * 100
        
        print(f"\n🎯 {pattern.upper()} - UNIVERSAL SUCCESS CONDITIONS:")
        print(f"   📊 Total patterns: {pattern_count:,}")
        print(f"   📊 Success rate: {pattern_success_rate:.1f}%")
        print(f"   📊 Tokens showing this pattern: {pattern_data['token'].nunique()}")
        print("-" * 60)
        
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
                print(f"      🎯 UNIVERSAL THRESHOLD: {data['success_threshold']:.3f}")
                
                # Interpret the condition
                if 'buy_ratio' in metric:
                    if data['difference'] > 0:
                        print(f"      💡 Higher buy ratio = Better success")
                        print(f"      🎯 UNIVERSAL RULE: > {data['success_threshold']:.3f}")
                    else:
                        print(f"      💡 Lower buy ratio = Better success")
                        print(f"      🎯 UNIVERSAL RULE: < {data['success_threshold']:.3f}")
                elif 'buy_dominance' in metric:
                    if data['difference'] > 0:
                        print(f"      💡 More buy dominance = Better success")
                        print(f"      🎯 UNIVERSAL RULE: > {data['success_threshold']:.0f}")
                    else:
                        print(f"      💡 Less buy dominance = Better success")
                        print(f"      🎯 UNIVERSAL RULE: < {data['success_threshold']:.0f}")
                elif 'volume' in metric:
                    if data['difference'] > 0:
                        print(f"      💡 Higher volume = Better success")
                        print(f"      🎯 UNIVERSAL RULE: > {data['success_threshold']:.0f}")
                    else:
                        print(f"      💡 Lower volume = Better success")
                        print(f"      🎯 UNIVERSAL RULE: < {data['success_threshold']:.0f}")
                
                print()
    
    # Create universal trading rules
    print(f"\n🚀 UNIVERSAL TRADING RULES (ALL 26 TOKENS):")
    print("=" * 80)
    
    # Find the most profitable patterns
    profitable_patterns = []
    for pattern, conditions in success_conditions.items():
        pattern_data = df_results[df_results['pattern'] == pattern]
        success_rate = pattern_data['success'].mean()
        profitable_patterns.append((pattern, success_rate, len(pattern_data)))
    
    # Sort by success rate
    profitable_patterns.sort(key=lambda x: x[1], reverse=True)
    
    for i, (pattern, success_rate, count) in enumerate(profitable_patterns[:3]):
        print(f"\n🥇 #{i+1} MOST PROFITABLE PATTERN: {pattern.upper()}")
        print(f"   📊 Success Rate: {success_rate*100:.1f}%")
        print(f"   📊 Pattern Count: {count:,}")
        print(f"   📊 Tokens: {df_results[df_results['pattern'] == pattern]['token'].nunique()}")
        
        # Show the key conditions for this pattern
        if pattern in success_conditions:
            conditions = success_conditions[pattern]
            print(f"   🎯 KEY UNIVERSAL CONDITIONS:")
            
            # Get top 3 most predictive conditions
            sorted_conditions = sorted(conditions.items(), 
                                     key=lambda x: abs(x[1]['difference']), 
                                     reverse=True)
            
            for j, (metric, data) in enumerate(sorted_conditions[:3]):
                if data['difference'] > 0:
                    print(f"      {j+1}. {metric}: > {data['success_threshold']:.3f}")
                else:
                    print(f"      {j+1}. {metric}: < {data['success_threshold']:.3f}")

def main():
    print("🚀 UNIVERSAL TRADING PATTERNS ANALYSIS - ALL 26 TOKENS")
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
    
    # Create universal trading report
    create_universal_trading_report(success_conditions, df_results)
    
    print(f"\n✅ UNIVERSAL TRADING PATTERNS ANALYSIS COMPLETE!")
    print(f"\n💾 Final Results Summary:")
    print(f"   • Total patterns analyzed: {len(all_results):,}")
    print(f"   • Total tokens analyzed: {df_results['token'].nunique()}")
    print(f"   • Pattern types: {df_results['pattern'].nunique()}")
    print(f"   • Overall success rate: {df_results['success'].mean()*100:.1f}%")
    
    # Show pattern distribution
    print(f"\n📊 Universal Pattern Distribution:")
    pattern_counts = df_results['pattern'].value_counts()
    for pattern, count in pattern_counts.items():
        success_rate = df_results[df_results['pattern'] == pattern]['success'].mean() * 100
        token_count = df_results[df_results['pattern'] == pattern]['token'].nunique()
        print(f"   • {pattern}: {count:,} patterns, {success_rate:.1f}% success, {token_count} tokens")

if __name__ == "__main__":
    main()
