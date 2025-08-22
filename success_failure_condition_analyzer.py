#!/usr/bin/env python3
"""
Success/Failure Condition Analyzer
Finds predictive conditions that determine pattern success vs failure
"""

import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SuccessFailureConditionAnalyzer:
    def __init__(self, tokens_webhook: str, trading_webhook: str, output_dir: str = "success_failure_conditions"):
        self.tokens_webhook = tokens_webhook
        self.trading_webhook = trading_webhook
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_tokens(self):
        """Fetch all tokens from the first webhook"""
        try:
            response = requests.get(self.tokens_webhook, timeout=30)
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
    
    def fetch_token_history(self, token_address: str, token_name: str):
        """Fetch historical data for a specific token"""
        url = f"{self.trading_webhook}?token={token_address}"
        
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
    
    def extract_transaction_metrics(self, data_point):
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
        metrics['fdv'] = data_point.get('fdv', 0)
        metrics['market_cap'] = data_point.get('market_cap', 0)
        metrics['timestamp'] = data_point.get('created_at', '')
        
        return metrics
    
    def identify_patterns_with_conditions(self, df):
        """Identify patterns and extract success/failure conditions"""
        patterns = []
        
        for i in range(50, len(df)):  # Start from 50 to have enough history
            current = df.iloc[i]
            
            # Define pattern conditions
            pattern = {
                'timestamp': current['timestamp'],
                'price': current['price'],
                'patterns': [],
                'signals': [],
                'conditions': {}
            }
            
            # Extract transaction metrics for this pattern
            tx_metrics = self.extract_transaction_metrics(current)
            
            # Golden Cross (Bullish)
            if (current['price'] > current['sma_5'] > current['sma_20'] and 
                current['rsi'] < 70 and current['rsi'] > 30):
                pattern['patterns'].append('golden_cross')
                pattern['signals'].append('bullish')
                
                # Add success/failure conditions
                pattern['conditions'] = {
                    'h1_buy_ratio': tx_metrics.get('h1_buy_ratio', 0),
                    'h1_buy_dominance': tx_metrics.get('h1_buy_dominance', 0),
                    'h1_total_tx': tx_metrics.get('h1_total', 0),
                    'h6_buy_ratio': tx_metrics.get('h6_buy_ratio', 0),
                    'h6_buy_dominance': tx_metrics.get('h6_buy_dominance', 0),
                    'h6_total_tx': tx_metrics.get('h6_total', 0),
                    'h24_buy_ratio': tx_metrics.get('h24_buy_ratio', 0),
                    'h24_buy_dominance': tx_metrics.get('h24_buy_dominance', 0),
                    'h24_total_tx': tx_metrics.get('h24_total', 0),
                    'volume_h1': tx_metrics.get('volume_h1', 0),
                    'volume_h6': tx_metrics.get('volume_h6', 0),
                    'volume_h24': tx_metrics.get('volume_h24', 0),
                    'rsi': current['rsi'],
                    'price_vs_sma20': current['price_vs_sma20'],
                    'volatility': current['volatility_20']
                }
            
            # Death Cross (Bearish)
            if (current['price'] < current['sma_5'] < current['sma_20'] and 
                current['rsi'] > 30):
                pattern['patterns'].append('death_cross')
                pattern['signals'].append('bearish')
                
                # Add success/failure conditions
                pattern['conditions'] = {
                    'h1_buy_ratio': tx_metrics.get('h1_buy_ratio', 0),
                    'h1_buy_dominance': tx_metrics.get('h1_buy_dominance', 0),
                    'h1_total_tx': tx_metrics.get('h1_total', 0),
                    'h6_buy_ratio': tx_metrics.get('h6_buy_ratio', 0),
                    'h6_buy_dominance': tx_metrics.get('h6_buy_dominance', 0),
                    'h6_total_tx': tx_metrics.get('h6_total', 0),
                    'h24_buy_ratio': tx_metrics.get('h24_buy_ratio', 0),
                    'h24_buy_dominance': tx_metrics.get('h24_buy_dominance', 0),
                    'h24_total_tx': tx_metrics.get('h24_total', 0),
                    'volume_h1': tx_metrics.get('volume_h1', 0),
                    'volume_h6': tx_metrics.get('volume_h6', 0),
                    'volume_h24': tx_metrics.get('volume_h24', 0),
                    'rsi': current['rsi'],
                    'price_vs_sma20': current['price_vs_sma20'],
                    'volatility': current['volatility_20']
                }
            
            # Bollinger Breakout (Bullish)
            if (current['price'] > current['bb_upper'] * 0.98 and 
                current['rsi'] > 60):
                pattern['patterns'].append('bollinger_breakout')
                pattern['signals'].append('bullish')
                
                # Add success/failure conditions
                pattern['conditions'] = {
                    'h1_buy_ratio': tx_metrics.get('h1_buy_ratio', 0),
                    'h1_buy_dominance': tx_metrics.get('h1_buy_dominance', 0),
                    'h1_total_tx': tx_metrics.get('h1_total', 0),
                    'h6_buy_ratio': tx_metrics.get('h6_buy_ratio', 0),
                    'h6_buy_dominance': tx_metrics.get('h6_buy_dominance', 0),
                    'h6_total_tx': tx_metrics.get('h6_total', 0),
                    'h24_buy_ratio': tx_metrics.get('h24_buy_ratio', 0),
                    'h24_buy_dominance': tx_metrics.get('h24_buy_dominance', 0),
                    'h24_total_tx': tx_metrics.get('h24_total', 0),
                    'volume_h1': tx_metrics.get('volume_h1', 0),
                    'volume_h6': tx_metrics.get('volume_h6', 0),
                    'volume_h24': tx_metrics.get('volume_h24', 0),
                    'rsi': current['rsi'],
                    'bb_position': current['bb_position'],
                    'volatility': current['volatility_20']
                }
            
            # Add pattern if any signals found
            if pattern['patterns']:
                patterns.append(pattern)
        
        return patterns
    
    def analyze_future_performance_with_conditions(self, df, patterns, lookforward_hours=24):
        """Analyze how patterns with different conditions perform in the future"""
        results = []
        
        for pattern in patterns:
            pattern_time = pattern['timestamp']
            future_time = pattern_time + timedelta(hours=lookforward_hours)
            
            # Find future price
            future_data = df[df['timestamp'] >= future_time]
            if len(future_data) > 0:
                future_price = future_data.iloc[0]['price']
                current_price = pattern['price']
                future_return = (future_price / current_price - 1) * 100
                
                result = {
                    'timestamp': pattern_time,
                    'pattern': pattern['patterns'][0],
                    'signal': pattern['signals'][0],
                    'entry_price': current_price,
                    'future_price': future_price,
                    'return_24h': future_return,
                    'success': future_return > 0 if pattern['signals'][0] == 'bullish' else future_return < 0,
                    **pattern['conditions']  # Include all condition metrics
                }
                results.append(result)
        
        return results
    
    def find_success_conditions(self, df_results):
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
                
                # Analyze technical conditions
                for metric in ['rsi', 'price_vs_sma20', 'volatility']:
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
    
    def create_success_condition_report(self, success_conditions):
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
        
        # Save detailed report
        self._save_success_conditions_report(success_conditions)
    
    def _save_success_conditions_report(self, success_conditions):
        """Save the success conditions report"""
        report_data = {}
        
        for pattern, conditions in success_conditions.items():
            pattern_data = {}
            for metric, data in conditions.items():
                pattern_data[metric] = {
                    'success_avg': float(data['success_avg']),
                    'failure_avg': float(data['failure_avg']),
                    'difference': float(data['difference']),
                    'success_threshold': float(data['success_threshold'])
                }
            report_data[pattern] = pattern_data
        
        # Save as JSON
        with open(self.output_dir / 'success_conditions_report.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Save as CSV for easy analysis
        rows = []
        for pattern, conditions in success_conditions.items():
            for metric, data in conditions.items():
                rows.append({
                    'pattern': pattern,
                    'metric': metric,
                    'success_avg': data['success_avg'],
                    'failure_avg': data['failure_avg'],
                    'difference': data['difference'],
                    'success_threshold': data['success_threshold']
                })
        
        df_report = pd.DataFrame(rows)
        df_report.to_csv(self.output_dir / 'success_conditions_report.csv', index=False)
        
        print(f"💾 Success conditions report saved to: {self.output_dir}")
    
    def run_comprehensive_analysis(self):
        """Run the complete success/failure condition analysis"""
        print("🚀 Starting Success/Failure Condition Analysis...")
        print("=" * 60)
        
        # Fetch tokens
        tokens = self.fetch_tokens()
        if not tokens:
            print("❌ No tokens found")
            return
        
        print(f"📋 Found {len(tokens)} tokens")
        
        all_results = []
        all_patterns = []
        
        # Analyze each token
        for i, token in enumerate(tokens):
            if isinstance(token, dict):
                address = token.get('address') or token.get('adress')
                name = token.get('name', f'Token_{i}')
                
                if not address:
                    continue
                
                print(f"\n🔍 Analyzing {i+1}/{len(tokens)}: {name}")
                
                # Fetch and process data
                history = self.fetch_token_history(address, name)
                if history:
                    print(f"   📊 Processing {len(history)} data points...")
                    
                    # Convert to DataFrame and add technical indicators
                    df = self._prepare_dataframe(history)
                    if df is not None and len(df) > 100:
                        # Identify patterns with conditions
                        patterns = self.identify_patterns_with_conditions(df)
                        print(f"   🎯 Found {len(patterns)} patterns with conditions")
                        
                        # Analyze future performance
                        results = self.analyze_future_performance_with_conditions(df, patterns)
                        print(f"   📈 Analyzed {len(results)} pattern outcomes")
                        
                        # Store results
                        for result in results:
                            result['token'] = name
                            all_results.append(result)
                        
                        for pattern in patterns:
                            pattern['token'] = name
                            all_patterns.append(pattern)
                        
                        print(f"   ✅ Analysis complete")
                    else:
                        print(f"   ⚠️ Insufficient data")
                else:
                    print(f"   ❌ No history data")
                
                time.sleep(0.5)  # Be respectful to API
        
        if not all_results:
            print("❌ No results to analyze")
            return
        
        # Convert to DataFrame
        df_results = pd.DataFrame(all_results)
        
        # Find success conditions
        success_conditions = self.find_success_conditions(df_results)
        
        # Create comprehensive report
        self.create_success_condition_report(success_conditions)
        
        # Save detailed results
        df_results.to_csv(self.output_dir / 'detailed_condition_results.csv', index=False)
        
        print(f"\n✅ Analysis complete! Results saved to: {self.output_dir}")
    
    def _prepare_dataframe(self, history):
        """Prepare DataFrame with technical indicators"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['created_at'])
            df = df.sort_values('timestamp')
            
            # Add basic technical indicators
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['price_change'] = df['price'].pct_change()
            df['sma_5'] = df['price'].rolling(window=5).mean()
            df['sma_5'] = df['price'].rolling(window=5).mean()
            df['sma_20'] = df['price'].rolling(window=20).mean()
            df['rsi'] = self._calculate_rsi(df['price'])
            df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['price'])
            df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['volatility_20'] = df['price_change'].rolling(window=20).std()
            df['price_vs_sma20'] = (df['price'] / df['sma_20'] - 1) * 100
            
            return df
        except Exception as e:
            print(f"   ❌ Error preparing DataFrame: {e}")
            return None
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + ( std * num_std)
        lower = sma - (std * num_std)
        return upper, lower

def main():
    tokens_webhook = "https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4"
    trading_webhook = "https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0"
    
    analyzer = SuccessFailureConditionAnalyzer(tokens_webhook, trading_webhook)
    analyzer.run_comprehensive_analysis()

if __name__ == "__main__":
    main()
