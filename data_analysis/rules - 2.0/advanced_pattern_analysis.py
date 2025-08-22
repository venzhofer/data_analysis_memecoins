#!/usr/bin/env python3
"""
Advanced Pattern Analysis for Token Trading
Analyzes historical data to find entry patterns and predictors for price movements
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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class AdvancedPatternAnalyzer:
    def __init__(self, tokens_webhook: str, trading_webhook: str, output_dir: str = "output/advanced_pattern_analysis"):
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
    
    def process_token_data(self, token_data: list, token_name: str):
        """Process raw token data into a DataFrame with technical indicators"""
        if not token_data:
            return None
            
        processed_data = []
        for item in token_data:
            if isinstance(item, dict) and 'created_at' in item and 'price' in item:
                try:
                    timestamp = datetime.fromisoformat(item['created_at'].replace('Z', '+00:00'))
                    price = float(item['price'])
                    fdv = float(item.get('fdv', 0))
                    market_cap = float(item.get('market_cap', 0))
                    
                    processed_data.append({
                        'timestamp': timestamp,
                        'price': price,
                        'fdv': fdv,
                        'market_cap': market_cap
                    })
                except (ValueError, TypeError):
                    continue
        
        if processed_data:
            df = pd.DataFrame(processed_data)
            df = df.sort_values('timestamp')
            df = self.add_technical_indicators(df)
            return df
        else:
            return None
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the DataFrame"""
        # Price changes
        df['price_change'] = df['price'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # Moving averages
        df['sma_5'] = df['price'].rolling(window=5).mean()
        df['sma_20'] = df['price'].rolling(window=20).mean()
        df['sma_50'] = df['price'].rolling(window=50).mean()
        
        # Price vs moving averages
        df['price_vs_sma5'] = (df['price'] / df['sma_5'] - 1) * 100
        df['price_vs_sma20'] = (df['price'] / df['sma_20'] - 1) * 100
        df['price_vs_sma50'] = (df['price'] / df['sma_50'] - 1) * 100
        
        # Volatility
        df['volatility_5'] = df['price_change'].rolling(window=5).std()
        df['volatility_20'] = df['price_change'].rolling(window=20).std()
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['price'], window=14)
        
        # MACD
        df['macd'], df['macd_signal'] = self.calculate_macd(df['price'])
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_upper'], df['bb_lower'] = self.calculate_bollinger_bands(df['price'])
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators (using price changes as proxy)
        df['volume_sma'] = df['price_change_abs'].rolling(window=20).mean()
        df['volume_ratio'] = df['price_change_abs'] / df['volume_sma']
        
        # Momentum indicators
        df['momentum_5'] = df['price'] / df['price'].shift(5) - 1
        df['momentum_20'] = df['price'] / df['price'].shift(20) - 1
        
        # Support/Resistance levels
        df['support_level'] = df['price'].rolling(window=20).min()
        df['resistance_level'] = df['price'].rolling(window=20).max()
        df['price_vs_support'] = (df['price'] / df['support_level'] - 1) * 100
        df['price_vs_resistance'] = (df['price'] / df['resistance_level'] - 1) * 100
        
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, lower
    
    def identify_patterns(self, df):
        """Identify trading patterns and signals"""
        patterns = []
        
        for i in range(50, len(df)):  # Start from 50 to have enough history
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Define pattern conditions
            pattern = {
                'timestamp': current['timestamp'],
                'price': current['price'],
                'patterns': [],
                'signals': []
            }
            
            # Bullish patterns
            if (current['price'] > current['sma_5'] > current['sma_20'] and 
                current['rsi'] < 70 and current['rsi'] > 30):
                pattern['patterns'].append('golden_cross')
                pattern['signals'].append('bullish')
            
            if (current['price'] > current['bb_upper'] * 0.98 and 
                current['rsi'] > 60):
                pattern['patterns'].append('breakout_above_bb')
                pattern['signals'].append('bullish')
            
            if (current['macd'] > current['macd_signal'] and 
                current['macd_histogram'] > prev['macd_histogram']):
                pattern['patterns'].append('macd_bullish_crossover')
                pattern['signals'].append('bullish')
            
            if (current['price_vs_support'] < 5 and 
                current['rsi'] < 40):
                pattern['patterns'].append('bounce_off_support')
                pattern['signals'].append('bullish')
            
            # Bearish patterns
            if (current['price'] < current['sma_5'] < current['sma_20'] and 
                current['rsi'] > 30):
                pattern['patterns'].append('death_cross')
                pattern['signals'].append('bearish')
            
            if (current['price'] < current['bb_lower'] * 1.02 and 
                current['rsi'] < 40):
                pattern['patterns'].append('breakdown_below_bb')
                pattern['signals'].append('bearish')
            
            if (current['macd'] < current['macd_signal'] and 
                current['macd_histogram'] < prev['macd_histogram']):
                pattern['patterns'].append('macd_bearish_crossover')
                pattern['signals'].append('bearish')
            
            if (current['price_vs_resistance'] > 95 and 
                current['rsi'] > 70):
                pattern['patterns'].append('rejection_at_resistance')
                pattern['signals'].append('bearish')
            
            # Add pattern if any signals found
            if pattern['patterns']:
                patterns.append(pattern)
        
        return patterns
    
    def analyze_future_performance(self, df, patterns, lookforward_hours=24):
        """Analyze how patterns perform in the future"""
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
                    'success': future_return > 0 if pattern['signals'][0] == 'bullish' else future_return < 0
                }
                results.append(result)
        
        return results
    
    def run_comprehensive_analysis(self):
        """Run the complete pattern analysis"""
        print("🚀 Starting Advanced Pattern Analysis...")
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
                    df = self.process_token_data(history, name)
                    if df is not None and len(df) > 100:
                        print(f"   📊 Processing {len(df)} data points...")
                        
                        # Identify patterns
                        patterns = self.identify_patterns(df)
                        print(f"   🎯 Found {len(patterns)} patterns")
                        
                        # Analyze future performance
                        results = self.analyze_future_performance(df, patterns)
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
        
        # Create comprehensive analysis
        self._create_analysis_reports(all_results, all_patterns)
    
    def _create_analysis_reports(self, results, patterns):
        """Create comprehensive analysis reports and visualizations"""
        print(f"\n📊 Creating Analysis Reports...")
        print(f"Total patterns analyzed: {len(results)}")
        
        if not results:
            print("❌ No results to analyze")
            return
        
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        
        # 1. Pattern Success Analysis
        self._analyze_pattern_success(df_results)
        
        # 2. Return Distribution Analysis
        self._analyze_return_distributions(df_results)
        
        # 3. Token Performance Analysis
        self._analyze_token_performance(df_results)
        
        # 4. Timing Analysis
        self._analyze_timing_patterns(df_results)
        
        # 5. Save comprehensive report
        self._save_comprehensive_report(df_results, patterns)
    
    def _analyze_pattern_success(self, df_results):
        """Analyze success rates of different patterns"""
        print("📈 Analyzing Pattern Success Rates...")
        
        # Overall success rate
        overall_success = df_results['success'].mean() * 100
        print(f"Overall success rate: {overall_success:.1f}%")
        
        # Success by pattern type
        pattern_success = df_results.groupby('pattern')['success'].agg(['mean', 'count']).round(3)
        pattern_success['success_rate'] = pattern_success['mean'] * 100
        print(f"\nPattern Success Rates:")
        print(pattern_success)
        
        # Success by signal type
        signal_success = df_results.groupby('signal')['success'].agg(['mean', 'count']).round(3)
        signal_success['success_rate'] = signal_success['mean'] * 100
        print(f"\nSignal Success Rates:")
        print(signal_success)
        
        # Save pattern success data
        pattern_success.to_csv(self.output_dir / 'pattern_success_rates.csv')
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Pattern success rates
        pattern_success['success_rate'].plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Pattern Success Rates', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Success Rate (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Signal success rates
        signal_success['success_rate'].plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_title('Signal Success Rates', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Success Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pattern_success_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_return_distributions(self, df_results):
        """Analyze return distributions for different patterns"""
        print("📊 Analyzing Return Distributions...")
        
        # Return statistics by pattern
        return_stats = df_results.groupby('pattern')['return_24h'].agg(['mean', 'std', 'min', 'max', 'count']).round(3)
        print(f"\nReturn Statistics by Pattern:")
        print(return_stats)
        
        # Create return distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.ravel()
        
        # Overall return distribution
        axes[0].hist(df_results['return_24h'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(df_results['return_24h'].mean(), color='red', linestyle='--', label=f'Mean: {df_results["return_24h"].mean():.2f}%')
        axes[0].set_title('Overall Return Distribution (24h)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Return (%)')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Returns by pattern
        pattern_returns = [df_results[df_results['pattern'] == pattern]['return_24h'] for pattern in df_results['pattern'].unique()]
        axes[1].boxplot(pattern_returns, labels=df_results['pattern'].unique())
        axes[1].set_title('Return Distribution by Pattern', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Return (%)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Returns by signal
        bullish_returns = df_results[df_results['signal'] == 'bullish']['return_24h']
        bearish_returns = df_results[df_results['signal'] == 'bearish']['return_24h']
        
        axes[2].hist(bullish_returns, bins=30, alpha=0.7, label='Bullish Signals', color='green')
        axes[2].hist(bearish_returns, bins=30, alpha=0.7, label='Bearish Signals', color='red')
        axes[2].set_title('Return Distribution by Signal Type', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Return (%)')
        axes[2].set_ylabel('Frequency')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Cumulative returns
        df_results_sorted = df_results.sort_values('timestamp')
        cumulative_returns = df_results_sorted.groupby('signal')['return_24h'].cumsum()
        for signal in df_results_sorted['signal'].unique():
            signal_data = df_results_sorted[df_results_sorted['signal'] == signal]
            cumulative = signal_data['return_24h'].cumsum()
            axes[3].plot(range(len(cumulative)), cumulative, label=f'{signal.capitalize()} Signals', linewidth=2)
        
        axes[3].set_title('Cumulative Returns by Signal Type', fontsize=14, fontweight='bold')
        axes[3].set_xlabel('Signal Number')
        axes[3].set_ylabel('Cumulative Return (%)')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'return_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save return statistics
        return_stats.to_csv(self.output_dir / 'return_statistics.csv')
    
    def _analyze_token_performance(self, df_results):
        """Analyze performance by token"""
        print("🏆 Analyzing Token Performance...")
        
        # Token success rates
        token_success = df_results.groupby('token')['success'].agg(['mean', 'count']).round(3)
        token_success['success_rate'] = token_success['mean'] * 100
        token_success = token_success.sort_values('success_rate', ascending=False)
        
        print(f"\nTop Performing Tokens:")
        print(token_success.head(10))
        
        # Token average returns
        token_returns = df_results.groupby('token')['return_24h'].agg(['mean', 'std', 'count']).round(3)
        token_returns = token_returns.sort_values('mean', ascending=False)
        
        print(f"\nTop Tokens by Average Return:")
        print(token_returns.head(10))
        
        # Create token performance visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))
        
        # Success rates by token
        top_tokens = token_success.head(15)
        top_tokens['success_rate'].plot(kind='barh', ax=ax1, color='lightgreen')
        ax1.set_title('Top 15 Tokens by Success Rate', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Success Rate (%)')
        ax1.grid(True, alpha=0.3)
        
        # Average returns by token
        top_return_tokens = token_returns.head(15)
        top_return_tokens['mean'].plot(kind='barh', ax=ax2, color='lightcoral')
        ax2.set_title('Top 15 Tokens by Average Return', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Average Return (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'token_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save token performance data
        token_success.to_csv(self.output_dir / 'token_success_rates.csv')
        token_returns.to_csv(self.output_dir / 'token_return_statistics.csv')
    
    def _analyze_timing_patterns(self, df_results):
        """Analyze timing patterns and optimal entry times"""
        print("⏰ Analyzing Timing Patterns...")
        
        # Extract hour from timestamp
        df_results['hour'] = pd.to_datetime(df_results['timestamp']).dt.hour
        
        # Success rate by hour
        hourly_success = df_results.groupby('hour')['success'].agg(['mean', 'count']).round(3)
        hourly_success['success_rate'] = hourly_success['mean'] * 100
        
        print(f"\nSuccess Rate by Hour:")
        print(hourly_success)
        
        # Return by hour
        hourly_returns = df_results.groupby('hour')['return_24h'].agg(['mean', 'std']).round(3)
        
        print(f"\nAverage Return by Hour:")
        print(hourly_returns)
        
        # Create timing analysis visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
        
        # Success rate by hour
        hourly_success['success_rate'].plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Success Rate by Hour of Day', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_xlabel('Hour (UTC)')
        ax1.grid(True, alpha=0.3)
        
        # Average return by hour
        hourly_returns['mean'].plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_title('Average Return by Hour of Day', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Return (%)')
        ax2.set_xlabel('Hour (UTC)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'timing_pattern_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save timing data
        hourly_success.to_csv(self.output_dir / 'hourly_success_rates.csv')
        hourly_returns.to_csv(self.output_dir / 'hourly_return_statistics.csv')
    
    def _save_comprehensive_report(self, df_results, patterns):
        """Save comprehensive analysis report"""
        print("💾 Saving Comprehensive Report...")
        
        # Summary statistics
        summary = {
            'total_patterns_analyzed': len(df_results),
            'overall_success_rate': df_results['success'].mean() * 100,
            'average_return': df_results['return_24h'].mean(),
            'best_pattern': df_results.groupby('pattern')['success'].mean().idxmax(),
            'best_token': df_results.groupby('token')['success'].mean().idxmax(),
            'best_hour': df_results.groupby(pd.to_datetime(df_results['timestamp']).dt.hour)['success'].mean().idxmax(),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        with open(self.output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed results
        df_results.to_csv(self.output_dir / 'detailed_pattern_results.csv', index=False)
        
        # Save patterns
        patterns_df = pd.DataFrame(patterns)
        patterns_df.to_csv(self.output_dir / 'all_identified_patterns.csv', index=False)
        
        # Create executive summary
        self._create_executive_summary(summary, df_results)
        
        print(f"✅ Analysis complete! Reports saved to: {self.output_dir}")
    
    def _create_executive_summary(self, summary, df_results):
        """Create an executive summary report"""
        print("\n" + "=" * 80)
        print("🎯 EXECUTIVE SUMMARY - TRADING PATTERN ANALYSIS")
        print("=" * 80)
        
        print(f"📊 Total Patterns Analyzed: {summary['total_patterns_analyzed']:,}")
        print(f"🎯 Overall Success Rate: {summary['overall_success_rate']:.1f}%")
        print(f"💰 Average Return: {summary['average_return']:.2f}%")
        print(f"🏆 Best Pattern: {summary['best_pattern']}")
        print(f"🚀 Best Token: {summary['best_token']}")
        print(f"⏰ Best Entry Hour: {summary['best_hour']}:00 UTC")
        
        print(f"\n📈 TOP 5 MOST PROFITABLE PATTERNS:")
        top_patterns = df_results.groupby('pattern')['return_24h'].mean().sort_values(ascending=False).head(5)
        for i, (pattern, avg_return) in enumerate(top_patterns.items(), 1):
            print(f"   {i}. {pattern}: {avg_return:.2f}%")
        
        print(f"\n🏆 TOP 5 MOST SUCCESSFUL TOKENS:")
        top_tokens = df_results.groupby('token')['success'].mean().sort_values(ascending=False).head(5)
        for i, (token, success_rate) in enumerate(top_tokens.items(), 1):
            print(f"   {i}. {token}: {success_rate*100:.1f}%")
        
        print(f"\n⏰ OPTIMAL ENTRY TIMES:")
        hourly_success = df_results.groupby(pd.to_datetime(df_results['timestamp']).dt.hour)['success'].mean()
        top_hours = hourly_success.sort_values(ascending=False).head(5)
        for i, (hour, success_rate) in enumerate(top_hours.items(), 1):
            print(f"   {i}. {hour:02d}:00 UTC: {success_rate*100:.1f}%")
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   • {summary['overall_success_rate']:.1f}% of identified patterns are profitable")
        print(f"   • Best performing pattern: {summary['best_pattern']}")
        print(f"   • Most reliable token: {summary['best_token']}")
        print(f"   • Optimal entry time: {summary['best_hour']}:00 UTC")
        
        print("=" * 80)

def main():
    tokens_webhook = "https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4"
    trading_webhook = "https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0"
    
    analyzer = AdvancedPatternAnalyzer(tokens_webhook, trading_webhook)
    analyzer.run_comprehensive_analysis()

if __name__ == "__main__":
    main()
