#!/usr/bin/env python3
"""
LIVE TRADING PATTERN DETECTION SYSTEM
Based on 365,195 patterns analyzed across all 26 tokens
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

class LiveTradingPatternDetector:
    def __init__(self):
        self.tokens_webhook = "https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4"
        self.trading_webhook = "https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0"
        
        # Universal thresholds from analysis
        self.bearish_breakdown_thresholds = {
            'volume_h24': 1309765,
            'volume_h6': 957612,
            'volume_h1': 264705,
            'buy_dominance_h6': 1030,
            'buy_dominance_h1': 325,
            'buy_ratio_h1': 0.523
        }
        
        self.bearish_momentum_thresholds = {
            'volume_h24': 1459366,
            'volume_h6': 914263,
            'volume_h1': 133013,
            'buy_dominance_h6': 1008,
            'buy_dominance_h1': 174,
            'buy_ratio_h6': 0.527
        }
        
        self.bullish_momentum_thresholds = {
            'volume_h24': 4287812,
            'volume_h6': 1330173,
            'volume_h1': 278264,
            'buy_dominance_h24': 2695,
            'buy_dominance_h6': 334,
            'buy_ratio_h1': 0.505
        }
        
        self.bullish_breakout_thresholds = {
            'volume_h24': 3413450,
            'volume_h6': 1135719,
            'volume_h1': 316185,
            'buy_dominance_h24': 2060,
            'buy_dominance_h6': 303,
            'buy_dominance_h1': 62,
            'buy_ratio_h1': 0.505
        }
        
        self.detected_patterns = []
        self.trading_signals = []
    
    def fetch_tokens(self):
        """Fetch all available tokens"""
        try:
            response = requests.get(self.tokens_webhook, timeout=30)
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
    
    def fetch_token_data(self, token_address, token_name):
        """Fetch current token data"""
        url = f"{self.trading_webhook}?token={token_address}"
        
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                return data[0]  # Get most recent data point
            elif isinstance(data, dict) and 'data' in data and len(data['data']) > 0:
                return data['data'][0]
            else:
                return None
                
        except Exception as e:
            print(f"   ❌ Error fetching {token_name}: {e}")
            return None
    
    def extract_metrics(self, data_point):
        """Extract trading metrics from data point"""
        metrics = {}
        
        try:
            # Extract transaction counts
            if 'transactions' in data_point:
                transactions = data_point['transactions']
                
                # H1 transactions
                if 'h1' in transactions:
                    h1 = transactions['h1']
                    metrics['h1_buys'] = h1.get('buys', 0)
                    metrics['h1_sells'] = h1.get('sells', 0)
                    metrics['h1_total'] = metrics['h1_buys'] + metrics['h1_sells']
                    metrics['h1_buy_ratio'] = metrics['h1_buys'] / max(metrics['h1_total'], 1)
                    metrics['h1_buy_dominance'] = metrics['h1_buys'] - metrics['h1_sells']
                
                # H6 transactions
                if 'h6' in transactions:
                    h6 = transactions['h6']
                    metrics['h6_buys'] = h6.get('buys', 0)
                    metrics['h6_sells'] = h6.get('sells', 0)
                    metrics['h6_total'] = metrics['h6_buys'] + metrics['h6_sells']
                    metrics['h6_buy_ratio'] = metrics['h6_buys'] / max(metrics['h6_total'], 1)
                    metrics['h6_buy_dominance'] = metrics['h6_buys'] - metrics['h6_sells']
                
                # H24 transactions
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
            
        except Exception as e:
            print(f"   ⚠️ Error extracting metrics: {e}")
        
        return metrics
    
    def detect_bearish_breakdown(self, metrics):
        """Detect Bearish Breakdown pattern (81.4% success rate)"""
        try:
            conditions_met = (
                metrics.get('volume_h24', 0) < self.bearish_breakdown_thresholds['volume_h24'] and
                metrics.get('volume_h6', 0) < self.bearish_breakdown_thresholds['volume_h6'] and
                metrics.get('volume_h1', 0) < self.bearish_breakdown_thresholds['volume_h1'] and
                metrics.get('h6_buy_dominance', 0) > self.bearish_breakdown_thresholds['buy_dominance_h6'] and
                metrics.get('h1_buy_dominance', 0) > self.bearish_breakdown_thresholds['buy_dominance_h1'] and
                metrics.get('h1_buy_ratio', 0) > self.bearish_breakdown_thresholds['buy_ratio_h1']
            )
            
            if conditions_met:
                return {
                    'pattern': 'bearish_breakdown',
                    'success_rate': 81.4,
                    'risk_level': 'LOW',
                    'action': 'AVOID BUYING',
                    'confidence': 'HIGH',
                    'metrics': metrics
                }
        except Exception as e:
            print(f"   ⚠️ Error in bearish breakdown detection: {e}")
        
        return None
    
    def detect_bearish_momentum(self, metrics):
        """Detect Bearish Momentum pattern (75.4% success rate)"""
        try:
            conditions_met = (
                metrics.get('volume_h24', 0) < self.bearish_momentum_thresholds['volume_h24'] and
                metrics.get('volume_h6', 0) < self.bearish_momentum_thresholds['volume_h6'] and
                metrics.get('volume_h1', 0) < self.bearish_momentum_thresholds['volume_h1'] and
                metrics.get('h6_buy_dominance', 0) > self.bearish_momentum_thresholds['buy_dominance_h6'] and
                metrics.get('h1_buy_dominance', 0) > self.bearish_momentum_thresholds['buy_dominance_h1'] and
                metrics.get('h6_buy_ratio', 0) > self.bearish_momentum_thresholds['buy_ratio_h6']
            )
            
            if conditions_met:
                return {
                    'pattern': 'bearish_momentum',
                    'success_rate': 75.4,
                    'risk_level': 'LOW-MEDIUM',
                    'action': 'AVOID BUYING',
                    'confidence': 'HIGH',
                    'metrics': metrics
                }
        except Exception as e:
            print(f"   ⚠️ Error in bearish momentum detection: {e}")
        
        return None
    
    def detect_bullish_momentum(self, metrics):
        """Detect Bullish Momentum pattern (23.3% success rate)"""
        try:
            conditions_met = (
                metrics.get('volume_h24', 0) > self.bullish_momentum_thresholds['volume_h24'] and
                metrics.get('volume_h6', 0) > self.bullish_momentum_thresholds['volume_h6'] and
                metrics.get('volume_h1', 0) > self.bullish_momentum_thresholds['volume_h1'] and
                metrics.get('h24_buy_dominance', 0) > self.bullish_momentum_thresholds['buy_dominance_h24'] and
                metrics.get('h6_buy_dominance', 0) < self.bullish_momentum_thresholds['buy_dominance_h6'] and
                metrics.get('h1_buy_ratio', 0) < self.bullish_momentum_thresholds['buy_ratio_h1']
            )
            
            if conditions_met:
                return {
                    'pattern': 'bullish_momentum',
                    'success_rate': 23.3,
                    'risk_level': 'HIGH',
                    'action': 'CONSIDER SMALL BUY',
                    'confidence': 'LOW',
                    'position_size': '5-8% portfolio',
                    'metrics': metrics
                }
        except Exception as e:
            print(f"   ⚠️ Error in bullish momentum detection: {e}")
        
        return None
    
    def detect_bullish_breakout(self, metrics):
        """Detect Bullish Breakout pattern (20.0% success rate)"""
        try:
            conditions_met = (
                metrics.get('volume_h24', 0) > self.bullish_breakout_thresholds['volume_h24'] and
                metrics.get('volume_h6', 0) > self.bullish_breakout_thresholds['volume_h6'] and
                metrics.get('volume_h1', 0) > self.bullish_breakout_thresholds['volume_h1'] and
                metrics.get('h24_buy_dominance', 0) > self.bullish_breakout_thresholds['buy_dominance_h24'] and
                metrics.get('h6_buy_dominance', 0) < self.bullish_breakout_thresholds['buy_dominance_h6'] and
                metrics.get('h1_buy_dominance', 0) < self.bullish_breakout_thresholds['buy_dominance_h1'] and
                metrics.get('h1_buy_ratio', 0) < self.bullish_breakout_thresholds['buy_ratio_h1']
            )
            
            if conditions_met:
                return {
                    'pattern': 'bullish_breakout',
                    'success_rate': 20.0,
                    'risk_level': 'HIGH',
                    'action': 'CONSIDER VERY SMALL BUY',
                    'confidence': 'LOW',
                    'position_size': '3-5% portfolio',
                    'metrics': metrics
                }
        except Exception as e:
            print(f"   ⚠️ Error in bullish breakout detection: {e}")
        
        return None
    
    def analyze_token(self, token_address, token_name):
        """Analyze a single token for all patterns"""
        print(f"\n🔍 Analyzing {token_name}...")
        
        # Fetch current data
        data = self.fetch_token_data(token_address, token_name)
        if not data:
            print(f"   ❌ No data available for {token_name}")
            return None
        
        # Extract metrics
        metrics = self.extract_metrics(data)
        if not metrics:
            print(f"   ❌ No metrics available for {token_name}")
            return None
        
        # Detect all patterns
        patterns = []
        
        # Check for bearish patterns (HIGH PRIORITY - AVOID BUYING)
        bearish_breakdown = self.detect_bearish_breakdown(metrics)
        if bearish_breakdown:
            patterns.append(bearish_breakdown)
            print(f"   🚨 BEARISH BREAKDOWN DETECTED! {bearish_breakdown['success_rate']}% success rate")
            print(f"      ACTION: {bearish_breakdown['action']}")
        
        bearish_momentum = self.detect_bearish_momentum(metrics)
        if bearish_momentum:
            patterns.append(bearish_momentum)
            print(f"   🚨 BEARISH MOMENTUM DETECTED! {bearish_momentum['success_rate']}% success rate")
            print(f"      ACTION: {bearish_momentum['action']}")
        
        # Check for bullish patterns (LOWER PRIORITY - CONSIDER SMALL BUYS)
        bullish_momentum = self.detect_bullish_momentum(metrics)
        if bullish_momentum:
            patterns.append(bullish_momentum)
            print(f"   🟢 BULLISH MOMENTUM DETECTED! {bullish_momentum['success_rate']}% success rate")
            print(f"      ACTION: {bullish_momentum['action']} ({bullish_momentum['position_size']})")
        
        bullish_breakout = self.detect_bullish_breakout(metrics)
        if bullish_breakout:
            patterns.append(bullish_breakout)
            print(f"   🟢 BULLISH BREAKOUT DETECTED! {bullish_breakout['success_rate']}% success rate")
            print(f"      ACTION: {bullish_breakout['action']} ({bullish_breakout['position_size']})")
        
        if not patterns:
            print(f"   ⚪ No clear patterns detected - monitor for changes")
        
        return {
            'token': token_name,
            'patterns': patterns,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_live_analysis(self, max_tokens=None):
        """Run live analysis on all tokens"""
        print("🚀 LIVE TRADING PATTERN DETECTION SYSTEM")
        print("=" * 60)
        print("Based on analysis of 365,195 patterns across all 26 tokens")
        print("=" * 60)
        
        # Fetch tokens
        tokens = self.fetch_tokens()
        if not tokens:
            print("❌ No tokens found")
            return
        
        # Limit tokens if specified
        if max_tokens:
            tokens = tokens[:max_tokens]
        
        all_results = []
        
        # Analyze each token
        for i, token in enumerate(tokens):
            if isinstance(token, dict):
                address = token.get('address') or token.get('adress')
                name = token.get('name', f'Token_{i}')
                
                if not address:
                    continue
                
                print(f"\n📊 Token {i+1}/{len(tokens)}: {name}")
                
                # Analyze token
                result = self.analyze_token(address, name)
                if result:
                    all_results.append(result)
                    self.detected_patterns.extend(result['patterns'])
                
                # Be respectful to API
                time.sleep(1)
        
        # Generate trading summary
        self.generate_trading_summary(all_results)
        
        return all_results
    
    def generate_trading_summary(self, results):
        """Generate comprehensive trading summary"""
        print("\n" + "=" * 80)
        print("📊 TRADING SUMMARY & RECOMMENDATIONS")
        print("=" * 80)
        
        if not self.detected_patterns:
            print("❌ No patterns detected - no trading opportunities")
            return
        
        # Count patterns by type
        pattern_counts = {}
        for pattern in self.detected_patterns:
            pattern_type = pattern['pattern']
            if pattern_type not in pattern_counts:
                pattern_counts[pattern_type] = 0
            pattern_counts[pattern_type] += 1
        
        print(f"\n🎯 PATTERNS DETECTED:")
        for pattern_type, count in pattern_counts.items():
            print(f"   • {pattern_type}: {count} tokens")
        
        # Generate trading recommendations
        print(f"\n🚫 HIGH PRIORITY - AVOID BUYING:")
        bearish_patterns = [p for p in self.detected_patterns if 'bearish' in p['pattern']]
        for pattern in bearish_patterns:
            print(f"   🚨 {pattern['pattern'].upper()}: {pattern['success_rate']}% success rate")
            print(f"      ACTION: {pattern['action']}")
            print(f"      RISK: {pattern['risk_level']}")
        
        print(f"\n🟢 LOWER PRIORITY - CONSIDER SMALL BUYS:")
        bullish_patterns = [p for p in self.detected_patterns if 'bullish' in p['pattern']]
        for pattern in bullish_patterns:
            print(f"   🟢 {pattern['pattern'].upper()}: {pattern['success_rate']}% success rate")
            print(f"      ACTION: {pattern['action']}")
            print(f"      POSITION SIZE: {pattern.get('position_size', 'N/A')}")
            print(f"      RISK: {pattern['risk_level']}")
        
        # Portfolio recommendations
        print(f"\n💰 PORTFOLIO RECOMMENDATIONS:")
        if bearish_patterns:
            print(f"   🚫 AVOID: {len(bearish_patterns)} tokens showing bearish patterns")
            print(f"      These have 75-81% chance of price going DOWN")
        if bullish_patterns:
            print(f"   🟢 CONSIDER: {len(bullish_patterns)} tokens showing bullish patterns")
            print(f"      These have 20-23% chance of price going UP")
            print(f"      Use very small position sizes (3-8% portfolio)")
        
        print(f"\n🛡️ RISK MANAGEMENT:")
        print(f"   • Maximum portfolio exposure: 20%")
        print(f"   • Stop loss: 5-8% for bullish positions")
        print(f"   • Take profit: 15-25% for bullish positions")
        print(f"   • Never buy tokens showing bearish patterns")

def main():
    """Main function to run the live trading system"""
    print("🎯 LIVE TRADING PATTERN DETECTION SYSTEM")
    print("Based on 365,195 patterns analyzed across all 26 tokens")
    print("\n🚀 Starting live analysis...")
    
    # Create detector
    detector = LiveTradingPatternDetector()
    
    # Run analysis (limit to first 5 tokens for demo)
    results = detector.run_live_analysis(max_tokens=5)
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Analyzed {len(results)} tokens")
    print(f"🎯 Detected {len(detector.detected_patterns)} patterns")
    
    # Save results to file
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"live_analysis_results_{timestamp}.json"
        
        # Convert to JSON-serializable format
        json_results = []
        for result in results:
            json_result = {
                'token': result['token'],
                'timestamp': result['timestamp'],
                'patterns': []
            }
            for pattern in result['patterns']:
                json_pattern = {k: v for k, v in pattern.items() if k != 'metrics'}
                json_result['patterns'].append(json_pattern)
            json_results.append(json_result)
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")

if __name__ == "__main__":
    main()
