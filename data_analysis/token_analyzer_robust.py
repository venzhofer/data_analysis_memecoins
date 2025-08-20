import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class RobustTokenAnalyzer:
    def __init__(self):
        self.tokens_webhook = "https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4"
        self.trading_webhook = "https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0"
        self.tokens_data = {}
        self.trading_data = {}
        self.analysis_results = {}
        
        # Fallback sample data for testing
        self.sample_tokens = self._generate_sample_tokens()
        self.sample_trading_data = self._generate_sample_trading_data()
        
    def _generate_sample_tokens(self) -> List[Dict]:
        """Generate sample token data for testing when webhooks fail"""
        return [
            {"address": "0x1234567890123456789012345678901234567890", "name": "Sample Token 1", "symbol": "ST1"},
            {"address": "0x2345678901234567890123456789012345678901", "name": "Sample Token 2", "symbol": "ST2"},
            {"address": "0x3456789012345678901234567890123456789012", "name": "Sample Token 3", "symbol": "ST3"},
            {"address": "0x4567890123456789012345678901234567890123", "name": "Sample Token 4", "symbol": "ST4"},
            {"address": "0x5678901234567890123456789012345678901234", "name": "Sample Token 5", "symbol": "ST5"},
        ]
    
    def _generate_sample_trading_data(self) -> Dict[str, List[Dict]]:
        """Generate sample trading data for testing"""
        trading_data = {}
        
        # Token 1: Strong rise
        trading_data["0x1234567890123456789012345678901234567890"] = [
            {"price": 1.0, "timestamp": "2024-01-01T00:00:00Z"},
            {"price": 1.2, "timestamp": "2024-01-01T00:00:05Z"},
            {"price": 1.5, "timestamp": "2024-01-01T00:00:10Z"},
            {"price": 1.8, "timestamp": "2024-01-01T00:00:15Z"},
            {"price": 2.1, "timestamp": "2024-01-01T00:00:20Z"},
            {"price": 2.5, "timestamp": "2024-01-01T00:00:25Z"},
            {"price": 2.8, "timestamp": "2024-01-01T00:00:30Z"},
            {"price": 3.2, "timestamp": "2024-01-01T00:00:35Z"},
            {"price": 3.5, "timestamp": "2024-01-01T00:00:40Z"},
            {"price": 3.8, "timestamp": "2024-01-01T00:00:45Z"},
        ]
        
        # Token 2: Died
        trading_data["0x2345678901234567890123456789012345678901"] = [
            {"price": 1.0, "timestamp": "2024-01-01T00:00:00Z"},
            {"price": 0.9, "timestamp": "2024-01-01T00:00:05Z"},
            {"price": 0.7, "timestamp": "2024-01-01T00:00:10Z"},
            {"price": 0.5, "timestamp": "2024-01-01T00:00:15Z"},
            {"price": 0.3, "timestamp": "2024-01-01T00:00:20Z"},
            {"price": 0.2, "timestamp": "2024-01-01T00:00:25Z"},
            {"price": 0.15, "timestamp": "2024-01-01T00:00:30Z"},
            {"price": 0.1, "timestamp": "2024-01-01T00:00:35Z"},
            {"price": 0.05, "timestamp": "2024-01-01T00:00:40Z"},
            {"price": 0.02, "timestamp": "2024-01-01T00:00:45Z"},
        ]
        
        # Token 3: Moderate rise
        trading_data["0x3456789012345678901234567890123456789012"] = [
            {"price": 1.0, "timestamp": "2024-01-01T00:00:00Z"},
            {"price": 1.05, "timestamp": "2024-01-01T00:00:05Z"},
            {"price": 1.1, "timestamp": "2024-01-01T00:00:10Z"},
            {"price": 1.15, "timestamp": "2024-01-01T00:00:15Z"},
            {"price": 1.2, "timestamp": "2024-01-01T00:00:20Z"},
            {"price": 1.18, "timestamp": "2024-01-01T00:00:25Z"},
            {"price": 1.22, "timestamp": "2024-01-01T00:00:30Z"},
            {"price": 1.25, "timestamp": "2024-01-01T00:00:35Z"},
            {"price": 1.28, "timestamp": "2024-01-01T00:00:40Z"},
            {"price": 1.3, "timestamp": "2024-01-01T00:00:45Z"},
        ]
        
        # Token 4: Significant drop
        trading_data["0x4567890123456789012345678901234567890123"] = [
            {"price": 1.0, "timestamp": "2024-01-01T00:00:00Z"},
            {"price": 0.95, "timestamp": "2024-01-01T00:00:05Z"},
            {"price": 0.85, "timestamp": "2024-01-01T00:00:10Z"},
            {"price": 0.75, "timestamp": "2024-01-01T00:00:15Z"},
            {"price": 0.65, "timestamp": "2024-01-01T00:00:20Z"},
            {"price": 0.55, "timestamp": "2024-01-01T00:00:25Z"},
            {"price": 0.45, "timestamp": "2024-01-01T00:00:30Z"},
            {"price": 0.35, "timestamp": "2024-01-01T00:00:35Z"},
            {"price": 0.3, "timestamp": "2024-01-01T00:00:40Z"},
            {"price": 0.25, "timestamp": "2024-01-01T00:00:45Z"},
        ]
        
        # Token 5: Stable
        trading_data["0x5678901234567890123456789012345678901234"] = [
            {"price": 1.0, "timestamp": "2024-01-01T00:00:00Z"},
            {"price": 1.01, "timestamp": "2024-01-01T00:00:05Z"},
            {"price": 0.99, "timestamp": "2024-01-01T00:00:10Z"},
            {"price": 1.02, "timestamp": "2024-01-01T00:00:15Z"},
            {"price": 0.98, "timestamp": "2024-01-01T00:00:20Z"},
            {"price": 1.03, "timestamp": "2024-01-01T00:00:25Z"},
            {"price": 0.97, "timestamp": "2024-01-01T00:00:30Z"},
            {"price": 1.01, "timestamp": "2024-01-01T00:00:35Z"},
            {"price": 0.99, "timestamp": "2024-01-01T00:00:40Z"},
            {"price": 1.0, "timestamp": "2024-01-01T00:00:45Z"},
        ]
        
        return trading_data
    
    def fetch_tokens_info(self, use_sample: bool = False) -> List[Dict]:
        """Fetch basic information about all tokens"""
        if use_sample:
            print("Using sample token data for testing...")
            tokens = self.sample_tokens
        else:
            try:
                print("Fetching tokens information from webhook...")
                response = requests.get(self.tokens_webhook, timeout=30)
                response.raise_for_status()
                tokens = response.json()
                print(f"Successfully fetched {len(tokens)} tokens from webhook")
            except Exception as e:
                print(f"Webhook failed: {e}")
                print("Falling back to sample data...")
                tokens = self.sample_tokens
        
        # Store token info
        for token in tokens:
            if isinstance(token, dict) and 'address' in token:
                self.tokens_data[token['address']] = token
                
        return tokens
    
    def fetch_trading_data(self, token_address: str, use_sample: bool = False, max_retries: int = 3) -> Optional[List[Dict]]:
        """Fetch 5-second trading data for a specific token"""
        if use_sample:
            return self.sample_trading_data.get(token_address, [])
        
        url = f"{self.trading_webhook}?token={token_address}"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0:
                    return data
                else:
                    print(f"No trading data for token {token_address[:10]}...")
                    return None
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed for {token_address[:10]}, retrying...")
                    time.sleep(1)
                else:
                    print(f"Failed to fetch trading data for {token_address[:10]}: {e}")
                    return None
        
        return None
    
    def analyze_token_performance(self, trading_data: List[Dict]) -> Dict:
        """Analyze the performance pattern of a token"""
        if not trading_data or len(trading_data) < 5:
            return {"status": "insufficient_data", "pattern": "unknown"}
        
        # Extract price and timestamp data
        prices = []
        timestamps = []
        
        for entry in trading_data:
            if isinstance(entry, dict):
                # Try different possible price field names
                price = None
                for field in ['price', 'value', 'amount', 'price_usd', 'price_eth']:
                    if field in entry and entry[field] is not None:
                        try:
                            price = float(entry[field])
                            break
                        except (ValueError, TypeError):
                            continue
                
                if price is not None and price > 0:
                    prices.append(price)
                    
                    # Try to get timestamp
                    timestamp = None
                    for field in ['timestamp', 'time', 'date', 'created_at']:
                        if field in entry and entry[field] is not None:
                            try:
                                if isinstance(entry[field], str):
                                    timestamp = pd.to_datetime(entry[field])
                                else:
                                    timestamp = pd.to_datetime(entry[field], unit='s')
                                break
                            except:
                                continue
                    
                    if timestamp is not None:
                        timestamps.append(timestamp)
        
        if len(prices) < 5:
            return {"status": "insufficient_data", "pattern": "unknown"}
        
        # Calculate performance metrics
        initial_price = prices[0]
        final_price = prices[-1]
        max_price = max(prices)
        min_price = min(prices)
        
        # Calculate percentage changes
        total_change_pct = ((final_price - initial_price) / initial_price) * 100
        max_gain_pct = ((max_price - initial_price) / initial_price) * 100
        max_loss_pct = ((min_price - initial_price) / initial_price) * 100
        
        # Determine pattern
        if total_change_pct > 20:
            pattern = "strong_rise"
        elif total_change_pct > 5:
            pattern = "moderate_rise"
        elif total_change_pct < -50:
            pattern = "died"
        elif total_change_pct < -20:
            pattern = "significant_drop"
        elif total_change_pct < -5:
            pattern = "moderate_drop"
        else:
            pattern = "stable"
        
        # Calculate volatility
        price_changes = np.diff(prices)
        volatility = np.std(price_changes) / np.mean(prices) if np.mean(prices) > 0 else 0
        
        # Check for recovery patterns
        recovery_threshold = 0.8  # 80% of max price
        recovery_point = None
        
        for i, price in enumerate(prices):
            if price >= initial_price * recovery_threshold:
                recovery_point = i
                break
        
        return {
            "status": "analyzed",
            "pattern": pattern,
            "initial_price": initial_price,
            "final_price": final_price,
            "max_price": max_price,
            "min_price": min_price,
            "total_change_pct": total_change_pct,
            "max_gain_pct": max_gain_pct,
            "max_loss_pct": max_loss_pct,
            "volatility": volatility,
            "recovery_point": recovery_point,
            "data_points": len(prices),
            "prices": prices,
            "timestamps": timestamps
        }
    
    def run_analysis(self, max_tokens: int = 100, use_sample: bool = False) -> Dict:
        """Run the complete analysis"""
        print("Starting token analysis...")
        
        # Fetch tokens
        tokens = self.fetch_tokens_info(use_sample=use_sample)
        if not tokens:
            print("No tokens found. Exiting.")
            return {}
        
        # Limit analysis to max_tokens
        token_addresses = list(self.tokens_data.keys())[:max_tokens]
        print(f"Analyzing {len(token_addresses)} tokens...")
        
        results = {
            "total_tokens": len(token_addresses),
            "analyzed_tokens": 0,
            "patterns": {},
            "summary": {},
            "detailed_results": {},
            "analysis_mode": "sample" if use_sample else "webhook"
        }
        
        for i, token_address in enumerate(token_addresses):
            print(f"Analyzing token {i+1}/{len(token_addresses)}: {token_address[:10]}...")
            
            # Fetch trading data
            trading_data = self.fetch_trading_data(token_address, use_sample=use_sample)
            if trading_data:
                # Analyze performance
                analysis = self.analyze_token_performance(trading_data)
                
                if analysis["status"] == "analyzed":
                    results["analyzed_tokens"] += 1
                    pattern = analysis["pattern"]
                    
                    # Count patterns
                    if pattern not in results["patterns"]:
                        results["patterns"][pattern] = 0
                    results["patterns"][pattern] += 1
                    
                    # Store detailed results
                    results["detailed_results"][token_address] = {
                        "analysis": analysis,
                        "token_info": self.tokens_data.get(token_address, {})
                    }
                    
                    # Add delay only for webhook calls
                    if not use_sample:
                        time.sleep(0.5)
            
            # Progress update every 5 tokens
            if (i + 1) % 5 == 0:
                print(f"Progress: {i+1}/{len(token_addresses)} tokens analyzed")
        
        # Generate summary statistics
        self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: Dict):
        """Generate summary statistics"""
        if results["analyzed_tokens"] == 0:
            return
        
        total = results["analyzed_tokens"]
        
        # Calculate percentages
        for pattern, count in results["patterns"].items():
            percentage = (count / total) * 100
            results["summary"][pattern] = {
                "count": count,
                "percentage": percentage
            }
        
        # Categorize results
        died_count = results["patterns"].get("died", 0)
        rose_count = (results["patterns"].get("strong_rise", 0) + 
                     results["patterns"].get("moderate_rise", 0))
        dropped_count = (results["patterns"].get("significant_drop", 0) + 
                        results["patterns"].get("moderate_drop", 0))
        stable_count = results["patterns"].get("stable", 0)
        
        results["summary"]["categorized"] = {
            "died": {"count": died_count, "percentage": (died_count / total) * 100},
            "rose": {"count": rose_count, "percentage": (rose_count / total) * 100},
            "dropped": {"count": dropped_count, "percentage": (dropped_count / total) * 100},
            "stable": {"count": stable_count, "percentage": (stable_count / total) * 100}
        }
    
    def save_results(self, results: Dict, filename: str = "token_analysis_results.json"):
        """Save analysis results to JSON file"""
        try:
            # Remove price arrays to reduce file size
            clean_results = results.copy()
            for token_data in clean_results.get("detailed_results", {}).values():
                if "analysis" in token_data and "prices" in token_data["analysis"]:
                    token_data["analysis"]["prices"] = f"Array of {len(token_data['analysis']['prices'])} prices"
                    token_data["analysis"]["timestamps"] = f"Array of {len(token_data['analysis']['timestamps'])} timestamps"
            
            with open(filename, 'w') as f:
                json.dump(clean_results, f, indent=2, default=str)
            
            print(f"Results saved to {filename}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def create_visualizations(self, results: Dict):
        """Create visualizations of the analysis results"""
        if results["analyzed_tokens"] == 0:
            print("No data to visualize")
            return
        
        # Set style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Token Performance Analysis ({results["analysis_mode"].upper()} MODE)', fontsize=16, fontweight='bold')
        
        # 1. Pattern Distribution Pie Chart
        if results["patterns"]:
            patterns = list(results["patterns"].keys())
            counts = list(results["patterns"].values())
            
            axes[0, 0].pie(counts, labels=patterns, autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Token Performance Patterns')
        
        # 2. Categorized Results Bar Chart
        if "categorized" in results["summary"]:
            categories = list(results["summary"]["categorized"].keys())
            percentages = [results["summary"]["categorized"][cat]["percentage"] for cat in categories]
            
            colors = ['red', 'green', 'orange', 'blue']
            bars = axes[0, 1].bar(categories, percentages, color=colors)
            axes[0, 1].set_title('Token Performance Categories')
            axes[0, 1].set_ylabel('Percentage (%)')
            axes[0, 1].set_ylim(0, max(percentages) * 1.1)
            
            # Add value labels on bars
            for bar, percentage in zip(bars, percentages):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{percentage:.1f}%', ha='center', va='bottom')
        
        # 3. Price Change Distribution
        if results["detailed_results"]:
            changes = []
            for token_data in results["detailed_results"].values():
                if "analysis" in token_data:
                    changes.append(token_data["analysis"]["total_change_pct"])
            
            if changes:
                axes[1, 0].hist(changes, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[1, 0].set_title('Distribution of Price Changes (%)')
                axes[1, 0].set_xlabel('Price Change (%)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Change')
                axes[1, 0].legend()
        
        # 4. Volatility vs Performance
        if results["detailed_results"]:
            volatilities = []
            changes = []
            for token_data in results["detailed_results"].values():
                if "analysis" in token_data:
                    volatilities.append(token_data["analysis"]["volatility"])
                    changes.append(token_data["analysis"]["total_change_pct"])
            
            if volatilities and changes:
                axes[1, 1].scatter(volatilities, changes, alpha=0.6, color='purple')
                axes[1, 1].set_title('Volatility vs Performance')
                axes[1, 1].set_xlabel('Volatility')
                axes[1, 1].set_ylabel('Price Change (%)')
                axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('token_analysis_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'token_analysis_charts.png'")
    
    def print_summary(self, results: Dict):
        """Print a summary of the analysis results"""
        if results["analyzed_tokens"] == 0:
            print("No tokens were analyzed successfully.")
            return
        
        print("\n" + "="*60)
        print("TOKEN ANALYSIS SUMMARY")
        print("="*60)
        print(f"Analysis Mode: {results['analysis_mode'].upper()}")
        print(f"Total tokens analyzed: {results['analyzed_tokens']}")
        print(f"Success rate: {(results['analyzed_tokens'] / results['total_tokens']) * 100:.1f}%")
        
        if "categorized" in results["summary"]:
            print("\nPERFORMANCE CATEGORIES:")
            print("-" * 30)
            for category, data in results["summary"]["categorized"].items():
                print(f"{category.upper()}: {data['count']} tokens ({data['percentage']:.1f}%)")
        
        if results["patterns"]:
            print("\nDETAILED PATTERNS:")
            print("-" * 20)
            for pattern, count in results["patterns"].items():
                percentage = (count / results["analyzed_tokens"]) * 100
                print(f"{pattern}: {count} tokens ({percentage:.1f}%)")
        
        print("\n" + "="*60)
        print("KEY INSIGHTS:")
        print("="*60)
        
        if "categorized" in results["summary"]:
            died_pct = results["summary"]["categorized"]["died"]["percentage"]
            rose_pct = results["summary"]["categorized"]["rose"]["percentage"]
            dropped_pct = results["summary"]["categorized"]["dropped"]["percentage"]
            
            print(f"• {died_pct:.1f}% of tokens DIED (no point of return)")
            print(f"• {rose_pct:.1f}% of tokens ROSE after launch")
            print(f"• {dropped_pct:.1f}% of tokens DROPPED significantly")
            print(f"• {100 - died_pct - rose_pct - dropped_pct:.1f}% of tokens were STABLE")
        
        print("\nAnalysis complete! Check the generated files for detailed results.")

def main():
    """Main function to run the analysis"""
    print("🚀 Starting Robust Token Performance Analysis")
    print("=" * 60)
    
    analyzer = RobustTokenAnalyzer()
    
    # Try webhook first, fallback to sample data
    print("Attempting to use webhook data...")
    results = analyzer.run_analysis(max_tokens=50, use_sample=False)
    
    if not results or results["analyzed_tokens"] == 0:
        print("\nWebhook analysis failed. Using sample data for demonstration...")
        results = analyzer.run_analysis(max_tokens=5, use_sample=True)
    
    if results and results["analyzed_tokens"] > 0:
        # Save results
        analyzer.save_results(results)
        
        # Create visualizations
        analyzer.create_visualizations(results)
        
        # Print summary
        analyzer.print_summary(results)
        
    else:
        print("❌ Analysis failed completely.")

if __name__ == "__main__":
    main()
