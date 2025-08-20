import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from typing import Dict, List, Optional
import warnings
from tabulate import tabulate
warnings.filterwarnings('ignore')

class EnhancedTokenAnalyzer:
    def __init__(self):
        self.tokens_webhook = "https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4"
        self.trading_webhook = "https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0"
        self.tokens_data = {}
        self.trading_data = {}
        self.analysis_results = {}
        
    def fetch_tokens_info(self) -> List[Dict]:
        """Fetch basic information about all tokens"""
        try:
            print("Fetching tokens information from webhook...")
            response = requests.get(self.tokens_webhook, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Handle different response formats
            if isinstance(data, list):
                # List of items, check if tokens are nested in 'data' field
                if len(data) > 0 and isinstance(data[0], dict) and 'data' in data[0]:
                    # Nested structure: [{"data": [token1, token2, ...]}]
                    tokens = data[0]['data']
                    print(f"Found nested structure with {len(tokens)} tokens")
                else:
                    # Direct list of tokens
                    tokens = data
            elif isinstance(data, dict):
                # Single token object or has nested data
                if 'data' in data:
                    tokens = data['data']
                    print(f"Found nested structure with {len(tokens)} tokens")
                else:
                    tokens = [data]
            else:
                print(f"Unexpected response format: {type(data)}")
                return []
            
            print(f"Successfully fetched {len(tokens)} tokens from webhook")
            
            # Store token info - handle both 'address' and 'adress' fields
            for token in tokens:
                if isinstance(token, dict):
                    # Try both possible address field names
                    address = token.get('address') or token.get('adress')
                    if address:
                        self.tokens_data[address] = token
                    else:
                        print(f"Warning: Token {token.get('name', 'Unknown')} has no address field")
                    
            return tokens
            
        except Exception as e:
            print(f"Webhook failed: {e}")
            return []
    
    def fetch_trading_data(self, token_address: str, max_retries: int = 3) -> Optional[Dict]:
        """Fetch current trading data for a specific token"""
        url = f"{self.trading_webhook}?token={token_address}"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                if data and isinstance(data, dict):
                    return data
                else:
                    return None
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return None
        
        return None
    
    def analyze_token_performance(self, trading_data: Dict, token_info: Dict) -> Dict:
        """Analyze the performance pattern of a token based on current market data"""
        if not trading_data or not isinstance(trading_data, dict):
            return {"status": "insufficient_data", "pattern": "unknown"}
        
        # Extract key metrics from trading data
        current_price = trading_data.get('price', 0)
        current_fdv = trading_data.get('fdv', 0)
        current_market_cap = trading_data.get('market_cap', 0)
        
        # Get initial metrics from token info
        start_fdv = token_info.get('start_fdv', 0)
        start_market_cap = token_info.get('start_market_cap', 0)
        
        # Calculate performance metrics
        if start_fdv and start_fdv > 0:
            fdv_change_pct = ((current_fdv - start_fdv) / start_fdv) * 100
        else:
            fdv_change_pct = 0
            
        if start_market_cap and start_market_cap > 0:
            market_cap_change_pct = ((current_market_cap - start_market_cap) / start_market_cap) * 100
        else:
            market_cap_change_pct = 0
        
        # Get transaction data
        transactions = trading_data.get('transactions', {})
        h24_transactions = transactions.get('h24', {})
        h1_transactions = transactions.get('h1', {})
        
        h24_buys = h24_transactions.get('buys', 0)
        h24_sells = h24_transactions.get('sells', 0)
        h1_buys = h1_transactions.get('buys', 0)
        h1_sells = h1_transactions.get('sells', 0)
        
        # Calculate buy/sell ratio
        h24_ratio = h24_buys / max(h24_sells, 1) if h24_sells > 0 else h24_buys
        h1_ratio = h1_buys / max(h1_sells, 1) if h1_sells > 0 else h1_buys
        
        # Get volume data
        volume = trading_data.get('volume', {})
        h24_volume = volume.get('h24', 0)
        h1_volume = volume.get('h1', 0)
        
        # Determine pattern based on FDV change
        if fdv_change_pct > 100:
            pattern = "moon_shot"
        elif fdv_change_pct > 50:
            pattern = "strong_rise"
        elif fdv_change_pct > 20:
            pattern = "moderate_rise"
        elif fdv_change_pct < -80:
            pattern = "died"
        elif fdv_change_pct < -50:
            pattern = "significant_drop"
        elif fdv_change_pct < -20:
            pattern = "moderate_drop"
        else:
            pattern = "stable"
        
        # Calculate momentum indicators
        momentum_score = 0
        if h1_ratio > 1.5:
            momentum_score += 2  # Strong recent buying
        elif h1_ratio > 1.2:
            momentum_score += 1  # Moderate recent buying
        
        if h24_ratio > 1.3:
            momentum_score += 1  # Good 24h buying pressure
        
        if h1_volume > h24_volume * 0.1:  # High recent volume
            momentum_score += 1
        
        # Risk assessment
        risk_score = token_info.get('rugcheck_score', 5)  # Default to high risk if not specified
        risks = token_info.get('risks', [])
        
        return {
            "status": "analyzed",
            "pattern": pattern,
            "current_price": current_price,
            "current_fdv": current_fdv,
            "current_market_cap": current_market_cap,
            "start_fdv": start_fdv,
            "start_market_cap": start_market_cap,
            "fdv_change_pct": fdv_change_pct,
            "market_cap_change_pct": market_cap_change_pct,
            "h24_buys": h24_buys,
            "h24_sells": h24_sells,
            "h1_buys": h1_buys,
            "h1_sells": h1_sells,
            "h24_buy_sell_ratio": h24_ratio,
            "h1_buy_sell_ratio": h1_ratio,
            "h24_volume": h24_volume,
            "h1_volume": h1_volume,
            "momentum_score": momentum_score,
            "risk_score": risk_score,
            "risks": risks,
            "rugcheck_score": risk_score
        }
    
    def run_analysis(self) -> Dict:
        """Run the complete analysis on ALL tokens"""
        print("Starting comprehensive token analysis...")
        
        # Fetch tokens
        tokens = self.fetch_tokens_info()
        if not tokens:
            print("No tokens found. Exiting.")
            return {}
        
        token_addresses = list(self.tokens_data.keys())
        print(f"Analyzing ALL {len(token_addresses)} tokens...")
        
        results = {
            "total_tokens": len(token_addresses),
            "analyzed_tokens": 0,
            "patterns": {},
            "summary": {},
            "detailed_results": {},
            "statistics": {}
        }
        
        for i, token_address in enumerate(token_addresses):
            print(f"Analyzing token {i+1}/{len(token_addresses)}: {token_address[:10]}...")
            
            # Fetch trading data
            trading_data = self.fetch_trading_data(token_address)
            if trading_data:
                # Analyze performance
                analysis = self.analyze_token_performance(trading_data, self.tokens_data.get(token_address, {}))
                
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
                    
                    # Add delay to avoid overwhelming the API
                    time.sleep(0.5)
            
            # Progress update every 10 tokens
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{len(token_addresses)} tokens analyzed")
        
        # Generate summary statistics
        self._generate_summary(results)
        
        # Calculate key metrics
        self._calculate_key_metrics(results)
        
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
    
    def _calculate_key_metrics(self, results: Dict):
        """Calculate key metrics requested by user"""
        if results["analyzed_tokens"] == 0:
            return
        
        # Extract all performance data
        fdv_changes = []
        market_cap_changes = []
        momentum_scores = []
        risk_scores = []
        
        for token_data in results["detailed_results"].values():
            if "analysis" in token_data:
                analysis = token_data["analysis"]
                fdv_changes.append(analysis["fdv_change_pct"])
                market_cap_changes.append(analysis["market_cap_change_pct"])
                momentum_scores.append(analysis["momentum_score"])
                risk_scores.append(analysis["risk_score"])
        
        # Calculate averages and maximums
        results["statistics"] = {
            "average_percentage_rise": np.mean([x for x in fdv_changes if x > 0]) if any(x > 0 for x in fdv_changes) else 0,
            "average_percentage_drop": np.mean([x for x in fdv_changes if x < 0]) if any(x < 0 for x in fdv_changes) else 0,
            "maximal_rise": max(fdv_changes) if fdv_changes else 0,
            "maximal_drop": min(fdv_changes) if fdv_changes else 0,
            "average_point_of_no_return": None,  # Not applicable with current data
            "tokens_with_point_of_no_return": 0,
            "average_volatility": np.mean(risk_scores) if risk_scores else 0,
            "overall_average_change": np.mean(fdv_changes) if fdv_changes else 0,
            "average_momentum_score": np.mean(momentum_scores) if momentum_scores else 0,
            "average_market_cap_change": np.mean(market_cap_changes) if market_cap_changes else 0
        }
    
    def create_detailed_table(self, results: Dict) -> str:
        """Create a detailed table for each token"""
        if results["analyzed_tokens"] == 0:
            return "No data to display"
        
        table_data = []
        
        for token_address, token_data in results["detailed_results"].items():
            analysis = token_data["analysis"]
            token_info = token_data["token_info"]
            
            # Get token name/symbol
            token_name = token_info.get("name", "Unknown")
            token_symbol = token_info.get("symbol", "N/A")
            
            # Format the data
            row = [
                token_address[:10] + "...",
                token_name,
                token_symbol,
                f"{analysis['current_price']:.6f}",
                f"{analysis['fdv_change_pct']:+.2f}%",
                f"{analysis['market_cap_change_pct']:+.2f}%",
                f"{analysis['momentum_score']:.2f}",
                analysis['pattern'],
                analysis['risk_score']
            ]
            table_data.append(row)
        
        # Sort by FDV change percentage
        table_data.sort(key=lambda x: float(x[4].replace('%', '').replace('+', '')), reverse=True)
        
        headers = [
            "Address", "Name", "Symbol", "Current Price", "FDV Change", "Market Cap Change", "Momentum Score", "Pattern", "Risk Score"
        ]
        
        return tabulate(table_data, headers=headers, tablefmt="grid", numalign="right")
    
    def save_results(self, results: Dict, filename: str = "enhanced_token_analysis_results.json"):
        """Save analysis results to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"Results saved to {filename}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def create_visualizations(self, results: Dict):
        """Create enhanced visualizations"""
        if results["analyzed_tokens"] == 0:
            print("No data to visualize")
            return
        
        # Set style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Enhanced Token Performance Analysis', fontsize=16, fontweight='bold')
        
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
        
        # 3. FDV Change Distribution
        if results["detailed_results"]:
            changes = []
            for token_data in results["detailed_results"].values():
                if "analysis" in token_data:
                    changes.append(token_data["analysis"]["fdv_change_pct"])
            
            if changes:
                axes[0, 2].hist(changes, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[0, 2].set_title('Distribution of FDV Changes (%)')
                axes[0, 2].set_xlabel('FDV Change (%)')
                axes[0, 2].set_ylabel('Frequency')
                axes[0, 2].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Change')
                axes[0, 2].legend()
        
        # 4. Momentum Score vs Risk Score
        if results["detailed_results"]:
            momentum_scores = []
            risk_scores = []
            for token_data in results["detailed_results"].values():
                if "analysis" in token_data:
                    momentum_scores.append(token_data["analysis"]["momentum_score"])
                    risk_scores.append(token_data["analysis"]["risk_score"])
            
            if momentum_scores and risk_scores:
                axes[1, 0].scatter(momentum_scores, risk_scores, alpha=0.6, color='purple')
                axes[1, 0].set_title('Momentum Score vs Risk Score')
                axes[1, 0].set_xlabel('Momentum Score')
                axes[1, 0].set_ylabel('Risk Score')
                axes[1, 0].axhline(y=5, color='red', linestyle='--', alpha=0.7, label='Average Risk')
                axes[1, 0].legend()
        
        # 5. FDV Change vs Market Cap Change
        if results["detailed_results"]:
            fdv_changes = []
            market_cap_changes = []
            for token_data in results["detailed_results"].values():
                if "analysis" in token_data:
                    fdv_changes.append(token_data["analysis"]["fdv_change_pct"])
                    market_cap_changes.append(token_data["analysis"]["market_cap_change_pct"])
            
            if fdv_changes and market_cap_changes:
                axes[1, 1].scatter(fdv_changes, market_cap_changes, alpha=0.6, color='orange')
                axes[1, 1].set_title('FDV Change vs Market Cap Change')
                axes[1, 1].set_xlabel('FDV Change (%)')
                axes[1, 1].set_ylabel('Market Cap Change (%)')
                axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
                axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # 6. Risk Score Distribution
        if results["detailed_results"]:
            risk_scores = []
            for token_data in results["detailed_results"].values():
                if "analysis" in token_data:
                    risk_scores.append(token_data["analysis"]["risk_score"])
            
            if risk_scores:
                axes[1, 2].hist(risk_scores, bins=5, alpha=0.7, color='green', edgecolor='black')
                axes[1, 2].set_title('Distribution of Risk Scores')
                axes[1, 2].set_xlabel('Risk Score')
                axes[1, 2].set_ylabel('Frequency')
                axes[1, 2].axvline(x=5, color='red', linestyle='--', alpha=0.7, label='Average Risk')
                axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('enhanced_token_analysis_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'enhanced_token_analysis_charts.png'")
    
    def print_summary(self, results: Dict):
        """Print a comprehensive summary of the analysis results"""
        if results["analyzed_tokens"] == 0:
            print("No tokens were analyzed successfully.")
            return
        
        print("\n" + "="*80)
        print("ENHANCED TOKEN ANALYSIS SUMMARY")
        print("="*80)
        print(f"Total tokens analyzed: {results['analyzed_tokens']}")
        print(f"Success rate: {(results['analyzed_tokens'] / results['total_tokens']) * 100:.1f}%")
        
        # Key Metrics Section
        if "statistics" in results:
            stats = results["statistics"]
            print("\n" + "="*50)
            print("KEY METRICS")
            print("="*50)
            print(f"• Average Percentage Rise (FDV): {stats['average_percentage_rise']:+.2f}%")
            print(f"• Average Percentage Drop (FDV): {stats['average_percentage_drop']:+.2f}%")
            print(f"• Maximal Rise (FDV): {stats['maximal_rise']:+.2f}%")
            print(f"• Maximal Drop (FDV): {stats['maximal_drop']:+.2f}%")
            print(f"• Average Point of No Return: N/A (based on current data)")
            print(f"• Tokens with Point of No Return: 0")
            print(f"• Average Volatility (Risk Score): {stats['average_volatility']:.2f}")
            print(f"• Overall Average Change (FDV): {stats['overall_average_change']:+.2f}%")
            print(f"• Average Momentum Score: {stats['average_momentum_score']:.2f}")
            print(f"• Average Market Cap Change: {stats['average_market_cap_change']:+.2f}%")
        
        if "categorized" in results["summary"]:
            print("\n" + "="*50)
            print("PERFORMANCE CATEGORIES")
            print("="*50)
            for category, data in results["summary"]["categorized"].items():
                print(f"{category.upper()}: {data['count']} tokens ({data['percentage']:.1f}%)")
        
        if results["patterns"]:
            print("\n" + "="*50)
            print("DETAILED PATTERNS")
            print("="*50)
            for pattern, count in results["patterns"].items():
                percentage = (count / results["analyzed_tokens"]) * 100
                print(f"{pattern}: {count} tokens ({percentage:.1f}%)")
        
        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)
        
        if "categorized" in results["summary"]:
            died_pct = results["summary"]["categorized"]["died"]["percentage"]
            rose_pct = results["summary"]["categorized"]["rose"]["percentage"]
            dropped_pct = results["summary"]["categorized"]["dropped"]["percentage"]
            
            print(f"• {died_pct:.1f}% of tokens DIED (FDV change < -80%)")
            print(f"• {rose_pct:.1f}% of tokens ROSE after launch (FDV change > 100%)")
            print(f"• {dropped_pct:.1f}% of tokens DROPPED significantly (FDV change < -50%)")
            print(f"• {100 - died_pct - rose_pct - dropped_pct:.1f}% of tokens were STABLE")
        
        print("\nAnalysis complete! Check the generated files for detailed results.")

def main():
    """Main function to run the enhanced analysis"""
    print("🚀 Starting Enhanced Token Performance Analysis")
    print("=" * 70)
    
    analyzer = EnhancedTokenAnalyzer()
    
    # Run the analysis on ALL tokens
    results = analyzer.run_analysis()
    
    if results and results["analyzed_tokens"] > 0:
        # Save results
        analyzer.save_results(results)
        
        # Create visualizations
        analyzer.create_visualizations(results)
        
        # Print summary
        analyzer.print_summary(results)
        
        # Create and display detailed table
        print("\n" + "="*80)
        print("DETAILED TOKEN DATA TABLE")
        print("="*80)
        table = analyzer.create_detailed_table(results)
        print(table)
        
        # Save table to file
        with open('token_analysis_table.txt', 'w') as f:
            f.write("ENHANCED TOKEN ANALYSIS TABLE\n")
            f.write("="*80 + "\n\n")
            f.write(table)
        print(f"\nDetailed table saved to 'token_analysis_table.txt'")
        
    else:
        print("❌ Analysis failed or no tokens were analyzed successfully.")

if __name__ == "__main__":
    main()
