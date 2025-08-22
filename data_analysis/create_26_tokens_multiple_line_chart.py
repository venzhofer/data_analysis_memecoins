#!/usr/bin/env python3
"""
26 Tokens Multiple Line Chart Creator
Creates a multiple line chart showing all 26 tokens as individual lines
Uses both webhooks: tokens info + trading data
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import time
import json
from typing import Dict, List, Optional
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultipleLineChartCreator:
    """Creates multiple line chart with all tokens as individual lines"""
    
    def __init__(self, 
                 tokens_webhook: str = "https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4",
                 trading_webhook: str = "https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0",
                 output_dir: str = "output/multiple_line_chart"):
        
        self.tokens_webhook = tokens_webhook
        self.trading_webhook = trading_webhook
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "TokenTS/1.0",
            "Accept": "application/json"
        })
        
        # Store data
        self.tokens_data = {}
        self.current_prices = {}
        
        # Color palette for 26 tokens
        self.colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#a6cee3', '#fb9a99', '#fdbf6f', '#cab2d6', '#ff9896',
            '#fdd49e', '#b3de69', '#fccde5', '#d9d9d9', '#ffed6f',
            '#c4e6f3', '#c9c9c9', '#f4a582', '#92c5de', '#fdb462',
            '#b3cd3e'
        ]
        
    def fetch_tokens_info(self) -> List[Dict]:
        """Fetch basic information about all tokens from first webhook"""
        try:
            logger.info("Fetching tokens information from first webhook...")
            response = self.session.get(self.tokens_webhook, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Handle nested structure
            if isinstance(data, list) and len(data) > 0 and 'data' in data[0]:
                tokens = data[0]['data']
                logger.info(f"Found nested structure with {len(tokens)} tokens")
            elif isinstance(data, list):
                tokens = data
            elif isinstance(data, dict) and 'data' in data:
                tokens = data['data']
            else:
                tokens = [data] if isinstance(data, dict) else []
            
            logger.info(f"Successfully fetched {len(tokens)} tokens from first webhook")
            return tokens
            
        except Exception as e:
            logger.error(f"Failed to fetch tokens: {e}")
            return []
    
    def fetch_token_price(self, token_address: str, retries: int = 3) -> Optional[Dict]:
        """Fetch current price data for a specific token from second webhook"""
        try:
            url = f"{self.trading_webhook}?token={token_address}"
            
            for attempt in range(retries):
                try:
                    response = self.session.get(url, timeout=30)
                    response.raise_for_status()
                    
                    if response.text.strip():
                        data = response.json()
                        return data
                    else:
                        return None
                        
                except Exception as e:
                    if attempt == retries - 1:
                        logger.error(f"Failed to fetch price data for {token_address[:10]} after {retries} attempts: {e}")
                        return None
                    time.sleep(1)
                    
        except Exception as e:
            logger.error(f"Failed to fetch price data for {token_address[:10]}: {e}")
            return None
    
    def collect_all_tokens_data(self):
        """Collect data for all tokens from both webhooks"""
        logger.info("Collecting data for all tokens from both webhooks...")
        
        # Fetch tokens info from first webhook
        tokens = self.fetch_tokens_info()
        if not tokens:
            logger.error("No tokens found from first webhook")
            return
        
        successful_tokens = 0
        
        for i, token in enumerate(tokens):
            try:
                # Handle both 'address' and 'adress' fields
                address = token.get('address') or token.get('adress')
                token_name = token.get('name', f'Token_{i}')
                
                if not address:
                    logger.warning(f"Token {token_name} has no address, skipping")
                    continue
                
                logger.info(f"Processing token {i+1}/{len(tokens)}: {token_name}")
                
                # Store token info from first webhook
                self.tokens_data[address] = {
                    'name': token_name,
                    'address': address,
                    'symbol': token.get('symbol', ''),
                    'index': i,
                    'start_fdv': token.get('start_fdv', 0),
                    'start_market_cap': token.get('start_market_cap', 0),
                    'created_at': token.get('created_at', ''),
                    'rugcheck_score': token.get('rugcheck_score', 0)
                }
                
                # Fetch current price data from second webhook
                price_data = self.fetch_token_price(address)
                if price_data:
                    # Extract price information
                    current_price = price_data.get('price', 0)
                    current_fdv = price_data.get('fdv', 0)
                    current_market_cap = price_data.get('market_cap', 0)
                    current_volume = price_data.get('volume', {})
                    
                    if current_price and current_price > 0:
                        # Calculate price change from start
                        start_fdv = token.get('start_fdv', current_fdv)
                        price_change_pct = ((current_fdv / start_fdv) - 1) * 100 if start_fdv > 0 else 0
                        
                        self.current_prices[address] = {
                            'current_price': current_price,
                            'current_fdv': current_fdv,
                            'current_market_cap': current_market_cap,
                            'start_fdv': start_fdv,
                            'price_change_pct': price_change_pct,
                            'volume_24h': current_volume.get('h24', 0),
                            'buy_sell_ratio': self._calculate_buy_sell_ratio(price_data.get('transactions', {}))
                        }
                        
                        successful_tokens += 1
                        logger.info(f"✓ {token_name}: ${current_price:.8f} ({price_change_pct:+.1f}%)")
                    else:
                        logger.warning(f"✗ {token_name}: invalid price data")
                else:
                    logger.warning(f"✗ {token_name}: no price data from second webhook")
                
                # Add delay to avoid overwhelming the API
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing token {token.get('name', 'Unknown')}: {e}")
                continue
        
        logger.info(f"Successfully collected data for {successful_tokens}/{len(tokens)} tokens")
    
    def _calculate_buy_sell_ratio(self, transactions: Dict) -> float:
        """Calculate buy/sell ratio from transaction data"""
        try:
            h24 = transactions.get('h24', {})
            buys = h24.get('buys', 0)
            sells = h24.get('sells', 0)
            
            if sells > 0:
                return buys / sells
            elif buys > 0:
                return float('inf')  # Only buys, no sells
            else:
                return 0  # No transactions
        except:
            return 0
    
    def create_multiple_line_chart(self):
        """Create the main multiple line chart with all tokens"""
        if not self.current_prices:
            logger.error("No price data available")
            return
        
        logger.info("Creating multiple line chart with all tokens...")
        
        # Create dataframe for easier manipulation
        data = []
        for address, price_info in self.current_prices.items():
            token_info = self.tokens_data[address]
            data.append({
                'name': token_info['name'],
                'address': address,
                'current_price': price_info['current_price'],
                'current_fdv': price_info['current_fdv'],
                'price_change_pct': price_info['price_change_pct'],
                'volume_24h': price_info['volume_24h'],
                'buy_sell_ratio': price_info['buy_sell_ratio'],
                'rugcheck_score': token_info['rugcheck_score']
            })
        
        df = pd.DataFrame(data)
        
        # Sort by price change percentage for better visualization
        df_sorted = df.sort_values('price_change_pct', ascending=False)
        
        # Create the main multiple line chart
        plt.figure(figsize=(24, 14))
        
        # Create x-axis positions (token indices)
        x_positions = np.arange(len(df_sorted))
        
        # Plot each token as a separate line
        for i, (_, token) in enumerate(df_sorted.iterrows()):
            # Get color for this token
            color = self.colors[i % len(self.colors)]
            
            # Plot the line for this token
            plt.plot([i], [token['current_price']], 
                    color=color, 
                    linewidth=3, 
                    marker='o', 
                    markersize=8, 
                    alpha=0.8, 
                    label=f"{token['name']} (${token['current_price']:.8f})")
            
            # Add token name label
            plt.annotate(f"{token['name'][:15]}...", 
                        (i, token['current_price']),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, rotation=45,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Connect all points with a trend line
        plt.plot(x_positions, df_sorted['current_price'], 
                color='blue', alpha=0.3, linewidth=2, linestyle='--', 
                label='Overall Price Trend')
        
        # Customize the chart
        plt.title('All 26 Tokens - Multiple Line Chart\n(Current Prices)', 
                 fontsize=24, fontweight='bold', pad=20)
        plt.xlabel('Token Index (Sorted by Performance)', fontsize=16, fontweight='bold')
        plt.ylabel('Current Price (USD)', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Set y-axis to log scale for better visualization of small prices
        plt.yscale('log')
        
        # Add reference lines for key price levels
        plt.axhline(y=0.0001, color='green', linestyle='--', alpha=0.7, linewidth=1, label='$0.0001')
        plt.axhline(y=0.00005, color='orange', linestyle='--', alpha=0.7, linewidth=1, label='$0.00005')
        plt.axhline(y=0.00001, color='red', linestyle='--', alpha=0.7, linewidth=1, label='$0.00001')
        
        # Create comprehensive legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, ncol=2)
        
        # Add summary statistics
        total_tokens = len(df_sorted)
        positive_tokens = len(df_sorted[df_sorted['price_change_pct'] > 0])
        negative_tokens = total_tokens - positive_tokens
        moon_shot_tokens = len(df_sorted[df_sorted['price_change_pct'] > 100])
        died_tokens = len(df_sorted[df_sorted['price_change_pct'] < -80])
        
        summary_text = f"""Summary:
Total Tokens: {total_tokens}
Positive Performance: {positive_tokens}
Negative Performance: {negative_tokens}
Moon Shot (>100%): {moon_shot_tokens}
Died (<-80%): {died_tokens}
Best: {df_sorted.iloc[0]['name']} ({df_sorted.iloc[0]['price_change_pct']:.1f}%)
Worst: {df_sorted.iloc[-1]['name']} ({df_sorted.iloc[-1]['price_change_pct']:.1f}%)"""
        
        plt.figtext(0.02, 0.02, summary_text, 
                   fontsize=12, bbox=dict(boxstyle='round,pad=0.5', 
                                        facecolor='lightblue', alpha=0.8))
        
        # Format x-axis
        plt.xticks(x_positions[::max(1, len(x_positions)//20)], 
                  [f"{i+1}" for i in x_positions[::max(1, len(x_positions)//20)]], 
                  rotation=0)
        
        plt.tight_layout()
        
        # Save the chart
        output_file = self.output_dir / 'all_26_tokens_multiple_line_chart.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Multiple line chart saved to: {output_file}")
        
        # Display the chart
        plt.show()
        
        # Save detailed data
        self._save_chart_data(df_sorted)
    
    def _save_chart_data(self, df: pd.DataFrame):
        """Save the chart data to a JSON file"""
        output_file = self.output_dir / 'multiple_line_chart_data.json'
        
        data_to_save = {
            'metadata': {
                'total_tokens': len(df),
                'analysis_date': datetime.now().isoformat(),
                'data_source': 'both_webhooks',
                'chart_type': 'multiple_line_chart'
            },
            'tokens': {}
        }
        
        for _, row in df.iterrows():
            data_to_save['tokens'][row['address']] = {
                'name': row['name'],
                'current_price': row['current_price'],
                'current_fdv': row['current_fdv'],
                'price_change_pct': row['price_change_pct'],
                'volume_24h': row['volume_24h'],
                'buy_sell_ratio': row['buy_sell_ratio'],
                'rugcheck_score': row['rugcheck_score']
            }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(data_to_save, f, indent=2, default=str)
            logger.info(f"Chart data saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving chart data: {e}")
    
    def create_alternative_visualization(self):
        """Create an alternative visualization showing price distribution"""
        if not self.current_prices:
            return
        
        # Create dataframe
        data = []
        for address, price_info in self.current_prices.items():
            token_info = self.tokens_data[address]
            data.append({
                'name': token_info['name'],
                'current_price': price_info['current_price'],
                'price_change_pct': price_info['price_change_pct']
            })
        
        df = pd.DataFrame(data)
        df_sorted = df.sort_values('current_price', ascending=False)
        
        # Create price distribution chart
        plt.figure(figsize=(20, 12))
        
        # Plot each token as a horizontal bar
        y_positions = range(len(df_sorted))
        colors = ['green' if x > 0 else 'red' for x in df_sorted['price_change_pct']]
        
        bars = plt.barh(y_positions, df_sorted['current_price'], 
                       color=colors, alpha=0.7, edgecolor='black')
        
        # Add token names
        plt.yticks(y_positions, [f"{name[:20]}..." for name in df_sorted['name']])
        
        # Customize
        plt.title('All 26 Tokens - Price Distribution Chart', fontsize=20, fontweight='bold')
        plt.xlabel('Current Price (USD)', fontsize=14, fontweight='bold')
        plt.ylabel('Token', fontsize=14, fontweight='bold')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, price) in enumerate(zip(bars, df_sorted['current_price'])):
            plt.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height()/2,
                    f'${price:.8f}', ha='left', va='center', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_file = self.output_dir / 'all_26_tokens_price_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Price distribution chart saved to: {output_file}")
        
        plt.show()

def main():
    """Main function to create the multiple line chart"""
    print("Creating 26 Tokens Multiple Line Chart...")
    print("=" * 50)
    
    creator = MultipleLineChartCreator()
    
    # Collect data for all tokens from both webhooks
    creator.collect_all_tokens_data()
    
    if creator.current_prices:
        # Create the main multiple line chart
        creator.create_multiple_line_chart()
        
        # Create alternative visualization
        creator.create_alternative_visualization()
        
        print("\n" + "=" * 50)
        print("✅ All 26 tokens multiple line chart completed!")
        print(f"📊 Chart saved to: {creator.output_dir}")
    else:
        print("❌ No price data collected. Check the webhook endpoints and try again.")

if __name__ == "__main__":
    main()
