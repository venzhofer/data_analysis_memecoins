#!/usr/bin/env python3
"""
Current Prices Chart Creator
Creates a chart showing current prices for all tokens together
Since we don't have historical time series data, this shows current price distribution
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

class CurrentPricesChartCreator:
    """Creates charts showing current prices for all tokens"""
    
    def __init__(self, 
                 tokens_webhook: str = "https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4",
                 trading_webhook: str = "https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0",
                 output_dir: str = "output/current_prices_chart"):
        
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
        
        # Color palette for tokens
        self.color_palette = plt.cm.Set3(np.linspace(0, 1, 50))
        
    def fetch_tokens_info(self) -> List[Dict]:
        """Fetch basic information about all tokens"""
        try:
            logger.info("Fetching tokens information...")
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
            
            logger.info(f"Successfully fetched {len(tokens)} tokens")
            return tokens
            
        except Exception as e:
            logger.error(f"Failed to fetch tokens: {e}")
            return []
    
    def fetch_current_price(self, token_address: str, retries: int = 3) -> Optional[Dict]:
        """Fetch current price data for a specific token"""
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
    
    def collect_all_tokens_prices(self, max_tokens: int = 20):
        """Collect current price data for all tokens"""
        logger.info("Collecting current price data for all tokens...")
        
        # Fetch tokens info
        tokens = self.fetch_tokens_info()
        if not tokens:
            logger.error("No tokens found")
            return
        
        # Limit number of tokens to analyze
        tokens = tokens[:max_tokens]
        
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
                
                # Store token info
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
                
                # Fetch current price data
                price_data = self.fetch_current_price(address)
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
                    logger.warning(f"✗ {token_name}: no price data")
                
                # Add delay to avoid overwhelming the API
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing token {token.get('name', 'Unknown')}: {e}")
                continue
        
        logger.info(f"Successfully collected price data for {successful_tokens}/{len(tokens)} tokens")
    
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
    
    def create_current_prices_chart(self):
        """Create the main current prices chart with all tokens"""
        if not self.current_prices:
            logger.error("No price data available")
            return
        
        logger.info("Creating current prices chart...")
        
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
        
        # Sort by price change percentage
        df_sorted = df.sort_values('price_change_pct', ascending=False)
        
        # Create the main chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 16))
        fig.suptitle('All Tokens Current Prices & Performance Overview', fontsize=24, fontweight='bold')
        
        # 1. Price Change Distribution (Main Chart)
        colors = ['green' if x > 0 else 'red' for x in df_sorted['price_change_pct']]
        bars = ax1.bar(range(len(df_sorted)), df_sorted['price_change_pct'], color=colors, alpha=0.7, edgecolor='black')
        
        # Add token names as labels
        ax1.set_xlabel('Token Index (Sorted by Performance)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price Change from Launch (%)', fontsize=14, fontweight='bold')
        ax1.set_title('All Tokens Price Performance (Sorted)', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
        
        # Add reference lines
        ax1.axhline(y=100, color='green', linestyle='--', alpha=0.7, linewidth=1, label='100% Gain')
        ax1.axhline(y=50, color='lightgreen', linestyle='--', alpha=0.7, linewidth=1, label='50% Gain')
        ax1.axhline(y=-50, color='red', linestyle='--', alpha=0.7, linewidth=1, label='50% Loss')
        ax1.axhline(y=-80, color='darkred', linestyle='--', alpha=0.7, linewidth=1, label='80% Loss')
        
        # Add value labels on bars
        for i, (bar, change) in enumerate(zip(bars, df_sorted['price_change_pct'])):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -5),
                    f'{change:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=8, fontweight='bold')
        
        # Add legend
        ax1.legend(loc='upper right')
        
        # 2. Current Price Distribution (Log Scale)
        ax2.hist(df_sorted['current_price'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Current Price ($)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Tokens', fontsize=14, fontweight='bold')
        ax2.set_title('Distribution of Current Prices', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # 3. Volume vs Price Change Scatter
        scatter = ax3.scatter(df_sorted['volume_24h'], df_sorted['price_change_pct'], 
                             c=df_sorted['price_change_pct'], cmap='RdYlGn', 
                             s=100, alpha=0.7, edgecolors='black')
        ax3.set_xlabel('24h Volume ($)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Price Change (%)', fontsize=14, fontweight='bold')
        ax3.set_title('Volume vs Price Performance', fontsize=16, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xscale('log')
        plt.colorbar(scatter, ax=ax3, label='Price Change (%)')
        
        # 4. Buy/Sell Ratio vs Performance
        ax4.scatter(df_sorted['buy_sell_ratio'], df_sorted['price_change_pct'], 
                   c=df_sorted['price_change_pct'], cmap='RdYlGn', 
                   s=100, alpha=0.7, edgecolors='black')
        ax4.set_xlabel('Buy/Sell Ratio (24h)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Price Change (%)', fontsize=14, fontweight='bold')
        ax4.set_title('Buy/Sell Ratio vs Performance', fontsize=16, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_xscale('log')
        
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
        
        plt.tight_layout()
        
        # Save the chart
        output_file = self.output_dir / 'all_tokens_current_prices_dashboard.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Dashboard saved to: {output_file}")
        
        # Display the chart
        plt.show()
        
        # Save detailed data
        self._save_price_data(df_sorted)
    
    def _save_price_data(self, df: pd.DataFrame):
        """Save the price data to a JSON file"""
        output_file = self.output_dir / 'current_prices_data.json'
        
        data_to_save = {
            'metadata': {
                'total_tokens': len(df),
                'analysis_date': datetime.now().isoformat(),
                'data_source': 'webhook_snapshots'
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
            logger.info(f"Price data saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving price data: {e}")
    
    def create_simple_line_chart(self):
        """Create a simple line chart showing all tokens in one view"""
        if not self.current_prices:
            return
        
        # Create dataframe
        data = []
        for address, price_info in self.current_prices.items():
            token_info = self.tokens_data[address]
            data.append({
                'name': token_info['name'],
                'price_change_pct': price_info['price_change_pct'],
                'current_price': price_info['current_price']
            })
        
        df = pd.DataFrame(data)
        df_sorted = df.sort_values('price_change_pct', ascending=False)
        
        # Create simple line chart
        plt.figure(figsize=(20, 10))
        
        # Plot each token as a point
        x_positions = range(len(df_sorted))
        colors = ['green' if x > 0 else 'red' for x in df_sorted['price_change_pct']]
        
        # Plot points
        plt.scatter(x_positions, df_sorted['price_change_pct'], 
                   c=colors, s=100, alpha=0.8, edgecolors='black', linewidth=1)
        
        # Connect points with lines
        plt.plot(x_positions, df_sorted['price_change_pct'], 
                color='blue', alpha=0.3, linewidth=1, linestyle='--')
        
        # Add token names
        for i, (_, token) in enumerate(df_sorted.iterrows()):
            plt.annotate(f"{token['name'][:15]}...", 
                        (i, token['price_change_pct']),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, rotation=45,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Customize
        plt.title('All Tokens Price Performance - Current Snapshot\n(Line Chart View)', 
                 fontsize=20, fontweight='bold', pad=20)
        plt.xlabel('Token Index (Sorted by Performance)', fontsize=14, fontweight='bold')
        plt.ylabel('Price Change from Launch (%)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
        
        # Add reference lines
        plt.axhline(y=100, color='green', linestyle='--', alpha=0.7, linewidth=1, label='100% Gain')
        plt.axhline(y=50, color='lightgreen', linestyle='--', alpha=0.7, linewidth=1, label='50% Gain')
        plt.axhline(y=-50, color='red', linestyle='--', alpha=0.7, linewidth=1, label='50% Loss')
        plt.axhline(y=-80, color='darkred', linestyle='--', alpha=0.7, linewidth=1, label='80% Loss')
        
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        # Save
        output_file = self.output_dir / 'all_tokens_simple_line_chart.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Simple line chart saved to: {output_file}")
        
        plt.show()

def main():
    """Main function to create the current prices charts"""
    print("Creating Current Prices Charts for All Tokens...")
    
    creator = CurrentPricesChartCreator()
    
    # Collect current price data for all tokens
    creator.collect_all_tokens_prices(max_tokens=20)
    
    if creator.current_prices:
        # Create the comprehensive dashboard
        creator.create_current_prices_chart()
        
        # Create the simple line chart
        creator.create_simple_line_chart()
        
        print("All current prices visualizations completed!")
    else:
        print("No price data collected. Check the webhook endpoints and try again.")

if __name__ == "__main__":
    main()
