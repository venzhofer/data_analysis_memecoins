#!/usr/bin/env python3
"""
Price Time Series Chart Creator
Creates a single line chart showing actual price data over time for all tokens
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Optional, Tuple
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PriceTimeSeriesChartCreator:
    """Creates time series line charts showing actual price data for all tokens"""
    
    def __init__(self, 
                 tokens_webhook: str = "https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4",
                 trading_webhook: str = "https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0",
                 output_dir: str = "output/price_timeseries_chart"):
        
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
        self.price_data = {}
        self.time_series_data = {}
        
        # Color palette for tokens
        self.color_palette = plt.cm.Set3(np.linspace(0, 1, 50))  # Support up to 50 tokens
        
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
    
    def fetch_trading_data(self, token_address: str, retries: int = 3) -> Optional[Dict]:
        """Fetch trading data for a specific token"""
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
                        logger.error(f"Failed to fetch trading data for {token_address[:10]} after {retries} attempts: {e}")
                        return None
                    time.sleep(1)  # Wait before retry
                    
        except Exception as e:
            logger.error(f"Failed to fetch trading data for {token_address[:10]}: {e}")
            return None
    
    def extract_price_timeseries(self, trading_data: Dict) -> List[Tuple[datetime, float]]:
        """Extract price and timestamp data from trading data"""
        timeseries = []
        
        try:
            # Handle different possible data structures
            if isinstance(trading_data, list):
                data_points = trading_data
            elif isinstance(trading_data, dict) and 'data' in trading_data:
                data_points = trading_data['data']
            elif isinstance(trading_data, dict) and 'trades' in trading_data:
                data_points = trading_data['trades']
            else:
                data_points = [trading_data]
            
            for point in data_points:
                if isinstance(point, dict):
                    # Try to extract timestamp and price
                    timestamp = None
                    price = None
                    
                    # Look for timestamp fields
                    for ts_field in ['timestamp', 'time', 'date', 'created_at']:
                        if ts_field in point:
                            ts_value = point[ts_field]
                            try:
                                if isinstance(ts_value, str):
                                    timestamp = datetime.fromisoformat(ts_value.replace('Z', '+00:00'))
                                elif isinstance(ts_value, (int, float)):
                                    timestamp = datetime.fromtimestamp(ts_value)
                                break
                            except:
                                continue
                    
                    # Look for price fields
                    for price_field in ['price', 'amount', 'value', 'fdv', 'market_cap']:
                        if price_field in point:
                            try:
                                price = float(point[price_field])
                                break
                            except:
                                continue
                    
                    if timestamp and price and price > 0:
                        timeseries.append((timestamp, price))
            
            # Sort by timestamp
            timeseries.sort(key=lambda x: x[0])
            
        except Exception as e:
            logger.error(f"Error extracting timeseries: {e}")
        
        return timeseries
    
    def collect_all_tokens_data(self, max_tokens: int = 20):
        """Collect price data for all tokens"""
        logger.info("Collecting price data for all tokens...")
        
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
                    'index': i
                }
                
                # Fetch trading data
                trading_data = self.fetch_trading_data(address)
                if trading_data:
                    # Extract price timeseries
                    timeseries = self.extract_price_timeseries(trading_data)
                    
                    if timeseries and len(timeseries) > 1:
                        self.time_series_data[address] = timeseries
                        successful_tokens += 1
                        logger.info(f"✓ {token_name}: {len(timeseries)} data points")
                    else:
                        logger.warning(f"✗ {token_name}: insufficient data points")
                else:
                    logger.warning(f"✗ {token_name}: no trading data")
                
                # Add delay to avoid overwhelming the API
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing token {token.get('name', 'Unknown')}: {e}")
                continue
        
        logger.info(f"Successfully collected data for {successful_tokens}/{len(tokens)} tokens")
    
    def create_price_timeseries_chart(self):
        """Create the main price timeseries chart with all tokens"""
        if not self.time_series_data:
            logger.error("No time series data available")
            return
        
        logger.info("Creating price timeseries chart...")
        
        # Create the main chart
        plt.figure(figsize=(20, 12))
        
        # Find global time range
        all_timestamps = []
        for timeseries in self.time_series_data.values():
            all_timestamps.extend([ts for ts, _ in timeseries])
        
        if not all_timestamps:
            logger.error("No valid timestamps found")
            return
        
        global_start = min(all_timestamps)
        global_end = max(all_timestamps)
        
        # Normalize all timeseries to the same time range
        normalized_data = {}
        for address, timeseries in self.time_series_data.items():
            if len(timeseries) > 1:
                # Convert to relative time (minutes from start)
                relative_times = [(ts - global_start).total_seconds() / 60 for ts, _ in timeseries]
                prices = [price for _, price in timeseries]
                
                # Normalize prices to percentage change from first price
                if prices[0] > 0:
                    normalized_prices = [(p / prices[0] - 1) * 100 for p in prices]
                    normalized_data[address] = (relative_times, normalized_prices, prices[0])
        
        # Plot each token
        for i, (address, (times, prices, initial_price)) in enumerate(normalized_data.items()):
            token_info = self.tokens_data[address]
            color = self.color_palette[i % len(self.color_palette)]
            
            # Plot the line
            plt.plot(times, prices, 
                    color=color, 
                    linewidth=2, 
                    alpha=0.8, 
                    label=f"{token_info['name']} (${initial_price:.6f})")
            
            # Add markers at key points
            if len(prices) > 0:
                # Start point
                plt.scatter(times[0], prices[0], color=color, s=50, alpha=0.8, edgecolors='black')
                # End point
                plt.scatter(times[-1], prices[-1], color=color, s=50, alpha=0.8, edgecolors='black')
        
        # Customize the chart
        plt.title('All Tokens Price Performance Over Time\n(Percentage Change from Initial Price)', 
                 fontsize=20, fontweight='bold', pad=20)
        plt.xlabel('Time (Minutes from Start)', fontsize=14, fontweight='bold')
        plt.ylabel('Price Change (%)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add reference lines
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1, label='Initial Price (0%)')
        plt.axhline(y=100, color='green', linestyle='--', alpha=0.7, linewidth=1, label='100% Gain')
        plt.axhline(y=50, color='lightgreen', linestyle='--', alpha=0.7, linewidth=1, label='50% Gain')
        plt.axhline(y=-50, color='red', linestyle='--', alpha=0.7, linewidth=1, label='50% Loss')
        plt.axhline(y=-80, color='darkred', linestyle='--', alpha=0.7, linewidth=1, label='80% Loss')
        
        # Format x-axis
        plt.xticks(rotation=45)
        
        # Add legend (outside the plot to avoid overlap)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Add summary statistics
        total_tokens = len(normalized_data)
        positive_tokens = sum(1 for _, (_, prices, _) in normalized_data.items() if prices[-1] > 0)
        negative_tokens = total_tokens - positive_tokens
        
        summary_text = f"""Summary:
Total Tokens: {total_tokens}
Positive Performance: {positive_tokens}
Negative Performance: {negative_tokens}
Time Range: {global_start.strftime('%H:%M')} - {global_end.strftime('%H:%M')}"""
        
        plt.figtext(0.02, 0.02, summary_text, 
                   fontsize=10, bbox=dict(boxstyle='round,pad=0.5', 
                                        facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the chart
        output_file = self.output_dir / 'all_tokens_price_timeseries.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Chart saved to: {output_file}")
        
        # Display the chart
        plt.show()
        
        # Save detailed data
        self._save_timeseries_data(normalized_data, global_start, global_end)
    
    def _save_timeseries_data(self, normalized_data: Dict, start_time: datetime, end_time: datetime):
        """Save the timeseries data to a JSON file"""
        output_file = self.output_dir / 'price_timeseries_data.json'
        
        data_to_save = {
            'metadata': {
                'total_tokens': len(normalized_data),
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_minutes': (end_time - start_time).total_seconds() / 60
            },
            'tokens': {}
        }
        
        for address, (times, prices, initial_price) in normalized_data.items():
            token_info = self.tokens_data[address]
            data_to_save['tokens'][address] = {
                'name': token_info['name'],
                'symbol': token_info['symbol'],
                'initial_price': initial_price,
                'timeseries': {
                    'minutes_from_start': times,
                    'price_change_percent': prices
                },
                'final_change_percent': prices[-1] if prices else 0,
                'max_gain': max(prices) if prices else 0,
                'max_loss': min(prices) if prices else 0
            }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(data_to_save, f, indent=2, default=str)
            logger.info(f"Timeseries data saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving timeseries data: {e}")
    
    def create_individual_token_charts(self):
        """Create individual charts for each token"""
        if not self.time_series_data:
            return
        
        logger.info("Creating individual token charts...")
        
        # Create subplots
        n_tokens = len(self.time_series_data)
        cols = 3
        rows = (n_tokens + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        fig.suptitle('Individual Token Price Performance', fontsize=20, fontweight='bold')
        
        # Flatten axes if needed
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (address, timeseries) in enumerate(self.time_series_data.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            token_info = self.tokens_data[address]
            
            # Extract data
            timestamps = [ts for ts, _ in timeseries]
            prices = [price for _, price in timeseries]
            
            # Convert to relative time
            start_time = timestamps[0]
            relative_times = [(ts - start_time).total_seconds() / 60 for ts in timestamps]
            
            # Plot
            ax.plot(relative_times, prices, linewidth=2, color='blue', alpha=0.8)
            ax.scatter(relative_times[0], prices[0], color='green', s=50, alpha=0.8, label='Start')
            ax.scatter(relative_times[-1], prices[-1], color='red', s=50, alpha=0.8, label='End')
            
            # Customize
            ax.set_title(f"{token_info['name'][:20]}...", fontsize=10, fontweight='bold')
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Price')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Add performance info
            if prices[0] > 0:
                change_pct = ((prices[-1] / prices[0]) - 1) * 100
                ax.text(0.02, 0.98, f'Change: {change_pct:.1f}%', 
                       transform=ax.transAxes, fontsize=8, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for i in range(len(self.time_series_data), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save
        output_file = self.output_dir / 'individual_token_charts.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Individual charts saved to: {output_file}")
        
        plt.show()

def main():
    """Main function to create the price timeseries chart"""
    print("Creating Price Time Series Chart for All Tokens...")
    
    creator = PriceTimeSeriesChartCreator()
    
    # Collect data for all tokens
    creator.collect_all_tokens_data(max_tokens=15)  # Limit to 15 tokens for better visualization
    
    if creator.time_series_data:
        # Create the main timeseries chart
        creator.create_price_timeseries_chart()
        
        # Create individual token charts
        creator.create_individual_token_charts()
        
        print("All price timeseries visualizations completed!")
    else:
        print("No time series data collected. Check the webhook endpoints and try again.")

if __name__ == "__main__":
    main()
