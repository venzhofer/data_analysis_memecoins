#!/usr/bin/env python3
"""
Create True Time Series Chart
Fetches historical price data and creates a line chart showing price evolution over time
"""

import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
import time
import numpy as np

class TrueTimeSeriesChartCreator:
    def __init__(self, tokens_webhook: str, trading_webhook: str, output_dir: str = "output/true_timeseries_chart"):
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
            
            # Handle nested structure
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
            print(f"📡 Fetching history for {token_name}...")
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            data = response.json()
            
            if isinstance(data, list):
                print(f"   ✅ Got {len(data)} data points")
                return data
            elif isinstance(data, dict) and 'data' in data:
                print(f"   ✅ Got {len(data['data'])} data points")
                return data['data']
            else:
                print(f"   ❌ Unexpected data format: {type(data)}")
                return []
                
        except Exception as e:
            print(f"   ❌ Error fetching {token_name}: {e}")
            return []
    
    def process_token_data(self, token_data: list, token_name: str):
        """Process raw token data into a DataFrame"""
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
                except (ValueError, TypeError) as e:
                    continue
        
        if processed_data:
            df = pd.DataFrame(processed_data)
            df = df.sort_values('timestamp')
            print(f"   📊 Processed {len(df)} valid data points")
            return df
        else:
            print(f"   ❌ No valid data points found")
            return None
    
    def create_timeseries_chart(self):
        """Create the main time series chart"""
        print("🚀 Starting True Time Series Chart Creation...")
        print("=" * 60)
        
        # Fetch all tokens
        tokens = self.fetch_tokens()
        if not tokens:
            print("❌ No tokens found")
            return
            
        print(f"📋 Found {len(tokens)} tokens")
        
        # Process each token
        all_data = {}
        successful_tokens = 0
        
        for i, token in enumerate(tokens):
            if isinstance(token, dict):
                address = token.get('address') or token.get('adress')
                name = token.get('name', f'Token_{i}')
                
                if not address:
                    continue
                    
                print(f"\n🔍 Processing {i+1}/{len(tokens)}: {name}")
                
                # Fetch historical data
                history = self.fetch_token_history(address, name)
                if history:
                    # Process the data
                    df = self.process_token_data(history, name)
                    if df is not None and len(df) > 1:  # Need at least 2 points for a line
                        all_data[name] = df
                        successful_tokens += 1
                        print(f"   ✅ Added to chart")
                    else:
                        print(f"   ⚠️ Insufficient data points")
                else:
                    print(f"   ❌ No history data")
                
                # Small delay to be respectful to the API
                time.sleep(0.5)
        
        print(f"\n📊 Successfully processed {successful_tokens} tokens")
        
        if not all_data:
            print("❌ No valid data to plot")
            return
        
        # Create the chart
        self._plot_timeseries(all_data)
        
    def _plot_timeseries(self, all_data: dict):
        """Create the actual time series plot"""
        print("\n🎨 Creating time series chart...")
        
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Plot each token
        colors = plt.cm.tab20(np.linspace(0, 1, len(all_data)))
        
        for (token_name, df), color in zip(all_data.items(), colors):
            if len(df) > 1:
                # Normalize price to start at 100 for better comparison
                start_price = df['price'].iloc[0]
                normalized_prices = (df['price'] / start_price) * 100
                
                ax.plot(df['timestamp'], normalized_prices, 
                       label=token_name, color=color, linewidth=1.5, alpha=0.8)
        
        # Customize the chart
        ax.set_title('Token Price Evolution Over Time (Normalized to 100)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Normalized Price (Base = 100)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        chart_path = self.output_dir / 'true_timeseries_chart.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"💾 Chart saved to: {chart_path}")
        
        # Also save the data
        data_path = self.output_dir / 'timeseries_data.json'
        data_to_save = {}
        for token_name, df in all_data.items():
            data_to_save[token_name] = {
                'data_points': len(df),
                'time_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat()
                },
                'price_range': {
                    'min': float(df['price'].min()),
                    'max': float(df['price'].max()),
                    'start': float(df['price'].iloc[0]),
                    'end': float(df['price'].iloc[-1])
                }
            }
        
        with open(data_path, 'w') as f:
            json.dump(data_to_save, f, indent=2, default=str)
        print(f"💾 Data summary saved to: {data_path}")
        
        plt.show()

def main():
    tokens_webhook = "https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4"
    trading_webhook = "https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0"
    
    creator = TrueTimeSeriesChartCreator(tokens_webhook, trading_webhook)
    creator.create_timeseries_chart()

if __name__ == "__main__":
    main()
