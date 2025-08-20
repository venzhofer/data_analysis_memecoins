"""
Time-Series Data Loader and Resampling
Handles loading data from partitioned Parquet files and resampling for analysis
"""

import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
from timeseries_schema import TimeSeriesSchema

logger = logging.getLogger(__name__)

class TimeSeriesLoader:
    """Loads and resamples time-series data from partitioned Parquet files"""
    
    def __init__(self, data_root: str = "data/ts"):
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise ValueError(f"Data directory {data_root} does not exist")
    
    def get_available_addresses(self) -> List[str]:
        """Get list of all available token addresses"""
        addresses = set()
        try:
            for address_dir in self.data_root.glob("address=*"):
                if address_dir.is_dir():
                    addr = address_dir.name.split("=", 1)[1]
                    addresses.add(addr)
        except Exception as e:
            logger.error(f"Failed to scan addresses: {e}")
        
        return sorted(list(addresses))
    
    def get_available_dates(self, address: str) -> List[str]:
        """Get list of available dates for a specific address"""
        dates = set()
        try:
            address_dir = self.data_root / f"address={address}"
            if address_dir.exists():
                for date_dir in address_dir.glob("date=*"):
                    if date_dir.is_dir():
                        date = date_dir.name.split("=", 1)[1]
                        dates.add(date)
        except Exception as e:
            logger.error(f"Failed to scan dates for {address}: {e}")
        
        return sorted(list(dates))
    
    def load_token_df(self, 
                     address: str, 
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """Load DataFrame for a specific token with optional date filtering"""
        try:
            # Build glob pattern
            if start_date and end_date:
                pattern = f"{self.data_root}/address={address}/date={start_date}/part-*.parquet"
                # Add intermediate dates if needed
                dates = self._get_date_range(start_date, end_date)
                all_files = []
                for date in dates:
                    date_pattern = f"{self.data_root}/address={address}/date={date}/part-*.parquet"
                    all_files.extend(glob.glob(date_pattern))
            elif start_date:
                pattern = f"{self.data_root}/address={address}/date={start_date}/part-*.parquet"
                all_files = glob.glob(pattern)
            elif end_date:
                # Load all dates up to end_date
                dates = self.get_available_dates(address)
                end_dt = pd.to_datetime(end_date).date()
                valid_dates = [d for d in dates if pd.to_datetime(d).date() <= end_dt]
                all_files = []
                for date in valid_dates:
                    date_pattern = f"{self.data_root}/address={address}/date={date}/part-*.parquet"
                    all_files.extend(glob.glob(date_pattern))
            else:
                pattern = f"{self.data_root}/address={address}/date=*/part-*.parquet"
                all_files = glob.glob(pattern)
            
            if not all_files:
                logger.warning(f"No data files found for {address}")
                return TimeSeriesSchema.create_empty_dataframe()
            
            # Load and concatenate all files
            dfs = []
            for file_path in all_files:
                try:
                    df = pd.read_parquet(file_path)
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
            
            if not dfs:
                logger.warning(f"No valid data files for {address}")
                return TimeSeriesSchema.create_empty_dataframe()
            
            # Concatenate and clean
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Normalize using schema
            combined_df = TimeSeriesSchema.normalize_dataframe(combined_df)
            
            # Apply date filtering if specified
            if start_date or end_date:
                combined_df = self._filter_by_date_range(combined_df, start_date, end_date)
            
            # Sort by timestamp and remove duplicates
            combined_df = combined_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
            
            # Apply sanity filters
            combined_df = self._apply_sanity_filters(combined_df)
            
            logger.info(f"Loaded {len(combined_df)} rows for {address}")
            return combined_df
            
        except Exception as e:
            logger.error(f"Failed to load data for {address}: {e}")
            return TimeSeriesSchema.create_empty_dataframe()
    
    def _get_date_range(self, start_date: str, end_date: str) -> List[str]:
        """Get list of dates between start and end (inclusive)"""
        try:
            start_dt = pd.to_datetime(start_date).date()
            end_dt = pd.to_datetime(end_date).date()
            
            dates = []
            current_dt = start_dt
            while current_dt <= end_dt:
                dates.append(current_dt.strftime("%Y-%m-%d"))
                current_dt += pd.Timedelta(days=1)
            
            return dates
        except Exception as e:
            logger.error(f"Failed to generate date range: {e}")
            return []
    
    def _filter_by_date_range(self, df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """Filter DataFrame by date range"""
        if df.empty:
            return df
        
        try:
            if start_date:
                start_dt = pd.to_datetime(start_date, utc=True)
                df = df[df['timestamp'] >= start_dt]
            
            if end_date:
                end_dt = pd.to_datetime(end_date, utc=True)
                df = df[df['timestamp'] <= end_dt]
            
            return df
        except Exception as e:
            logger.error(f"Failed to filter by date range: {e}")
            return df
    
    def _apply_sanity_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic sanity filters to the data"""
        if df.empty:
            return df
        
        try:
            # Price must be positive
            df = df[df['price'] > 0]
            
            # FDV and market cap must be non-negative
            df = df[df['fdv'] >= 0]
            df = df[df['market_cap'] >= 0]
            
            # Transaction counts must be non-negative
            df = df[df['buys'] >= 0]
            df = df[df['sells'] >= 0]
            
            # Volumes must be non-negative
            df = df[df['buy_volume'] >= 0]
            df = df[df['sell_volume'] >= 0]
            df = df[df['total_volume'] >= 0]
            
            logger.debug(f"Applied sanity filters, remaining rows: {len(df)}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to apply sanity filters: {e}")
            return df
    
    def resample_bars(self, df: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
        """Resample tick data to OHLC bars"""
        if df.empty:
            return df
        
        try:
            # Set timestamp as index for resampling
            df_resampled = df.set_index('timestamp').copy()
            
            # Resample price data to OHLC
            o = df_resampled['price'].resample(rule).first()
            h = df_resampled['price'].resample(rule).max()
            l = df_resampled['price'].resample(rule).min()
            c = df_resampled['price'].resample(rule).last()
            
            # Resample volume and transaction data
            v = df_resampled['total_volume'].resample(rule).sum(min_count=1)
            b = df_resampled['buys'].resample(rule).sum(min_count=1)
            s = df_resampled['sells'].resample(rule).sum(min_count=1)
            
            # Resample other metrics (use last value for most recent)
            fdv = df_resampled['fdv'].resample(rule).last()
            market_cap = df_resampled['market_cap'].resample(rule).last()
            buy_volume = df_resampled['buy_volume'].resample(rule).sum(min_count=1)
            sell_volume = df_resampled['sell_volume'].resample(rule).sum(min_count=1)
            
            # Create resampled DataFrame
            bars = pd.DataFrame({
                'open': o,
                'high': h,
                'low': l,
                'close': c,
                'volume': v,
                'buys': b,
                'sells': s,
                'fdv': fdv,
                'market_cap': market_cap,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume
            })
            
            # Remove rows with no close price
            bars = bars.dropna(subset=['close'])
            
            # Reset index to get timestamp as column
            bars = bars.reset_index()
            
            logger.info(f"Resampled {len(df)} ticks to {len(bars)} {rule} bars")
            return bars
            
        except Exception as e:
            logger.error(f"Failed to resample data: {e}")
            return df
    
    def get_data_summary(self, address: str) -> Dict[str, Any]:
        """Get summary statistics for a token's data"""
        try:
            df = self.load_token_df(address)
            if df.empty:
                return {"status": "no_data"}
            
            summary = {
                "status": "success",
                "address": address,
                "total_rows": len(df),
                "date_range": {
                    "start": df['timestamp'].min().isoformat(),
                    "end": df['timestamp'].max().isoformat()
                },
                "time_span_hours": (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600,
                "price_stats": {
                    "min": float(df['price'].min()),
                    "max": float(df['price'].max()),
                    "mean": float(df['price'].mean()),
                    "std": float(df['price'].std())
                },
                "volume_stats": {
                    "total_volume": float(df['total_volume'].sum()),
                    "avg_volume": float(df['total_volume'].mean()),
                    "max_volume": float(df['total_volume'].max())
                },
                "transaction_stats": {
                    "total_buys": int(df['buys'].sum()),
                    "total_sells": int(df['sells'].sum()),
                    "buy_sell_ratio": float(df['buys'].sum() / max(df['sells'].sum(), 1))
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get data summary for {address}: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_all_summaries(self) -> Dict[str, Dict]:
        """Get summaries for all available tokens"""
        addresses = self.get_available_addresses()
        summaries = {}
        
        for address in addresses:
            summaries[address] = self.get_data_summary(address)
        
        return summaries

# Example usage
if __name__ == "__main__":
    # Test the loader
    try:
        loader = TimeSeriesLoader("data/ts")
        
        # Get available addresses
        addresses = loader.get_available_addresses()
        print(f"Available addresses: {len(addresses)}")
        
        if addresses:
            # Test with first address
            test_address = addresses[0]
            print(f"\nTesting with address: {test_address[:10]}...")
            
            # Get data summary
            summary = loader.get_data_summary(test_address)
            print(f"Summary: {summary}")
            
            # Load and resample data
            df = loader.load_token_df(test_address)
            if not df.empty:
                print(f"Loaded {len(df)} rows")
                
                # Resample to 1-minute bars
                bars = loader.resample_bars(df, "1min")
                print(f"Resampled to {len(bars)} 1-minute bars")
                
                if not bars.empty:
                    print(f"First bar: {bars.iloc[0].to_dict()}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run the ingestor first to create data files")
