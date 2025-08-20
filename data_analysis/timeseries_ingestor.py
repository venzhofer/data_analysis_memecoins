"""
Time-Series Data Ingestor for Token Analysis
Handles data fetching, validation, and storage to partitioned Parquet files
"""

import requests
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
from timeseries_schema import TimeSeriesSchema

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeSeriesIngestor:
    """Robust ingestor for time-series token data"""
    
    def __init__(self, 
                 tokens_webhook: str,
                 trading_webhook: str,
                 out_dir: str = "data/ts",
                 max_workers: int = 4):
        self.tokens_webhook = tokens_webhook
        self.trading_webhook = trading_webhook
        self.out_dir = Path(out_dir)
        self.max_workers = max_workers
        
        # Create output directory
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup session with retry logic
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "TokenTS/1.0",
            "Accept": "application/json"
        })
        
        # Statistics
        self.stats = {
            "tokens_fetched": 0,
            "data_points_collected": 0,
            "files_written": 0,
            "errors": 0
        }
    
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
    
    def fetch_trading_data(self, 
                          token_address: str, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          retries: int = 4) -> List[Dict]:
        """Fetch trading data for a specific token with time range support"""
        
        # Build URL with parameters
        url = f"{self.trading_webhook}?token={token_address}"
        if start_time:
            url += f"&start={start_time.isoformat()}"
        if end_time:
            url += f"&end={end_time.isoformat()}"
        
        delay = 0.6
        for attempt in range(retries):
            try:
                logger.debug(f"Fetching trading data for {token_address[:10]}... (attempt {attempt + 1})")
                response = self.session.get(url, timeout=45)
                response.raise_for_status()
                
                data = response.json()
                
                # Handle different response formats
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    # If single data point, wrap in list
                    return [data] if data else []
                else:
                    return []
                    
            except Exception as e:
                if attempt < retries - 1:
                    sleep_time = delay + random.uniform(0, 0.3)
                    logger.warning(f"Attempt {attempt + 1} failed for {token_address[:10]}: {e}. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    delay *= 1.7
                else:
                    logger.error(f"All attempts failed for {token_address[:10]}: {e}")
                    return []
        
        return []
    
    def to_dataframe(self, rows: List[Dict], address: str) -> pd.DataFrame:
        """Convert raw data rows to normalized DataFrame"""
        if not rows:
            return TimeSeriesSchema.create_empty_dataframe()
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Add address if not present
        if 'address' not in df.columns:
            df['address'] = address
        
        # Normalize using schema
        try:
            df = TimeSeriesSchema.normalize_dataframe(df)
            logger.debug(f"Normalized {len(df)} rows for {address[:10]}")
            return df
        except Exception as e:
            logger.error(f"Failed to normalize data for {address[:10]}: {e}")
            return TimeSeriesSchema.create_empty_dataframe()
    
    def write_parquet_partitioned(self, df: pd.DataFrame) -> None:
        """Write DataFrame to partitioned Parquet files"""
        if df.empty:
            return
        
        # Add date partition column
        df = df.copy()
        df['date'] = df['timestamp'].dt.date.astype(str)
        
        # Group by address and date
        for (addr, date), group in df.groupby(['address', 'date'], as_index=False):
            # Create partition directory
            partition_dir = self.out_dir / f"address={addr}" / f"date={date}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            timestamp_ms = int(time.time() * 1000)
            filename = f"part-{timestamp_ms}.parquet"
            filepath = partition_dir / filename
            
            try:
                # Remove date column before writing
                group_to_write = group.drop(columns=['date'])
                group_to_write.to_parquet(filepath, index=False)
                
                self.stats['files_written'] += 1
                logger.debug(f"Wrote {len(group_to_write)} rows to {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to write {filepath}: {e}")
                self.stats['errors'] += 1
    
    def ingest_token(self, token_info: Dict) -> bool:
        """Ingest data for a single token"""
        try:
            # Extract token address
            address = token_info.get('address') or token_info.get('adress')
            if not address:
                logger.warning(f"Token {token_info.get('name', 'Unknown')} has no address")
                return False
            
            token_name = token_info.get('name', 'Unknown')
            logger.info(f"Ingesting data for {token_name} ({address[:10]}...)")
            
            # Fetch current trading data
            trading_data = self.fetch_trading_data(address)
            if not trading_data:
                logger.warning(f"No trading data for {token_name}")
                return False
            
            # Convert to DataFrame
            df = self.to_dataframe(trading_data, address)
            if df.empty:
                logger.warning(f"Empty DataFrame for {token_name}")
                return False
            
            # Write to Parquet
            self.write_parquet_partitioned(df)
            
            # Update statistics
            self.stats['tokens_fetched'] += 1
            self.stats['data_points_collected'] += len(df)
            
            logger.info(f"Successfully ingested {len(df)} data points for {token_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest token {token_info.get('name', 'Unknown')}: {e}")
            self.stats['errors'] += 1
            return False
    
    def ingest_all_tokens(self, batch_size: int = 5) -> Dict:
        """Ingest data for all available tokens"""
        logger.info("Starting ingestion of all tokens...")
        
        # Fetch token list
        tokens = self.fetch_tokens_info()
        if not tokens:
            logger.error("No tokens found")
            return self.stats
        
        logger.info(f"Found {len(tokens)} tokens to process")
        
        # Process tokens in batches
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(tokens) + batch_size - 1)//batch_size}")
            
            for token in batch:
                self.ingest_token(token)
                # Small delay between tokens to avoid overwhelming the API
                time.sleep(0.5)
        
        # Log final statistics
        logger.info("Ingestion completed!")
        logger.info(f"Tokens processed: {self.stats['tokens_fetched']}")
        logger.info(f"Data points collected: {self.stats['data_points_collected']}")
        logger.info(f"Files written: {self.stats['files_written']}")
        logger.info(f"Errors: {self.stats['errors']}")
        
        return self.stats
    
    def get_storage_info(self) -> Dict:
        """Get information about stored data"""
        info = {
            "total_files": 0,
            "total_size_mb": 0,
            "addresses": set(),
            "date_range": {"min": None, "max": None}
        }
        
        try:
            # Count files and sizes
            for parquet_file in self.out_dir.rglob("*.parquet"):
                info["total_files"] += 1
                info["total_size_mb"] += parquet_file.stat().st_size / (1024 * 1024)
                
                # Extract address and date from path
                parts = parquet_file.parts
                if len(parts) >= 3:
                    addr_part = parts[-3]
                    date_part = parts[-2]
                    
                    if addr_part.startswith("address="):
                        info["addresses"].add(addr_part.split("=", 1)[1])
                    
                    if date_part.startswith("date="):
                        date_str = date_part.split("=", 1)[1]
                        try:
                            date = datetime.strptime(date_str, "%Y-%m-%d").date()
                            if info["date_range"]["min"] is None or date < info["date_range"]["min"]:
                                info["date_range"]["min"] = date
                            if info["date_range"]["max"] is None or date > info["date_range"]["max"]:
                                info["date_range"]["max"] = date
                        except ValueError:
                            pass
            
            # Convert set to list for JSON serialization
            info["addresses"] = list(info["addresses"])
            info["total_size_mb"] = round(info["total_size_mb"], 2)
            
        except Exception as e:
            logger.error(f"Failed to get storage info: {e}")
        
        return info

# Example usage
if __name__ == "__main__":
    # Initialize ingestor
    ingestor = TimeSeriesIngestor(
        tokens_webhook="https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4",
        trading_webhook="https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0",
        out_dir="data/ts"
    )
    
    # Test with a small batch first
    print("Testing ingestor...")
    tokens = ingestor.fetch_tokens_info()
    if tokens:
        print(f"Found {len(tokens)} tokens")
        # Test with first token
        if len(tokens) > 0:
            success = ingestor.ingest_token(tokens[0])
            print(f"First token ingestion: {'Success' if success else 'Failed'}")
    
    # Show storage info
    storage_info = ingestor.get_storage_info()
    print(f"Storage info: {storage_info}")
