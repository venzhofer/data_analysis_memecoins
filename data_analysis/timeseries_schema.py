"""
Time-Series Data Schema for Token Analysis
Defines the structure and validation for historical token data
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

@dataclass
class TokenTimeSeriesRow:
    """Single row in the time-series data"""
    timestamp: datetime  # UTC ISO8601
    address: str         # Token address
    price: float         # Token price
    fdv: float          # Fully Diluted Valuation
    market_cap: float   # Market Capitalization
    buys: int           # Number of buy transactions
    sells: int          # Number of sell transactions
    buy_volume: float   # Buy volume
    sell_volume: float  # Sell volume
    total_volume: float # Total volume
    
    def __post_init__(self):
        """Validate and normalize data types"""
        # Ensure timestamp is UTC
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        elif self.timestamp.tzinfo != timezone.utc:
            self.timestamp = self.timestamp.astimezone(timezone.utc)
        
        # Ensure numeric fields are proper types
        self.price = float(self.price) if self.price is not None else 0.0
        self.fdv = float(self.fdv) if self.fdv is not None else 0.0
        self.market_cap = float(self.market_cap) if self.market_cap is not None else 0.0
        self.buys = int(self.buys) if self.buys is not None else 0
        self.sells = int(self.sells) if self.sells is not None else 0
        self.buy_volume = float(self.buy_volume) if self.buy_volume is not None else 0.0
        self.sell_volume = float(self.sell_volume) if self.sell_volume is not None else 0.0
        self.total_volume = float(self.total_volume) if self.total_volume is not None else 0.0

class TimeSeriesSchema:
    """Schema validation and conversion utilities"""
    
    REQUIRED_COLUMNS = [
        'timestamp', 'address', 'price', 'fdv', 'market_cap',
        'buys', 'sells', 'buy_volume', 'sell_volume', 'total_volume'
    ]
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> bool:
        """Validate that DataFrame has required schema"""
        missing_cols = set(TimeSeriesSchema.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
        return True
    
    @staticmethod
    def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame to proper schema"""
        if not TimeSeriesSchema.validate_dataframe(df):
            raise ValueError("DataFrame missing required columns")
        
        # Create normalized copy
        normalized = df.copy()
        
        # Convert timestamp to UTC datetime
        normalized['timestamp'] = pd.to_datetime(normalized['timestamp'], utc=True, errors='coerce')
        
        # Convert numeric columns
        numeric_cols = ['price', 'fdv', 'market_cap', 'buys', 'sells', 'buy_volume', 'sell_volume', 'total_volume']
        for col in numeric_cols:
            if col in normalized.columns:
                normalized[col] = pd.to_numeric(normalized[col], errors='coerce')
        
        # Remove rows with invalid timestamps
        normalized = normalized.dropna(subset=['timestamp'])
        
        # Remove duplicates by address and timestamp
        normalized = normalized.drop_duplicates(subset=['address', 'timestamp'])
        
        # Sort by timestamp
        normalized = normalized.sort_values('timestamp')
        
        return normalized
    
    @staticmethod
    def create_empty_dataframe() -> pd.DataFrame:
        """Create empty DataFrame with correct schema"""
        return pd.DataFrame(columns=TimeSeriesSchema.REQUIRED_COLUMNS)

# Example usage and testing
if __name__ == "__main__":
    # Test schema validation
    print("Testing TimeSeriesSchema...")
    
    # Create sample data
    sample_data = {
        'timestamp': ['2025-08-20T10:00:00Z', '2025-08-20T10:01:00Z'],
        'address': ['0xABC123', '0xABC123'],
        'price': [1.0, 1.1],
        'fdv': [1000000, 1100000],
        'market_cap': [500000, 550000],
        'buys': [10, 15],
        'sells': [5, 8],
        'buy_volume': [1000, 1500],
        'sell_volume': [500, 800],
        'total_volume': [1500, 2300]
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Sample DataFrame shape: {df.shape}")
    print(f"Schema validation: {TimeSeriesSchema.validate_dataframe(df)}")
    
    # Test normalization
    normalized = TimeSeriesSchema.normalize_dataframe(df)
    print(f"Normalized DataFrame shape: {normalized.shape}")
    print(f"Timestamp timezone: {normalized['timestamp'].iloc[0].tzinfo}")
    print(f"Price type: {type(normalized['price'].iloc[0])}")
