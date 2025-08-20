"""
Feature Engineering for Token Time-Series Analysis
Adds technical indicators and analysis features to time-series data
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Adds technical analysis features to time-series data"""
    
    def __init__(self):
        self.feature_columns = []
    
    def add_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical features to the bars DataFrame"""
        if bars.empty:
            return bars
        
        try:
            bars = bars.copy()
            logger.info(f"Adding features to {len(bars)} bars")
            
            # Basic price features
            bars = self._add_price_features(bars)
            
            # Moving averages and trends
            bars = self._add_moving_averages(bars)
            
            # MACD
            bars = self._add_macd(bars)
            
            # RSI
            bars = self._add_rsi(bars)
            
            # Bollinger Bands
            bars = self._add_bollinger_bands(bars)
            
            # Volatility measures
            bars = self._add_volatility_features(bars)
            
            # Volume analysis
            bars = self._add_volume_features(bars)
            
            # Buy/sell imbalance
            bars = self._add_imbalance_features(bars)
            
            # Momentum and trend features
            bars = self._add_momentum_features(bars)
            
            # Clean up any infinite or NaN values
            bars = self._clean_features(bars)
            
            logger.info(f"Added {len(self.feature_columns)} features")
            return bars
            
        except Exception as e:
            logger.error(f"Failed to add features: {e}")
            return bars
    
    def _add_price_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features"""
        try:
            # Returns
            bars['ret'] = bars['close'].pct_change()
            bars['logret'] = np.log(bars['close']).diff()
            
            # Price changes
            bars['price_change'] = bars['close'].diff()
            bars['price_change_pct'] = bars['close'].pct_change() * 100
            
            # High-Low spread
            bars['hl_spread'] = bars['high'] - bars['low']
            bars['hl_spread_pct'] = (bars['hl_spread'] / bars['close']) * 100
            
            # Open-Close spread
            bars['oc_spread'] = bars['close'] - bars['open']
            bars['oc_spread_pct'] = (bars['oc_spread'] / bars['open']) * 100
            
            self.feature_columns.extend(['ret', 'logret', 'price_change', 'price_change_pct', 
                                       'hl_spread', 'hl_spread_pct', 'oc_spread', 'oc_spread_pct'])
            
        except Exception as e:
            logger.error(f"Failed to add price features: {e}")
        
        return bars
    
    def _add_moving_averages(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Add moving average features"""
        try:
            # Simple moving averages
            for period in [5, 10, 20, 50]:
                bars[f'sma_{period}'] = bars['close'].rolling(period).mean()
                bars[f'ema_{period}'] = bars['close'].ewm(span=period, adjust=False).mean()
            
            # Price vs moving averages
            for period in [5, 10, 20, 50]:
                bars[f'price_vs_sma_{period}'] = (bars['close'] / bars[f'sma_{period}'] - 1) * 100
                bars[f'price_vs_ema_{period}'] = (bars['close'] / bars[f'ema_{period}'] - 1) * 100
            
            # Moving average crossovers
            bars['sma_5_vs_20'] = (bars['sma_5'] / bars['sma_20'] - 1) * 100
            bars['ema_5_vs_20'] = (bars['ema_5'] / bars['ema_20'] - 1) * 100
            
            self.feature_columns.extend([f'sma_{p}' for p in [5, 10, 20, 50]] +
                                      [f'ema_{p}' for p in [5, 10, 20, 50]] +
                                      [f'price_vs_sma_{p}' for p in [5, 10, 20, 50]] +
                                      [f'price_vs_ema_{p}' for p in [5, 10, 20, 50]] +
                                      ['sma_5_vs_20', 'ema_5_vs_20'])
            
        except Exception as e:
            logger.error(f"Failed to add moving averages: {e}")
        
        return bars
    
    def _add_macd(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Add MACD features"""
        try:
            # MACD components
            bars['macd'] = bars['ema_12'] - bars['ema_26']
            bars['macd_signal'] = bars['macd'].ewm(span=9, adjust=False).mean()
            bars['macd_hist'] = bars['macd'] - bars['macd_signal']
            
            # MACD signals
            bars['macd_above_signal'] = (bars['macd'] > bars['macd_signal']).astype(int)
            bars['macd_cross_up'] = ((bars['macd'] > bars['macd_signal']) & 
                                   (bars['macd'].shift(1) <= bars['macd_signal'].shift(1))).astype(int)
            bars['macd_cross_down'] = ((bars['macd'] < bars['macd_signal']) & 
                                     (bars['macd'].shift(1) >= bars['macd_signal'].shift(1))).astype(int)
            
            self.feature_columns.extend(['macd', 'macd_signal', 'macd_hist', 
                                       'macd_above_signal', 'macd_cross_up', 'macd_cross_down'])
            
        except Exception as e:
            logger.error(f"Failed to add MACD: {e}")
        
        return bars
    
    def _add_rsi(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Add RSI features"""
        try:
            # RSI calculation
            delta = bars['close'].diff()
            up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
            rs = up / (down.replace(0, np.nan))
            bars['rsi_14'] = 100 - (100 / (1 + rs))
            
            # RSI signals
            bars['rsi_overbought'] = (bars['rsi_14'] > 70).astype(int)
            bars['rsi_oversold'] = (bars['rsi_14'] < 30).astype(int)
            bars['rsi_neutral'] = ((bars['rsi_14'] >= 30) & (bars['rsi_14'] <= 70)).astype(int)
            
            self.feature_columns.extend(['rsi_14', 'rsi_overbought', 'rsi_oversold', 'rsi_neutral'])
            
        except Exception as e:
            logger.error(f"Failed to add RSI: {e}")
        
        return bars
    
    def _add_bollinger_bands(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands features"""
        try:
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            
            bars['bb_mid'] = bars['close'].rolling(bb_period).mean()
            bars['bb_std'] = bars['close'].rolling(bb_period).std(ddof=0)
            bars['bb_upper'] = bars['bb_mid'] + bb_std * bars['bb_std']
            bars['bb_lower'] = bars['bb_mid'] - bb_std * bars['bb_std']
            
            # Bollinger Band positions
            bars['bb_position'] = (bars['close'] - bars['bb_lower']) / (bars['bb_upper'] - bars['bb_lower'])
            bars['bb_width'] = (bars['bb_upper'] - bars['bb_lower']) / bars['bb_mid']
            
            # Z-score (price position within bands)
            bars['bb_z_score'] = (bars['close'] - bars['bb_mid']) / bars['bb_std']
            
            # Band touches
            bars['bb_upper_touch'] = (bars['high'] >= bars['bb_upper']).astype(int)
            bars['bb_lower_touch'] = (bars['low'] <= bars['bb_lower']).astype(int)
            
            self.feature_columns.extend(['bb_mid', 'bb_std', 'bb_upper', 'bb_lower', 
                                       'bb_position', 'bb_width', 'bb_z_score',
                                       'bb_upper_touch', 'bb_lower_touch'])
            
        except Exception as e:
            logger.error(f"Failed to add Bollinger Bands: {e}")
        
        return bars
    
    def _add_volatility_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Add volatility measures"""
        try:
            # Rolling volatility (standard deviation of log returns)
            for period in [10, 20, 50]:
                bars[f'volatility_{period}'] = bars['logret'].rolling(period).std(ddof=0)
            
            # True Range and ATR
            bars['tr1'] = bars['high'] - bars['low']
            bars['tr2'] = abs(bars['high'] - bars['close'].shift(1))
            bars['tr3'] = abs(bars['low'] - bars['close'].shift(1))
            bars['true_range'] = bars[['tr1', 'tr2', 'tr3']].max(axis=1)
            bars['atr_14'] = bars['true_range'].rolling(14).mean()
            
            # Remove temporary columns
            bars = bars.drop(columns=['tr1', 'tr2', 'tr3'])
            
            # Volatility ratio
            bars['volatility_ratio'] = bars['volatility_20'] / bars['volatility_50']
            
            self.feature_columns.extend([f'volatility_{p}' for p in [10, 20, 50]] +
                                      ['true_range', 'atr_14', 'volatility_ratio'])
            
        except Exception as e:
            logger.error(f"Failed to add volatility features: {e}")
        
        return bars
    
    def _add_volume_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Add volume analysis features"""
        try:
            # Volume moving averages
            for period in [5, 10, 20]:
                bars[f'volume_sma_{period}'] = bars['volume'].rolling(period).mean()
                bars[f'volume_ema_{period}'] = bars['volume'].ewm(span=period, adjust=False).mean()
            
            # Volume ratios
            bars['volume_ratio_5'] = bars['volume'] / bars['volume_sma_5']
            bars['volume_ratio_20'] = bars['volume'] / bars['volume_sma_20']
            
            # Volume surge detection
            bars['volume_surge_5'] = (bars['volume'] > bars['volume_sma_5'] * 2).astype(int)
            bars['volume_surge_20'] = (bars['volume'] > bars['volume_sma_20'] * 3).astype(int)
            
            # Volume Z-score
            bars['volume_z_score_20'] = (bars['volume'] - bars['volume_sma_20']) / bars['volume'].rolling(20).std(ddof=0)
            
            # Buy vs Sell volume
            if 'buy_volume' in bars.columns and 'sell_volume' in bars.columns:
                bars['buy_sell_volume_ratio'] = bars['buy_volume'] / bars['sell_volume'].replace(0, np.nan)
                bars['buy_volume_pct'] = (bars['buy_volume'] / bars['total_volume']) * 100
                bars['sell_volume_pct'] = (bars['sell_volume'] / bars['total_volume']) * 100
            
            self.feature_columns.extend([f'volume_sma_{p}' for p in [5, 10, 20]] +
                                      [f'volume_ema_{p}' for p in [5, 10, 20]] +
                                      ['volume_ratio_5', 'volume_ratio_20', 'volume_surge_5', 
                                       'volume_surge_20', 'volume_z_score_20'])
            
        except Exception as e:
            logger.error(f"Failed to add volume features: {e}")
        
        return bars
    
    def _add_imbalance_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Add buy/sell imbalance features"""
        try:
            if 'buys' in bars.columns and 'sells' in bars.columns:
                # Buy/sell imbalance
                total_tx = bars['buys'] + bars['sells']
                bars['imbalance'] = (bars['buys'] - bars['sells']) / total_tx.replace(0, np.nan)
                
                # Moving average of imbalance
                bars['imbalance_ma_5'] = bars['imbalance'].rolling(5).mean()
                bars['imbalance_ma_20'] = bars['imbalance'].rolling(20).mean()
                
                # Imbalance signals
                bars['imbalance_bullish'] = (bars['imbalance'] > 0.2).astype(int)
                bars['imbalance_bearish'] = (bars['imbalance'] < -0.2).astype(int)
                bars['imbalance_neutral'] = ((bars['imbalance'] >= -0.2) & (bars['imbalance'] <= 0.2)).astype(int)
                
                # Buy/sell ratio
                bars['buy_sell_ratio'] = bars['buys'] / bars['sells'].replace(0, np.nan)
                bars['buy_sell_ratio_ma'] = bars['buy_sell_ratio'].rolling(20).mean()
                
                self.feature_columns.extend(['imbalance', 'imbalance_ma_5', 'imbalance_ma_20',
                                           'imbalance_bullish', 'imbalance_bearish', 'imbalance_neutral',
                                           'buy_sell_ratio', 'buy_sell_ratio_ma'])
            
        except Exception as e:
            logger.error(f"Failed to add imbalance features: {e}")
        
        return bars
    
    def _add_momentum_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Add momentum and trend features"""
        try:
            # Price momentum
            for period in [5, 10, 20]:
                bars[f'momentum_{period}'] = bars['close'].diff(period)
                bars[f'momentum_{period}_pct'] = (bars['close'] / bars['close'].shift(period) - 1) * 100
            
            # Rate of change
            for period in [5, 10, 20]:
                bars[f'roc_{period}'] = ((bars['close'] - bars['close'].shift(period)) / 
                                       bars['close'].shift(period)) * 100
            
            # Trend strength
            bars['trend_strength'] = abs(bars['ema_20'] - bars['ema_50']) / bars['ema_50'] * 100
            
            # Price acceleration
            bars['price_acceleration'] = bars['momentum_5'].diff()
            
            # Support and resistance levels (simplified)
            bars['support_level'] = bars['low'].rolling(20).min()
            bars['resistance_level'] = bars['high'].rolling(20).max()
            bars['price_vs_support'] = (bars['close'] / bars['support_level'] - 1) * 100
            bars['price_vs_resistance'] = (bars['close'] / bars['resistance_level'] - 1) * 100
            
            self.feature_columns.extend([f'momentum_{p}' for p in [5, 10, 20]] +
                                      [f'momentum_{p}_pct' for p in [5, 10, 20]] +
                                      [f'roc_{p}' for p in [5, 10, 20]] +
                                      ['trend_strength', 'price_acceleration', 'support_level', 
                                       'resistance_level', 'price_vs_support', 'price_vs_resistance'])
            
        except Exception as e:
            logger.error(f"Failed to add momentum features: {e}")
        
        return bars
    
    def _clean_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Clean up feature columns by removing infinite and extreme values"""
        try:
            # Replace infinite values with NaN
            bars = bars.replace([np.inf, -np.inf], np.nan)
            
            # Cap extreme values (e.g., > 1000% or < -1000%)
            percentage_columns = [col for col in bars.columns if 'pct' in col or 'ratio' in col]
            for col in percentage_columns:
                if col in bars.columns:
                    bars[col] = bars[col].clip(-1000, 1000)
            
            # Forward fill NaN values for some features
            fill_columns = ['bb_position', 'imbalance', 'buy_sell_ratio']
            for col in fill_columns:
                if col in bars.columns:
                    bars[col] = bars[col].fillna(method='ffill')
            
            # Drop rows with too many NaN values (e.g., > 50% of features)
            feature_cols = [col for col in bars.columns if col in self.feature_columns]
            if feature_cols:
                min_features = len(feature_cols) * 0.5
                bars = bars.dropna(thresh=min_features)
            
            logger.info(f"Cleaned features, remaining rows: {len(bars)}")
            
        except Exception as e:
            logger.error(f"Failed to clean features: {e}")
        
        return bars
    
    def get_feature_summary(self, bars: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of all features"""
        if bars.empty:
            return {"status": "no_data"}
        
        try:
            feature_cols = [col for col in bars.columns if col in self.feature_columns]
            
            summary = {
                "status": "success",
                "total_features": len(feature_cols),
                "feature_categories": {
                    "price": len([col for col in feature_cols if 'price' in col or 'ret' in col]),
                    "moving_averages": len([col for col in feature_cols if 'sma' in col or 'ema' in col]),
                    "oscillators": len([col for col in feature_cols if 'rsi' in col or 'macd' in col]),
                    "volatility": len([col for col in feature_cols if 'volatility' in col or 'bb_' in col]),
                    "volume": len([col for col in feature_cols if 'volume' in col]),
                    "momentum": len([col for col in feature_cols if 'momentum' in col or 'roc' in col])
                },
                "data_quality": {
                    "total_rows": len(bars),
                    "complete_rows": len(bars.dropna(subset=feature_cols)),
                    "missing_data_pct": (bars[feature_cols].isna().sum().sum() / (len(bars) * len(feature_cols))) * 100
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get feature summary: {e}")
            return {"status": "error", "error": str(e)}

# Example usage
if __name__ == "__main__":
    # Test feature engineering
    print("Testing FeatureEngineer...")
    
    # Create sample data
    dates = pd.date_range('2025-08-20', periods=100, freq='1min', tz='UTC')
    sample_data = {
        'timestamp': dates,
        'open': np.random.uniform(1.0, 1.1, 100),
        'high': np.random.uniform(1.1, 1.2, 100),
        'low': np.random.uniform(0.9, 1.0, 100),
        'close': np.random.uniform(1.0, 1.1, 100),
        'volume': np.random.uniform(1000, 5000, 100),
        'buys': np.random.randint(10, 50, 100),
        'sells': np.random.randint(10, 50, 100),
        'fdv': np.random.uniform(1000000, 1100000, 100),
        'market_cap': np.random.uniform(500000, 550000, 100),
        'buy_volume': np.random.uniform(500, 2500, 100),
        'sell_volume': np.random.uniform(500, 2500, 100),
        'total_volume': np.random.uniform(1000, 5000, 100)
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Add features
    engineer = FeatureEngineer()
    df_with_features = engineer.add_features(df)
    
    print(f"Original shape: {df.shape}")
    print(f"With features: {df_with_features.shape}")
    
    # Get feature summary
    summary = engineer.get_feature_summary(df_with_features)
    print(f"Feature summary: {summary}")
