"""
Enhanced Time-Series Token Analyzer
Integrates all components for comprehensive time-series analysis
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from tabulate import tabulate

# Import our custom modules
from timeseries_ingestor import TimeSeriesIngestor
from timeseries_loader import TimeSeriesLoader
from feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedTimeSeriesAnalyzer:
    """Comprehensive time-series analysis for token performance"""
    
    def __init__(self, 
                 tokens_webhook: str,
                 trading_webhook: str,
                 data_dir: str = "data/ts",
                 output_dir: str = "output"):
        
        self.tokens_webhook = tokens_webhook
        self.trading_webhook = trading_webhook
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.ingestor = TimeSeriesIngestor(tokens_webhook, trading_webhook, data_dir)
        self.loader = TimeSeriesLoader(data_dir)
        self.feature_engineer = FeatureEngineer()
        
        # Analysis results
        self.analysis_results = {}
        self.token_metadata = {}
    
    def ingest_data(self, batch_size: int = 5) -> Dict:
        """Ingest data for all tokens"""
        logger.info("Starting data ingestion...")
        
        try:
            stats = self.ingestor.ingest_all_tokens(batch_size)
            logger.info("Data ingestion completed")
            return stats
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            return {"error": str(e)}
    
    def analyze_token(self, 
                     address: str, 
                     token_info: Dict,
                     resample_rule: str = "1min") -> Dict[str, Any]:
        """Analyze a single token with comprehensive time-series analysis"""
        try:
            logger.info(f"Analyzing token {address[:10]}...")
            
            # Load raw data
            df = self.loader.load_token_df(address)
            if df.empty:
                return {"status": "no_data", "address": address}
            
            # Resample to bars
            bars = self.loader.resample_bars(df, resample_rule)
            if bars.empty:
                return {"status": "resampling_failed", "address": address}
            
            # Add technical features
            bars_with_features = self.feature_engineer.add_features(bars)
            
            # Get feature summary
            feature_summary = self.feature_engineer.get_feature_summary(bars_with_features)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(bars_with_features, token_info)
            
            # Classify performance pattern
            pattern = self._classify_performance(performance_metrics)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(bars_with_features, token_info)
            
            # Calculate momentum metrics
            momentum_metrics = self._calculate_momentum_metrics(bars_with_features)
            
            # Compile analysis results
            analysis = {
                "status": "analyzed",
                "address": address,
                "token_info": token_info,
                "data_summary": {
                    "total_rows": len(df),
                    "total_bars": len(bars),
                    "bars_with_features": len(bars_with_features),
                    "date_range": {
                        "start": bars['timestamp'].min().isoformat() if not bars.empty else None,
                        "end": bars['timestamp'].max().isoformat() if not bars.empty else None
                    }
                },
                "feature_summary": feature_summary,
                "performance_metrics": performance_metrics,
                "pattern": pattern,
                "risk_metrics": risk_metrics,
                "momentum_metrics": momentum_metrics,
                "technical_indicators": self._extract_key_indicators(bars_with_features)
            }
            
            logger.info(f"Analysis completed for {address[:10]}")
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis failed for {address}: {e}")
            return {"status": "error", "address": address, "error": str(e)}
    
    def _calculate_performance_metrics(self, bars: pd.DataFrame, token_info: Dict) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if bars.empty:
            return {}
        
        try:
            first_bar = bars.iloc[0]
            last_bar = bars.iloc[-1]
            
            # Price-based metrics
            price_change = (last_bar['close'] / first_bar['close'] - 1) * 100
            price_volatility = bars['close'].pct_change().std() * 100
            
            # FDV and market cap changes
            fdv_change = None
            market_cap_change = None
            
            if 'fdv' in bars.columns and 'fdv' in first_bar and 'fdv' in last_bar:
                if first_bar['fdv'] > 0:
                    fdv_change = (last_bar['fdv'] / first_bar['fdv'] - 1) * 100
            
            if 'market_cap' in bars.columns and 'market_cap' in first_bar and 'market_cap' in last_bar:
                if first_bar['market_cap'] > 0:
                    market_cap_change = (last_bar['market_cap'] / first_bar['market_cap'] - 1) * 100
            
            # Drawdown analysis
            cumulative_returns = (bars['close'] / first_bar['close']).cummax()
            drawdown = ((bars['close'] / cumulative_returns) - 1) * 100
            max_drawdown = drawdown.min()
            
            # Volume analysis
            total_volume = bars['volume'].sum() if 'volume' in bars.columns else 0
            avg_volume = bars['volume'].mean() if 'volume' in bars.columns else 0
            
            # Transaction analysis
            total_buys = bars['buys'].sum() if 'buys' in bars.columns else 0
            total_sells = bars['sells'].sum() if 'sells' in bars.columns else 0
            buy_sell_ratio = total_buys / max(total_sells, 1)
            
            return {
                "price_change_pct": price_change,
                "price_volatility_pct": price_volatility,
                "fdv_change_pct": fdv_change,
                "market_cap_change_pct": market_cap_change,
                "max_drawdown_pct": max_drawdown,
                "total_volume": total_volume,
                "avg_volume": avg_volume,
                "total_buys": total_buys,
                "total_sells": total_sells,
                "buy_sell_ratio": buy_sell_ratio,
                "bars_analyzed": len(bars)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            return {}
    
    def _classify_performance(self, metrics: Dict[str, Any]) -> str:
        """Classify token performance based on metrics"""
        try:
            # Use FDV change if available, otherwise price change
            change_pct = metrics.get('fdv_change_pct')
            if change_pct is None:
                change_pct = metrics.get('price_change_pct', 0)
            
            if change_pct is None:
                return "unknown"
            
            # Classification logic
            if change_pct > 100:
                return "moon_shot"
            elif change_pct > 50:
                return "strong_rise"
            elif change_pct > 20:
                return "moderate_rise"
            elif change_pct < -80:
                return "died"
            elif change_pct < -50:
                return "significant_drop"
            elif change_pct < -20:
                return "moderate_drop"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Failed to classify performance: {e}")
            return "unknown"
    
    def _calculate_risk_metrics(self, bars: pd.DataFrame, token_info: Dict) -> Dict[str, Any]:
        """Calculate risk assessment metrics"""
        try:
            # Base risk from token metadata
            base_risk = token_info.get('rugcheck_score', 5)
            
            # Volatility-based risk
            volatility_risk = 0
            if 'volatility_20' in bars.columns:
                avg_volatility = bars['volatility_20'].mean()
                if avg_volatility > 0.1:  # High volatility
                    volatility_risk = 3
                elif avg_volatility > 0.05:  # Medium volatility
                    volatility_risk = 2
                else:
                    volatility_risk = 1
            
            # Drawdown risk
            drawdown_risk = 0
            if 'close' in bars.columns:
                cumulative_returns = (bars['close'] / bars['close'].iloc[0]).cummax()
                drawdown = (bars['close'] / cumulative_returns - 1)
                max_drawdown = abs(drawdown.min())
                if max_drawdown > 0.5:  # >50% drawdown
                    drawdown_risk = 3
                elif max_drawdown > 0.2:  # >20% drawdown
                    drawdown_risk = 2
                else:
                    drawdown_risk = 1
            
            # Volume risk
            volume_risk = 0
            if 'volume' in bars.columns:
                volume_std = bars['volume'].std()
                volume_mean = bars['volume'].mean()
                if volume_mean > 0:
                    volume_cv = volume_std / volume_mean
                    if volume_cv > 2:  # High volume variability
                        volume_risk = 2
                    else:
                        volume_risk = 1
            
            # Composite risk score
            total_risk = min(base_risk + volatility_risk + drawdown_risk + volume_risk, 10)
            
            return {
                "base_risk": base_risk,
                "volatility_risk": volatility_risk,
                "drawdown_risk": drawdown_risk,
                "volume_risk": volume_risk,
                "total_risk_score": total_risk,
                "risk_level": "high" if total_risk >= 7 else "medium" if total_risk >= 4 else "low"
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate risk metrics: {e}")
            return {"total_risk_score": 5, "risk_level": "medium"}
    
    def _calculate_momentum_metrics(self, bars: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum and trend metrics"""
        try:
            momentum_score = 0
            
            # RSI momentum
            if 'rsi_14' in bars.columns:
                last_rsi = bars['rsi_14'].iloc[-1]
                if not pd.isna(last_rsi):
                    if last_rsi > 70:
                        momentum_score += 1  # Overbought
                    elif last_rsi < 30:
                        momentum_score += 2  # Oversold (potential reversal)
                    else:
                        momentum_score += 1  # Neutral
            
            # MACD momentum
            if 'macd' in bars.columns and 'macd_signal' in bars.columns:
                last_macd = bars['macd'].iloc[-1]
                last_signal = bars['macd_signal'].iloc[-1]
                if not pd.isna(last_macd) and not pd.isna(last_signal):
                    if last_macd > last_signal:
                        momentum_score += 1  # Bullish MACD
            
            # Volume momentum
            if 'volume_ratio_20' in bars.columns:
                last_volume_ratio = bars['volume_ratio_20'].iloc[-1]
                if not pd.isna(last_volume_ratio) and last_volume_ratio > 1.5:
                    momentum_score += 1  # High volume
            
            # Buy/sell imbalance momentum
            if 'imbalance' in bars.columns:
                last_imbalance = bars['imbalance'].iloc[-1]
                if not pd.isna(last_imbalance):
                    if last_imbalance > 0.2:
                        momentum_score += 1  # Bullish imbalance
                    elif last_imbalance < -0.2:
                        momentum_score += 1  # Bearish imbalance
            
            # Moving average momentum
            if 'ema_5' in bars.columns and 'ema_20' in bars.columns:
                last_ema_5 = bars['ema_5'].iloc[-1]
                last_ema_20 = bars['ema_20'].iloc[-1]
                if not pd.isna(last_ema_5) and not pd.isna(last_ema_20):
                    if last_ema_5 > last_ema_20:
                        momentum_score += 1  # Short-term above long-term
            
            return {
                "momentum_score": momentum_score,
                "momentum_level": "high" if momentum_score >= 4 else "medium" if momentum_score >= 2 else "low"
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate momentum metrics: {e}")
            return {"momentum_score": 0, "momentum_level": "low"}
    
    def _extract_key_indicators(self, bars: pd.DataFrame) -> Dict[str, Any]:
        """Extract key technical indicators for reporting"""
        if bars.empty:
            return {}
        
        try:
            indicators = {}
            
            # Get last values of key indicators
            key_indicators = [
                'rsi_14', 'macd', 'bb_position', 'volatility_20', 
                'imbalance', 'volume_ratio_20', 'ema_20', 'sma_20'
            ]
            
            for indicator in key_indicators:
                if indicator in bars.columns:
                    last_value = bars[indicator].iloc[-1]
                    if not pd.isna(last_value):
                        indicators[indicator] = float(last_value)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to extract indicators: {e}")
            return {}
    
    def analyze_all_tokens(self, max_workers: int = 4) -> Dict[str, Any]:
        """Analyze all available tokens in parallel"""
        logger.info("Starting analysis of all tokens...")
        
        try:
            # Get available addresses
            addresses = self.loader.get_available_addresses()
            if not addresses:
                logger.warning("No token addresses found")
                return {"status": "no_tokens"}
            
            logger.info(f"Found {len(addresses)} tokens to analyze")
            
            # Get token metadata
            self.token_metadata = self._get_token_metadata()
            
            # Analyze tokens in parallel
            results = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit analysis tasks
                future_to_address = {}
                for address in addresses:
                    token_info = self.token_metadata.get(address, {})
                    future = executor.submit(self.analyze_token, address, token_info)
                    future_to_address[future] = address
                
                # Collect results
                for future in as_completed(future_to_address):
                    address = future_to_address[future]
                    try:
                        result = future.result()
                        results[address] = result
                        logger.info(f"Completed analysis for {address[:10]}")
                    except Exception as e:
                        logger.error(f"Analysis failed for {address}: {e}")
                        results[address] = {"status": "error", "address": address, "error": str(e)}
            
            # Store results
            self.analysis_results = results
            
            # Generate summary statistics
            summary = self._generate_summary_statistics(results)
            
            logger.info("Analysis of all tokens completed")
            return {
                "status": "completed",
                "total_tokens": len(addresses),
                "analyzed_tokens": len([r for r in results.values() if r.get('status') == 'analyzed']),
                "results": results,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _get_token_metadata(self) -> Dict[str, Dict]:
        """Get metadata for all tokens"""
        try:
            tokens = self.ingestor.fetch_tokens_info()
            metadata = {}
            
            for token in tokens:
                address = token.get('address') or token.get('adress')
                if address:
                    metadata[address] = token
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get token metadata: {e}")
            return {}
    
    def _generate_summary_statistics(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate summary statistics from analysis results"""
        try:
            analyzed_results = [r for r in results.values() if r.get('status') == 'analyzed']
            
            if not analyzed_results:
                return {"status": "no_analyzed_tokens"}
            
            # Pattern distribution
            patterns = [r.get('pattern', 'unknown') for r in analyzed_results]
            pattern_counts = pd.Series(patterns).value_counts().to_dict()
            
            # Performance metrics
            fdv_changes = [r.get('performance_metrics', {}).get('fdv_change_pct') for r in analyzed_results]
            fdv_changes = [x for x in fdv_changes if x is not None]
            
            price_changes = [r.get('performance_metrics', {}).get('price_change_pct') for r in analyzed_results]
            price_changes = [x for x in price_changes if x is not None]
            
            # Risk scores
            risk_scores = [r.get('risk_metrics', {}).get('total_risk_score') for r in analyzed_results]
            risk_scores = [x for x in risk_scores if x is not None]
            
            # Momentum scores
            momentum_scores = [r.get('momentum_metrics', {}).get('momentum_score') for r in analyzed_results]
            momentum_scores = [x for x in momentum_scores if x is not None]
            
            summary = {
                "pattern_distribution": pattern_counts,
                "performance_summary": {
                    "avg_fdv_change": np.mean(fdv_changes) if fdv_changes else 0,
                    "max_fdv_rise": max(fdv_changes) if fdv_changes else 0,
                    "max_fdv_drop": min(fdv_changes) if fdv_changes else 0,
                    "avg_price_change": np.mean(price_changes) if price_changes else 0,
                    "tokens_with_positive_fdv": len([x for x in fdv_changes if x > 0]),
                    "tokens_with_negative_fdv": len([x for x in fdv_changes if x < 0])
                },
                "risk_summary": {
                    "avg_risk_score": np.mean(risk_scores) if risk_scores else 0,
                    "high_risk_tokens": len([x for x in risk_scores if x >= 7]),
                    "medium_risk_tokens": len([x for x in risk_scores if 4 <= x < 7]),
                    "low_risk_tokens": len([x for x in risk_scores if x < 4])
                },
                "momentum_summary": {
                    "avg_momentum_score": np.mean(momentum_scores) if momentum_scores else 0,
                    "high_momentum_tokens": len([x for x in momentum_scores if x >= 4]),
                    "medium_momentum_tokens": len([x for x in momentum_scores if 2 <= x < 4]),
                    "low_momentum_tokens": len([x for x in momentum_scores if x < 2])
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary statistics: {e}")
            return {"status": "error", "error": str(e)}
    
    def save_results(self, filename: str = "enhanced_timeseries_analysis.json"):
        """Save analysis results to file"""
        try:
            output_file = self.output_dir / filename
            
            # Prepare data for JSON serialization
            serializable_results = {}
            for address, result in self.analysis_results.items():
                serializable_results[address] = self._make_serializable(result)
            
            output_data = {
                "analysis_timestamp": datetime.now().isoformat(),
                "total_tokens": len(self.analysis_results),
                "results": serializable_results
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            logger.info(f"Results saved to {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return None
    
    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def create_summary_table(self) -> str:
        """Create a summary table of all analysis results"""
        try:
            analyzed_results = [r for r in self.analysis_results.values() if r.get('status') == 'analyzed']
            
            if not analyzed_results:
                return "No analyzed results to display"
            
            # Prepare table data
            table_data = []
            for result in analyzed_results:
                address = result.get('address', 'Unknown')
                token_info = result.get('token_info', {})
                performance = result.get('performance_metrics', {})
                risk = result.get('risk_metrics', {})
                momentum = result.get('momentum_metrics', {})
                
                row = [
                    address[:10] + "...",
                    token_info.get('name', 'Unknown'),
                    f"{performance.get('fdv_change_pct', 0):+.2f}%" if performance.get('fdv_change_pct') is not None else "N/A",
                    f"{performance.get('price_change_pct', 0):+.2f}%" if performance.get('price_change_pct') is not None else "N/A",
                    result.get('pattern', 'unknown'),
                    risk.get('total_risk_score', 0),
                    risk.get('risk_level', 'unknown'),
                    momentum.get('momentum_score', 0),
                    momentum.get('momentum_level', 'unknown')
                ]
                table_data.append(row)
            
            # Sort by FDV change
            table_data.sort(key=lambda x: float(x[2].replace('%', '').replace('+', '').replace('N/A', '0')), reverse=True)
            
            headers = [
                "Address", "Name", "FDV Change", "Price Change", "Pattern", 
                "Risk Score", "Risk Level", "Momentum Score", "Momentum Level"
            ]
            
            table = tabulate(table_data, headers=headers, tablefmt="grid", numalign="right")
            
            # Save table to file
            table_file = self.output_dir / "enhanced_timeseries_analysis_table.txt"
            with open(table_file, 'w') as f:
                f.write("ENHANCED TIME-SERIES ANALYSIS TABLE\n")
                f.write("="*80 + "\n\n")
                f.write(table)
            
            logger.info(f"Summary table saved to {table_file}")
            return table
            
        except Exception as e:
            logger.error(f"Failed to create summary table: {e}")
            return f"Error creating table: {e}"

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = EnhancedTimeSeriesAnalyzer(
        tokens_webhook="https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4",
        trading_webhook="https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0",
        data_dir="data/ts",
        output_dir="output"
    )
    
    # First, ingest data if needed
    print("Checking if data exists...")
    try:
        addresses = analyzer.loader.get_available_addresses()
        if not addresses:
            print("No data found. Starting ingestion...")
            stats = analyzer.ingest_data()
            print(f"Ingestion stats: {stats}")
        else:
            print(f"Found {len(addresses)} existing token addresses")
    except Exception as e:
        print(f"Error checking data: {e}")
        print("Starting fresh ingestion...")
        stats = analyzer.ingest_data()
        print(f"Ingestion stats: {stats}")
    
    # Run analysis
    print("\nStarting analysis...")
    results = analyzer.analyze_all_tokens()
    print(f"Analysis completed: {results.get('status')}")
    
    if results.get('status') == 'completed':
        # Save results
        output_file = analyzer.save_results()
        print(f"Results saved to: {output_file}")
        
        # Create summary table
        table = analyzer.create_summary_table()
        print("\nSummary Table:")
        print(table)
        
        # Show summary statistics
        summary = results.get('summary', {})
        if summary:
            print(f"\nSummary Statistics:")
            print(f"Pattern Distribution: {summary.get('pattern_distribution', {})}")
            print(f"Performance Summary: {summary.get('performance_summary', {})}")
            print(f"Risk Summary: {summary.get('risk_summary', {})}")
            print(f"Momentum Summary: {summary.get('momentum_summary', {})}")
