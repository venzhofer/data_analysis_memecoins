"""
Enhanced Time-Series Token Analyzer - Adapted for Real Webhook Data
Works with the actual data structure returned by the webhooks
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
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptedTimeSeriesAnalyzer:
    """Analyzer adapted for the actual webhook data structure"""
    
    def __init__(self, 
                 tokens_webhook: str,
                 trading_webhook: str,
                 output_dir: str = "output"):
        
        self.tokens_webhook = tokens_webhook
        self.trading_webhook = trading_webhook
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "TokenTS/1.0",
            "Accept": "application/json"
        })
        
        # Analysis results
        self.analysis_results = {}
        self.token_metadata = {}
    
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
            else:
                tokens = data if isinstance(data, list) else [data]
            
            logger.info(f"Successfully fetched {len(tokens)} tokens")
            return tokens
            
        except Exception as e:
            logger.error(f"Failed to fetch tokens: {e}")
            return []
    
    def fetch_trading_data(self, token_address: str) -> Optional[Dict]:
        """Fetch trading data for a specific token"""
        try:
            url = f"{self.trading_webhook}?token={token_address}"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            if response.text.strip():
                data = response.json()
                return data
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch trading data for {token_address[:10]}: {e}")
            return None
    
    def analyze_token(self, token_info: Dict) -> Dict[str, Any]:
        """Analyze a single token with the actual data structure"""
        try:
            # Handle both 'address' and 'adress' (typo in the API)
            address = token_info.get('address') or token_info.get('adress')
            token_name = token_info.get('name', 'Unknown')
            
            logger.info(f"Analyzing token {token_name} ({address[:10]}...)")
            
            # Fetch current trading data
            trading_data = self.fetch_trading_data(address)
            if not trading_data:
                return {
                    "status": "no_trading_data",
                    "address": address,
                    "token_name": token_name
                }
            
            # Extract key metrics
            current_fdv = trading_data.get('fdv', 0)
            current_market_cap = trading_data.get('market_cap', 0)
            current_price = trading_data.get('price', 0)
            
            # Get start values from token metadata
            start_fdv = token_info.get('start_fdv', 0)
            start_market_cap = token_info.get('start_market_cap', 0)
            
            # Calculate performance metrics
            fdv_change_pct = 0
            market_cap_change_pct = 0
            
            if start_fdv and start_fdv > 0:
                fdv_change_pct = ((current_fdv - start_fdv) / start_fdv) * 100
            
            if start_market_cap and start_market_cap > 0:
                market_cap_change_pct = ((current_market_cap - start_market_cap) / start_market_cap) * 100
            
            # Analyze transaction data
            transaction_analysis = self._analyze_transactions(trading_data.get('transactions', {}))
            
            # Classify performance
            pattern = self._classify_performance(fdv_change_pct)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(token_info, trading_data)
            
            # Calculate momentum metrics
            momentum_metrics = self._calculate_momentum_metrics(trading_data)
            
            # Compile analysis results
            analysis = {
                "status": "analyzed",
                "address": address,
                "token_name": token_name,
                "current_metrics": {
                    "fdv": current_fdv,
                    "market_cap": current_market_cap,
                    "price": current_price
                },
                "start_metrics": {
                    "start_fdv": start_fdv,
                    "start_market_cap": start_market_cap
                },
                "performance_metrics": {
                    "fdv_change_pct": fdv_change_pct,
                    "market_cap_change_pct": market_cap_change_pct
                },
                "pattern": pattern,
                "transaction_analysis": transaction_analysis,
                "risk_metrics": risk_metrics,
                "momentum_metrics": momentum_metrics
            }
            
            logger.info(f"Analysis completed for {token_name}")
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis failed for {token_info.get('name', 'Unknown')}: {e}")
            return {
                "status": "error",
                "address": token_info.get('address', 'Unknown'),
                "token_name": token_info.get('name', 'Unknown'),
                "error": str(e)
            }
    
    def _analyze_transactions(self, transactions: Dict) -> Dict[str, Any]:
        """Analyze transaction data from different time periods"""
        try:
            analysis = {}
            
            # Extract time periods (h1, h6, m5, etc.)
            for period, data in transactions.items():
                if isinstance(data, dict):
                    buys = data.get('buys', 0)
                    sells = data.get('sells', 0)
                    
                    analysis[period] = {
                        "buys": buys,
                        "sells": sells,
                        "total_transactions": buys + sells,
                        "buy_sell_ratio": buys / max(sells, 1),
                        "buy_percentage": (buys / max(buys + sells, 1)) * 100
                    }
            
            # Calculate overall metrics
            all_buys = sum(data.get('buys', 0) for data in transactions.values() if isinstance(data, dict))
            all_sells = sum(data.get('sells', 0) for data in transactions.values() if isinstance(data, dict))
            
            analysis["overall"] = {
                "total_buys": all_buys,
                "total_sells": all_sells,
                "total_transactions": all_buys + all_sells,
                "buy_sell_ratio": all_buys / max(all_sells, 1),
                "buy_percentage": (all_buys / max(all_buys + all_sells, 1)) * 100
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze transactions: {e}")
            return {}
    
    def _classify_performance(self, fdv_change_pct: float) -> str:
        """Classify token performance based on FDV change"""
        try:
            if fdv_change_pct > 100:
                return "moon_shot"
            elif fdv_change_pct > 50:
                return "strong_rise"
            elif fdv_change_pct > 20:
                return "moderate_rise"
            elif fdv_change_pct < -80:
                return "died"
            elif fdv_change_pct < -50:
                return "significant_drop"
            elif fdv_change_pct < -20:
                return "moderate_drop"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Failed to classify performance: {e}")
            return "unknown"
    
    def _calculate_risk_metrics(self, token_info: Dict, trading_data: Dict) -> Dict[str, Any]:
        """Calculate risk assessment metrics"""
        try:
            # Base risk from token metadata
            base_risk = token_info.get('rugcheck_score', 5)
            
            # Transaction risk (based on buy/sell imbalance)
            transaction_risk = 0
            transactions = trading_data.get('transactions', {})
            
            if transactions:
                # Check if there are any periods with extreme imbalances
                for period, data in transactions.items():
                    if isinstance(data, dict):
                        buys = data.get('buys', 0)
                        sells = data.get('sells', 0)
                        total = buys + sells
                        
                        if total > 0:
                            imbalance = abs(buys - sells) / total
                            if imbalance > 0.8:  # Very imbalanced
                                transaction_risk = 3
                            elif imbalance > 0.5:  # Moderately imbalanced
                                transaction_risk = 2
                            else:
                                transaction_risk = 1
                            break
            
            # Composite risk score
            total_risk = min(base_risk + transaction_risk, 10)
            
            return {
                "base_risk": base_risk,
                "transaction_risk": transaction_risk,
                "total_risk_score": total_risk,
                "risk_level": "high" if total_risk >= 7 else "medium" if total_risk >= 4 else "low"
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate risk metrics: {e}")
            return {"total_risk_score": 5, "risk_level": "medium"}
    
    def _calculate_momentum_metrics(self, trading_data: Dict) -> Dict[str, Any]:
        """Calculate momentum and trend metrics"""
        try:
            momentum_score = 0
            
            # Analyze transaction momentum across time periods
            transactions = trading_data.get('transactions', {})
            periods = list(transactions.keys())
            
            if len(periods) >= 2:
                # Compare recent vs older periods
                recent_period = periods[0]  # Assuming periods are ordered by recency
                older_period = periods[-1]
                
                if isinstance(transactions[recent_period], dict) and isinstance(transactions[older_period], dict):
                    recent_buys = transactions[recent_period].get('buys', 0)
                    recent_sells = transactions[recent_period].get('sells', 0)
                    older_buys = transactions[older_period].get('buys', 0)
                    older_sells = transactions[older_period].get('sells', 0)
                    
                    # Recent buy momentum
                    if recent_buys > older_buys:
                        momentum_score += 1
                    
                    # Recent sell momentum
                    if recent_sells < older_sells:
                        momentum_score += 1
                    
                    # Buy/sell ratio momentum
                    recent_ratio = recent_buys / max(recent_sells, 1)
                    older_ratio = older_buys / max(older_sells, 1)
                    
                    if recent_ratio > older_ratio:
                        momentum_score += 1
            
            # Volume momentum (if available)
            if 'volume' in trading_data:
                momentum_score += 1
            
            return {
                "momentum_score": momentum_score,
                "momentum_level": "high" if momentum_score >= 3 else "medium" if momentum_score >= 1 else "low"
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate momentum metrics: {e}")
            return {"momentum_score": 0, "momentum_level": "low"}
    
    def analyze_all_tokens(self, max_workers: int = 4) -> Dict[str, Any]:
        """Analyze all available tokens in parallel"""
        logger.info("Starting analysis of all tokens...")
        
        try:
            # Fetch token list
            tokens = self.fetch_tokens_info()
            if not tokens:
                logger.warning("No tokens found")
                return {"status": "no_tokens"}
            
            logger.info(f"Found {len(tokens)} tokens to analyze")
            
            # Analyze tokens in parallel
            results = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit analysis tasks
                future_to_token = {}
                for token in tokens:
                    future = executor.submit(self.analyze_token, token)
                    future_to_token[future] = token
                
                # Collect results
                for future in as_completed(future_to_token):
                    token = future_to_token[future]
                    try:
                        result = future.result()
                        # Handle both 'address' and 'adress' (typo in the API)
                        token_address = token.get('address') or token.get('adress')
                        results[token_address] = result
                        logger.info(f"Completed analysis for {token.get('name', 'Unknown')}")
                    except Exception as e:
                        logger.error(f"Analysis failed for {token.get('name', 'Unknown')}: {e}")
                        token_address = token.get('address') or token.get('adress')
                        results[token_address] = {
                            "status": "error",
                            "address": token_address,
                            "token_name": token.get('name', 'Unknown'),
                            "error": str(e)
                        }
            
            # Store results
            self.analysis_results = results
            
            # Generate summary statistics
            summary = self._generate_summary_statistics(results)
            
            logger.info("Analysis of all tokens completed")
            return {
                "status": "completed",
                "total_tokens": len(tokens),
                "analyzed_tokens": len([r for r in results.values() if r.get('status') == 'analyzed']),
                "results": results,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"status": "error", "error": str(e)}
    
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
                    "high_momentum_tokens": len([x for x in momentum_scores if x >= 3]),
                    "medium_momentum_tokens": len([x for x in momentum_scores if 1 <= x < 3]),
                    "low_momentum_tokens": len([x for x in momentum_scores if x < 1])
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary statistics: {e}")
            return {"status": "error", "error": str(e)}
    
    def save_results(self, filename: str = "adapted_timeseries_analysis.json"):
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
                token_name = result.get('token_name', 'Unknown')
                performance = result.get('performance_metrics', {})
                risk = result.get('risk_metrics', {})
                momentum = result.get('momentum_metrics', {})
                
                row = [
                    address[:10] + "...",
                    token_name,
                    f"{performance.get('fdv_change_pct', 0):+.2f}%" if performance.get('fdv_change_pct') is not None else "N/A",
                    f"{performance.get('market_cap_change_pct', 0):+.2f}%" if performance.get('market_cap_change_pct') is not None else "N/A",
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
                "Address", "Name", "FDV Change", "Market Cap Change", "Pattern", 
                "Risk Score", "Risk Level", "Momentum Score", "Momentum Level"
            ]
            
            table = tabulate(table_data, headers=headers, tablefmt="grid", numalign="right")
            
            # Save table to file
            table_file = self.output_dir / "adapted_timeseries_analysis_table.txt"
            with open(table_file, 'w') as f:
                f.write("ADAPTED TIME-SERIES ANALYSIS TABLE\n")
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
    analyzer = AdaptedTimeSeriesAnalyzer(
        tokens_webhook="https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4",
        trading_webhook="https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0",
        output_dir="output"
    )
    
    # Run analysis
    print("Starting adapted analysis...")
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
