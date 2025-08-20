#!/usr/bin/env python3
"""
Human-Readable Correlation Analysis for Memecoin Patterns
Creates actionable insights connecting patterns to actual performance outcomes
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class HumanReadableCorrelationAnalyzer:
    """Creates human-readable correlation analysis with actionable insights"""
    
    def __init__(self, results_file: str = "output/adapted_timeseries_analysis.json"):
        self.results_file = Path(results_file)
        self.output_dir = Path("output/human_readable_insights")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.results = self._load_results()
        self.df = self._create_dataframe()
        
        # Pattern definitions
        self.patterns = {
            'moon_shot': '🚀 Moon Shot (>100% gain)',
            'strong_rise': '📈 Strong Rise (50-100% gain)',
            'moderate_rise': '📊 Moderate Rise (20-50% gain)',
            'stable': '⚖️ Stable (-20% to +20%)',
            'moderate_drop': '📉 Moderate Drop (-20% to -50%)',
            'significant_drop': '💸 Significant Drop (-50% to -80%)',
            'died': '⚰️ Died (>80% loss)'
        }
    
    def _load_results(self) -> dict:
        """Load analysis results from JSON file"""
        try:
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading results: {e}")
            return {}
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for easier analysis"""
        if not self.results or 'results' not in self.results:
            return pd.DataFrame()
        
        rows = []
        for address, result in self.results['results'].items():
            if result.get('status') == 'analyzed':
                # Extract all available metrics
                row = {
                    'address': address,
                    'token_name': result.get('token_name', 'Unknown'),
                    'fdv_change_pct': result.get('performance_metrics', {}).get('fdv_change_pct', 0),
                    'market_cap_change_pct': result.get('performance_metrics', {}).get('market_cap_change_pct', 0),
                    'pattern': result.get('pattern', 'unknown'),
                    'risk_score': result.get('risk_metrics', {}).get('total_risk_score', 0),
                    'risk_level': result.get('risk_metrics', {}).get('risk_level', 'unknown'),
                    'momentum_score': result.get('momentum_metrics', {}).get('momentum_score', 0),
                    'momentum_level': result.get('momentum_metrics', {}).get('momentum_level', 'unknown'),
                    'current_fdv': result.get('current_metrics', {}).get('fdv', 0),
                    'start_fdv': result.get('start_metrics', {}).get('start_fdv', 0),
                    'current_market_cap': result.get('current_metrics', {}).get('market_cap', 0),
                    'start_market_cap': result.get('start_metrics', {}).get('start_market_cap', 0),
                    'current_price': result.get('current_metrics', {}).get('price', 0)
                }
                
                # Add transaction analysis metrics
                transaction_data = result.get('transaction_analysis', {})
                if 'overall' in transaction_data:
                    overall = transaction_data['overall']
                    row.update({
                        'total_transactions': overall.get('total_transactions', 0),
                        'buy_sell_ratio': overall.get('buy_sell_ratio', 0),
                        'buy_percentage': overall.get('buy_percentage', 0),
                        'total_buys': overall.get('total_buys', 0),
                        'total_sells': overall.get('total_sells', 0)
                    })
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Add derived metrics
        if not df.empty:
            df['fdv_volatility'] = abs(df['fdv_change_pct'])
            df['risk_performance_ratio'] = df['risk_score'] / (df['fdv_change_pct'] + 100)
            df['momentum_efficiency'] = df['momentum_score'] / (df['fdv_change_pct'] + 100)
            df['transaction_intensity'] = df['total_transactions'] / (df['start_fdv'] + 1)
            df['buy_pressure'] = df['buy_percentage'] - 50
            df['is_profitable'] = df['fdv_change_pct'] > 0
            df['is_moon_shot'] = df['fdv_change_pct'] > 100
            df['is_dead'] = df['fdv_change_pct'] < -80
        
        return df
    
    def create_human_readable_correlations(self):
        """Create comprehensive human-readable correlation analysis"""
        print("🧠 Creating human-readable correlation analysis...")
        
        if self.df.empty:
            print("❌ No data available for analysis")
            return
        
        # Generate all correlation reports
        reports = {}
        
        # 1. Pattern Success Analysis
        reports['pattern_success'] = self._analyze_pattern_success()
        
        # 2. Risk-Performance Correlations
        reports['risk_performance'] = self._analyze_risk_performance()
        
        # 3. Momentum-Performance Correlations
        reports['momentum_performance'] = self._analyze_momentum_performance()
        
        # 4. Transaction Pattern Correlations
        reports['transaction_patterns'] = self._analyze_transaction_patterns()
        
        # 5. Buy/Sell Pressure Analysis
        reports['buy_sell_pressure'] = self._analyze_buy_sell_pressure()
        
        # 6. Market Cap vs FDV Correlations
        reports['market_cap_correlations'] = self._analyze_market_cap_correlations()
        
        # 7. Predictive Power Analysis
        reports['predictive_power'] = self._analyze_predictive_power()
        
        # 8. Actionable Insights
        reports['actionable_insights'] = self._generate_actionable_insights()
        
        # Save all reports
        self._save_reports(reports)
        
        # Display summary
        self._display_summary(reports)
        
        return reports
    
    def _analyze_pattern_success(self):
        """Analyze how well each pattern predicts actual performance"""
        analysis = {
            'title': '🎯 PATTERN SUCCESS ANALYSIS',
            'description': 'How well each pattern predicts actual performance outcomes',
            'insights': []
        }
        
        total_tokens = len(self.df)
        
        for pattern in self.df['pattern'].unique():
            pattern_data = self.df[self.df['pattern'] == pattern]
            pattern_count = len(pattern_data)
            pattern_percentage = (pattern_count / total_tokens) * 100
            
            # Performance metrics for this pattern
            avg_performance = pattern_data['fdv_change_pct'].mean()
            success_rate = (pattern_data['fdv_change_pct'] > 0).mean() * 100
            moon_shot_rate = (pattern_data['fdv_change_pct'] > 100).mean() * 100
            death_rate = (pattern_data['fdv_change_pct'] < -80).mean() * 100
            
            # Pattern accuracy (how often the pattern correctly predicted the outcome)
            if pattern == 'moon_shot':
                accuracy = moon_shot_rate
                prediction = "100%+ gains"
            elif pattern == 'strong_rise':
                accuracy = (pattern_data['fdv_change_pct'] >= 50).mean() * 100
                prediction = "50%+ gains"
            elif pattern == 'moderate_rise':
                accuracy = ((pattern_data['fdv_change_pct'] >= 20) & (pattern_data['fdv_change_pct'] < 50)).mean() * 100
                prediction = "20-50% gains"
            elif pattern == 'stable':
                accuracy = ((pattern_data['fdv_change_pct'] >= -20) & (pattern_data['fdv_change_pct'] < 20)).mean() * 100
                prediction = "Stable performance"
            elif pattern == 'moderate_drop':
                accuracy = ((pattern_data['fdv_change_pct'] >= -50) & (pattern_data['fdv_change_pct'] < -20)).mean() * 100
                prediction = "20-50% drops"
            elif pattern == 'significant_drop':
                accuracy = ((pattern_data['fdv_change_pct'] >= -80) & (pattern_data['fdv_change_pct'] < -50)).mean() * 100
                prediction = "50-80% drops"
            elif pattern == 'died':
                accuracy = death_rate
                prediction = "80%+ losses"
            else:
                accuracy = 0
                prediction = "Unknown"
            
            insight = {
                'pattern': pattern,
                'emoji': self.patterns.get(pattern, '❓'),
                'occurrence': f"{pattern_count} tokens ({pattern_percentage:.1f}%)",
                'prediction': prediction,
                'accuracy': f"{accuracy:.1f}%",
                'avg_performance': f"{avg_performance:+.1f}%",
                'success_rate': f"{success_rate:.1f}%",
                'moon_shot_rate': f"{moon_shot_rate:.1f}%",
                'death_rate': f"{death_rate:.1f}%",
                'reliability': self._get_reliability_score(accuracy)
            }
            
            analysis['insights'].append(insight)
        
        return analysis
    
    def _analyze_risk_performance(self):
        """Analyze risk vs performance correlations"""
        analysis = {
            'title': '⚠️ RISK-PERFORMANCE CORRELATIONS',
            'description': 'How risk levels correlate with actual performance outcomes',
            'insights': []
        }
        
        # Risk level analysis
        for risk_level in self.df['risk_level'].unique():
            risk_data = self.df[self.df['risk_level'] == risk_level]
            risk_count = len(risk_data)
            risk_percentage = (risk_count / len(self.df)) * 100
            
            # Performance metrics
            avg_performance = risk_data['fdv_change_pct'].mean()
            success_rate = (risk_data['fdv_change_pct'] > 0).mean() * 100
            moon_shot_rate = (risk_data['fdv_change_pct'] > 100).mean() * 100
            death_rate = (risk_data['fdv_change_pct'] < -80).mean() * 100
            
            # Risk-adjusted returns
            risk_adjusted_return = avg_performance / (risk_data['risk_score'].mean() + 1)
            
            insight = {
                'risk_level': risk_level,
                'occurrence': f"{risk_count} tokens ({risk_percentage:.1f}%)",
                'avg_performance': f"{avg_performance:+.1f}%",
                'success_rate': f"{success_rate:.1f}%",
                'moon_shot_rate': f"{moon_shot_rate:.1f}%",
                'death_rate': f"{death_rate:.1f}%",
                'risk_adjusted_return': f"{risk_adjusted_return:.1f}",
                'recommendation': self._get_risk_recommendation(risk_level, avg_performance, success_rate)
            }
            
            analysis['insights'].append(insight)
        
        # Overall risk correlation
        risk_corr = self.df['risk_score'].corr(self.df['fdv_change_pct'])
        analysis['overall_correlation'] = f"{risk_corr:.3f}"
        analysis['correlation_interpretation'] = self._interpret_correlation(risk_corr)
        
        return analysis
    
    def _analyze_momentum_performance(self):
        """Analyze momentum vs performance correlations"""
        analysis = {
            'title': '📈 MOMENTUM-PERFORMANCE CORRELATIONS',
            'description': 'How momentum indicators correlate with actual performance',
            'insights': []
        }
        
        # Momentum level analysis
        for momentum_level in self.df['momentum_level'].unique():
            momentum_data = self.df[self.df['momentum_level'] == momentum_level]
            momentum_count = len(momentum_data)
            momentum_percentage = (momentum_count / len(self.df)) * 100
            
            # Performance metrics
            avg_performance = momentum_data['fdv_change_pct'].mean()
            success_rate = (momentum_data['fdv_change_pct'] > 0).mean() * 100
            moon_shot_rate = (momentum_data['fdv_change_pct'] > 100).mean() * 100
            death_rate = (momentum_data['fdv_change_pct'] < -80).mean() * 100
            
            # Momentum efficiency
            momentum_efficiency = avg_performance / (momentum_data['momentum_score'].mean() + 1)
            
            insight = {
                'momentum_level': momentum_level,
                'occurrence': f"{momentum_count} tokens ({momentum_percentage:.1f}%)",
                'avg_performance': f"{avg_performance:+.1f}%",
                'success_rate': f"{success_rate:.1f}%",
                'moon_shot_rate': f"{moon_shot_rate:.1f}%",
                'death_rate': f"{death_rate:.1f}%",
                'momentum_efficiency': f"{momentum_efficiency:.1f}",
                'recommendation': self._get_momentum_recommendation(momentum_level, avg_performance, success_rate)
            }
            
            analysis['insights'].append(insight)
        
        # Overall momentum correlation
        momentum_corr = self.df['momentum_score'].corr(self.df['fdv_change_pct'])
        analysis['overall_correlation'] = f"{momentum_corr:.3f}"
        analysis['correlation_interpretation'] = self._interpret_correlation(momentum_corr)
        
        return analysis
    
    def _analyze_transaction_patterns(self):
        """Analyze transaction pattern correlations"""
        analysis = {
            'title': '💱 TRANSACTION PATTERN CORRELATIONS',
            'description': 'How transaction patterns correlate with performance outcomes',
            'insights': []
        }
        
        # Buy/sell ratio analysis
        high_buy_ratio = self.df[self.df['buy_sell_ratio'] > 1.2]
        low_buy_ratio = self.df[self.df['buy_sell_ratio'] < 0.8]
        balanced_ratio = self.df[(self.df['buy_sell_ratio'] >= 0.8) & (self.df['buy_sell_ratio'] <= 1.2)]
        
        patterns = [
            ('High Buy Pressure', high_buy_ratio, 'buy_sell_ratio > 1.2'),
            ('Low Buy Pressure', low_buy_ratio, 'buy_sell_ratio < 0.8'),
            ('Balanced Pressure', balanced_ratio, '0.8 ≤ buy_sell_ratio ≤ 1.2')
        ]
        
        for pattern_name, pattern_data, condition in patterns:
            if len(pattern_data) > 0:
                pattern_count = len(pattern_data)
                pattern_percentage = (pattern_count / len(self.df)) * 100
                
                avg_performance = pattern_data['fdv_change_pct'].mean()
                success_rate = (pattern_data['fdv_change_pct'] > 0).mean() * 100
                moon_shot_rate = (pattern_data['fdv_change_pct'] > 100).mean() * 100
                death_rate = (pattern_data['fdv_change_pct'] < -80).mean() * 100
                
                insight = {
                    'pattern': pattern_name,
                    'condition': condition,
                    'occurrence': f"{pattern_count} tokens ({pattern_percentage:.1f}%)",
                    'avg_performance': f"{avg_performance:+.1f}%",
                    'success_rate': f"{success_rate:.1f}%",
                    'moon_shot_rate': f"{moon_shot_rate:.1f}%",
                    'death_rate': f"{death_rate:.1f}%",
                    'recommendation': self._get_transaction_recommendation(pattern_name, avg_performance, success_rate)
                }
                
                analysis['insights'].append(insight)
        
        # Buy pressure correlation
        buy_pressure_corr = self.df['buy_pressure'].corr(self.df['fdv_change_pct'])
        analysis['buy_pressure_correlation'] = f"{buy_pressure_corr:.3f}"
        analysis['buy_pressure_interpretation'] = self._interpret_correlation(buy_pressure_corr)
        
        return analysis
    
    def _analyze_buy_sell_pressure(self):
        """Analyze buy/sell pressure patterns"""
        analysis = {
            'title': '🔄 BUY/SELL PRESSURE ANALYSIS',
            'description': 'Detailed analysis of buy vs sell pressure patterns',
            'insights': []
        }
        
        # Buy pressure categories
        strong_buy = self.df[self.df['buy_pressure'] > 10]
        moderate_buy = self.df[(self.df['buy_pressure'] > 0) & (self.df['buy_pressure'] <= 10)]
        neutral = self.df[self.df['buy_pressure'] == 0]
        moderate_sell = self.df[(self.df['buy_pressure'] < 0) & (self.df['buy_pressure'] >= -10)]
        strong_sell = self.df[self.df['buy_pressure'] < -10]
        
        pressure_patterns = [
            ('Strong Buy Pressure', strong_buy, 'buy_pressure > 10%'),
            ('Moderate Buy Pressure', moderate_buy, '0% < buy_pressure ≤ 10%'),
            ('Neutral Pressure', neutral, 'buy_pressure = 0%'),
            ('Moderate Sell Pressure', moderate_sell, '-10% ≤ buy_pressure < 0%'),
            ('Strong Sell Pressure', strong_sell, 'buy_pressure < -10%')
        ]
        
        for pattern_name, pattern_data, condition in pressure_patterns:
            if len(pattern_data) > 0:
                pattern_count = len(pattern_data)
                pattern_percentage = (pattern_count / len(self.df)) * 100
                
                avg_performance = pattern_data['fdv_change_pct'].mean()
                success_rate = (pattern_data['fdv_change_pct'] > 0).mean() * 100
                moon_shot_rate = (pattern_data['fdv_change_pct'] > 100).mean() * 100
                death_rate = (pattern_data['fdv_change_pct'] < -80).mean() * 100
                
                insight = {
                    'pressure_type': pattern_name,
                    'condition': condition,
                    'occurrence': f"{pattern_count} tokens ({pattern_percentage:.1f}%)",
                    'avg_performance': f"{avg_performance:+.1f}%",
                    'success_rate': f"{success_rate:.1f}%",
                    'moon_shot_rate': f"{moon_shot_rate:.1f}%",
                    'death_rate': f"{death_rate:.1f}%",
                    'effectiveness': self._get_pressure_effectiveness(pattern_name, avg_performance, success_rate)
                }
                
                analysis['insights'].append(insight)
        
        return analysis
    
    def _analyze_market_cap_correlations(self):
        """Analyze market cap vs FDV correlations"""
        analysis = {
            'title': '💰 MARKET CAP vs FDV CORRELATIONS',
            'description': 'How market cap changes correlate with FDV changes',
            'insights': []
        }
        
        # Correlation analysis
        market_cap_corr = self.df['market_cap_change_pct'].corr(self.df['fdv_change_pct'])
        analysis['correlation'] = f"{market_cap_corr:.3f}"
        analysis['interpretation'] = self._interpret_correlation(market_cap_corr)
        
        # Market cap change categories
        strong_growth = self.df[self.df['market_cap_change_pct'] > 50]
        moderate_growth = self.df[(self.df['market_cap_change_pct'] > 0) & (self.df['market_cap_change_pct'] <= 50)]
        stable = self.df[(self.df['market_cap_change_pct'] >= -20) & (self.df['market_cap_change_pct'] <= 0)]
        decline = self.df[self.df['market_cap_change_pct'] < -20]
        
        cap_patterns = [
            ('Strong Market Cap Growth', strong_growth, 'market_cap_change > 50%'),
            ('Moderate Market Cap Growth', moderate_growth, '0% < market_cap_change ≤ 50%'),
            ('Stable Market Cap', stable, '-20% ≤ market_cap_change ≤ 0%'),
            ('Market Cap Decline', decline, 'market_cap_change < -20%')
        ]
        
        for pattern_name, pattern_data, condition in cap_patterns:
            if len(pattern_data) > 0:
                pattern_count = len(pattern_data)
                pattern_percentage = (pattern_count / len(self.df)) * 100
                
                avg_fdv_change = pattern_data['fdv_change_pct'].mean()
                success_rate = (pattern_data['fdv_change_pct'] > 0).mean() * 100
                
                insight = {
                    'market_cap_pattern': pattern_name,
                    'condition': condition,
                    'occurrence': f"{pattern_count} tokens ({pattern_percentage:.1f}%)",
                    'avg_fdv_change': f"{avg_fdv_change:+.1f}%",
                    'success_rate': f"{success_rate:.1f}%",
                    'consistency': self._get_market_cap_consistency(pattern_name, avg_fdv_change, success_rate)
                }
                
                analysis['insights'].append(insight)
        
        return analysis
    
    def _analyze_predictive_power(self):
        """Analyze the predictive power of different indicators"""
        analysis = {
            'title': '🔮 PREDICTIVE POWER ANALYSIS',
            'description': 'How well different indicators predict actual outcomes',
            'insights': []
        }
        
        # Calculate predictive power for different metrics
        metrics = {
            'Risk Score': 'risk_score',
            'Momentum Score': 'momentum_score',
            'Buy/Sell Ratio': 'buy_sell_ratio',
            'Buy Pressure': 'buy_pressure',
            'Transaction Intensity': 'transaction_intensity'
        }
        
        for metric_name, metric_col in metrics.items():
            if metric_col in self.df.columns:
                # Correlation with performance
                correlation = self.df[metric_col].corr(self.df['fdv_change_pct'])
                
                # Predictive accuracy for positive performance
                if metric_col == 'risk_score':
                    # Lower risk should predict better performance
                    high_confidence = self.df[self.df[metric_col] <= 2]
                    low_confidence = self.df[self.df[metric_col] >= 4]
                elif metric_col == 'momentum_score':
                    # Higher momentum should predict better performance
                    high_confidence = self.df[self.df[metric_col] >= 2]
                    low_confidence = self.df[self.df[metric_col] <= 0]
                elif metric_col == 'buy_sell_ratio':
                    # Higher buy ratio should predict better performance
                    high_confidence = self.df[self.df[metric_col] > 1.2]
                    low_confidence = self.df[self.df[metric_col] < 0.8]
                elif metric_col == 'buy_pressure':
                    # Higher buy pressure should predict better performance
                    high_confidence = self.df[self.df[metric_col] > 5]
                    low_confidence = self.df[self.df[metric_col] < -5]
                else:
                    continue
                
                if len(high_confidence) > 0 and len(low_confidence) > 0:
                    high_success = (high_confidence['fdv_change_pct'] > 0).mean() * 100
                    low_success = (low_confidence['fdv_change_pct'] > 0).mean() * 100
                    
                    predictive_power = high_success - low_success
                    
                    insight = {
                        'metric': metric_name,
                        'correlation': f"{correlation:.3f}",
                        'correlation_strength': self._get_correlation_strength(abs(correlation)),
                        'high_confidence_success': f"{high_success:.1f}%",
                        'low_confidence_success': f"{low_success:.1f}%",
                        'predictive_power': f"{predictive_power:+.1f}%",
                        'reliability': self._get_predictive_reliability(predictive_power)
                    }
                    
                    analysis['insights'].append(insight)
        
        return analysis
    
    def _generate_actionable_insights(self):
        """Generate actionable insights based on all analysis"""
        insights = {
            'title': '💡 ACTIONABLE INSIGHTS & RECOMMENDATIONS',
            'description': 'Practical recommendations based on pattern analysis',
            'insights': []
        }
        
        # Overall market conditions
        success_rate = (self.df['fdv_change_pct'] > 0).mean() * 100
        avg_performance = self.df['fdv_change_pct'].mean()
        
        if success_rate > 60:
            market_condition = "BULLISH - High success rate suggests favorable conditions"
            strategy = "Consider aggressive strategies, focus on momentum plays"
        elif success_rate > 40:
            market_condition = "NEUTRAL - Moderate success rate, mixed conditions"
            strategy = "Balanced approach, focus on risk management"
        else:
            market_condition = "BEARISH - Low success rate suggests challenging conditions"
            strategy = "Conservative approach, wait for better opportunities"
        
        insights['market_conditions'] = {
            'condition': market_condition,
            'success_rate': f"{success_rate:.1f}%",
            'avg_performance': f"{avg_performance:+.1f}%",
            'strategy': strategy
        }
        
        # Pattern-specific recommendations
        pattern_recommendations = []
        for pattern in self.df['pattern'].unique():
            pattern_data = self.df[self.df['pattern'] == pattern]
            success_rate = (pattern_data['fdv_change_pct'] > 0).mean() * 100
            
            if success_rate > 70:
                recommendation = f"✅ {pattern.replace('_', ' ').title()} - High success rate, consider targeting"
            elif success_rate > 50:
                recommendation = f"⚠️ {pattern.replace('_', ' ').title()} - Moderate success, proceed with caution"
            else:
                recommendation = f"❌ {pattern.replace('_', ' ').title()} - Low success rate, avoid or short"
            
            pattern_recommendations.append({
                'pattern': pattern,
                'success_rate': f"{success_rate:.1f}%",
                'recommendation': recommendation
            })
        
        insights['pattern_recommendations'] = pattern_recommendations
        
        # Risk level recommendations
        risk_recommendations = []
        for risk_level in self.df['risk_level'].unique():
            risk_data = self.df[self.df['risk_level'] == risk_level]
            avg_performance = risk_data['fdv_change_pct'].mean()
            success_rate = (risk_data['fdv_change_pct'] > 0).mean() * 100
            
            if avg_performance > 0 and success_rate > 50:
                recommendation = f"✅ {risk_level.title()} Risk - Positive returns, good risk/reward"
            elif avg_performance < 0 and success_rate < 30:
                recommendation = f"❌ {risk_level.title()} Risk - Poor performance, avoid"
            else:
                recommendation = f"⚠️ {risk_level.title()} Risk - Mixed results, proceed carefully"
            
            risk_recommendations.append({
                'risk_level': risk_level,
                'avg_performance': f"{avg_performance:+.1f}%",
                'success_rate': f"{success_rate:.1f}%",
                'recommendation': recommendation
            })
        
        insights['risk_recommendations'] = risk_recommendations
        
        # Top performing combinations
        top_combinations = self._find_top_performing_combinations()
        insights['top_combinations'] = top_combinations
        
        return insights
    
    def _find_top_performing_combinations(self):
        """Find the best performing combinations of indicators"""
        combinations = []
        
        # Risk + Momentum combinations
        for risk_level in self.df['risk_level'].unique():
            for momentum_level in self.df['momentum_level'].unique():
                combo_data = self.df[(self.df['risk_level'] == risk_level) & 
                                   (self.df['momentum_level'] == momentum_level)]
                
                if len(combo_data) >= 2:  # Only consider combinations with enough data
                    avg_performance = combo_data['fdv_change_pct'].mean()
                    success_rate = (combo_data['fdv_change_pct'] > 0).mean() * 100
                    
                    if success_rate > 60 or avg_performance > 50:
                        combinations.append({
                            'combination': f"{risk_level.title()} Risk + {momentum_level.title()} Momentum",
                            'tokens': len(combo_data),
                            'avg_performance': f"{avg_performance:+.1f}%",
                            'success_rate': f"{success_rate:.1f}%",
                            'rating': self._get_combination_rating(success_rate, avg_performance)
                        })
        
        # Sort by success rate
        combinations.sort(key=lambda x: float(x['success_rate'].replace('%', '')), reverse=True)
        return combinations[:5]  # Top 5 combinations
    
    def _get_reliability_score(self, accuracy):
        """Get reliability score for pattern accuracy"""
        if accuracy >= 80:
            return "🟢 HIGH - Very reliable pattern"
        elif accuracy >= 60:
            return "🟡 MEDIUM - Moderately reliable pattern"
        elif accuracy >= 40:
            return "🟠 LOW - Low reliability pattern"
        else:
            return "🔴 POOR - Unreliable pattern"
    
    def _get_risk_recommendation(self, risk_level, avg_performance, success_rate):
        """Get recommendation based on risk level"""
        if risk_level == 'low' and avg_performance > 0:
            return "✅ Excellent choice for conservative strategies"
        elif risk_level == 'medium' and success_rate > 50:
            return "⚠️ Moderate risk with decent returns"
        elif risk_level == 'high' and avg_performance > 100:
            return "🚀 High risk but high potential returns"
        else:
            return "❌ Poor risk/reward ratio, avoid"
    
    def _get_momentum_recommendation(self, momentum_level, avg_performance, success_rate):
        """Get recommendation based on momentum level"""
        if momentum_level == 'high' and avg_performance > 0:
            return "🚀 Strong momentum with positive returns"
        elif momentum_level == 'medium' and success_rate > 50:
            return "📊 Moderate momentum, balanced approach"
        elif momentum_level == 'low' and avg_performance < 0:
            return "📉 Weak momentum, avoid or short"
        else:
            return "❓ Mixed signals, proceed with caution"
    
    def _get_transaction_recommendation(self, pattern_name, avg_performance, success_rate):
        """Get recommendation based on transaction pattern"""
        if 'High Buy Pressure' in pattern_name and avg_performance > 0:
            return "✅ Strong buying pressure correlates with gains"
        elif 'Low Buy Pressure' in pattern_name and avg_performance < 0:
            return "❌ Low buying pressure correlates with losses"
        elif 'Balanced' in pattern_name and abs(avg_performance) < 30:
            return "⚖️ Balanced pressure leads to stable performance"
        else:
            return "❓ Mixed correlation, not a strong predictor"
    
    def _get_pressure_effectiveness(self, pattern_name, avg_performance, success_rate):
        """Get effectiveness rating for pressure patterns"""
        if 'Buy' in pattern_name and avg_performance > 0:
            return "✅ Effective - Buy pressure predicts gains"
        elif 'Sell' in pattern_name and avg_performance < 0:
            return "✅ Effective - Sell pressure predicts losses"
        elif abs(avg_performance) < 20:
            return "⚖️ Neutral - Pressure doesn't strongly predict performance"
        else:
            return "❓ Mixed - Inconsistent correlation"
    
    def _get_market_cap_consistency(self, pattern_name, avg_fdv_change, success_rate):
        """Get consistency rating for market cap patterns"""
        if 'Growth' in pattern_name and avg_fdv_change > 0:
            return "✅ Consistent - Market cap growth aligns with FDV gains"
        elif 'Decline' in pattern_name and avg_fdv_change < 0:
            return "✅ Consistent - Market cap decline aligns with FDV losses"
        else:
            return "❓ Inconsistent - Market cap and FDV changes don't align"
    
    def _interpret_correlation(self, correlation):
        """Interpret correlation coefficient"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            return "STRONG correlation"
        elif abs_corr >= 0.5:
            return "MODERATE correlation"
        elif abs_corr >= 0.3:
            return "WEAK correlation"
        else:
            return "VERY WEAK or no correlation"
    
    def _get_correlation_strength(self, correlation):
        """Get correlation strength description"""
        if correlation >= 0.7:
            return "STRONG"
        elif correlation >= 0.5:
            return "MODERATE"
        elif correlation >= 0.3:
            return "WEAK"
        else:
            return "VERY WEAK"
    
    def _get_predictive_reliability(self, predictive_power):
        """Get reliability rating for predictive power"""
        if predictive_power >= 30:
            return "🟢 HIGH - Very reliable predictor"
        elif predictive_power >= 15:
            return "🟡 MEDIUM - Moderately reliable predictor"
        elif predictive_power >= 5:
            return "🟠 LOW - Weak predictor"
        else:
            return "🔴 POOR - Unreliable predictor"
    
    def _get_combination_rating(self, success_rate, avg_performance):
        """Get rating for indicator combinations"""
        if success_rate >= 80 and avg_performance >= 50:
            return "🏆 EXCELLENT - High success rate with strong returns"
        elif success_rate >= 70 and avg_performance >= 30:
            return "🥇 GOLD - Very good combination"
        elif success_rate >= 60 and avg_performance >= 20:
            return "🥈 SILVER - Good combination"
        elif success_rate >= 50 and avg_performance >= 10:
            return "🥉 BRONZE - Decent combination"
        else:
            return "❌ POOR - Avoid this combination"
    
    def _save_reports(self, reports):
        """Save all reports to files"""
        # Save detailed report
        detailed_report = self._format_detailed_report(reports)
        with open(self.output_dir / 'detailed_correlation_report.txt', 'w') as f:
            f.write(detailed_report)
        
        # Save summary report
        summary_report = self._format_summary_report(reports)
        with open(self.output_dir / 'summary_correlation_report.txt', 'w') as f:
            f.write(summary_report)
        
        # Save JSON version
        with open(self.output_dir / 'correlation_analysis.json', 'w') as f:
            json.dump(reports, f, indent=2, default=str)
        
        print(f"✅ Reports saved to: {self.output_dir}")
    
    def _format_detailed_report(self, reports):
        """Format detailed report"""
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("🧠 HUMAN-READABLE CORRELATION ANALYSIS REPORT")
        report_lines.append("=" * 100)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Tokens Analyzed: {len(self.df)}")
        report_lines.append("")
        
        for report_key, report_data in reports.items():
            if isinstance(report_data, dict) and 'title' in report_data:
                report_lines.append(report_data['title'])
                report_lines.append("-" * len(report_data['title']))
                report_lines.append(report_data['description'])
                report_lines.append("")
                
                if 'insights' in report_data:
                    for insight in report_data['insights']:
                        for key, value in insight.items():
                            if key != 'title' and key != 'description':
                                report_lines.append(f"{key.replace('_', ' ').title()}: {value}")
                        report_lines.append("")
                
                if 'overall_correlation' in report_data:
                    report_lines.append(f"Overall Correlation: {report_data['overall_correlation']}")
                    report_lines.append(f"Interpretation: {report_data['correlation_interpretation']}")
                    report_lines.append("")
        
        return "\n".join(report_lines)
    
    def _format_summary_report(self, reports):
        """Format summary report"""
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("📊 CORRELATION ANALYSIS SUMMARY")
        summary_lines.append("=" * 80)
        summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")
        
        # Key insights summary
        if 'actionable_insights' in reports:
            insights = reports['actionable_insights']
            if 'market_conditions' in insights:
                mc = insights['market_conditions']
                summary_lines.append("🎯 MARKET CONDITIONS:")
                summary_lines.append(f"   {mc['condition']}")
                summary_lines.append(f"   Success Rate: {mc['success_rate']}")
                summary_lines.append(f"   Strategy: {mc['strategy']}")
                summary_lines.append("")
        
        # Top patterns
        if 'pattern_success' in reports:
            summary_lines.append("🏆 TOP PERFORMING PATTERNS:")
            pattern_insights = reports['pattern_success']['insights']
            # Sort by success rate
            sorted_patterns = sorted(pattern_insights, 
                                   key=lambda x: float(x['success_rate'].replace('%', '')), 
                                   reverse=True)
            
            for i, pattern in enumerate(sorted_patterns[:3], 1):
                summary_lines.append(f"   {i}. {pattern['emoji']} {pattern['pattern'].replace('_', ' ').title()}")
                summary_lines.append(f"      Success Rate: {pattern['success_rate']}")
                summary_lines.append(f"      Accuracy: {pattern['accuracy']}")
                summary_lines.append("")
        
        # Top combinations
        if 'actionable_insights' in reports and 'top_combinations' in reports['actionable_insights']:
            summary_lines.append("🔥 TOP INDICATOR COMBINATIONS:")
            combinations = reports['actionable_insights']['top_combinations']
            for i, combo in enumerate(combinations[:3], 1):
                summary_lines.append(f"   {i}. {combo['combination']}")
                summary_lines.append(f"      Success Rate: {combo['success_rate']}")
                summary_lines.append(f"      Rating: {combo['rating']}")
                summary_lines.append("")
        
        return "\n".join(summary_lines)
    
    def _display_summary(self, reports):
        """Display summary of all reports"""
        print("\n" + "="*80)
        print("🧠 HUMAN-READABLE CORRELATION ANALYSIS COMPLETE!")
        print("="*80)
        
        # Display key insights
        if 'actionable_insights' in reports:
            insights = reports['actionable_insights']
            if 'market_conditions' in insights:
                mc = insights['market_conditions']
                print(f"🎯 MARKET CONDITIONS: {mc['condition']}")
                print(f"   Success Rate: {mc['success_rate']}")
                print(f"   Strategy: {mc['strategy']}")
                print("")
        
        # Display top patterns
        if 'pattern_success' in reports:
            print("🏆 TOP PERFORMING PATTERNS:")
            pattern_insights = reports['pattern_success']['insights']
            sorted_patterns = sorted(pattern_insights, 
                                   key=lambda x: float(x['success_rate'].replace('%', '')), 
                                   reverse=True)
            
            for i, pattern in enumerate(sorted_patterns[:3], 1):
                print(f"   {i}. {pattern['emoji']} {pattern['pattern'].replace('_', ' ').title()}")
                print(f"      Success Rate: {pattern['success_rate']} | Accuracy: {pattern['accuracy']}")
            print("")
        
        print("📁 Check the 'output/human_readable_insights' folder for detailed reports")
        print("📊 Summary report: summary_correlation_report.txt")
        print("📝 Detailed report: detailed_correlation_report.txt")
        print("🔧 JSON data: correlation_analysis.json")

def main():
    """Main function to create human-readable correlations"""
    print("🧠 Starting Human-Readable Correlation Analysis...")
    
    # Create analyzer
    analyzer = HumanReadableCorrelationAnalyzer()
    
    # Generate all correlations
    reports = analyzer.create_human_readable_correlations()
    
    print("\n" + "="*80)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
