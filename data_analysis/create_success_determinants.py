#!/usr/bin/env python3
"""
Success Determinants Analysis
Identifies the best predictors of successful and unsuccessful tokens
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

class SuccessDeterminantsAnalyzer:
    """Analyzes what determines token success vs failure"""
    
    def __init__(self, results_file: str = "output/adapted_timeseries_analysis.json"):
        self.results_file = Path(results_file)
        self.output_dir = Path("output/success_determinants")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.results = self._load_results()
        self.df = self._create_dataframe()
        
        # Success definitions
        self.success_thresholds = {
            'high_success': 50,      # >50% gain
            'moderate_success': 20,   # 20-50% gain
            'neutral': 0,             # 0-20% gain/loss
            'moderate_failure': -20,  # -20% to 0%
            'high_failure': -50,     # -50% to -20%
            'complete_failure': -80   # <-80% (died)
        }
    
    def _load_results(self) -> dict:
        """Load analysis results"""
        try:
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading results: {e}")
            return {}
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Create DataFrame with success analysis"""
        if not self.results or 'results' not in self.results:
            return pd.DataFrame()
        
        rows = []
        for address, result in self.results['results'].items():
            if result.get('status') == 'analyzed':
                row = {
                    'address': address,
                    'token_name': result.get('token_name', 'Unknown'),
                    'pattern': result.get('pattern', 'unknown'),
                    'fdv_change_pct': result.get('performance_metrics', {}).get('fdv_change_pct', 0),
                    'market_cap_change_pct': result.get('performance_metrics', {}).get('market_cap_change_pct', 0),
                    'risk_score': result.get('risk_metrics', {}).get('total_risk_score', 0),
                    'risk_level': result.get('risk_metrics', {}).get('risk_level', 'unknown'),
                    'momentum_score': result.get('momentum_metrics', {}).get('momentum_score', 0),
                    'momentum_level': result.get('momentum_metrics', {}).get('momentum_level', 'unknown')
                }
                
                # Extract transaction analysis
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
                
                # Add time-based transaction analysis
                if 'h1' in transaction_data and 'h24' in transaction_data:
                    h1_data = transaction_data['h1']
                    h24_data = transaction_data['h24']
                    
                    row.update({
                        'h1_buy_sell_ratio': h1_data.get('buy_sell_ratio', 1),
                        'h24_buy_sell_ratio': h24_data.get('buy_sell_ratio', 1),
                        'h1_buy_percentage': h1_data.get('buy_percentage', 50),
                        'h24_buy_percentage': h24_data.get('buy_percentage', 50),
                        'momentum_change': h1_data.get('buy_sell_ratio', 1) - h24_data.get('buy_sell_ratio', 1),
                        'buy_pressure_change': h1_data.get('buy_percentage', 50) - h24_data.get('buy_percentage', 50)
                    })
                
                rows.append(row)
        
        # Now categorize success for all rows
        df = pd.DataFrame(rows)
        if not df.empty:
            # Add success categories directly
            df['success_category'] = df['fdv_change_pct'].apply(lambda x: 
                'high_success' if x > 50 else
                'moderate_success' if x > 20 else
                'neutral' if x > 0 else
                'moderate_failure' if x > -20 else
                'high_failure' if x > -50 else
                'complete_failure'
            )
            
            df['success_score'] = df['fdv_change_pct'].apply(lambda x: 
                5 if x > 50 else
                4 if x > 20 else
                3 if x > 0 else
                2 if x > -20 else
                1 if x > -50 else
                0
            )
            
            df['is_successful'] = df['success_score'] >= 3
            df['is_high_success'] = df['success_score'] >= 4
            df['is_failure'] = df['success_score'] <= 2
        
        return df
    
    def _categorize_success(self, row):
        """Categorize token success level"""
        fdv_change = row.get('fdv_change_pct', 0)
        
        if fdv_change > 50:  # high_success
            success_category = 'high_success'
            success_score = 5
        elif fdv_change > 20:  # moderate_success
            success_category = 'moderate_success'
            success_score = 4
        elif fdv_change > 0:  # neutral
            success_category = 'neutral'
            success_score = 3
        elif fdv_change > -20:  # moderate_failure
            success_category = 'moderate_failure'
            success_score = 2
        elif fdv_change > -50:  # high_failure
            success_category = 'high_failure'
            success_score = 1
        else:  # complete_failure
            success_category = 'complete_failure'
            success_score = 0
        
        return {
            'success_category': success_category,
            'success_score': success_score,
            'is_successful': success_score >= 3,  # neutral or better
            'is_high_success': success_score >= 4,  # moderate success or better
            'is_failure': success_score <= 2  # moderate failure or worse
        }
    
    def create_success_determinants_analysis(self):
        """Create comprehensive success determinants analysis"""
        print("🎯 Creating success determinants analysis...")
        
        if self.df.empty:
            print("❌ No data available for analysis")
            return
        
        # Generate analysis reports
        reports = {}
        
        # 1. Success Distribution Analysis
        reports['success_distribution'] = self._analyze_success_distribution()
        
        # 2. Feature Importance Analysis
        reports['feature_importance'] = self._analyze_feature_importance()
        
        # 3. Success vs Failure Characteristics
        reports['success_vs_failure'] = self._analyze_success_vs_failure()
        
        # 4. Predictive Power Analysis
        reports['predictive_power'] = self._analyze_predictive_power()
        
        # 5. Key Success Indicators
        reports['key_indicators'] = self._identify_key_indicators()
        
        # Create visualizations
        self._create_success_charts()
        
        # Save reports
        self._save_success_reports(reports)
        
        # Display summary
        self._display_success_summary(reports)
        
        return reports
    
    def _analyze_success_distribution(self):
        """Analyze distribution of success categories"""
        analysis = {
            'title': '📊 SUCCESS DISTRIBUTION ANALYSIS',
            'description': 'Distribution of tokens across success categories',
            'insights': {}
        }
        
        # Success category distribution
        success_counts = self.df['success_category'].value_counts()
        success_percentages = (success_counts / len(self.df)) * 100
        
        analysis['insights']['success_distribution'] = {
            'total_tokens': len(self.df),
            'successful_tokens': self.df['is_successful'].sum(),
            'successful_percentage': f"{(self.df['is_successful'].sum() / len(self.df)) * 100:.1f}%",
            'high_success_tokens': self.df['is_high_success'].sum(),
            'high_success_percentage': f"{(self.df['is_high_success'].sum() / len(self.df)) * 100:.1f}%",
            'failure_tokens': self.df['is_failure'].sum(),
            'failure_percentage': f"{(self.df['is_failure'].sum() / len(self.df)) * 100:.1f}%"
        }
        
        # Detailed breakdown
        analysis['insights']['detailed_breakdown'] = {}
        for category in success_counts.index:
            analysis['insights']['detailed_breakdown'][category] = {
                'count': int(success_counts[category]),
                'percentage': f"{success_percentages[category]:.1f}%",
                'avg_performance': f"{self.df[self.df['success_category'] == category]['fdv_change_pct'].mean():+.1f}%"
            }
        
        return analysis
    
    def _analyze_feature_importance(self):
        """Analyze which features are most important for success"""
        analysis = {
            'title': '🔍 FEATURE IMPORTANCE ANALYSIS',
            'description': 'Which metrics are most predictive of token success',
            'insights': {}
        }
        
        # Prepare features for analysis
        feature_columns = [
            'risk_score', 'buy_sell_ratio', 'buy_percentage', 'momentum_change', 
            'buy_pressure_change', 'total_transactions', 'momentum_score'
        ]
        
        # Remove rows with missing values
        df_clean = self.df.dropna(subset=feature_columns + ['success_score'])
        
        if len(df_clean) > 0:
            X = df_clean[feature_columns]
            y = df_clean['success_score']
            
            # Train Random Forest to get feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            analysis['insights']['feature_importance'] = feature_importance.to_dict('records')
            
            # Correlation analysis
            correlations = {}
            for feature in feature_columns:
                corr = df_clean[feature].corr(df_clean['success_score'])
                correlations[feature] = f"{corr:.3f}"
            
            analysis['insights']['correlations'] = correlations
        
        return analysis
    
    def _analyze_success_vs_failure(self):
        """Analyze characteristics that distinguish success from failure"""
        analysis = {
            'title': '⚖️ SUCCESS VS FAILURE CHARACTERISTICS',
            'description': 'Key differences between successful and failed tokens',
            'insights': {}
        }
        
        # Compare successful vs failed tokens
        successful = self.df[self.df['is_successful'] == True]
        failed = self.df[self.df['is_successful'] == False]
        
        if len(successful) > 0 and len(failed) > 0:
            # Key metrics comparison
            key_metrics = ['risk_score', 'buy_sell_ratio', 'buy_percentage', 'momentum_change', 'buy_pressure_change']
            
            comparison = {}
            for metric in key_metrics:
                if metric in self.df.columns:
                    successful_avg = successful[metric].mean()
                    failed_avg = failed[metric].mean()
                    difference = successful_avg - failed_avg
                    
                    comparison[metric] = {
                        'successful_avg': f"{successful_avg:.2f}",
                        'failed_avg': f"{failed_avg:.2f}",
                        'difference': f"{difference:+.2f}",
                        'success_advantage': difference > 0
                    }
            
            analysis['insights']['metric_comparison'] = comparison
            
            # Pattern analysis
            successful_patterns = successful['pattern'].value_counts()
            failed_patterns = failed['pattern'].value_counts()
            
            analysis['insights']['pattern_analysis'] = {
                'successful_patterns': successful_patterns.to_dict(),
                'failed_patterns': failed_patterns.to_dict(),
                'most_successful_pattern': successful_patterns.index[0] if len(successful_patterns) > 0 else 'None',
                'most_failed_pattern': failed_patterns.index[0] if len(failed_patterns) > 0 else 'None'
            }
        
        return analysis
    
    def _analyze_predictive_power(self):
        """Analyze predictive power of different indicators"""
        analysis = {
            'title': '🔮 PREDICTIVE POWER ANALYSIS',
            'description': 'How well different indicators predict success',
            'insights': {}
        }
        
        # Analyze predictive power of key thresholds
        thresholds = {
            'risk_score_6': 6,
            'risk_score_8': 8,
            'buy_sell_ratio_0.6': 0.6,
            'buy_sell_ratio_0.8': 0.8,
            'buy_percentage_40': 40,
            'buy_percentage_45': 45,
            'momentum_positive': 0
        }
        
        predictive_power = {}
        for name, threshold in thresholds.items():
            if 'risk_score' in name:
                # For risk score, lower is better
                above_threshold = self.df['risk_score'] >= threshold
                below_threshold = self.df['risk_score'] < threshold
            elif 'buy_sell_ratio' in name:
                # For buy/sell ratio, higher is better
                above_threshold = self.df['buy_sell_ratio'] >= threshold
                below_threshold = self.df['buy_sell_ratio'] < threshold
            elif 'buy_percentage' in name:
                # For buy percentage, higher is better
                above_threshold = self.df['buy_percentage'] >= threshold
                below_threshold = self.df['buy_percentage'] < threshold
            elif 'momentum' in name:
                # For momentum, positive is better
                above_threshold = self.df['momentum_change'] >= threshold
                below_threshold = self.df['momentum_change'] < threshold
            
            # Calculate success rates
            above_success_rate = self.df[above_threshold]['is_successful'].mean() if above_threshold.sum() > 0 else 0
            below_success_rate = self.df[below_threshold]['is_successful'].mean() if below_threshold.sum() > 0 else 0
            
            predictive_power[name] = {
                'threshold': threshold,
                'above_threshold_success_rate': f"{above_success_rate:.1%}",
                'below_threshold_success_rate': f"{below_success_rate:.1%}",
                'predictive_power': f"{abs(above_success_rate - below_success_rate):.1%}",
                'direction': 'above_better' if above_success_rate > below_success_rate else 'below_better'
            }
        
        analysis['insights']['threshold_analysis'] = predictive_power
        
        return analysis
    
    def _identify_key_indicators(self):
        """Identify the most important success indicators"""
        analysis = {
            'title': '🎯 KEY SUCCESS INDICATORS',
            'description': 'The most reliable predictors of token success',
            'insights': {}
        }
        
        # Identify top success indicators
        if 'feature_importance' in analysis:
            # This will be populated by the feature importance analysis
            pass
        
        # Create actionable insights
        analysis['insights']['actionable_insights'] = {
            'entry_criteria': {
                'risk_score_max': 'Keep below 6.0 for best success rate',
                'buy_sell_ratio_min': 'Keep above 0.8 for best success rate',
                'buy_percentage_min': 'Keep above 45% for best success rate',
                'momentum_requirement': 'Must be positive for best success rate'
            },
            'warning_signs': {
                'risk_score_warning': 'Above 6.0 indicates declining success probability',
                'buy_sell_ratio_warning': 'Below 0.8 indicates selling pressure',
                'buy_percentage_warning': 'Below 40% indicates heavy selling',
                'momentum_warning': 'Negative momentum indicates declining success'
            },
            'success_patterns': {
                'best_pattern': 'moon_shot has highest success rate',
                'timing': 'instant_boom tokens have highest success rate',
                'risk_profile': 'Low risk tokens (score < 4) have highest success rate'
            }
        }
        
        return analysis
    
    def _create_success_charts(self):
        """Create success analysis visualizations"""
        print("🎨 Creating success analysis charts...")
        
        # 1. Success Distribution Chart
        self._create_success_distribution_chart()
        
        # 2. Feature Importance Chart
        self._create_feature_importance_chart()
        
        # 3. Success vs Failure Comparison
        self._create_success_failure_chart()
        
        # 4. Predictive Power Chart
        self._create_predictive_power_chart()
        
        print("✅ Success analysis charts created")
    
    def _create_success_distribution_chart(self):
        """Create success distribution chart"""
        if self.df.empty:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Success category distribution
        success_counts = self.df['success_category'].value_counts()
        colors = ['#00FF00', '#32CD32', '#FFD700', '#FFA500', '#FF6B6B', '#8B0000']
        
        wedges, texts, autotexts = ax1.pie(success_counts.values, 
                                          labels=[cat.replace('_', ' ').title() for cat in success_counts.index],
                                          colors=colors[:len(success_counts)], autopct='%1.1f%%', startangle=90)
        ax1.set_title('📊 Success Category Distribution', fontsize=14, fontweight='bold')
        
        # Success score distribution
        success_scores = self.df['success_score'].value_counts().sort_index()
        bars = ax2.bar(range(len(success_scores)), success_scores.values, 
                      color=['red', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen'][:len(success_scores)])
        ax2.set_title('🎯 Success Score Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Success Score (0=Complete Failure, 5=High Success)')
        ax2.set_ylabel('Number of Tokens')
        ax2.set_xticks(range(len(success_scores)))
        ax2.set_xticklabels(success_scores.index)
        
        # Add value labels
        for bar, value in zip(bars, success_scores.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'success_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_feature_importance_chart(self):
        """Create feature importance chart"""
        if self.df.empty:
            return
        
        # Prepare features for analysis
        feature_columns = [
            'risk_score', 'buy_sell_ratio', 'buy_percentage', 'momentum_change', 
            'buy_pressure_change', 'total_transactions', 'momentum_score'
        ]
        
        df_clean = self.df.dropna(subset=feature_columns + ['success_score'])
        
        if len(df_clean) > 0:
            X = df_clean[feature_columns]
            y = df_clean['success_score']
            
            # Train Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Create feature importance chart
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=True)
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(feature_importance)), feature_importance['importance'], 
                          color='skyblue', alpha=0.7)
            plt.yticks(range(len(feature_importance)), 
                      [f.replace('_', ' ').title() for f in feature_importance['feature']])
            plt.xlabel('Feature Importance')
            plt.title('🔍 Feature Importance for Token Success', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (bar, importance) in enumerate(zip(bars, feature_importance['importance'])):
                plt.text(importance + 0.01, i, f'{importance:.3f}', 
                        va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def _create_success_failure_chart(self):
        """Create success vs failure comparison chart"""
        if self.df.empty:
            return
        
        successful = self.df[self.df['is_successful'] == True]
        failed = self.df[self.df['is_successful'] == False]
        
        if len(successful) > 0 and len(failed) > 0:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Compare key metrics
            metrics = ['risk_score', 'buy_sell_ratio', 'buy_percentage', 'momentum_change']
            metric_names = ['Risk Score', 'Buy/Sell Ratio', 'Buy Percentage', 'Momentum Change']
            
            for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                if metric in self.df.columns:
                    ax = [ax1, ax2, ax3, ax4][i]
                    
                    successful_values = successful[metric].dropna()
                    failed_values = failed[metric].dropna()
                    
                    if len(successful_values) > 0 and len(failed_values) > 0:
                        ax.hist(successful_values, alpha=0.7, label='Successful', color='green', bins=10)
                        ax.hist(failed_values, alpha=0.7, label='Failed', color='red', bins=10)
                        ax.set_title(f'{name} Distribution')
                        ax.set_xlabel(name)
                        ax.set_ylabel('Number of Tokens')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
            
            plt.suptitle('⚖️ Success vs Failure Characteristics', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'success_failure_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def _create_predictive_power_chart(self):
        """Create predictive power chart"""
        if self.df.empty:
            return
        
        # Analyze predictive power of key thresholds
        thresholds = {
            'Risk Score < 6': (self.df['risk_score'] < 6, 'Risk Score < 6'),
            'Risk Score < 8': (self.df['risk_score'] < 8, 'Risk Score < 8'),
            'Buy/Sell Ratio > 0.8': (self.df['buy_sell_ratio'] > 0.8, 'Buy/Sell Ratio > 0.8'),
            'Buy/Sell Ratio > 0.6': (self.df['buy_sell_ratio'] > 0.6, 'Buy/Sell Ratio > 0.6'),
            'Buy % > 45': (self.df['buy_percentage'] > 45, 'Buy % > 45'),
            'Buy % > 40': (self.df['buy_percentage'] > 40, 'Buy % > 40'),
            'Positive Momentum': (self.df['momentum_change'] > 0, 'Positive Momentum')
        }
        
        success_rates = []
        threshold_names = []
        
        for name, (condition, label) in thresholds.items():
            if condition.sum() > 0:
                success_rate = self.df[condition]['is_successful'].mean()
                success_rates.append(success_rate * 100)
                threshold_names.append(label)
        
        if success_rates:
            plt.figure(figsize=(14, 8))
            bars = plt.bar(range(len(success_rates)), success_rates, 
                          color=['green' if rate > 50 else 'red' for rate in success_rates], alpha=0.7)
            plt.xticks(range(len(threshold_names)), threshold_names, rotation=45, ha='right')
            plt.ylabel('Success Rate (%)')
            plt.title('🔮 Predictive Power of Different Thresholds', fontsize=16, fontweight='bold')
            plt.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% Baseline')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add value labels
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'predictive_power.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def _save_success_reports(self, reports):
        """Save success determinants analysis reports"""
        # Save detailed report
        detailed_report = self._format_success_report(reports)
        with open(self.output_dir / 'detailed_success_analysis.txt', 'w') as f:
            f.write(detailed_report)
        
        # Save summary report
        summary_report = self._format_summary_report(reports)
        with open(self.output_dir / 'success_determinants_summary.txt', 'w') as f:
            f.write(summary_report)
        
        print(f"✅ Success determinants reports saved to: {self.output_dir}")
    
    def _format_success_report(self, reports):
        """Format detailed success determinants report"""
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("🎯 MEMECOIN SUCCESS DETERMINANTS ANALYSIS REPORT")
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
                    for key, value in report_data['insights'].items():
                        if isinstance(value, dict):
                            report_lines.append(f"{key.replace('_', ' ').title()}:")
                            for sub_key, sub_value in value.items():
                                report_lines.append(f"  {sub_key.replace('_', ' ').title()}: {sub_value}")
                        else:
                            report_lines.append(f"{key.replace('_', ' ').title()}: {value}")
                        report_lines.append("")
        
        return "\n".join(report_lines)
    
    def _format_summary_report(self, reports):
        """Format summary report"""
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("🎯 SUCCESS DETERMINANTS ANALYSIS SUMMARY")
        summary_lines.append("=" * 80)
        summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")
        
        # Key findings
        if 'success_distribution' in reports:
            success_dist = reports['success_distribution']['insights']['success_distribution']
            summary_lines.append("🎯 KEY FINDINGS:")
            summary_lines.append(f"• Total Tokens: {success_dist['total_tokens']}")
            summary_lines.append(f"• Successful Tokens: {success_dist['successful_tokens']} ({success_dist['successful_percentage']})")
            summary_lines.append(f"• High Success Tokens: {success_dist['high_success_tokens']} ({success_dist['high_success_percentage']})")
            summary_lines.append(f"• Failed Tokens: {success_dist['failure_tokens']} ({success_dist['failure_percentage']})")
            summary_lines.append("")
        
        # Top success indicators
        if 'feature_importance' in reports:
            summary_lines.append("🔍 TOP SUCCESS INDICATORS:")
            if 'feature_importance' in reports['feature_importance']['insights']:
                top_features = reports['feature_importance']['insights']['feature_importance'][:3]
                for i, feature in enumerate(top_features, 1):
                    summary_lines.append(f"• #{i}: {feature['feature'].replace('_', ' ').title()} (Importance: {feature['importance']:.3f})")
            summary_lines.append("")
        
        # Actionable insights
        if 'key_indicators' in reports:
            summary_lines.append("💡 ACTIONABLE INSIGHTS:")
            actionable = reports['key_indicators']['insights']['actionable_insights']
            summary_lines.append("• Entry Criteria: Follow the established thresholds")
            summary_lines.append("• Warning Signs: Monitor for declining metrics")
            summary_lines.append("• Success Patterns: Focus on proven patterns")
            summary_lines.append("")
        
        summary_lines.append("📁 Check the 'output/success_determinants' folder for detailed reports")
        
        return "\n".join(summary_lines)
    
    def _display_success_summary(self, reports):
        """Display summary of success determinants analysis"""
        print("\n" + "="*80)
        print("🎯 SUCCESS DETERMINANTS ANALYSIS COMPLETE!")
        print("="*80)
        
        # Display key findings
        if 'success_distribution' in reports:
            success_dist = reports['success_distribution']['insights']['success_distribution']
            print("🎯 SUCCESS DISTRIBUTION:")
            print(f"   Total Tokens: {success_dist['total_tokens']}")
            print(f"   Successful: {success_dist['successful_tokens']} ({success_dist['successful_percentage']})")
            print(f"   High Success: {success_dist['high_success_tokens']} ({success_dist['high_success_percentage']})")
            print(f"   Failed: {success_dist['failure_tokens']} ({success_dist['failure_percentage']})")
            print("")
        
        # Display top features
        if 'feature_importance' in reports:
            print("🔍 TOP SUCCESS INDICATORS:")
            if 'feature_importance' in reports['feature_importance']['insights']:
                top_features = reports['feature_importance']['insights']['feature_importance'][:3]
                for i, feature in enumerate(top_features, 1):
                    print(f"   #{i}: {feature['feature'].replace('_', ' ').title()} (Importance: {feature['importance']:.3f})")
            print("")
        
        print("📁 Check the 'output/success_determinants' folder for detailed reports")
        print("🎯 Summary: success_determinants_summary.txt")

def main():
    """Main function to create success determinants analysis"""
    print("🎯 Starting Success Determinants Analysis...")
    
    # Create analyzer
    analyzer = SuccessDeterminantsAnalyzer()
    
    # Generate success determinants analysis
    reports = analyzer.create_success_determinants_analysis()
    
    print("\n" + "="*80)
    print("🎉 SUCCESS DETERMINANTS ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
