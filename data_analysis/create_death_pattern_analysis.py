#!/usr/bin/env python3
"""
Death Pattern Analysis for Memecoin Trading Safeguards
Identifies point of no return conditions and warning signs
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

class DeathPatternAnalyzer:
    """Analyzes death patterns and identifies point of no return conditions"""
    
    def __init__(self, results_file: str = "output/adapted_timeseries_analysis.json"):
        self.results_file = Path(results_file)
        self.output_dir = Path("output/death_pattern_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.results = self._load_results()
        self.df = self._create_dataframe()
        
        # Death pattern definitions
        self.death_patterns = {
            'died': '⚰️ Died (>80% loss)',
            'significant_drop': '💸 Significant Drop (-50% to -80%)',
            'moderate_drop': '📉 Moderate Drop (-20% to -50%)'
        }
        
        # Warning level colors
        self.warning_colors = {
            'low': '#96CEB4',      # Green - Low risk
            'medium': '#FFEAA7',   # Yellow - Medium risk
            'high': '#FF6B6B',     # Red - High risk
            'critical': '#8B0000'  # Dark red - Critical risk
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
        """Create DataFrame with death pattern analysis"""
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
                
                # Categorize death risk
                row.update(self._categorize_death_risk(row))
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _categorize_death_risk(self, row):
        """Categorize death risk based on various indicators"""
        risk_indicators = {}
        
        # 1. Performance-based risk
        fdv_change = row.get('fdv_change_pct', 0)
        if fdv_change < -80:
            risk_indicators['death_risk_level'] = 'critical'
            risk_indicators['death_risk_score'] = 10
        elif fdv_change < -50:
            risk_indicators['death_risk_level'] = 'high'
            risk_indicators['death_risk_score'] = 8
        elif fdv_change < -20:
            risk_indicators['death_risk_level'] = 'medium'
            risk_indicators['death_risk_score'] = 6
        elif fdv_change < 0:
            risk_indicators['death_risk_level'] = 'low'
            risk_indicators['death_risk_score'] = 4
        else:
            risk_indicators['death_risk_level'] = 'low'
            risk_indicators['death_risk_score'] = 2
        
        # 2. Transaction-based risk
        buy_sell_ratio = row.get('buy_sell_ratio', 1)
        if buy_sell_ratio < 0.5:
            risk_indicators['transaction_risk_level'] = 'critical'
            risk_indicators['transaction_risk_score'] = 10
        elif buy_sell_ratio < 0.8:
            risk_indicators['transaction_risk_level'] = 'high'
            risk_indicators['transaction_risk_score'] = 8
        elif buy_sell_ratio < 1.0:
            risk_indicators['transaction_risk_level'] = 'medium'
            risk_indicators['transaction_risk_score'] = 6
        else:
            risk_indicators['transaction_risk_level'] = 'low'
            risk_indicators['transaction_risk_score'] = 4
        
        # 3. Momentum-based risk
        momentum_change = row.get('momentum_change', 0)
        if momentum_change < -0.5:
            risk_indicators['momentum_risk_level'] = 'critical'
            risk_indicators['momentum_risk_score'] = 10
        elif momentum_change < -0.2:
            risk_indicators['momentum_risk_level'] = 'high'
            risk_indicators['momentum_risk_score'] = 8
        elif momentum_change < 0:
            risk_indicators['momentum_risk_level'] = 'medium'
            risk_indicators['momentum_risk_score'] = 6
        else:
            risk_indicators['momentum_risk_level'] = 'low'
            risk_indicators['momentum_risk_score'] = 4
        
        # 4. Buy pressure risk
        buy_pressure_change = row.get('buy_pressure_change', 0)
        if buy_pressure_change < -20:
            risk_indicators['pressure_risk_level'] = 'critical'
            risk_indicators['pressure_risk_score'] = 10
        elif buy_pressure_change < -10:
            risk_indicators['pressure_risk_level'] = 'high'
            risk_indicators['pressure_risk_score'] = 8
        elif buy_pressure_change < 0:
            risk_indicators['pressure_risk_level'] = 'medium'
            risk_indicators['pressure_risk_score'] = 6
        else:
            risk_indicators['pressure_risk_level'] = 'low'
            risk_indicators['pressure_risk_score'] = 4
        
        # 5. Combined risk score
        total_risk_score = (risk_indicators.get('death_risk_score', 0) + 
                           risk_indicators.get('transaction_risk_score', 0) + 
                           risk_indicators.get('momentum_risk_score', 0) + 
                           risk_indicators.get('pressure_risk_score', 0)) / 4
        
        if total_risk_score >= 8:
            risk_indicators['overall_risk_level'] = 'critical'
        elif total_risk_score >= 6:
            risk_indicators['overall_risk_level'] = 'high'
        elif total_risk_score >= 4:
            risk_indicators['overall_risk_level'] = 'medium'
        else:
            risk_indicators['overall_risk_level'] = 'low'
        
        risk_indicators['total_risk_score'] = total_risk_score
        
        return risk_indicators
    
    def create_death_pattern_analysis(self):
        """Create comprehensive death pattern analysis"""
        print("⚰️ Creating death pattern analysis...")
        
        if self.df.empty:
            print("❌ No data available for analysis")
            return
        
        # Generate death pattern reports
        reports = {}
        
        # 1. Death Pattern Analysis
        reports['death_patterns'] = self._analyze_death_patterns()
        
        # 2. Point of No Return Analysis
        reports['point_of_no_return'] = self._analyze_point_of_no_return()
        
        # 3. Warning Signs Analysis
        reports['warning_signs'] = self._analyze_warning_signs()
        
        # 4. Safeguard Rails
        reports['safeguard_rails'] = self._create_safeguard_rails()
        
        # 5. Risk Assessment Framework
        reports['risk_framework'] = self._create_risk_framework()
        
        # Create visualizations
        self._create_death_pattern_charts()
        
        # Save reports
        self._save_death_reports(reports)
        
        # Display summary
        self._display_death_summary(reports)
        
        return reports
    
    def _analyze_death_patterns(self):
        """Analyze death patterns in detail"""
        analysis = {
            'title': '⚰️ DEATH PATTERN ANALYSIS',
            'description': 'Detailed analysis of how tokens die and enter different death categories',
            'insights': []
        }
        
        # Analyze each death pattern
        death_patterns = ['died', 'significant_drop', 'moderate_drop']
        
        for pattern in death_patterns:
            pattern_data = self.df[self.df['pattern'] == pattern]
            pattern_count = len(pattern_data)
            
            if pattern_count == 0:
                continue
            
            # Performance metrics
            avg_performance = pattern_data['fdv_change_pct'].mean()
            min_performance = pattern_data['fdv_change_pct'].min()
            max_performance = pattern_data['fdv_change_pct'].max()
            
            # Risk metrics
            avg_risk_score = pattern_data['total_risk_score'].mean()
            risk_distribution = pattern_data['overall_risk_level'].value_counts()
            
            # Transaction metrics
            avg_buy_sell_ratio = pattern_data['buy_sell_ratio'].mean()
            avg_buy_percentage = pattern_data['buy_percentage'].mean()
            
            # Momentum metrics
            avg_momentum_change = pattern_data['momentum_change'].mean()
            avg_pressure_change = pattern_data['buy_pressure_change'].mean()
            
            insight = {
                'pattern': pattern,
                'emoji': self.death_patterns.get(pattern, '❓'),
                'total_tokens': pattern_count,
                'avg_performance': f"{avg_performance:+.1f}%",
                'performance_range': f"{min_performance:+.1f}% to {max_performance:+.1f}%",
                'avg_risk_score': f"{avg_risk_score:.1f}",
                'risk_distribution': risk_distribution.to_dict(),
                'avg_buy_sell_ratio': f"{avg_buy_sell_ratio:.2f}",
                'avg_buy_percentage': f"{avg_buy_percentage:.1f}%",
                'avg_momentum_change': f"{avg_momentum_change:+.2f}",
                'avg_pressure_change': f"{avg_pressure_change:+.1f}%",
                'death_characteristics': self._get_death_characteristics(pattern, avg_performance, avg_buy_sell_ratio)
            }
            
            analysis['insights'].append(insight)
        
        return analysis
    
    def _analyze_point_of_no_return(self):
        """Analyze the point of no return conditions"""
        analysis = {
            'title': '🚫 POINT OF NO RETURN ANALYSIS',
            'description': 'Conditions that indicate a token has passed the point of no return',
            'insights': []
        }
        
        # Find tokens that are in critical death risk
        critical_death = self.df[self.df['overall_risk_level'] == 'critical']
        
        if len(critical_death) > 0:
            # Analyze common characteristics
            common_indicators = {
                'avg_fdv_change': critical_death['fdv_change_pct'].mean(),
                'avg_buy_sell_ratio': critical_death['buy_sell_ratio'].mean(),
                'avg_momentum_change': critical_death['momentum_change'].mean(),
                'avg_pressure_change': critical_death['buy_pressure_change'].mean(),
                'avg_risk_score': critical_death['total_risk_score'].mean()
            }
            
            analysis['critical_death_indicators'] = common_indicators
            
            # Identify point of no return thresholds
            analysis['point_of_no_return_thresholds'] = {
                'fdv_drop_threshold': f"{critical_death['fdv_change_pct'].quantile(0.25):.1f}%",
                'buy_sell_ratio_threshold': f"{critical_death['buy_sell_ratio'].quantile(0.25):.2f}",
                'momentum_change_threshold': f"{critical_death['momentum_change'].quantile(0.25):.2f}",
                'pressure_change_threshold': f"{critical_death['buy_pressure_change'].quantile(0.25):.1f}%"
            }
        
        # Analyze warning progression
        warning_progression = self._analyze_warning_progression()
        analysis['warning_progression'] = warning_progression
        
        return analysis
    
    def _analyze_warning_signs(self):
        """Analyze early warning signs of death"""
        analysis = {
            'title': '⚠️ WARNING SIGNS ANALYSIS',
            'description': 'Early indicators that suggest a token is heading towards death',
            'insights': []
        }
        
        # Analyze tokens that eventually died
        death_tokens = self.df[self.df['pattern'].isin(['died', 'significant_drop'])]
        
        if len(death_tokens) > 0:
            # Find common warning signs
            warning_indicators = {
                'buy_sell_ratio_warning': f"{death_tokens['buy_sell_ratio'].quantile(0.75):.2f}",
                'momentum_change_warning': f"{death_tokens['momentum_change'].quantile(0.75):.2f}",
                'pressure_change_warning': f"{death_tokens['buy_pressure_change'].quantile(0.75):.1f}%",
                'risk_score_warning': f"{death_tokens['total_risk_score'].quantile(0.75):.1f}"
            }
            
            analysis['warning_thresholds'] = warning_indicators
            
            # Categorize warning levels
            analysis['warning_levels'] = self._categorize_warning_levels(death_tokens)
        
        return analysis
    
    def _analyze_warning_progression(self):
        """Analyze how warnings progress towards death"""
        progression = {
            'early_warning': 'Buy/sell ratio drops below 0.8',
            'moderate_warning': 'Momentum becomes negative',
            'severe_warning': 'Buy pressure drops below 40%',
            'critical_warning': 'Risk score exceeds 7.0',
            'point_of_no_return': 'FDV drops below -50%'
        }
        
        return progression
    
    def _categorize_warning_levels(self, death_tokens):
        """Categorize warning levels based on death token analysis"""
        warning_levels = {}
        
        # Early warning (still recoverable)
        early_warning = death_tokens[
            (death_tokens['buy_sell_ratio'] >= 0.8) & 
            (death_tokens['fdv_change_pct'] > -30)
        ]
        warning_levels['early_warning'] = {
            'count': len(early_warning),
            'description': 'Still recoverable with intervention',
            'action': 'Monitor closely, consider exit'
        }
        
        # Moderate warning (difficult to recover)
        moderate_warning = death_tokens[
            (death_tokens['buy_sell_ratio'] < 0.8) & 
            (death_tokens['fdv_change_pct'] > -50)
        ]
        warning_levels['moderate_warning'] = {
            'count': len(moderate_warning),
            'description': 'Difficult to recover',
            'action': 'Exit position, avoid further investment'
        }
        
        # Severe warning (point of no return)
        severe_warning = death_tokens[
            (death_tokens['fdv_change_pct'] <= -50) | 
            (death_tokens['total_risk_score'] >= 8)
        ]
        warning_levels['severe_warning'] = {
            'count': len(severe_warning),
            'description': 'Point of no return reached',
            'action': 'Exit immediately, avoid completely'
        }
        
        return warning_levels
    
    def _create_safeguard_rails(self):
        """Create trading safeguard rails"""
        safeguards = {
            'title': '🛡️ TRADING SAFEGUARD RAILS',
            'description': 'Risk management rules to prevent entering death spiral tokens',
            'rails': {}
        }
        
        # Entry safeguards
        safeguards['rails']['entry_safeguards'] = {
            'buy_sell_ratio_minimum': 'Must be >= 0.8',
            'buy_percentage_minimum': 'Must be >= 45%',
            'momentum_change_minimum': 'Must be >= -0.2',
            'risk_score_maximum': 'Must be <= 6.0',
            'fdv_change_minimum': 'Must be >= -20%'
        }
        
        # Exit safeguards
        safeguards['rails']['exit_safeguards'] = {
            'buy_sell_ratio_exit': 'Exit if drops below 0.6',
            'buy_percentage_exit': 'Exit if drops below 40%',
            'momentum_change_exit': 'Exit if becomes negative',
            'risk_score_exit': 'Exit if exceeds 7.0',
            'fdv_change_exit': 'Exit if drops below -30%'
        }
        
        # Monitoring safeguards
        safeguards['rails']['monitoring_safeguards'] = {
            'check_frequency': 'Every 15-30 minutes',
            'key_metrics': 'Buy/sell ratio, momentum, buy pressure',
            'alert_triggers': 'Any metric hits warning threshold',
            'action_timeframe': 'Within 5 minutes of warning'
        }
        
        return safeguards
    
    def _create_risk_framework(self):
        """Create comprehensive risk assessment framework"""
        framework = {
            'title': '📊 RISK ASSESSMENT FRAMEWORK',
            'description': 'Systematic approach to assessing token death risk',
            'framework': {}
        }
        
        # Risk scoring system
        framework['framework']['risk_scoring'] = {
            'low_risk': 'Score 2-4: Safe to enter, monitor normally',
            'medium_risk': 'Score 4-6: Exercise caution, monitor closely',
            'high_risk': 'Score 6-8: High risk, consider exit',
            'critical_risk': 'Score 8-10: Critical risk, exit immediately'
        }
        
        # Risk factors and weights
        framework['framework']['risk_factors'] = {
            'performance_risk': 'Weight: 25% - Based on FDV change',
            'transaction_risk': 'Weight: 25% - Based on buy/sell ratio',
            'momentum_risk': 'Weight: 25% - Based on momentum change',
            'pressure_risk': 'Weight: 25% - Based on buy pressure change'
        }
        
        # Risk mitigation strategies
        framework['framework']['risk_mitigation'] = {
            'position_sizing': 'Reduce position size as risk increases',
            'stop_losses': 'Set tighter stop losses for high-risk tokens',
            'diversification': 'Limit exposure to any single risk category',
            'monitoring': 'Increase monitoring frequency for high-risk positions'
        }
        
        return framework
    
    def _get_death_characteristics(self, pattern, avg_performance, avg_buy_sell_ratio):
        """Get characteristics of death patterns"""
        if pattern == 'died':
            if avg_buy_sell_ratio < 0.5:
                return "Complete rug pull - Extreme selling pressure"
            else:
                return "Gradual death - Sustained negative momentum"
        elif pattern == 'significant_drop':
            if avg_buy_sell_ratio < 0.7:
                return "Heavy selling - Difficult to recover"
            else:
                return "Moderate selling - Some recovery possible"
        else:  # moderate_drop
            return "Light selling - Recovery likely with intervention"
    
    def _create_death_pattern_charts(self):
        """Create death pattern visualizations"""
        print("🎨 Creating death pattern charts...")
        
        # 1. Death Pattern Distribution
        self._create_death_distribution_chart()
        
        # 2. Risk Score Analysis
        self._create_risk_score_chart()
        
        # 3. Warning Signs Heatmap
        self._create_warning_signs_heatmap()
        
        # 4. Point of No Return Timeline
        self._create_point_of_no_return_chart()
        
        print("✅ Death pattern charts created")
    
    def _create_death_distribution_chart(self):
        """Create death pattern distribution chart"""
        if self.df.empty:
            return
        
        # Count death patterns
        death_counts = self.df[self.df['pattern'].isin(['died', 'significant_drop', 'moderate_drop'])]['pattern'].value_counts()
        
        plt.figure(figsize=(12, 8))
        colors = ['#8B0000', '#FF6B6B', '#FFEAA7']
        wedges, texts, autotexts = plt.pie(death_counts.values, 
                                          labels=[self.death_patterns.get(p, p) for p in death_counts.index],
                                          colors=colors, autopct='%1.1f%%', startangle=90)
        
        plt.title('⚰️ Death Pattern Distribution', fontsize=16, fontweight='bold', pad=20)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'death_pattern_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_risk_score_chart(self):
        """Create risk score analysis chart"""
        if self.df.empty:
            return
        
        # Risk score distribution
        risk_scores = self.df['total_risk_score']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram of risk scores
        ax1.hist(risk_scores, bins=20, alpha=0.7, color='red', edgecolor='black')
        ax1.set_title('📊 Risk Score Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Risk Score')
        ax1.set_ylabel('Number of Tokens')
        ax1.axvline(x=6, color='orange', linestyle='--', alpha=0.8, label='High Risk Threshold')
        ax1.axvline(x=8, color='red', linestyle='--', alpha=0.8, label='Critical Risk Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Risk level distribution
        risk_levels = self.df['overall_risk_level'].value_counts()
        colors = ['#96CEB4', '#FFEAA7', '#FF6B6B', '#8B0000']
        
        bars = ax2.bar(range(len(risk_levels)), risk_levels.values, 
                      color=colors[:len(risk_levels)])
        ax2.set_title('⚠️ Risk Level Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Risk Level')
        ax2.set_ylabel('Number of Tokens')
        ax2.set_xticks(range(len(risk_levels)))
        ax2.set_xticklabels([level.replace('_', ' ').title() for level in risk_levels.index])
        
        # Add value labels
        for bar, value in zip(bars, risk_levels.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_score_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_warning_signs_heatmap(self):
        """Create warning signs heatmap"""
        if self.df.empty:
            return
        
        # Create correlation matrix of warning indicators
        warning_columns = ['fdv_change_pct', 'buy_sell_ratio', 'momentum_change', 'buy_pressure_change', 'total_risk_score']
        warning_data = self.df[warning_columns].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(warning_data, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0,
                   cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('🔥 Warning Signs Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'warning_signs_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_point_of_no_return_chart(self):
        """Create point of no return analysis chart"""
        if self.df.empty:
            return
        
        # Scatter plot of risk vs performance
        plt.figure(figsize=(12, 8))
        
        # Color by risk level
        for risk_level in self.df['overall_risk_level'].unique():
            risk_data = self.df[self.df['overall_risk_level'] == risk_level]
            color = self.warning_colors.get(risk_level, '#CCCCCC')
            plt.scatter(risk_data['total_risk_score'], risk_data['fdv_change_pct'], 
                       c=color, label=risk_level.replace('_', ' ').title(),
                       alpha=0.7, s=60)
        
        # Add threshold lines
        plt.axhline(y=-50, color='red', linestyle='--', alpha=0.8, label='Point of No Return (-50%)')
        plt.axhline(y=-80, color='darkred', linestyle='--', alpha=0.8, label='Death Threshold (-80%)')
        plt.axvline(x=6, color='orange', linestyle='--', alpha=0.8, label='High Risk Threshold')
        plt.axvline(x=8, color='red', linestyle='--', alpha=0.8, label='Critical Risk Threshold')
        
        plt.title('🚫 Point of No Return Analysis', fontsize=16, fontweight='bold')
        plt.xlabel('Risk Score')
        plt.ylabel('FDV Change (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'point_of_no_return_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _save_death_reports(self, reports):
        """Save death pattern analysis reports"""
        # Save detailed report
        detailed_report = self._format_death_report(reports)
        with open(self.output_dir / 'detailed_death_analysis.txt', 'w') as f:
            f.write(detailed_report)
        
        # Save safeguard rails
        safeguard_report = self._format_safeguard_report(reports)
        with open(self.output_dir / 'safeguard_rails.txt', 'w') as f:
            f.write(safeguard_report)
        
        print(f"✅ Death pattern reports saved to: {self.output_dir}")
    
    def _format_death_report(self, reports):
        """Format detailed death pattern report"""
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("⚰️ MEMECOIN DEATH PATTERN ANALYSIS REPORT")
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
        
        return "\n".join(report_lines)
    
    def _format_safeguard_report(self, reports):
        """Format safeguard rails report"""
        safeguard_lines = []
        safeguard_lines.append("=" * 80)
        safeguard_lines.append("🛡️ TRADING SAFEGUARD RAILS")
        safeguard_lines.append("=" * 80)
        safeguard_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        safeguard_lines.append("")
        
        if 'safeguard_rails' in reports:
            safeguards = reports['safeguard_rails']
            if 'rails' in safeguards:
                for category, rules in safeguards['rails'].items():
                    safeguard_lines.append(f"📋 {category.replace('_', ' ').title()}:")
                    safeguard_lines.append("-" * len(category.replace('_', ' ').title()))
                    for rule, value in rules.items():
                        safeguard_lines.append(f"• {rule.replace('_', ' ').title()}: {value}")
                    safeguard_lines.append("")
        
        return "\n".join(safeguard_lines)
    
    def _display_death_summary(self, reports):
        """Display summary of death pattern analysis"""
        print("\n" + "="*80)
        print("⚰️ DEATH PATTERN ANALYSIS COMPLETE!")
        print("="*80)
        
        # Display key insights
        if 'death_patterns' in reports:
            print("⚰️ DEATH PATTERN INSIGHTS:")
            death_insights = reports['death_patterns']['insights']
            for insight in death_insights:
                print(f"   {insight['emoji']} {insight['pattern'].replace('_', ' ').title()}")
                print(f"      Tokens: {insight['total_tokens']}")
                print(f"      Avg Performance: {insight['avg_performance']}")
                print(f"      Risk Level: {insight['risk_distribution']}")
                print("")
        
        # Display safeguard rails
        if 'safeguard_rails' in reports:
            print("🛡️ SAFEGUARD RAILS CREATED:")
            safeguards = reports['safeguard_rails']
            if 'rails' in safeguards:
                for category in safeguards['rails'].keys():
                    print(f"   ✅ {category.replace('_', ' ').title()}")
            print("")
        
        print("📁 Check the 'output/death_pattern_analysis' folder for detailed reports")
        print("🛡️ Safeguard rails: safeguard_rails.txt")

def main():
    """Main function to create death pattern analysis"""
    print("⚰️ Starting Death Pattern Analysis...")
    
    # Create analyzer
    analyzer = DeathPatternAnalyzer()
    
    # Generate death pattern analysis
    reports = analyzer.create_death_pattern_analysis()
    
    print("\n" + "="*80)
    print("🎉 DEATH PATTERN ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
