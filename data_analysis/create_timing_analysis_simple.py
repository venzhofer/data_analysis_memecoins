#!/usr/bin/env python3
"""
Simple Timing Pattern Analysis for Memecoin Lifecycle
Uses available data to analyze timing patterns
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

class SimpleTimingAnalyzer:
    """Analyzes timing patterns using available data"""
    
    def __init__(self, results_file: str = "output/adapted_timeseries_analysis.json"):
        self.results_file = Path(results_file)
        self.output_dir = Path("output/timing_analysis_simple")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.results = self._load_results()
        self.df = self._create_dataframe()
        
        # Pattern definitions
        self.patterns = {
            'moon_shot': '🚀 Moon Shot',
            'strong_rise': '📈 Strong Rise',
            'moderate_rise': '📊 Moderate Rise',
            'stable': '⚖️ Stable',
            'moderate_drop': '📉 Moderate Drop',
            'significant_drop': '💸 Significant Drop',
            'died': '⚰️ Died'
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
        """Create DataFrame with timing inferences"""
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
                    'risk_score': result.get('risk_metrics', {}).get('total_risk_score', 0),
                    'risk_level': result.get('risk_metrics', {}).get('risk_level', 'unknown'),
                    'momentum_score': result.get('momentum_metrics', {}).get('momentum_score', 0),
                    'momentum_level': result.get('momentum_metrics', {}).get('momentum_level', 'unknown')
                }
                
                # Extract transaction timing data
                transaction_data = result.get('transaction_analysis', {})
                if 'overall' in transaction_data:
                    overall = transaction_data['overall']
                    row.update({
                        'total_transactions': overall.get('total_transactions', 0),
                        'buy_sell_ratio': overall.get('buy_sell_ratio', 0),
                        'buy_percentage': overall.get('buy_percentage', 0)
                    })
                
                # Infer timing based on transaction patterns
                row.update(self._infer_timing_patterns(transaction_data))
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _infer_timing_patterns(self, transaction_data):
        """Infer timing patterns from transaction data"""
        timing_info = {}
        
        if 'h1' in transaction_data and 'h24' in transaction_data:
            h1_data = transaction_data['h1']
            h24_data = transaction_data['h24']
            
            # Calculate transaction velocity
            h1_volume = h1_data.get('total_transactions', 0)
            h24_volume = h24_data.get('total_transactions', 0)
            
            # Infer timing based on transaction patterns
            if h1_volume > h24_volume * 0.3:  # High recent activity
                timing_info['timing_category'] = 'instant_boom'
                timing_info['timing_description'] = 'Instant boom - High recent activity'
                timing_info['entry_timing'] = 'Immediate entry required'
            elif h1_volume > h24_volume * 0.1:  # Moderate recent activity
                timing_info['timing_category'] = 'early_boom'
                timing_info['timing_description'] = 'Early boom - Good entry window'
                timing_info['entry_timing'] = 'Entry within 1-6 hours'
            else:  # Low recent activity
                timing_info['timing_category'] = 'gradual_development'
                timing_info['timing_description'] = 'Gradual development - Patient entry'
                timing_info['entry_timing'] = 'Entry within 6-24 hours'
            
            # Calculate momentum timing
            h1_buy_ratio = h1_data.get('buy_sell_ratio', 1)
            h24_buy_ratio = h24_data.get('buy_sell_ratio', 1)
            
            if h1_buy_ratio > h24_buy_ratio * 1.2:
                timing_info['momentum_timing'] = 'accelerating'
                timing_info['momentum_description'] = 'Momentum accelerating - Enter now'
            elif h1_buy_ratio < h24_buy_ratio * 0.8:
                timing_info['momentum_timing'] = 'decelerating'
                timing_info['momentum_description'] = 'Momentum decelerating - Exit soon'
            else:
                timing_info['momentum_timing'] = 'stable'
                timing_info['momentum_description'] = 'Momentum stable - Monitor closely'
        
        return timing_info
    
    def create_timing_analysis(self):
        """Create timing pattern analysis"""
        print("⏰ Creating timing pattern analysis...")
        
        if self.df.empty:
            print("❌ No data available for analysis")
            return
        
        # Generate timing reports
        reports = {}
        
        # 1. Pattern Timing Analysis
        reports['pattern_timing'] = self._analyze_pattern_timing()
        
        # 2. Transaction Timing Analysis
        reports['transaction_timing'] = self._analyze_transaction_timing()
        
        # 3. Momentum Timing Analysis
        reports['momentum_timing'] = self._analyze_momentum_timing()
        
        # 4. Optimal Entry/Exit Timing
        reports['optimal_timing'] = self._analyze_optimal_timing()
        
        # Create visualizations
        self._create_timing_charts()
        
        # Save reports
        self._save_reports(reports)
        
        # Display summary
        self._display_summary(reports)
        
        return reports
    
    def _analyze_pattern_timing(self):
        """Analyze timing patterns for each pattern type"""
        analysis = {
            'title': '⏰ PATTERN TIMING ANALYSIS',
            'description': 'When different patterns typically occur',
            'insights': []
        }
        
        for pattern in self.df['pattern'].unique():
            pattern_data = self.df[self.df['pattern'] == pattern]
            pattern_count = len(pattern_data)
            
            if pattern_count == 0:
                continue
            
            # Analyze timing categories for this pattern
            timing_dist = pattern_data['timing_category'].value_counts()
            
            # Calculate average performance
            avg_performance = pattern_data['fdv_change_pct'].mean()
            success_rate = (pattern_data['fdv_change_pct'] > 0).mean() * 100
            
            insight = {
                'pattern': pattern,
                'emoji': self.patterns.get(pattern, '❓'),
                'total_tokens': pattern_count,
                'avg_performance': f"{avg_performance:+.1f}%",
                'success_rate': f"{success_rate:.1f}%",
                'timing_distribution': timing_dist.to_dict(),
                'recommendation': self._get_pattern_timing_recommendation(pattern, timing_dist)
            }
            
            analysis['insights'].append(insight)
        
        return analysis
    
    def _analyze_transaction_timing(self):
        """Analyze transaction timing patterns"""
        analysis = {
            'title': '💱 TRANSACTION TIMING ANALYSIS',
            'description': 'How transaction patterns relate to timing',
            'insights': []
        }
        
        # Analyze each timing category
        for category in self.df['timing_category'].unique():
            category_data = self.df[self.df['timing_category'] == category]
            category_count = len(category_data)
            
            if category_count == 0:
                continue
            
            # Performance metrics
            avg_performance = category_data['fdv_change_pct'].mean()
            success_rate = (category_data['fdv_change_pct'] > 0).mean() * 100
            moon_shot_rate = (category_data['fdv_change_pct'] > 100).mean() * 100
            
            # Pattern distribution
            pattern_dist = category_data['pattern'].value_counts()
            
            insight = {
                'timing_category': category,
                'description': category_data['timing_description'].iloc[0],
                'total_tokens': category_count,
                'avg_performance': f"{avg_performance:+.1f}%",
                'success_rate': f"{success_rate:.1f}%",
                'moon_shot_rate': f"{moon_shot_rate:.1f}%",
                'pattern_distribution': pattern_dist.to_dict(),
                'entry_timing': category_data['entry_timing'].iloc[0]
            }
            
            analysis['insights'].append(insight)
        
        return analysis
    
    def _analyze_momentum_timing(self):
        """Analyze momentum timing patterns"""
        analysis = {
            'title': '📈 MOMENTUM TIMING ANALYSIS',
            'description': 'How momentum changes over time',
            'insights': []
        }
        
        # Analyze each momentum timing category
        for momentum_timing in self.df['momentum_timing'].unique():
            momentum_data = self.df[self.df['momentum_timing'] == momentum_timing]
            momentum_count = len(momentum_data)
            
            if momentum_count == 0:
                continue
            
            # Performance metrics
            avg_performance = momentum_data['fdv_change_pct'].mean()
            success_rate = (momentum_data['fdv_change_pct'] > 0).mean() * 100
            
            # Pattern distribution
            pattern_dist = momentum_data['pattern'].value_counts()
            
            insight = {
                'momentum_timing': momentum_timing,
                'description': momentum_data['momentum_description'].iloc[0],
                'total_tokens': momentum_count,
                'avg_performance': f"{avg_performance:+.1f}%",
                'success_rate': f"{success_rate:.1f}%",
                'pattern_distribution': pattern_dist.to_dict()
            }
            
            analysis['insights'].append(insight)
        
        return analysis
    
    def _analyze_optimal_timing(self):
        """Analyze optimal entry and exit timing"""
        analysis = {
            'title': '🎯 OPTIMAL TIMING RECOMMENDATIONS',
            'description': 'Best times to enter and exit based on analysis',
            'insights': []
        }
        
        # Best entry timing
        positive_patterns = self.df[self.df['fdv_change_pct'] > 0]
        if len(positive_patterns) > 0:
            # Find best timing category for positive patterns
            timing_success = positive_patterns.groupby('timing_category')['fdv_change_pct'].agg(['mean', 'count'])
            if not timing_success.empty:
                best_timing = timing_success.loc[timing_success['mean'].idxmax()]
                
                analysis['best_entry_timing'] = {
                    'timing_category': timing_success['mean'].idxmax(),
                    'avg_performance': f"{best_timing['mean']:+.1f}%",
                    'token_count': int(best_timing['count']),
                    'recommendation': f"Best entry timing: {timing_success['mean'].idxmax()}"
                }
        
        # Momentum-based timing
        accelerating_momentum = self.df[self.df['momentum_timing'] == 'accelerating']
        if len(accelerating_momentum) > 0:
            analysis['momentum_timing'] = {
                'accelerating_count': len(accelerating_momentum),
                'avg_performance': f"{accelerating_momentum['fdv_change_pct'].mean():+.1f}%",
                'success_rate': f"{(accelerating_momentum['fdv_change_pct'] > 0).mean() * 100:.1f}%",
                'recommendation': 'Enter when momentum is accelerating'
            }
        
        return analysis
    
    def _get_pattern_timing_recommendation(self, pattern, timing_dist):
        """Get timing recommendation for a pattern"""
        if pattern in ['moon_shot', 'strong_rise']:
            if 'instant_boom' in timing_dist:
                return "🚀 Instant entry required - These patterns happen fast"
            elif 'early_boom' in timing_dist:
                return "📈 Early entry recommended - Good entry window"
            else:
                return "⏰ Patient entry - Monitor for opportunities"
        elif pattern in ['died', 'significant_drop']:
            if 'instant_boom' in timing_dist:
                return "⚰️ Instant death - Avoid completely"
            else:
                return "📉 Exit quickly if detected"
        else:
            return "⚖️ Standard timing - Monitor closely"
    
    def _create_timing_charts(self):
        """Create timing pattern visualizations"""
        print("🎨 Creating timing pattern charts...")
        
        # 1. Pattern Timing Distribution
        self._create_pattern_timing_chart()
        
        # 2. Transaction Timing Performance
        self._create_transaction_timing_chart()
        
        # 3. Momentum Timing Analysis
        self._create_momentum_timing_chart()
        
        print("✅ Timing charts created")
    
    def _create_pattern_timing_chart(self):
        """Create pattern timing distribution chart"""
        if self.df.empty:
            return
        
        # Create cross-tabulation
        timing_pattern_cross = pd.crosstab(self.df['timing_category'], self.df['pattern'])
        
        plt.figure(figsize=(14, 8))
        timing_pattern_cross.plot(kind='bar', stacked=True, colormap='Set3')
        plt.title('⏰ Pattern Distribution by Timing Category', fontsize=16, fontweight='bold')
        plt.xlabel('Timing Category')
        plt.ylabel('Number of Tokens')
        plt.legend(title='Pattern', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pattern_timing_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_transaction_timing_chart(self):
        """Create transaction timing performance chart"""
        if self.df.empty:
            return
        
        # Group by timing category and calculate performance
        timing_performance = self.df.groupby('timing_category')['fdv_change_pct'].agg(['mean', 'count'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Performance by timing
        bars = ax1.bar(range(len(timing_performance)), timing_performance['mean'],
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('📊 Performance by Timing Category', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Timing Category')
        ax1.set_ylabel('Average FDV Change (%)')
        ax1.set_xticks(range(len(timing_performance)))
        ax1.set_xticklabels([cat.replace('_', ' ').title() for cat in timing_performance.index], rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars, timing_performance['mean']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -15),
                    f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold')
        
        ax1.grid(True, alpha=0.3)
        
        # Token count by timing
        ax2.pie(timing_performance['count'], labels=[cat.replace('_', ' ').title() for cat in timing_performance.index],
                autopct='%1.1f%%', startangle=90, colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_title('📈 Token Distribution by Timing', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'transaction_timing_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_momentum_timing_chart(self):
        """Create momentum timing analysis chart"""
        if self.df.empty:
            return
        
        # Group by momentum timing and calculate performance
        momentum_performance = self.df.groupby('momentum_timing')['fdv_change_pct'].agg(['mean', 'count'])
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(momentum_performance)), momentum_performance['mean'],
                      color=['#4ECDC4', '#FF6B6B', '#96CEB4'])
        plt.title('📈 Performance by Momentum Timing', fontsize=16, fontweight='bold')
        plt.xlabel('Momentum Timing')
        plt.ylabel('Average FDV Change (%)')
        plt.xticks(range(len(momentum_performance)), 
                  [timing.replace('_', ' ').title() for timing in momentum_performance.index])
        
        # Add value labels
        for bar, value in zip(bars, momentum_performance['mean']):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -15),
                    f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'momentum_timing_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _save_reports(self, reports):
        """Save timing analysis reports"""
        # Save detailed report
        detailed_report = self._format_timing_report(reports)
        with open(self.output_dir / 'detailed_timing_report.txt', 'w') as f:
            f.write(detailed_report)
        
        # Save summary report
        summary_report = self._format_timing_summary(reports)
        with open(self.output_dir / 'summary_timing_report.txt', 'w') as f:
            f.write(summary_report)
        
        print(f"✅ Timing reports saved to: {self.output_dir}")
    
    def _format_timing_report(self, reports):
        """Format detailed timing report"""
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("⏰ MEMECOIN TIMING PATTERN ANALYSIS REPORT")
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
    
    def _format_timing_summary(self, reports):
        """Format timing summary report"""
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("⏰ TIMING PATTERN ANALYSIS SUMMARY")
        summary_lines.append("=" * 80)
        summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")
        
        # Key timing insights
        if 'optimal_timing' in reports:
            timing = reports['optimal_timing']
            if 'best_entry_timing' in timing:
                summary_lines.append("🎯 BEST ENTRY TIMING:")
                summary_lines.append(f"   Category: {timing['best_entry_timing']['timing_category']}")
                summary_lines.append(f"   Performance: {timing['best_entry_timing']['avg_performance']}")
                summary_lines.append(f"   Tokens: {timing['best_entry_timing']['token_count']}")
                summary_lines.append("")
        
        # Pattern timing insights
        if 'pattern_timing' in reports:
            summary_lines.append("⏰ PATTERN TIMING INSIGHTS:")
            pattern_insights = reports['pattern_timing']['insights']
            for insight in pattern_insights[:3]:  # Top 3
                summary_lines.append(f"   {insight['emoji']} {insight['pattern'].replace('_', ' ').title()}")
                summary_lines.append(f"      Success Rate: {insight['success_rate']}")
                summary_lines.append(f"      Timing: {insight['timing_distribution']}")
                summary_lines.append("")
        
        return "\n".join(summary_lines)
    
    def _display_summary(self, reports):
        """Display summary of timing analysis"""
        print("\n" + "="*80)
        print("⏰ TIMING PATTERN ANALYSIS COMPLETE!")
        print("="*80)
        
        # Display key insights
        if 'optimal_timing' in reports:
            timing = reports['optimal_timing']
            if 'best_entry_timing' in timing:
                print(f"🎯 BEST ENTRY TIMING: {timing['best_entry_timing']['timing_category']}")
                print(f"   Performance: {timing['best_entry_timing']['avg_performance']}")
                print("")
        
        # Display pattern timing
        if 'pattern_timing' in reports:
            print("⏰ PATTERN TIMING INSIGHTS:")
            pattern_insights = reports['pattern_timing']['insights']
            for insight in pattern_insights[:3]:
                print(f"   {insight['emoji']} {insight['pattern'].replace('_', ' ').title()}")
                print(f"      Success Rate: {insight['success_rate']}")
                print(f"      Timing Distribution: {insight['timing_distribution']}")
            print("")
        
        print("📁 Check the 'output/timing_analysis_simple' folder for detailed reports")

def main():
    """Main function to create timing pattern analysis"""
    print("⏰ Starting Simple Timing Pattern Analysis...")
    
    # Create analyzer
    analyzer = SimpleTimingAnalyzer()
    
    # Generate timing analysis
    reports = analyzer.create_timing_analysis()
    
    print("\n" + "="*80)
    print("🎉 TIMING ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
