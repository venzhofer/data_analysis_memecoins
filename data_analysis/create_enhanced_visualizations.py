#!/usr/bin/env python3
"""
Enhanced Visualization Creator for Memecoin Analysis
Creates beautiful, publication-ready charts and pattern discovery insights
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class EnhancedVisualizationCreator:
    """Creates comprehensive visualizations for memecoin analysis"""
    
    def __init__(self, results_file: str = "output/adapted_timeseries_analysis.json"):
        self.results_file = Path(results_file)
        self.output_dir = Path("output/enhanced_visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.results = self._load_results()
        self.df = self._create_dataframe()
        
        # Color schemes
        self.colors = {
            'moon_shot': '#FF6B6B',      # Red for moon shots
            'strong_rise': '#4ECDC4',    # Teal for strong rises
            'moderate_rise': '#45B7D1',  # Blue for moderate rises
            'stable': '#96CEB4',         # Green for stable
            'moderate_drop': '#FFEAA7',  # Yellow for moderate drops
            'significant_drop': '#DDA0DD', # Plum for significant drops
            'died': '#A8E6CF'            # Light green for died (ironic)
        }
        
        # Pattern emojis
        self.pattern_emojis = {
            'moon_shot': '🚀',
            'strong_rise': '📈',
            'moderate_rise': '📊',
            'stable': '⚖️',
            'moderate_drop': '📉',
            'significant_drop': '💸',
            'died': '⚰️'
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
        
        return df
    
    def create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard with all visualizations"""
        print("🎨 Creating comprehensive visualization dashboard...")
        
        # Create subplots
        fig = plt.figure(figsize=(24, 20))
        fig.suptitle('🚀 Memecoin Performance Analysis Dashboard', fontsize=24, fontweight='bold', y=0.98)
        
        # 1. Pattern Distribution (Top Left)
        ax1 = plt.subplot(3, 4, 1)
        self._plot_pattern_distribution(ax1)
        
        # 2. Performance Distribution (Top Center Left)
        ax2 = plt.subplot(3, 4, 2)
        self._plot_performance_distribution(ax2)
        
        # 3. Risk vs Performance (Top Center Right)
        ax3 = plt.subplot(3, 4, 3)
        self._plot_risk_vs_performance(ax3)
        
        # 4. Momentum Analysis (Top Right)
        ax4 = plt.subplot(3, 4, 4)
        self._plot_momentum_analysis(ax4)
        
        # 5. Transaction Patterns (Middle Left)
        ax5 = plt.subplot(3, 4, 5)
        self._plot_transaction_patterns(ax5)
        
        # 6. Market Cap vs FDV Changes (Middle Center)
        ax6 = plt.subplot(3, 4, 6)
        self._plot_market_cap_vs_fdv(ax6)
        
        # 7. Buy/Sell Pressure Analysis (Middle Right)
        ax7 = plt.subplot(3, 4, 7)
        self._plot_buy_sell_pressure(ax7)
        
        # 8. Risk Level Distribution (Bottom Left)
        ax8 = plt.subplot(3, 4, 9)
        self._plot_risk_distribution(ax8)
        
        # 9. Performance by Risk Level (Bottom Center)
        ax9 = plt.subplot(3, 4, 10)
        self._plot_performance_by_risk(ax9)
        
        # 10. Transaction Intensity vs Performance (Bottom Right)
        ax10 = plt.subplot(3, 4, 11)
        self._plot_transaction_intensity(ax10)
        
        # 11. Summary Statistics (Bottom Center Right)
        ax11 = plt.subplot(3, 4, 12)
        self._plot_summary_stats(ax11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Comprehensive dashboard saved to: {self.output_dir / 'comprehensive_dashboard.png'}")
    
    def _plot_pattern_distribution(self, ax):
        """Plot pattern distribution with emojis"""
        if self.df.empty:
            return
        
        pattern_counts = self.df['pattern'].value_counts()
        colors = [self.colors.get(pattern, '#CCCCCC') for pattern in pattern_counts.index]
        
        wedges, texts, autotexts = ax.pie(pattern_counts.values, 
                                          labels=[f"{self.pattern_emojis.get(p, '❓')} {p.replace('_', ' ').title()}" 
                                                 for p in pattern_counts.index],
                                          colors=colors, autopct='%1.1f%%', startangle=90)
        
        ax.set_title('🎯 Token Performance Patterns', fontsize=14, fontweight='bold', pad=20)
        
        # Make text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _plot_performance_distribution(self, ax):
        """Plot FDV change distribution"""
        if self.df.empty:
            return
        
        # Create histogram with better bins
        bins = np.linspace(self.df['fdv_change_pct'].min(), self.df['fdv_change_pct'].max(), 20)
        ax.hist(self.df['fdv_change_pct'], bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Break-even')
        ax.axvline(x=100, color='green', linestyle='--', alpha=0.8, linewidth=2, label='100% Gain')
        ax.axvline(x=-50, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='50% Loss')
        
        ax.set_title('📊 FDV Change Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('FDV Change (%)')
        ax.set_ylabel('Number of Tokens')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_risk_vs_performance(self, ax):
        """Plot risk vs performance scatter"""
        if self.df.empty:
            return
        
        # Color by pattern
        for pattern in self.df['pattern'].unique():
            mask = self.df['pattern'] == pattern
            ax.scatter(self.df[mask]['risk_score'], self.df[mask]['fdv_change_pct'], 
                      c=self.colors.get(pattern, '#CCCCCC'), label=pattern.replace('_', ' ').title(),
                      alpha=0.7, s=60)
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
        ax.set_title('⚖️ Risk vs Performance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Risk Score')
        ax.set_ylabel('FDV Change (%)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_momentum_analysis(self, ax):
        """Plot momentum analysis"""
        if self.df.empty:
            return
        
        momentum_data = self.df.groupby('momentum_level')['fdv_change_pct'].mean().sort_values()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        bars = ax.bar(range(len(momentum_data)), momentum_data.values, color=colors[:len(momentum_data)])
        ax.set_title('📈 Performance by Momentum Level', fontsize=14, fontweight='bold')
        ax.set_xlabel('Momentum Level')
        ax.set_ylabel('Average FDV Change (%)')
        ax.set_xticks(range(len(momentum_data)))
        ax.set_xticklabels([level.replace('_', ' ').title() for level in momentum_data.index])
        
        # Add value labels on bars
        for bar, value in zip(bars, momentum_data.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -15),
                   f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                   fontweight='bold')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_transaction_patterns(self, ax):
        """Plot transaction patterns"""
        if self.df.empty:
            return
        
        # Buy vs Sell analysis
        buy_sell_data = self.df[['total_buys', 'total_sells']].sum()
        colors = ['#4ECDC4', '#FF6B6B']
        
        wedges, texts, autotexts = ax.pie(buy_sell_data.values, 
                                          labels=['Buys', 'Sells'], 
                                          colors=colors, autopct='%1.1f%%', startangle=90)
        
        ax.set_title('💱 Transaction Patterns', fontsize=14, fontweight='bold', pad=20)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _plot_market_cap_vs_fdv(self, ax):
        """Plot market cap vs FDV changes"""
        if self.df.empty:
            return
        
        ax.scatter(self.df['fdv_change_pct'], self.df['market_cap_change_pct'], 
                  alpha=0.7, s=60, c='purple')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        ax.set_title('💰 Market Cap vs FDV Changes', fontsize=14, fontweight='bold')
        ax.set_xlabel('FDV Change (%)')
        ax.set_ylabel('Market Cap Change (%)')
        ax.grid(True, alpha=0.3)
    
    def _plot_buy_sell_pressure(self, ax):
        """Plot buy/sell pressure analysis"""
        if self.df.empty:
            return
        
        # Buy pressure distribution
        ax.hist(self.df['buy_pressure'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Neutral')
        ax.axvline(x=10, color='green', linestyle='--', alpha=0.8, linewidth=2, label='10% Buy Pressure')
        ax.axvline(x=-10, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='10% Sell Pressure')
        
        ax.set_title('🔄 Buy/Sell Pressure Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Buy Pressure (%)')
        ax.set_ylabel('Number of Tokens')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_risk_distribution(self, ax):
        """Plot risk distribution"""
        if self.df.empty:
            return
        
        risk_counts = self.df['risk_level'].value_counts()
        colors = ['#96CEB4', '#FFEAA7', '#FF6B6B']  # Low, Medium, High
        
        bars = ax.bar(range(len(risk_counts)), risk_counts.values, 
                     color=colors[:len(risk_counts)])
        ax.set_title('⚠️ Risk Level Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Risk Level')
        ax.set_ylabel('Number of Tokens')
        ax.set_xticks(range(len(risk_counts)))
        ax.set_xticklabels([level.replace('_', ' ').title() for level in risk_counts.index])
        
        # Add value labels
        for bar, value in zip(bars, risk_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   str(value), ha='center', va='bottom', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_by_risk(self, ax):
        """Plot performance grouped by risk level"""
        if self.df.empty:
            return
        
        risk_performance = self.df.groupby('risk_level')['fdv_change_pct'].agg(['mean', 'count']).sort_values('mean')
        
        bars = ax.bar(range(len(risk_performance)), risk_performance['mean'], 
                     color=['#96CEB4', '#FFEAA7', '#FF6B6B'][:len(risk_performance)])
        ax.set_title('📊 Performance by Risk Level', fontsize=14, fontweight='bold')
        ax.set_xlabel('Risk Level')
        ax.set_ylabel('Average FDV Change (%)')
        ax.set_xticks(range(len(risk_performance)))
        ax.set_xticklabels([level.replace('_', ' ').title() for level in risk_performance.index])
        
        # Add value labels
        for bar, value in zip(bars, risk_performance['mean']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -15),
                   f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                   fontweight='bold')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_transaction_intensity(self, ax):
        """Plot transaction intensity vs performance"""
        if self.df.empty:
            return
        
        ax.scatter(self.df['transaction_intensity'], self.df['fdv_change_pct'], 
                  alpha=0.7, s=60, c='orange')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        ax.set_title('🔥 Transaction Intensity vs Performance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Transaction Intensity (transactions/FDV)')
        ax.set_ylabel('FDV Change (%)')
        ax.grid(True, alpha=0.3)
    
    def _plot_summary_stats(self, ax):
        """Plot summary statistics"""
        if self.df.empty:
            return
        
        ax.axis('off')
        
        # Calculate summary stats
        total_tokens = len(self.df)
        avg_fdv_change = self.df['fdv_change_pct'].mean()
        success_rate = (self.df['fdv_change_pct'] > 0).mean() * 100
        moon_shot_rate = (self.df['pattern'] == 'moon_shot').mean() * 100
        died_rate = (self.df['pattern'] == 'died').mean() * 100
        
        stats_text = f"""
📊 SUMMARY STATISTICS

🎯 Total Tokens: {total_tokens}
📈 Average FDV Change: {avg_fdv_change:.1f}%
✅ Success Rate: {success_rate:.1f}%
🚀 Moon Shot Rate: {moon_shot_rate:.1f}%
⚰️ Died Rate: {died_rate:.1f}%

💰 Top Performers:
{self._get_top_performers()}

⚠️ Risk Analysis:
{self._get_risk_summary()}
        """
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    def _get_top_performers(self):
        """Get top performing tokens"""
        if self.df.empty:
            return "No data available"
        
        top_5 = self.df.nlargest(5, 'fdv_change_pct')[['token_name', 'fdv_change_pct']]
        result = []
        for _, row in top_5.iterrows():
            result.append(f"• {row['token_name']}: {row['fdv_change_pct']:.1f}%")
        return "\n".join(result)
    
    def _get_risk_summary(self):
        """Get risk summary"""
        if self.df.empty:
            return "No data available"
        
        risk_dist = self.df['risk_level'].value_counts()
        result = []
        for level, count in risk_dist.items():
            result.append(f"• {level.title()}: {count}")
        return "\n".join(result)
    
    def create_pattern_discovery_report(self):
        """Create a comprehensive pattern discovery report"""
        print("🔍 Creating pattern discovery report...")
        
        if self.df.empty:
            print("❌ No data available for analysis")
            return
        
        # Create the report
        report = self._generate_pattern_report()
        
        # Save to file
        report_file = self.output_dir / 'pattern_discovery_report.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Pattern discovery report saved to: {report_file}")
        
        # Also create a JSON version for programmatic use
        json_report = self._generate_json_report()
        json_file = self.output_dir / 'pattern_discovery_report.json'
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        print(f"✅ JSON report saved to: {json_file}")
        
        return report
    
    def _generate_pattern_report(self):
        """Generate the pattern discovery report"""
        report = []
        report.append("=" * 80)
        report.append("🚀 MEMECOIN PATTERN DISCOVERY REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Tokens Analyzed: {len(self.df)}")
        report.append("")
        
        # 1. Executive Summary
        report.append("📊 EXECUTIVE SUMMARY")
        report.append("-" * 40)
        success_rate = (self.df['fdv_change_pct'] > 0).mean() * 100
        avg_performance = self.df['fdv_change_pct'].mean()
        report.append(f"• Overall Success Rate: {success_rate:.1f}%")
        report.append(f"• Average Performance: {avg_performance:.1f}%")
        report.append(f"• Best Performer: {self.df['fdv_change_pct'].max():.1f}%")
        report.append(f"• Worst Performer: {self.df['fdv_change_pct'].min():.1f}%")
        report.append("")
        
        # 2. Pattern Analysis
        report.append("🎯 PATTERN ANALYSIS")
        report.append("-" * 40)
        pattern_analysis = self.df['pattern'].value_counts()
        for pattern, count in pattern_analysis.items():
            percentage = (count / len(self.df)) * 100
            emoji = self.pattern_emojis.get(pattern, '❓')
            report.append(f"{emoji} {pattern.replace('_', ' ').title()}: {count} tokens ({percentage:.1f}%)")
        report.append("")
        
        # 3. Performance Insights
        report.append("📈 PERFORMANCE INSIGHTS")
        report.append("-" * 40)
        report.append(self._get_performance_insights())
        report.append("")
        
        # 4. Risk Analysis
        report.append("⚠️ RISK ANALYSIS")
        report.append("-" * 40)
        report.append(self._get_risk_insights())
        report.append("")
        
        # 5. Transaction Pattern Insights
        report.append("💱 TRANSACTION PATTERN INSIGHTS")
        report.append("-" * 40)
        report.append(self._get_transaction_insights())
        report.append("")
        
        # 6. Top Performers Analysis
        report.append("🏆 TOP PERFORMERS ANALYSIS")
        report.append("-" * 40)
        report.append(self._get_top_performers_analysis())
        report.append("")
        
        # 7. Risk-Reward Analysis
        report.append("⚖️ RISK-REWARD ANALYSIS")
        report.append("-" * 40)
        report.append(self._get_risk_reward_insights())
        report.append("")
        
        # 8. Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        report.append(self._get_recommendations())
        report.append("")
        
        report.append("=" * 80)
        report.append("Report generated by Enhanced Memecoin Analysis System")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _get_performance_insights(self):
        """Get performance insights"""
        insights = []
        
        # Success rate by momentum
        momentum_success = self.df.groupby('momentum_level')['fdv_change_pct'].apply(
            lambda x: (x > 0).mean() * 100
        ).sort_values(ascending=False)
        
        insights.append("• Success Rate by Momentum Level:")
        for level, rate in momentum_success.items():
            insights.append(f"  - {level.replace('_', ' ').title()}: {rate:.1f}%")
        
        # Performance by risk level
        risk_performance = self.df.groupby('risk_level')['fdv_change_pct'].mean().sort_values(ascending=False)
        insights.append("\n• Average Performance by Risk Level:")
        for level, perf in risk_performance.items():
            insights.append(f"  - {level.replace('_', ' ').title()}: {perf:.1f}%")
        
        return "\n".join(insights)
    
    def _get_risk_insights(self):
        """Get risk insights"""
        insights = []
        
        # Risk distribution
        risk_dist = self.df['risk_level'].value_counts()
        insights.append("• Risk Level Distribution:")
        for level, count in risk_dist.items():
            percentage = (count / len(self.df)) * 100
            insights.append(f"  - {level.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        # Risk vs performance correlation
        risk_corr = self.df['risk_score'].corr(self.df['fdv_change_pct'])
        insights.append(f"\n• Risk-Performance Correlation: {risk_corr:.3f}")
        
        return "\n".join(insights)
    
    def _get_transaction_insights(self):
        """Get transaction insights"""
        insights = []
        
        # Buy/sell patterns
        avg_buy_pressure = self.df['buy_pressure'].mean()
        insights.append(f"• Average Buy Pressure: {avg_buy_pressure:.1f}%")
        
        # Transaction intensity
        avg_intensity = self.df['transaction_intensity'].mean()
        insights.append(f"• Average Transaction Intensity: {avg_intensity:.3f}")
        
        # Buy/sell ratio insights
        buy_sell_ratio = self.df['buy_sell_ratio'].mean()
        insights.append(f"• Average Buy/Sell Ratio: {buy_sell_ratio:.2f}")
        
        return "\n".join(insights)
    
    def _get_top_performers_analysis(self):
        """Get top performers analysis"""
        insights = []
        
        top_10 = self.df.nlargest(10, 'fdv_change_pct')
        insights.append("• Top 10 Performers:")
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            insights.append(f"  {i:2d}. {row['token_name']}: {row['fdv_change_pct']:+.1f}%")
        
        # Pattern analysis of top performers
        top_patterns = top_10['pattern'].value_counts()
        insights.append(f"\n• Pattern Distribution Among Top Performers:")
        for pattern, count in top_patterns.items():
            percentage = (count / len(top_10)) * 100
            emoji = self.pattern_emojis.get(pattern, '❓')
            insights.append(f"  {emoji} {pattern.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        return "\n".join(insights)
    
    def _get_risk_reward_insights(self):
        """Get risk-reward insights"""
        insights = []
        
        # Calculate risk-adjusted returns
        self.df['risk_adjusted_return'] = self.df['fdv_change_pct'] / (self.df['risk_score'] + 1)
        
        # Best risk-adjusted performers
        best_risk_adjusted = self.df.nlargest(5, 'risk_adjusted_return')
        insights.append("• Top 5 Risk-Adjusted Performers:")
        for i, (_, row) in enumerate(best_risk_adjusted.iterrows(), 1):
            insights.append(f"  {i}. {row['token_name']}: {row['risk_adjusted_return']:.1f} return/risk")
        
        # Risk-reward by pattern
        pattern_risk_reward = self.df.groupby('pattern')['risk_adjusted_return'].mean().sort_values(ascending=False)
        insights.append(f"\n• Risk-Adjusted Returns by Pattern:")
        for pattern, rar in pattern_risk_reward.items():
            emoji = self.pattern_emojis.get(pattern, '❓')
            insights.append(f"  {emoji} {pattern.replace('_', ' ').title()}: {rar:.1f}")
        
        return "\n".join(insights)
    
    def _get_recommendations(self):
        """Get recommendations based on analysis"""
        recommendations = []
        
        # Success rate analysis
        success_rate = (self.df['fdv_change_pct'] > 0).mean() * 100
        
        if success_rate > 60:
            recommendations.append("✅ High success rate suggests favorable market conditions")
        elif success_rate > 40:
            recommendations.append("⚠️ Moderate success rate - exercise caution")
        else:
            recommendations.append("❌ Low success rate - consider waiting for better conditions")
        
        # Risk level recommendations
        low_risk_success = self.df[self.df['risk_level'] == 'low']['fdv_change_pct'].mean()
        if low_risk_success > 0:
            recommendations.append("✅ Low-risk tokens showing positive returns - good for conservative strategies")
        
        # Momentum recommendations
        high_momentum_success = self.df[self.df['momentum_level'] == 'high']['fdv_change_pct'].mean()
        if high_momentum_success > 0:
            recommendations.append("🚀 High momentum tokens performing well - momentum strategies may be effective")
        
        # Pattern-specific recommendations
        moon_shot_rate = (self.df['pattern'] == 'moon_shot').mean() * 100
        if moon_shot_rate > 20:
            recommendations.append("🚀 High moon shot rate - aggressive strategies may be profitable")
        
        return "\n".join(recommendations)
    
    def _generate_json_report(self):
        """Generate JSON version of the report"""
        if self.df.empty:
            return {}
        
        return {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_tokens": len(self.df),
                "analysis_version": "1.0"
            },
            "summary_statistics": {
                "success_rate": (self.df['fdv_change_pct'] > 0).mean() * 100,
                "average_performance": self.df['fdv_change_pct'].mean(),
                "best_performer": self.df['fdv_change_pct'].max(),
                "worst_performer": self.df['fdv_change_pct'].min(),
                "total_tokens": len(self.df)
            },
            "pattern_analysis": self.df['pattern'].value_counts().to_dict(),
            "risk_analysis": {
                "risk_distribution": self.df['risk_level'].value_counts().to_dict(),
                "risk_performance": self.df.groupby('risk_level')['fdv_change_pct'].mean().to_dict()
            },
            "momentum_analysis": {
                "momentum_distribution": self.df['momentum_level'].value_counts().to_dict(),
                "momentum_performance": self.df.groupby('momentum_level')['fdv_change_pct'].mean().to_dict()
            },
            "top_performers": self.df.nlargest(10, 'fdv_change_pct')[['token_name', 'fdv_change_pct', 'pattern', 'risk_level']].to_dict('records'),
            "risk_reward_analysis": {
                "risk_adjusted_returns": self.df.groupby('pattern')['risk_adjusted_return'].mean().to_dict()
            }
        }

def main():
    """Main function to create all visualizations"""
    print("🎨 Starting Enhanced Visualization Creation...")
    
    # Create visualizer
    visualizer = EnhancedVisualizationCreator()
    
    # Create comprehensive dashboard
    visualizer.create_comprehensive_dashboard()
    
    # Create pattern discovery report
    report = visualizer.create_pattern_discovery_report()
    
    print("\n" + "="*80)
    print("🎉 VISUALIZATION CREATION COMPLETE!")
    print("="*80)
    print("📊 Created comprehensive dashboard")
    print("📝 Generated pattern discovery report")
    print("📁 Check the 'output/enhanced_visualizations' folder for all results")
    print("\n" + "="*80)
    print("PATTERN DISCOVERY REPORT PREVIEW:")
    print("="*80)
    print(report[:1000] + "..." if len(report) > 1000 else report)

if __name__ == "__main__":
    main()
