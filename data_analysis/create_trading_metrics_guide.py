#!/usr/bin/env python3
"""
Trading Metrics Guide
Comprehensive overview of basic trading metrics for memecoin trading
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

class TradingMetricsGuide:
    """Comprehensive guide to trading metrics"""
    
    def __init__(self, results_file: str = "output/adapted_timeseries_analysis.json"):
        self.results_file = Path(results_file)
        self.output_dir = Path("output/trading_metrics_guide")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.results = self._load_results()
        self.df = self._create_dataframe()
        
        # Color scheme for visualizations
        self.colors = {
            'profit': '#00FF00',      # Green
            'loss': '#FF0000',        # Red
            'neutral': '#FFFF00',     # Yellow
            'drawdown': '#FF6B6B',    # Light red
            'recovery': '#4ECDC4'     # Teal
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
        """Create DataFrame with trading metrics"""
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
                    'momentum_score': result.get('momentum_metrics', {}).get('momentum_score', 0)
                }
                
                # Extract transaction analysis
                transaction_data = result.get('transaction_analysis', {})
                if 'overall' in transaction_data:
                    overall = transaction_data['overall']
                    row.update({
                        'buy_sell_ratio': overall.get('buy_sell_ratio', 0),
                        'buy_percentage': overall.get('buy_percentage', 0)
                    })
                
                # Add time-based analysis
                if 'h1' in transaction_data and 'h24' in transaction_data:
                    h1_data = transaction_data['h1']
                    h24_data = transaction_data['h24']
                    
                    row.update({
                        'momentum_change': h1_data.get('buy_sell_ratio', 1) - h24_data.get('buy_sell_ratio', 1),
                        'buy_pressure_change': h1_data.get('buy_percentage', 50) - h24_data.get('buy_percentage', 50)
                    })
                
                # Calculate trading metrics
                row.update(self._calculate_trading_metrics(row))
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _calculate_trading_metrics(self, row):
        """Calculate various trading metrics for a token"""
        metrics = {}
        
        # Basic performance metrics
        fdv_change = row.get('fdv_change_pct', 0)
        
        # 1. Return (already available)
        metrics['return_pct'] = fdv_change
        
        # 2. Absolute Return
        metrics['absolute_return'] = abs(fdv_change)
        
        # 3. Risk-Adjusted Return (Return / Risk Score)
        risk_score = row.get('risk_score', 1)
        metrics['risk_adjusted_return'] = fdv_change / risk_score if risk_score > 0 else 0
        
        # 4. Volatility (using momentum change as proxy)
        momentum_change = row.get('momentum_change', 0)
        metrics['volatility'] = abs(momentum_change)
        
        # 5. Sharpe Ratio (Return / Volatility)
        metrics['sharpe_ratio'] = fdv_change / metrics['volatility'] if metrics['volatility'] > 0 else 0
        
        # 6. Maximum Drawdown (simplified calculation)
        if fdv_change < 0:
            metrics['max_drawdown'] = abs(fdv_change)
            metrics['drawdown_duration'] = 'Ongoing' if fdv_change < -50 else 'Temporary'
        else:
            metrics['max_drawdown'] = risk_score * 5  # Risk score * 5% as estimate
            metrics['drawdown_duration'] = 'Minimal'
        
        # 7. Recovery Time (estimate)
        if fdv_change < 0:
            if fdv_change < -80:
                metrics['recovery_time'] = 'Unlikely'
            elif fdv_change < -50:
                metrics['recovery_time'] = '6+ months'
            elif fdv_change < -20:
                metrics['recovery_time'] = '1-3 months'
            else:
                metrics['recovery_time'] = '1-4 weeks'
        else:
            metrics['recovery_time'] = 'N/A (profitable)'
        
        return metrics
    
    def create_trading_metrics_guide(self):
        """Create comprehensive trading metrics guide"""
        print("📊 Creating Trading Metrics Guide...")
        
        if self.df.empty:
            print("❌ No data available for analysis")
            return
        
        # Generate guide sections
        guide = {}
        
        # 1. Basic Metrics Overview
        guide['basic_metrics'] = self._create_basic_metrics_overview()
        
        # 2. Drawdown Analysis
        guide['drawdown_analysis'] = self._create_drawdown_analysis()
        
        # 3. Risk Metrics
        guide['risk_metrics'] = self._create_risk_metrics()
        
        # 4. Performance Metrics
        guide['performance_metrics'] = self._create_performance_metrics()
        
        # Create visualizations
        self._create_metrics_charts()
        
        # Save guide
        self._save_metrics_guide(guide)
        
        # Display summary
        self._display_metrics_summary(guide)
        
        return guide
    
    def _create_basic_metrics_overview(self):
        """Create overview of basic trading metrics"""
        overview = {
            'title': '📊 BASIC TRADING METRICS OVERVIEW',
            'description': 'Essential metrics every trader should understand',
            'metrics': {}
        }
        
        # Define and explain each metric
        overview['metrics'] = {
            'return': {
                'name': 'Return (%)',
                'description': 'Percentage change in investment value',
                'calculation': 'Final Value - Initial Value / Initial Value × 100',
                'example': 'If 0.5 SOL becomes 1.0 SOL, return = +100%',
                'importance': 'Primary measure of trading success'
            },
            'drawdown': {
                'name': 'Maximum Drawdown (%)',
                'description': 'Largest peak-to-trough decline in investment value',
                'calculation': 'Peak Value - Trough Value / Peak Value × 100',
                'example': 'If investment drops from 1.0 SOL to 0.6 SOL, drawdown = 40%',
                'importance': 'Critical risk metric - how much you can lose'
            },
            'sharpe_ratio': {
                'name': 'Sharpe Ratio',
                'description': 'Risk-adjusted return measure',
                'calculation': 'Return / Volatility (higher is better)',
                'example': 'Sharpe > 1.0 = good, > 2.0 = excellent',
                'importance': 'Compares returns relative to risk taken'
            },
            'volatility': {
                'name': 'Volatility',
                'description': 'Measure of price/investment variability',
                'calculation': 'Standard deviation of returns',
                'example': 'High volatility = large price swings',
                'importance': 'Higher volatility = higher risk and potential reward'
            },
            'win_rate': {
                'name': 'Win Rate (%)',
                'description': 'Percentage of profitable trades',
                'calculation': 'Winning Trades / Total Trades × 100',
                'example': '8 wins out of 10 trades = 80% win rate',
                'importance': 'Consistency indicator, but not the whole story'
            }
        }
        
        return overview
    
    def _create_drawdown_analysis(self):
        """Create detailed drawdown analysis"""
        analysis = {
            'title': '📉 DRAWDOWN ANALYSIS',
            'description': 'Understanding and managing drawdowns in memecoin trading',
            'insights': {}
        }
        
        # Calculate drawdown statistics
        drawdowns = self.df['max_drawdown']
        
        analysis['insights']['drawdown_stats'] = {
            'total_tokens': len(self.df),
            'avg_drawdown': f"{drawdowns.mean():.1f}%",
            'median_drawdown': f"{drawdowns.median():.1f}%",
            'max_drawdown': f"{drawdowns.max():.1f}%",
            'min_drawdown': f"{drawdowns.min():.1f}%"
        }
        
        # Categorize drawdowns
        drawdown_categories = {
            'minimal': drawdowns[drawdowns < 10].count(),
            'moderate': drawdowns[(drawdowns >= 10) & (drawdowns < 25)].count(),
            'significant': drawdowns[(drawdowns >= 25) & (drawdowns < 50)].count(),
            'severe': drawdowns[(drawdowns >= 50) & (drawdowns < 80)].count(),
            'catastrophic': drawdowns[drawdowns >= 80].count()
        }
        
        analysis['insights']['drawdown_categories'] = drawdown_categories
        
        # Recovery analysis
        recovery_times = self.df['recovery_time'].value_counts()
        analysis['insights']['recovery_analysis'] = recovery_times.to_dict()
        
        return analysis
    
    def _create_risk_metrics(self):
        """Create risk metrics analysis"""
        analysis = {
            'title': '⚠️ RISK METRICS ANALYSIS',
            'description': 'Comprehensive risk assessment for memecoin trading',
            'insights': {}
        }
        
        # Risk score analysis
        risk_scores = self.df['risk_score']
        analysis['insights']['risk_score_stats'] = {
            'avg_risk_score': f"{risk_scores.mean():.1f}",
            'median_risk_score': f"{risk_scores.median():.1f}",
            'high_risk_tokens': (risk_scores > 6).sum(),
            'critical_risk_tokens': (risk_scores > 8).sum()
        }
        
        # Risk-adjusted returns
        risk_adjusted = self.df['risk_adjusted_return']
        analysis['insights']['risk_adjusted_stats'] = {
            'avg_risk_adjusted_return': f"{risk_adjusted.mean():.1f}%",
            'best_risk_adjusted': f"{risk_adjusted.max():.1f}%",
            'worst_risk_adjusted': f"{risk_adjusted.min():.1f}%"
        }
        
        # Sharpe ratio analysis
        sharpe_ratios = self.df['sharpe_ratio']
        analysis['insights']['sharpe_analysis'] = {
            'avg_sharpe': f"{sharpe_ratios.mean():.2f}",
            'excellent_sharpe': (sharpe_ratios > 2.0).sum(),
            'good_sharpe': (sharpe_ratios > 1.0).sum(),
            'poor_sharpe': (sharpe_ratios < 0).sum()
        }
        
        return analysis
    
    def _create_performance_metrics(self):
        """Create performance metrics analysis"""
        analysis = {
            'title': '📈 PERFORMANCE METRICS ANALYSIS',
            'description': 'Performance analysis across different metrics',
            'insights': {}
        }
        
        # Return analysis
        returns = self.df['return_pct']
        analysis['insights']['return_stats'] = {
            'avg_return': f"{returns.mean():.1f}%",
            'median_return': f"{returns.median():.1f}%",
            'best_return': f"{returns.max():.1f}%",
            'worst_return': f"{returns.min():.1f}%",
            'profitable_tokens': (returns > 0).sum(),
            'losing_tokens': (returns < 0).sum(),
            'win_rate': f"{(returns > 0).mean() * 100:.1f}%"
        }
        
        return analysis
    
    def _create_metrics_charts(self):
        """Create visualizations for trading metrics"""
        print("🎨 Creating trading metrics charts...")
        
        # 1. Drawdown Distribution
        self._create_drawdown_chart()
        
        # 2. Risk vs Return Scatter
        self._create_risk_return_chart()
        
        # 3. Sharpe Ratio Distribution
        self._create_sharpe_chart()
        
        print("✅ Trading metrics charts created")
    
    def _create_drawdown_chart(self):
        """Create drawdown analysis chart"""
        if self.df.empty:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Drawdown distribution
        drawdowns = self.df['max_drawdown']
        ax1.hist(drawdowns, bins=15, alpha=0.7, color=self.colors['drawdown'], edgecolor='black')
        ax1.set_title('📉 Maximum Drawdown Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Drawdown (%)')
        ax1.set_ylabel('Number of Tokens')
        ax1.axvline(x=drawdowns.mean(), color='red', linestyle='--', alpha=0.8,
                   label=f'Average: {drawdowns.mean():.1f}%')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Recovery time analysis
        recovery_times = self.df['recovery_time'].value_counts()
        colors = [self.colors['loss'], self.colors['neutral'], self.colors['profit']]
        wedges, texts, autotexts = ax2.pie(recovery_times.values, 
                                          labels=recovery_times.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('⏰ Recovery Time Analysis', fontsize=14, fontweight='bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'drawdown_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_risk_return_chart(self):
        """Create risk vs return scatter plot"""
        if self.df.empty:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Color by performance
        colors = ['red' if x < 0 else 'green' for x in self.df['return_pct']]
        
        scatter = plt.scatter(self.df['risk_score'], self.df['return_pct'], 
                            c=colors, alpha=0.7, s=80)
        plt.xlabel('Risk Score')
        plt.ylabel('Return (%)')
        plt.title('🎯 Risk vs Return Analysis', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.axvline(x=6, color='orange', linestyle='--', alpha=0.8, label='High Risk Threshold')
        plt.axvline(x=8, color='red', linestyle='--', alpha=0.8, label='Critical Risk Threshold')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_return_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_sharpe_chart(self):
        """Create Sharpe ratio analysis chart"""
        if self.df.empty:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Sharpe ratio distribution
        sharpe_ratios = self.df['sharpe_ratio']
        plt.hist(sharpe_ratios, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('📊 Sharpe Ratio Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Sharpe Ratio')
        plt.ylabel('Number of Tokens')
        plt.axvline(x=1.0, color='green', linestyle='--', alpha=0.8, label='Good (1.0)')
        plt.axvline(x=2.0, color='darkgreen', linestyle='--', alpha=0.8, label='Excellent (2.0)')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Poor (<0)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sharpe_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _save_metrics_guide(self, guide):
        """Save trading metrics guide"""
        # Save summary guide
        summary = self._format_metrics_summary(guide)
        with open(self.output_dir / 'metrics_summary.txt', 'w') as f:
            f.write(summary)
        
        print(f"✅ Trading metrics guide saved to: {self.output_dir}")
    
    def _format_metrics_summary(self, guide):
        """Format metrics summary"""
        lines = []
        lines.append("=" * 80)
        lines.append("📊 TRADING METRICS GUIDE SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Basic metrics overview
        if 'basic_metrics' in guide:
            lines.append("📊 BASIC TRADING METRICS:")
            metrics = guide['basic_metrics']['metrics']
            for key, metric in metrics.items():
                lines.append(f"• {metric['name']}: {metric['description']}")
                lines.append(f"  Example: {metric['example']}")
                lines.append("")
        
        # Key insights
        if 'drawdown_analysis' in guide:
            drawdown_stats = guide['drawdown_analysis']['insights']['drawdown_stats']
            lines.append("📉 DRAWDOWN INSIGHTS:")
            lines.append(f"• Average Drawdown: {drawdown_stats['avg_drawdown']}")
            lines.append(f"• Maximum Drawdown: {drawdown_stats['max_drawdown']}")
            lines.append("")
        
        if 'risk_metrics' in guide:
            risk_stats = guide['risk_metrics']['insights']['risk_score_stats']
            lines.append("⚠️ RISK INSIGHTS:")
            lines.append(f"• Average Risk Score: {risk_stats['avg_risk_score']}")
            lines.append(f"• High Risk Tokens: {risk_stats['high_risk_tokens']}")
            lines.append("")
        
        lines.append("💡 PRACTICAL USAGE:")
        lines.append("• Use drawdown limits to manage risk")
        lines.append("• Aim for Sharpe ratio > 1.0")
        lines.append("• Monitor risk-adjusted returns")
        lines.append("• Set stop-losses based on drawdown analysis")
        
        return "\n".join(lines)
    
    def _display_metrics_summary(self, guide):
        """Display metrics summary"""
        print("\n" + "="*80)
        print("📊 TRADING METRICS GUIDE COMPLETE!")
        print("="*80)
        
        # Display key metrics
        if 'basic_metrics' in guide:
            print("📊 BASIC TRADING METRICS COVERED:")
            metrics = guide['basic_metrics']['metrics']
            for key, metric in metrics.items():
                print(f"   ✅ {metric['name']}")
            print("")
        
        # Display key insights
        if 'drawdown_analysis' in guide:
            drawdown_stats = guide['drawdown_analysis']['insights']['drawdown_stats']
            print("📉 DRAWDOWN ANALYSIS:")
            print(f"   Average Drawdown: {drawdown_stats['avg_drawdown']}")
            print(f"   Maximum Drawdown: {drawdown_stats['max_drawdown']}")
            print("")
        
        if 'risk_metrics' in guide:
            risk_stats = guide['risk_metrics']['insights']['risk_score_stats']
            print("⚠️ RISK METRICS:")
            print(f"   Average Risk Score: {risk_stats['avg_risk_score']}")
            print(f"   High Risk Tokens: {risk_stats['high_risk_tokens']}")
            print("")
        
        print("📁 Check the 'output/trading_metrics_guide' folder for complete guide")
        print("📊 Summary: metrics_summary.txt")

def main():
    """Main function to create trading metrics guide"""
    print("📊 Starting Trading Metrics Guide Creation...")
    
    # Create guide
    guide = TradingMetricsGuide()
    
    # Generate guide
    guide.create_trading_metrics_guide()
    
    print("\n" + "="*80)
    print("🎉 TRADING METRICS GUIDE COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
