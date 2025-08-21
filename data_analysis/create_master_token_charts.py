#!/usr/bin/env python3
"""
Master Token Charts Creator
Creates comprehensive charts for all tokens showing patterns, timing, exit analysis, and death patterns
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

class MasterTokenChartsCreator:
    """Creates comprehensive charts for all tokens"""
    
    def __init__(self, results_file: str = "output/adapted_timeseries_analysis.json"):
        self.results_file = Path(results_file)
        self.output_dir = Path("output/master_token_charts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.results = self._load_results()
        self.df = self._create_comprehensive_dataframe()
        
        # Color schemes
        self.pattern_colors = {
            'moon_shot': '#00FF00',        # Bright green
            'moderate_rise': '#32CD32',    # Lime green
            'stable': '#FFD700',           # Gold
            'moderate_drop': '#FFA500',    # Orange
            'significant_drop': '#FF6B6B', # Red
            'died': '#8B0000',             # Dark red
            'unknown': '#CCCCCC'           # Gray
        }
        
        self.timing_colors = {
            'instant_boom': '#FF1493',     # Deep pink
            'early_boom': '#FF69B4',      # Hot pink
            'gradual_development': '#DDA0DD' # Plum
        }
        
        self.risk_colors = {
            'low': '#96CEB4',      # Green
            'medium': '#FFEAA7',   # Yellow
            'high': '#FF6B6B',     # Red
            'critical': '#8B0000'  # Dark red
        }
    
    def _load_results(self) -> dict:
        """Load analysis results"""
        try:
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading results: {e}")
            return {}
    
    def _create_comprehensive_dataframe(self) -> pd.DataFrame:
        """Create comprehensive DataFrame with all analysis data"""
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
                
                # Add timing analysis
                row.update(self._analyze_timing_pattern(row))
                
                # Add exit analysis
                row.update(self._analyze_exit_scenarios(row))
                
                # Add death risk analysis
                row.update(self._analyze_death_risk(row))
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _analyze_timing_pattern(self, row):
        """Analyze timing pattern for a token"""
        timing_info = {}
        
        # Infer timing based on transaction data
        h1_volume = row.get('h1_buy_sell_ratio', 1)
        h24_volume = row.get('h24_buy_sell_ratio', 1)
        
        if h1_volume > 1.5:  # High recent activity
            timing_info['timing_category'] = 'instant_boom'
            timing_info['timing_description'] = 'Instant Boom (High immediate activity)'
        elif h1_volume > 1.0:  # Moderate recent activity
            timing_info['timing_category'] = 'early_boom'
            timing_info['timing_description'] = 'Early Boom (Moderate early activity)'
        else:  # Lower activity
            timing_info['timing_category'] = 'gradual_development'
            timing_info['timing_description'] = 'Gradual Development (Slower growth)'
        
        return timing_info
    
    def _analyze_exit_scenarios(self, row):
        """Analyze exit scenarios for a token"""
        exit_info = {}
        
        current_performance = row.get('fdv_change_pct', 0)
        exit_info['current_performance'] = current_performance
        
        # Calculate exit scenarios
        exit_info['exit_at_ratio'] = -20 if row.get('buy_sell_ratio', 1) < 0.6 else current_performance
        exit_info['exit_at_percentage'] = -25 if row.get('buy_percentage', 50) < 40 else current_performance
        exit_info['exit_at_momentum'] = -15 if row.get('momentum_change', 0) < 0 else current_performance
        exit_info['exit_at_risk'] = -35 if row.get('risk_score', 0) > 7.0 else current_performance
        exit_info['exit_at_fdv'] = -30 if row.get('fdv_change_pct', 0) < -30 else current_performance
        
        # Best exit scenario
        exit_performances = [
            exit_info['exit_at_ratio'],
            exit_info['exit_at_percentage'], 
            exit_info['exit_at_momentum'],
            exit_info['exit_at_risk'],
            exit_info['exit_at_fdv']
        ]
        exit_info['best_exit_performance'] = max(exit_performances)
        exit_info['exit_improvement'] = exit_info['best_exit_performance'] - current_performance
        
        return exit_info
    
    def _analyze_death_risk(self, row):
        """Analyze death risk for a token"""
        death_info = {}
        
        # Point of no return check
        death_info['hit_point_of_no_return'] = (
            row.get('fdv_change_pct', 0) <= -50 or
            row.get('risk_score', 0) >= 8.0 or
            row.get('buy_sell_ratio', 1) <= 0.5
        )
        
        # Death category
        fdv_change = row.get('fdv_change_pct', 0)
        if fdv_change <= -80:
            death_info['death_category'] = 'died'
        elif fdv_change <= -50:
            death_info['death_category'] = 'significant_drop'
        elif fdv_change <= -20:
            death_info['death_category'] = 'moderate_drop'
        else:
            death_info['death_category'] = 'alive'
        
        return death_info
    
    def create_master_token_charts(self):
        """Create comprehensive charts for all tokens"""
        print("🎨 Creating master token charts...")
        
        if self.df.empty:
            print("❌ No data available for analysis")
            return
        
        # 1. Individual Token Performance Chart
        self._create_individual_token_chart()
        
        # 2. Pattern vs Timing Analysis
        self._create_pattern_timing_chart()
        
        # 3. Exit Analysis Comparison
        self._create_exit_comparison_chart()
        
        # 4. Risk vs Performance Matrix
        self._create_risk_performance_matrix()
        
        # 5. Token Summary Dashboard
        self._create_token_dashboard()
        
        # 6. Complete Token Analysis Grid
        self._create_token_analysis_grid()
        
        # 7. Performance Distribution Charts
        self._create_performance_distributions()
        
        # 8. Correlation Analysis Charts
        self._create_correlation_charts()
        
        print("✅ Master token charts created successfully!")
        print(f"📁 Charts saved to: {self.output_dir}")
        
        # Generate summary
        self._generate_chart_summary()
    
    def _create_individual_token_chart(self):
        """Create individual token performance chart"""
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Sort tokens by performance
        df_sorted = self.df.sort_values('fdv_change_pct', ascending=True)
        
        # Create bar chart
        bars = ax.barh(range(len(df_sorted)), df_sorted['fdv_change_pct'], 
                      color=[self.pattern_colors.get(p, '#CCCCCC') for p in df_sorted['pattern']])
        
        # Customize chart
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels([f"{name[:15]}..." if len(name) > 15 else name 
                           for name in df_sorted['token_name']], fontsize=10)
        ax.set_xlabel('FDV Change (%)', fontsize=14, fontweight='bold')
        ax.set_title('🚀 Individual Token Performance (All 26 Tokens)', fontsize=18, fontweight='bold', pad=20)
        
        # Add value labels
        for i, (bar, value, pattern) in enumerate(zip(bars, df_sorted['fdv_change_pct'], df_sorted['pattern'])):
            width = bar.get_width()
            label_x = width + (10 if width >= 0 else -10)
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'{value:+.1f}% ({pattern})', 
                   ha='left' if width >= 0 else 'right', va='center', 
                   fontsize=9, fontweight='bold')
        
        # Add legend
        legend_elements = [mpatches.Patch(color=color, label=pattern.replace('_', ' ').title()) 
                          for pattern, color in self.pattern_colors.items() if pattern in df_sorted['pattern'].values]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'individual_token_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_pattern_timing_chart(self):
        """Create pattern vs timing analysis chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Pattern distribution
        pattern_counts = self.df['pattern'].value_counts()
        wedges, texts, autotexts = ax1.pie(pattern_counts.values, 
                                          labels=[p.replace('_', ' ').title() for p in pattern_counts.index],
                                          colors=[self.pattern_colors.get(p, '#CCCCCC') for p in pattern_counts.index],
                                          autopct='%1.1f%%', startangle=90)
        ax1.set_title('📊 Pattern Distribution', fontsize=14, fontweight='bold')
        
        # 2. Timing distribution
        timing_counts = self.df['timing_category'].value_counts()
        wedges2, texts2, autotexts2 = ax2.pie(timing_counts.values,
                                              labels=[t.replace('_', ' ').title() for t in timing_counts.index],
                                              colors=[self.timing_colors.get(t, '#CCCCCC') for t in timing_counts.index],
                                              autopct='%1.1f%%', startangle=90)
        ax2.set_title('⏰ Timing Distribution', fontsize=14, fontweight='bold')
        
        # 3. Pattern vs Performance
        patterns = self.df['pattern'].unique()
        pattern_performance = [self.df[self.df['pattern'] == p]['fdv_change_pct'].mean() for p in patterns]
        bars3 = ax3.bar(range(len(patterns)), pattern_performance, 
                       color=[self.pattern_colors.get(p, '#CCCCCC') for p in patterns])
        ax3.set_xticks(range(len(patterns)))
        ax3.set_xticklabels([p.replace('_', ' ').title() for p in patterns], rotation=45, ha='right')
        ax3.set_ylabel('Average FDV Change (%)')
        ax3.set_title('📈 Average Performance by Pattern', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars3, pattern_performance):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -5),
                    f'{value:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                    fontweight='bold')
        
        # 4. Timing vs Performance
        timings = self.df['timing_category'].unique()
        timing_performance = [self.df[self.df['timing_category'] == t]['fdv_change_pct'].mean() for t in timings]
        bars4 = ax4.bar(range(len(timings)), timing_performance,
                       color=[self.timing_colors.get(t, '#CCCCCC') for t in timings])
        ax4.set_xticks(range(len(timings)))
        ax4.set_xticklabels([t.replace('_', ' ').title() for t in timings], rotation=45, ha='right')
        ax4.set_ylabel('Average FDV Change (%)')
        ax4.set_title('⏰ Average Performance by Timing', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars4, timing_performance):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -5),
                    f'{value:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                    fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pattern_timing_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_exit_comparison_chart(self):
        """Create exit analysis comparison chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Hold vs Best Exit comparison
        scenarios = ['current_performance', 'best_exit_performance']
        scenario_names = ['Hold to End', 'Best Exit Strategy']
        
        data_to_plot = [self.df[scenario] for scenario in scenarios]
        bp1 = ax1.boxplot(data_to_plot, labels=scenario_names, patch_artist=True)
        colors1 = ['#FF6B6B', '#4ECDC4']
        for patch, color in zip(bp1['boxes'], colors1):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('💰 Hold vs Exit Strategy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Performance (%)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Exit improvement per token
        exit_improvement = self.df['exit_improvement']
        colors_improvement = ['green' if x > 0 else 'red' for x in exit_improvement]
        
        bars2 = ax2.bar(range(len(exit_improvement)), exit_improvement, color=colors_improvement, alpha=0.7)
        ax2.set_title('📈 Exit Strategy Improvement per Token', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Token Index')
        ax2.set_ylabel('Improvement (%)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # 3. Pattern vs Exit effectiveness
        patterns = self.df['pattern'].unique()
        pattern_exit_improvement = [self.df[self.df['pattern'] == p]['exit_improvement'].mean() for p in patterns]
        
        bars3 = ax3.bar(range(len(patterns)), pattern_exit_improvement,
                       color=[self.pattern_colors.get(p, '#CCCCCC') for p in patterns])
        ax3.set_xticks(range(len(patterns)))
        ax3.set_xticklabels([p.replace('_', ' ').title() for p in patterns], rotation=45, ha='right')
        ax3.set_ylabel('Average Exit Improvement (%)')
        ax3.set_title('🎯 Exit Strategy Effectiveness by Pattern', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 4. Point of no return distribution
        point_of_no_return = self.df['hit_point_of_no_return'].sum()
        safe_tokens = len(self.df) - point_of_no_return
        
        ax4.pie([point_of_no_return, safe_tokens], 
               labels=[f'Hit Point of No Return ({point_of_no_return})', f'Safe Tokens ({safe_tokens})'],
               colors=['#8B0000', '#32CD32'], autopct='%1.1f%%', startangle=90)
        ax4.set_title('🚫 Point of No Return Analysis', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exit_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_risk_performance_matrix(self):
        """Create risk vs performance matrix"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Risk vs Performance scatter
        for risk_level in self.df['risk_level'].unique():
            risk_data = self.df[self.df['risk_level'] == risk_level]
            color = self.risk_colors.get(risk_level, '#CCCCCC')
            ax1.scatter(risk_data['risk_score'], risk_data['fdv_change_pct'], 
                       c=color, label=risk_level.replace('_', ' ').title(), alpha=0.7, s=60)
        
        ax1.set_xlabel('Risk Score')
        ax1.set_ylabel('FDV Change (%)')
        ax1.set_title('🎯 Risk vs Performance Matrix', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.axvline(x=6, color='orange', linestyle='--', alpha=0.8, label='High Risk Threshold')
        ax1.axvline(x=8, color='red', linestyle='--', alpha=0.8, label='Critical Risk Threshold')
        
        # 2. Risk level distribution
        risk_counts = self.df['risk_level'].value_counts()
        bars2 = ax2.bar(range(len(risk_counts)), risk_counts.values,
                       color=[self.risk_colors.get(r, '#CCCCCC') for r in risk_counts.index])
        ax2.set_xticks(range(len(risk_counts)))
        ax2.set_xticklabels([r.replace('_', ' ').title() for r in risk_counts.index])
        ax2.set_ylabel('Number of Tokens')
        ax2.set_title('⚠️ Risk Level Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars2, risk_counts.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # 3. Buy/Sell ratio vs Performance
        ax3.scatter(self.df['buy_sell_ratio'], self.df['fdv_change_pct'], 
                   c=[self.pattern_colors.get(p, '#CCCCCC') for p in self.df['pattern']], 
                   alpha=0.7, s=60)
        ax3.set_xlabel('Buy/Sell Ratio')
        ax3.set_ylabel('FDV Change (%)')
        ax3.set_title('💹 Buy/Sell Ratio vs Performance', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.axvline(x=0.6, color='red', linestyle='--', alpha=0.8, label='Exit Threshold')
        ax3.axvline(x=0.8, color='orange', linestyle='--', alpha=0.8, label='Warning Threshold')
        
        # 4. Momentum vs Performance
        ax4.scatter(self.df['momentum_change'], self.df['fdv_change_pct'],
                   c=[self.timing_colors.get(t, '#CCCCCC') for t in self.df['timing_category']], 
                   alpha=0.7, s=60)
        ax4.set_xlabel('Momentum Change')
        ax4.set_ylabel('FDV Change (%)')
        ax4.set_title('🚀 Momentum vs Performance', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Exit Threshold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_performance_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_token_dashboard(self):
        """Create comprehensive token dashboard"""
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Main performance chart (spans 2x2)
        ax_main = fig.add_subplot(gs[0:2, 0:2])
        df_sorted = self.df.sort_values('fdv_change_pct', ascending=False)
        bars = ax_main.bar(range(len(df_sorted)), df_sorted['fdv_change_pct'],
                          color=[self.pattern_colors.get(p, '#CCCCCC') for p in df_sorted['pattern']])
        ax_main.set_title('🚀 Token Performance Ranking', fontsize=16, fontweight='bold')
        ax_main.set_ylabel('FDV Change (%)')
        ax_main.set_xlabel('Token Rank')
        ax_main.grid(True, alpha=0.3)
        ax_main.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Pattern distribution
        ax1 = fig.add_subplot(gs[0, 2])
        pattern_counts = self.df['pattern'].value_counts()
        ax1.pie(pattern_counts.values, labels=pattern_counts.index, autopct='%1.0f%%')
        ax1.set_title('📊 Patterns', fontsize=12, fontweight='bold')
        
        # Timing distribution
        ax2 = fig.add_subplot(gs[0, 3])
        timing_counts = self.df['timing_category'].value_counts()
        ax2.pie(timing_counts.values, labels=timing_counts.index, autopct='%1.0f%%')
        ax2.set_title('⏰ Timing', fontsize=12, fontweight='bold')
        
        # Risk distribution
        ax3 = fig.add_subplot(gs[1, 2])
        risk_counts = self.df['risk_level'].value_counts()
        ax3.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.0f%%')
        ax3.set_title('⚠️ Risk Levels', fontsize=12, fontweight='bold')
        
        # Death analysis
        ax4 = fig.add_subplot(gs[1, 3])
        death_counts = self.df['death_category'].value_counts()
        ax4.pie(death_counts.values, labels=death_counts.index, autopct='%1.0f%%')
        ax4.set_title('💀 Death Analysis', fontsize=12, fontweight='bold')
        
        # Performance metrics (spans full width)
        ax5 = fig.add_subplot(gs[2, :])
        metrics = ['Hold Strategy', 'Best Exit Strategy', 'Average Risk Score', 'Point of No Return %']
        values = [
            self.df['current_performance'].mean(),
            self.df['best_exit_performance'].mean(),
            self.df['risk_score'].mean(),
            (self.df['hit_point_of_no_return'].sum() / len(self.df)) * 100
        ]
        bars5 = ax5.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#FFEAA7', '#8B0000'])
        ax5.set_title('📈 Key Performance Metrics', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Value')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars5, values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -1),
                    f'{value:.1f}', ha='center', va='bottom' if height >= 0 else 'top', 
                    fontweight='bold')
        
        # Summary statistics (spans full width)
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        # Create summary text
        summary_text = f"""
        📊 COMPREHENSIVE TOKEN ANALYSIS SUMMARY
        
        🎯 Total Tokens Analyzed: {len(self.df)}
        
        📈 Performance Summary:
        • Best Performer: {df_sorted.iloc[0]['token_name']} ({df_sorted.iloc[0]['fdv_change_pct']:+.1f}%)
        • Worst Performer: {df_sorted.iloc[-1]['token_name']} ({df_sorted.iloc[-1]['fdv_change_pct']:+.1f}%)
        • Average Performance: {self.df['fdv_change_pct'].mean():+.1f}%
        • Median Performance: {self.df['fdv_change_pct'].median():+.1f}%
        
        🎨 Pattern Distribution:
        • Moon Shots: {pattern_counts.get('moon_shot', 0)} tokens
        • Moderate Rise: {pattern_counts.get('moderate_rise', 0)} tokens  
        • Stable: {pattern_counts.get('stable', 0)} tokens
        • Drops/Deaths: {pattern_counts.get('moderate_drop', 0) + pattern_counts.get('significant_drop', 0) + pattern_counts.get('died', 0)} tokens
        
        💰 Exit Strategy Impact:
        • Average Hold Return: {self.df['current_performance'].mean():+.1f}%
        • Average Best Exit Return: {self.df['best_exit_performance'].mean():+.1f}%
        • Average Improvement: {self.df['exit_improvement'].mean():+.1f}%
        • Tokens Improved by Exit: {(self.df['exit_improvement'] > 0).sum()}/{len(self.df)}
        
        🚫 Risk Analysis:
        • Point of No Return Tokens: {self.df['hit_point_of_no_return'].sum()}/{len(self.df)} ({(self.df['hit_point_of_no_return'].sum()/len(self.df)*100):.1f}%)
        • Average Risk Score: {self.df['risk_score'].mean():.1f}
        • High Risk Tokens: {(self.df['risk_score'] > 6).sum()} tokens
        • Critical Risk Tokens: {(self.df['risk_score'] > 8).sum()} tokens
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.suptitle('🎨 MASTER TOKEN ANALYSIS DASHBOARD', fontsize=20, fontweight='bold', y=0.98)
        plt.savefig(self.output_dir / 'master_token_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_token_analysis_grid(self):
        """Create detailed token analysis grid"""
        # Calculate number of rows needed (4 tokens per row)
        tokens_per_row = 4
        num_rows = (len(self.df) + tokens_per_row - 1) // tokens_per_row
        
        fig, axes = plt.subplots(num_rows, tokens_per_row, figsize=(24, 6*num_rows))
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Sort tokens by performance for better visualization
        df_sorted = self.df.sort_values('fdv_change_pct', ascending=False)
        
        for idx, (_, token) in enumerate(df_sorted.iterrows()):
            row = idx // tokens_per_row
            col = idx % tokens_per_row
            
            if idx >= len(df_sorted):
                axes[row, col].axis('off')
                continue
            
            ax = axes[row, col]
            
            # Create mini analysis for each token
            metrics = ['FDV Change', 'Hold Return', 'Exit Return', 'Risk Score', 'Buy/Sell Ratio']
            values = [
                token['fdv_change_pct'],
                token['current_performance'],
                token['best_exit_performance'],
                token['risk_score'] * 10,  # Scale for visibility
                token['buy_sell_ratio'] * 100  # Scale for visibility
            ]
            
            # Create radar-style chart
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            bars = ax.bar(metrics, values, color=colors, alpha=0.7)
            
            # Customize
            ax.set_title(f"{token['token_name'][:12]}...\n{token['pattern'].replace('_', ' ').title()}", 
                        fontsize=10, fontweight='bold')
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Add performance indicator
            perf_color = self.pattern_colors.get(token['pattern'], '#CCCCCC')
            ax.set_facecolor(perf_color + '20')  # Add transparency
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -1),
                       f'{value:.0f}', ha='center', va='bottom' if height >= 0 else 'top', 
                       fontsize=8, fontweight='bold')
        
        # Hide empty subplots
        for idx in range(len(df_sorted), num_rows * tokens_per_row):
            row = idx // tokens_per_row
            col = idx % tokens_per_row
            axes[row, col].axis('off')
        
        plt.suptitle('🔍 DETAILED TOKEN ANALYSIS GRID', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'token_analysis_grid.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_performance_distributions(self):
        """Create performance distribution charts"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Performance histogram
        ax1.hist(self.df['fdv_change_pct'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=self.df['fdv_change_pct'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.df["fdv_change_pct"].mean():.1f}%')
        ax1.axvline(x=self.df['fdv_change_pct'].median(), color='green', linestyle='--', 
                   label=f'Median: {self.df["fdv_change_pct"].median():.1f}%')
        ax1.set_xlabel('FDV Change (%)')
        ax1.set_ylabel('Number of Tokens')
        ax1.set_title('📊 Performance Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Risk score distribution
        ax2.hist(self.df['risk_score'], bins=15, alpha=0.7, color='orange', edgecolor='black')
        ax2.axvline(x=6, color='red', linestyle='--', label='High Risk Threshold')
        ax2.axvline(x=8, color='darkred', linestyle='--', label='Critical Risk Threshold')
        ax2.set_xlabel('Risk Score')
        ax2.set_ylabel('Number of Tokens')
        ax2.set_title('⚠️ Risk Score Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Buy/Sell ratio distribution
        ax3.hist(self.df['buy_sell_ratio'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.axvline(x=0.6, color='red', linestyle='--', label='Exit Threshold')
        ax3.axvline(x=0.8, color='orange', linestyle='--', label='Warning Threshold')
        ax3.axvline(x=1.0, color='blue', linestyle='--', label='Neutral')
        ax3.set_xlabel('Buy/Sell Ratio')
        ax3.set_ylabel('Number of Tokens')
        ax3.set_title('💹 Buy/Sell Ratio Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Exit improvement distribution
        ax4.hist(self.df['exit_improvement'], bins=15, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(x=0, color='black', linestyle='-', label='No Improvement')
        ax4.axvline(x=self.df['exit_improvement'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.df["exit_improvement"].mean():.1f}%')
        ax4.set_xlabel('Exit Strategy Improvement (%)')
        ax4.set_ylabel('Number of Tokens')
        ax4.set_title('💰 Exit Strategy Improvement Distribution', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_correlation_charts(self):
        """Create correlation analysis charts"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Correlation heatmap
        correlation_columns = ['fdv_change_pct', 'risk_score', 'buy_sell_ratio', 'momentum_change', 
                              'buy_pressure_change', 'current_performance', 'best_exit_performance']
        corr_matrix = self.df[correlation_columns].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
                   cbar_kws={'label': 'Correlation Coefficient'}, ax=ax1)
        ax1.set_title('🔥 Correlation Matrix', fontsize=14, fontweight='bold')
        
        # 2. Performance vs Risk scatter
        scatter = ax2.scatter(self.df['risk_score'], self.df['fdv_change_pct'], 
                             c=self.df['fdv_change_pct'], cmap='RdYlGn', alpha=0.7, s=80)
        ax2.set_xlabel('Risk Score')
        ax2.set_ylabel('FDV Change (%)')
        ax2.set_title('🎯 Performance vs Risk', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='FDV Change (%)')
        
        # 3. Buy/Sell ratio vs Momentum
        scatter2 = ax3.scatter(self.df['buy_sell_ratio'], self.df['momentum_change'], 
                              c=self.df['fdv_change_pct'], cmap='RdYlGn', alpha=0.7, s=80)
        ax3.set_xlabel('Buy/Sell Ratio')
        ax3.set_ylabel('Momentum Change')
        ax3.set_title('💹 Buy/Sell vs Momentum', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax3, label='FDV Change (%)')
        
        # 4. Exit improvement vs original performance
        ax4.scatter(self.df['current_performance'], self.df['exit_improvement'], 
                   c=[self.pattern_colors.get(p, '#CCCCCC') for p in self.df['pattern']], 
                   alpha=0.7, s=80)
        ax4.set_xlabel('Original Performance (%)')
        ax4.set_ylabel('Exit Improvement (%)')
        ax4.set_title('💰 Original vs Exit Improvement', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _generate_chart_summary(self):
        """Generate summary of all charts created"""
        summary_text = f"""
================================================================================
🎨 MASTER TOKEN CHARTS SUMMARY
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Tokens Analyzed: {len(self.df)}

📊 CHARTS CREATED:
✅ individual_token_performance.png - Performance ranking of all 26 tokens
✅ pattern_timing_analysis.png - Pattern and timing distribution analysis
✅ exit_comparison_analysis.png - Exit strategy vs hold comparison
✅ risk_performance_matrix.png - Risk analysis and performance correlation
✅ master_token_dashboard.png - Comprehensive overview dashboard
✅ token_analysis_grid.png - Detailed grid view of each token
✅ performance_distributions.png - Statistical distributions
✅ correlation_analysis.png - Correlation and scatter plots

🎯 KEY INSIGHTS:
• Best Performer: {self.df.loc[self.df['fdv_change_pct'].idxmax(), 'token_name']} ({self.df['fdv_change_pct'].max():+.1f}%)
• Worst Performer: {self.df.loc[self.df['fdv_change_pct'].idxmin(), 'token_name']} ({self.df['fdv_change_pct'].min():+.1f}%)
• Average Performance: {self.df['fdv_change_pct'].mean():+.1f}%
• Exit Strategy Improvement: {self.df['exit_improvement'].mean():+.1f}%
• Point of No Return Tokens: {self.df['hit_point_of_no_return'].sum()}/{len(self.df)}

📁 All charts saved to: {self.output_dir.absolute()}
        """
        
        with open(self.output_dir / 'charts_summary.txt', 'w') as f:
            f.write(summary_text)
        
        print(summary_text)

def main():
    """Main function to create master token charts"""
    print("🎨 Starting Master Token Charts Creation...")
    
    # Create chart creator
    creator = MasterTokenChartsCreator()
    
    # Generate all charts
    creator.create_master_token_charts()
    
    print("\n" + "="*80)
    print("🎉 MASTER TOKEN CHARTS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
