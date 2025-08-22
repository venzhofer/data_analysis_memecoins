#!/usr/bin/env python3
"""
All Tokens Line Chart Creator
Creates a single line chart showing all tokens' performance metrics together
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

class AllTokensLineChartCreator:
    """Creates a single line chart showing all tokens together"""
    
    def __init__(self, results_file: str = "output/adapted_timeseries_analysis.json"):
        self.results_file = Path(results_file)
        self.output_dir = Path("output/all_tokens_chart")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.results = self._load_results()
        self.df = self._create_dataframe()
        
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
        
        # Risk colors
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
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Create a comprehensive dataframe from results"""
        if not self.results or 'results' not in self.results:
            return pd.DataFrame()
        
        data = []
        for address, token_data in self.results['results'].items():
            if token_data.get('status') == 'analyzed':
                row = {
                    'address': address,
                    'token_name': token_data.get('token_name', 'Unknown'),
                    'pattern': token_data.get('pattern', 'unknown'),
                    'fdv_change_pct': token_data.get('performance_metrics', {}).get('fdv_change_pct', 0),
                    'market_cap_change_pct': token_data.get('performance_metrics', {}).get('market_cap_change_pct', 0),
                    'risk_level': token_data.get('risk_metrics', {}).get('risk_level', 'unknown'),
                    'total_risk_score': token_data.get('risk_metrics', {}).get('total_risk_score', 0),
                    'momentum_score': token_data.get('momentum_metrics', {}).get('momentum_score', 0),
                    'momentum_level': token_data.get('momentum_metrics', {}).get('momentum_level', 'unknown'),
                    'buy_sell_ratio': token_data.get('transaction_analysis', {}).get('overall', {}).get('buy_sell_ratio', 0),
                    'total_transactions': token_data.get('transaction_analysis', {}).get('overall', {}).get('total_transactions', 0)
                }
                data.append(row)
        
        return pd.DataFrame(data)
    
    def create_all_tokens_line_chart(self):
        """Create a single line chart showing all tokens together"""
        if self.df.empty:
            print("No data to visualize")
            return
        
        # Sort tokens by performance for better visualization
        df_sorted = self.df.sort_values('fdv_change_pct', ascending=False)
        
        # Create the main line chart
        plt.figure(figsize=(20, 12))
        
        # Create x-axis positions (token indices)
        x_positions = np.arange(len(df_sorted))
        
        # Plot FDV change percentage for each token
        for i, (_, token) in enumerate(df_sorted.iterrows()):
            # Get color based on pattern
            color = self.pattern_colors.get(token['pattern'], '#CCCCCC')
            
            # Plot point
            plt.scatter(i, token['fdv_change_pct'], 
                       color=color, s=100, alpha=0.8, 
                       edgecolors='black', linewidth=1)
            
            # Add token name label
            plt.annotate(f"{token['token_name'][:15]}...", 
                        (i, token['fdv_change_pct']),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, rotation=45,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Connect points with lines
        plt.plot(x_positions, df_sorted['fdv_change_pct'], 
                color='blue', alpha=0.3, linewidth=1, linestyle='--')
        
        # Add horizontal reference lines
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        plt.axhline(y=100, color='green', linestyle='--', alpha=0.7, linewidth=1, label='100% (Moon Shot)')
        plt.axhline(y=50, color='orange', linestyle='--', alpha=0.7, linewidth=1, label='50% (Strong Rise)')
        plt.axhline(y=20, color='yellow', linestyle='--', alpha=0.7, linewidth=1, label='20% (Moderate Rise)')
        plt.axhline(y=-20, color='red', linestyle='--', alpha=0.7, linewidth=1, label='-20% (Moderate Drop)')
        plt.axhline(y=-50, color='darkred', linestyle='--', alpha=0.7, linewidth=1, label='-50% (Significant Drop)')
        plt.axhline(y=-80, color='maroon', linestyle='--', alpha=0.7, linewidth=1, label='-80% (Died)')
        
        # Customize the chart
        plt.title('All Tokens Performance Overview - FDV Change Percentage', 
                 fontsize=20, fontweight='bold', pad=20)
        plt.xlabel('Token Index (Sorted by Performance)', fontsize=14, fontweight='bold')
        plt.ylabel('FDV Change Percentage (%)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Set y-axis limits with some padding
        y_min = min(df_sorted['fdv_change_pct'].min(), -100)
        y_max = max(df_sorted['fdv_change_pct'].max(), 100)
        plt.ylim(y_min - 20, y_max + 20)
        
        # Create legend for patterns
        pattern_legend_elements = []
        for pattern, color in self.pattern_colors.items():
            if pattern in df_sorted['pattern'].values:
                pattern_legend_elements.append(
                    mpatches.Patch(color=color, label=pattern.replace('_', ' ').title())
                )
        
        # Add legends
        plt.legend(handles=pattern_legend_elements, 
                  title='Performance Patterns', 
                  loc='upper right', 
                  bbox_to_anchor=(1.15, 1))
        
        # Add performance summary
        total_tokens = len(df_sorted)
        moon_shot_count = len(df_sorted[df_sorted['pattern'] == 'moon_shot'])
        died_count = len(df_sorted[df_sorted['pattern'] == 'died'])
        positive_count = len(df_sorted[df_sorted['fdv_change_pct'] > 0])
        negative_count = len(df_sorted[df_sorted['fdv_change_pct'] < 0])
        
        summary_text = f"""Performance Summary:
Total Tokens: {total_tokens}
Moon Shot (>100%): {moon_shot_count}
Positive Performance: {positive_count}
Negative Performance: {negative_count}
Died Tokens: {died_count}"""
        
        plt.figtext(0.02, 0.02, summary_text, 
                   fontsize=10, bbox=dict(boxstyle='round,pad=0.5', 
                                        facecolor='lightblue', alpha=0.8))
        
        # Rotate x-axis labels for better readability
        plt.xticks(x_positions[::max(1, len(x_positions)//20)], 
                  [f"{i+1}" for i in x_positions[::max(1, len(x_positions)//20)]], 
                  rotation=0)
        
        plt.tight_layout()
        
        # Save the chart
        output_file = self.output_dir / 'all_tokens_performance_line_chart.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {output_file}")
        
        # Display the chart
        plt.show()
        
        # Save summary statistics
        self._save_summary_stats(df_sorted)
    
    def _save_summary_stats(self, df_sorted: pd.DataFrame):
        """Save summary statistics to a text file"""
        summary_file = self.output_dir / 'all_tokens_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("ALL TOKENS PERFORMANCE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Tokens Analyzed: {len(df_sorted)}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("PERFORMANCE PATTERNS:\n")
            f.write("-" * 30 + "\n")
            pattern_counts = df_sorted['pattern'].value_counts()
            for pattern, count in pattern_counts.items():
                percentage = (count / len(df_sorted)) * 100
                f.write(f"{pattern.replace('_', ' ').title()}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\nPERFORMANCE STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average FDV Change: {df_sorted['fdv_change_pct'].mean():.2f}%\n")
            f.write(f"Median FDV Change: {df_sorted['fdv_change_pct'].median():.2f}%\n")
            f.write(f"Best Performer: {df_sorted.iloc[0]['token_name']} ({df_sorted.iloc[0]['fdv_change_pct']:.2f}%)\n")
            f.write(f"Worst Performer: {df_sorted.iloc[-1]['token_name']} ({df_sorted.iloc[-1]['fdv_change_pct']:.2f}%)\n")
            f.write(f"Standard Deviation: {df_sorted['fdv_change_pct'].std():.2f}%\n")
            
            f.write(f"\nRISK ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            risk_counts = df_sorted['risk_level'].value_counts()
            for risk_level, count in risk_counts.items():
                percentage = (count / len(df_sorted)) * 100
                f.write(f"{risk_level.title()}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\nTOP 10 PERFORMERS:\n")
            f.write("-" * 30 + "\n")
            for i, (_, token) in enumerate(df_sorted.head(10).iterrows()):
                f.write(f"{i+1}. {token['token_name']}: {token['fdv_change_pct']:.2f}% ({token['pattern']})\n")
            
            f.write(f"\nBOTTOM 10 PERFORMERS:\n")
            f.write("-" * 30 + "\n")
            for i, (_, token) in enumerate(df_sorted.tail(10).iterrows()):
                f.write(f"{i+1}. {token['token_name']}: {token['fdv_change_pct']:.2f}% ({token['pattern']})\n")
        
        print(f"Summary statistics saved to: {summary_file}")
    
    def create_alternative_chart(self):
        """Create an alternative visualization showing tokens by category"""
        if self.df.empty:
            print("No data to visualize")
            return
        
        # Group tokens by pattern
        pattern_groups = self.df.groupby('pattern')
        
        plt.figure(figsize=(16, 10))
        
        # Create subplots for different visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('All Tokens Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Performance by Pattern (Box Plot)
        pattern_data = [group['fdv_change_pct'].values for name, group in pattern_groups]
        pattern_names = [name.replace('_', ' ').title() for name in pattern_groups.groups.keys()]
        
        bp = ax1.boxplot(pattern_data, labels=pattern_names, patch_artist=True)
        colors = [self.pattern_colors.get(pattern.lower().replace(' ', '_'), '#CCCCCC') for pattern in pattern_names]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('Performance Distribution by Pattern', fontweight='bold')
        ax1.set_ylabel('FDV Change (%)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Risk vs Performance Scatter
        scatter = ax2.scatter(self.df['total_risk_score'], self.df['fdv_change_pct'], 
                             c=self.df['fdv_change_pct'], cmap='RdYlGn', 
                             s=100, alpha=0.7, edgecolors='black')
        ax2.set_xlabel('Risk Score')
        ax2.set_ylabel('FDV Change (%)')
        ax2.set_title('Risk vs Performance', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.colorbar(scatter, ax=ax2, label='FDV Change (%)')
        
        # 3. Performance Histogram
        ax3.hist(self.df['fdv_change_pct'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_xlabel('FDV Change (%)')
        ax3.set_ylabel('Number of Tokens')
        ax3.set_title('Performance Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
        ax3.legend()
        
        # 4. Pattern Counts
        pattern_counts = self.df['pattern'].value_counts()
        colors = [self.pattern_colors.get(pattern, '#CCCCCC') for pattern in pattern_counts.index]
        bars = ax4.bar(range(len(pattern_counts)), pattern_counts.values, color=colors, alpha=0.7)
        ax4.set_xlabel('Performance Pattern')
        ax4.set_ylabel('Number of Tokens')
        ax4.set_title('Token Count by Pattern', fontweight='bold')
        ax4.set_xticks(range(len(pattern_counts)))
        ax4.set_xticklabels([p.replace('_', ' ').title() for p in pattern_counts.index], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, pattern_counts.values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the dashboard
        output_file = self.output_dir / 'all_tokens_dashboard.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to: {output_file}")
        
        plt.show()

def main():
    """Main function to create the all tokens line chart"""
    print("Creating All Tokens Line Chart...")
    
    creator = AllTokensLineChartCreator()
    
    # Create the main line chart
    creator.create_all_tokens_line_chart()
    
    # Create alternative dashboard
    creator.create_alternative_chart()
    
    print("All visualizations completed!")

if __name__ == "__main__":
    main()
