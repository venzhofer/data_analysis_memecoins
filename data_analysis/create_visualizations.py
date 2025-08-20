"""
Token Analysis Visualization Script
Creates comprehensive charts and graphs from the analysis results
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TokenAnalysisVisualizer:
    """Creates visualizations for token analysis results"""
    
    def __init__(self, results_file: str = "output/adapted_timeseries_analysis.json"):
        self.results_file = Path(results_file)
        self.output_dir = Path("output/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.results = self._load_results()
        self.df = self._create_dataframe()
        
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
                    'start_fdv': result.get('start_metrics', {}).get('start_fdv', 0)
                }
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_all_visualizations(self):
        """Create all visualizations"""
        if self.df.empty:
            print("No data to visualize")
            return
        
        print("Creating visualizations...")
        
        # 1. Performance Distribution
        self._create_performance_distribution()
        
        # 2. Pattern Distribution
        self._create_pattern_distribution()
        
        # 3. Risk vs Performance Scatter
        self._create_risk_performance_scatter()
        
        # 4. Momentum vs Performance Scatter
        self._create_momentum_performance_scatter()
        
        # 5. Performance by Risk Level
        self._create_performance_by_risk()
        
        # 6. Top and Bottom Performers
        self._create_top_bottom_performers()
        
        # 7. Risk Level Distribution
        self._create_risk_distribution()
        
        # 8. Momentum Distribution
        self._create_momentum_distribution()
        
        # 9. Performance Histogram
        self._create_performance_histogram()
        
        # 10. Comprehensive Dashboard
        self._create_comprehensive_dashboard()
        
        print(f"All visualizations saved to: {self.output_dir}")
    
    def _create_performance_distribution(self):
        """Create performance distribution chart"""
        plt.figure(figsize=(12, 8))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # FDV Change Distribution
        ax1.hist(self.df['fdv_change_pct'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(self.df['fdv_change_pct'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {self.df["fdv_change_pct"].mean():.1f}%')
        ax1.axvline(self.df['fdv_change_pct'].median(), color='green', linestyle='--', 
                    label=f'Median: {self.df["fdv_change_pct"].median():.1f}%')
        ax1.set_xlabel('FDV Change (%)')
        ax1.set_ylabel('Number of Tokens')
        ax1.set_title('Distribution of FDV Changes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(self.df['fdv_change_pct'], vert=False)
        ax2.set_xlabel('FDV Change (%)')
        ax2.set_title('Box Plot of FDV Changes')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pattern_distribution(self):
        """Create pattern distribution pie chart"""
        plt.figure(figsize=(12, 8))
        
        # Count patterns
        pattern_counts = self.df['pattern'].value_counts()
        
        # Create pie chart
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0', '#ffb366']
        wedges, texts, autotexts = plt.pie(pattern_counts.values, labels=pattern_counts.index, 
                                          autopct='%1.1f%%', colors=colors[:len(pattern_counts)])
        
        plt.title('Token Performance Pattern Distribution', fontsize=16, fontweight='bold')
        plt.axis('equal')
        
        # Add legend
        plt.legend(wedges, pattern_counts.index, title="Patterns", 
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pattern_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_risk_performance_scatter(self):
        """Create risk vs performance scatter plot"""
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot with color coding by pattern
        patterns = self.df['pattern'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(patterns)))
        
        for i, pattern in enumerate(patterns):
            mask = self.df['pattern'] == pattern
            plt.scatter(self.df[mask]['risk_score'], self.df[mask]['fdv_change_pct'], 
                       c=[colors[i]], label=pattern, alpha=0.7, s=100)
        
        plt.xlabel('Risk Score (1-10)', fontsize=12)
        plt.ylabel('FDV Change (%)', fontsize=12)
        plt.title('Risk Score vs Performance', fontsize=16, fontweight='bold')
        plt.legend(title='Performance Pattern', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(self.df['risk_score'], self.df['fdv_change_pct'], 1)
        p = np.poly1d(z)
        plt.plot(self.df['risk_score'], p(self.df['risk_score']), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_performance_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_momentum_performance_scatter(self):
        """Create momentum vs performance scatter plot"""
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot with color coding by risk level
        risk_levels = self.df['risk_level'].unique()
        colors = ['green', 'orange', 'red']
        
        for i, level in enumerate(risk_levels):
            mask = self.df['risk_level'] == level
            plt.scatter(self.df[mask]['momentum_score'], self.df[mask]['fdv_change_pct'], 
                       c=colors[i], label=level, alpha=0.7, s=100)
        
        plt.xlabel('Momentum Score (0-3)', fontsize=12)
        plt.ylabel('FDV Change (%)', fontsize=12)
        plt.title('Momentum Score vs Performance', fontsize=16, fontweight='bold')
        plt.legend(title='Risk Level', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(self.df['momentum_score'], self.df['fdv_change_pct'], 1)
        p = np.poly1d(z)
        plt.plot(self.df['momentum_score'], p(self.df['momentum_score']), "b--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'momentum_performance_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_performance_by_risk(self):
        """Create performance comparison by risk level"""
        plt.figure(figsize=(12, 8))
        
        # Group by risk level and calculate statistics
        risk_stats = self.df.groupby('risk_level')['fdv_change_pct'].agg(['mean', 'std', 'count']).reset_index()
        
        # Create bar chart
        bars = plt.bar(risk_stats['risk_level'], risk_stats['mean'], 
                      yerr=risk_stats['std'], capsize=5, alpha=0.7, 
                      color=['green', 'orange', 'red'])
        
        # Add value labels on bars
        for bar, count in zip(bars, risk_stats['count']):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'n={count}', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Risk Level', fontsize=12)
        plt.ylabel('Average FDV Change (%)', fontsize=12)
        plt.title('Average Performance by Risk Level', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_by_risk.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_top_bottom_performers(self):
        """Create top and bottom performers chart"""
        plt.figure(figsize=(14, 10))
        
        # Get top 10 and bottom 10 performers
        top_10 = self.df.nlargest(10, 'fdv_change_pct')
        bottom_10 = self.df.nsmallest(10, 'fdv_change_pct')
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Top performers
        bars1 = ax1.barh(range(len(top_10)), top_10['fdv_change_pct'], 
                         color='green', alpha=0.7)
        ax1.set_yticks(range(len(top_10)))
        ax1.set_yticklabels([f"{name[:15]}..." if len(name) > 15 else name 
                             for name in top_10['token_name']])
        ax1.set_xlabel('FDV Change (%)')
        ax1.set_title('Top 10 Performers', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%', ha='left', va='center')
        
        # Bottom performers
        bars2 = ax2.barh(range(len(bottom_10)), bottom_10['fdv_change_pct'], 
                         color='red', alpha=0.7)
        ax2.set_yticks(range(len(bottom_10)))
        ax2.set_yticklabels([f"{name[:15]}..." if len(name) > 15 else name 
                             for name in bottom_10['token_name']])
        ax2.set_xlabel('FDV Change (%)')
        ax2.set_title('Bottom 10 Performers', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax2.text(width - 1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%', ha='right', va='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_bottom_performers.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_risk_distribution(self):
        """Create risk distribution chart"""
        plt.figure(figsize=(12, 8))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Risk score distribution
        ax1.hist(self.df['risk_score'], bins=10, alpha=0.7, color='orange', edgecolor='black')
        ax1.axvline(self.df['risk_score'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {self.df["risk_score"].mean():.1f}')
        ax1.set_xlabel('Risk Score')
        ax1.set_ylabel('Number of Tokens')
        ax1.set_title('Distribution of Risk Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Risk level pie chart
        risk_level_counts = self.df['risk_level'].value_counts()
        colors = ['green', 'orange', 'red']
        wedges, texts, autotexts = ax2.pie(risk_level_counts.values, labels=risk_level_counts.index, 
                                          autopct='%1.1f%%', colors=colors[:len(risk_level_counts)])
        ax2.set_title('Risk Level Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_momentum_distribution(self):
        """Create momentum distribution chart"""
        plt.figure(figsize=(12, 8))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Momentum score distribution
        ax1.hist(self.df['momentum_score'], bins=4, alpha=0.7, color='purple', edgecolor='black')
        ax1.axvline(self.df['momentum_score'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {self.df["momentum_score"].mean():.1f}')
        ax1.set_xlabel('Momentum Score')
        ax1.set_ylabel('Number of Tokens')
        ax1.set_title('Distribution of Momentum Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Momentum level pie chart
        momentum_level_counts = self.df['momentum_level'].value_counts()
        colors = ['red', 'orange', 'green']
        wedges, texts, autotexts = ax2.pie(momentum_level_counts.values, labels=momentum_level_counts.index, 
                                          autopct='%1.1f%%', colors=colors[:len(momentum_level_counts)])
        ax2.set_title('Momentum Level Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'momentum_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_performance_histogram(self):
        """Create detailed performance histogram with pattern overlay"""
        plt.figure(figsize=(14, 8))
        
        # Create histogram with pattern overlay
        patterns = self.df['pattern'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(patterns)))
        
        # Plot histogram for each pattern
        for i, pattern in enumerate(patterns):
            mask = self.df['pattern'] == pattern
            plt.hist(self.df[mask]['fdv_change_pct'], bins=20, alpha=0.6, 
                    label=pattern, color=colors[i], edgecolor='black')
        
        plt.xlabel('FDV Change (%)', fontsize=12)
        plt.ylabel('Number of Tokens', fontsize=12)
        plt.title('Performance Distribution by Pattern', fontsize=16, fontweight='bold')
        plt.legend(title='Performance Pattern', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add vertical lines for key thresholds
        plt.axvline(100, color='red', linestyle='--', alpha=0.7, label='100% (Moon Shot)')
        plt.axvline(50, color='orange', linestyle='--', alpha=0.7, label='50% (Strong Rise)')
        plt.axvline(20, color='yellow', linestyle='--', alpha=0.7, label='20% (Moderate Rise)')
        plt.axvline(-20, color='yellow', linestyle='--', alpha=0.7, label='-20% (Moderate Drop)')
        plt.axvline(-50, color='orange', linestyle='--', alpha=0.7, label='-50% (Significant Drop)')
        plt.axvline(-80, color='red', linestyle='--', alpha=0.7, label='-80% (Died)')
        
        plt.legend(title='Thresholds', bbox_to_anchor=(1.05, 0), loc='lower left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard with multiple charts"""
        plt.figure(figsize=(20, 16))
        
        # Create subplot grid
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Performance distribution (top left)
        ax1 = plt.subplot(3, 3, 1)
        ax1.hist(self.df['fdv_change_pct'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Performance Distribution', fontweight='bold')
        ax1.set_xlabel('FDV Change (%)')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
        
        # 2. Pattern distribution (top center)
        ax2 = plt.subplot(3, 3, 2)
        pattern_counts = self.df['pattern'].value_counts()
        ax2.pie(pattern_counts.values, labels=pattern_counts.index, autopct='%1.1f%%')
        ax2.set_title('Pattern Distribution', fontweight='bold')
        
        # 3. Risk vs Performance (top right)
        ax3 = plt.subplot(3, 3, 3)
        scatter = ax3.scatter(self.df['risk_score'], self.df['fdv_change_pct'], 
                             c=self.df['momentum_score'], cmap='viridis', alpha=0.7)
        ax3.set_xlabel('Risk Score')
        ax3.set_ylabel('FDV Change (%)')
        ax3.set_title('Risk vs Performance', fontweight='bold')
        plt.colorbar(scatter, ax=ax3, label='Momentum Score')
        
        # 4. Performance by Risk Level (middle left)
        ax4 = plt.subplot(3, 3, 4)
        risk_performance = self.df.groupby('risk_level')['fdv_change_pct'].mean()
        bars = ax4.bar(risk_performance.index, risk_performance.values, 
                       color=['green', 'orange', 'red'], alpha=0.7)
        ax4.set_title('Avg Performance by Risk', fontweight='bold')
        ax4.set_ylabel('Avg FDV Change (%)')
        
        # 5. Momentum vs Performance (middle center)
        ax5 = plt.subplot(3, 3, 5)
        ax5.scatter(self.df['momentum_score'], self.df['fdv_change_pct'], alpha=0.7)
        ax5.set_xlabel('Momentum Score')
        ax5.set_ylabel('FDV Change (%)')
        ax5.set_title('Momentum vs Performance', fontweight='bold')
        
        # 6. Risk Level Distribution (middle right)
        ax6 = plt.subplot(3, 3, 6)
        risk_level_counts = self.df['risk_level'].value_counts()
        ax6.pie(risk_level_counts.values, labels=risk_level_counts.index, autopct='%1.1f%%')
        ax6.set_title('Risk Level Distribution', fontweight='bold')
        
        # 7. Top Performers (bottom left)
        ax7 = plt.subplot(3, 3, 7)
        top_5 = self.df.nlargest(5, 'fdv_change_pct')
        bars = ax7.barh(range(len(top_5)), top_5['fdv_change_pct'], color='green', alpha=0.7)
        ax7.set_yticks(range(len(top_5)))
        ax7.set_yticklabels([name[:12] + "..." if len(name) > 12 else name 
                             for name in top_5['token_name']])
        ax7.set_xlabel('FDV Change (%)')
        ax7.set_title('Top 5 Performers', fontweight='bold')
        
        # 8. Bottom Performers (bottom center)
        ax8 = plt.subplot(3, 3, 8)
        bottom_5 = self.df.nsmallest(5, 'fdv_change_pct')
        bars = ax8.barh(range(len(bottom_5)), bottom_5['fdv_change_pct'], color='red', alpha=0.7)
        ax8.set_yticks(range(len(bottom_5)))
        ax8.set_yticklabels([name[:12] + "..." if len(name) > 12 else name 
                             for name in bottom_5['token_name']])
        ax8.set_xlabel('FDV Change (%)')
        ax8.set_title('Bottom 5 Performers', fontweight='bold')
        
        # 9. Summary Statistics (bottom right)
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Calculate summary statistics
        total_tokens = len(self.df)
        positive_tokens = len(self.df[self.df['fdv_change_pct'] > 0])
        negative_tokens = len(self.df[self.df['fdv_change_pct'] < 0])
        avg_change = self.df['fdv_change_pct'].mean()
        avg_risk = self.df['risk_score'].mean()
        avg_momentum = self.df['momentum_score'].mean()
        
        summary_text = f"""
        SUMMARY STATISTICS
        
        Total Tokens: {total_tokens}
        Positive Performance: {positive_tokens} ({positive_tokens/total_tokens*100:.1f}%)
        Negative Performance: {negative_tokens} ({negative_tokens/total_tokens*100:.1f}%)
        
        Average FDV Change: {avg_change:.1f}%
        Average Risk Score: {avg_risk:.1f}/10
        Average Momentum: {avg_momentum:.1f}/3
        
        Success Rate: {positive_tokens/total_tokens*100:.1f}%
        """
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Token Analysis Dashboard', fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        plt.savefig(self.output_dir / 'comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to create all visualizations"""
    try:
        # Initialize visualizer
        visualizer = TokenAnalysisVisualizer()
        
        # Create all visualizations
        visualizer.create_all_visualizations()
        
        print("✅ All visualizations created successfully!")
        print(f"📁 Check the 'output/visualizations' folder for the charts")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
