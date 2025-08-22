"""
Token Correlation and Pattern Analysis
Analyzes relationships between different metrics and identifies patterns
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TokenCorrelationAnalyzer:
    """Analyzes correlations and patterns in token data"""
    
    def __init__(self, results_file: str = "output/adapted_timeseries_analysis.json"):
        self.results_file = Path(results_file)
        self.output_dir = Path("output/correlation_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.results = self._load_results()
        self.df = self._create_dataframe()
        
        # Analysis results
        self.correlation_matrix = None
        self.pattern_insights = {}
        self.cluster_analysis = {}
        
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
            df['risk_performance_ratio'] = df['risk_score'] / (df['fdv_change_pct'] + 100)  # Avoid division by zero
            df['momentum_efficiency'] = df['momentum_score'] / (df['fdv_change_pct'] + 100)
            df['transaction_intensity'] = df['total_transactions'] / (df['start_fdv'] + 1)
            df['buy_pressure'] = df['buy_percentage'] - 50  # Deviation from 50%
            
        return df
    
    def run_comprehensive_analysis(self):
        """Run all correlation and pattern analyses"""
        if self.df.empty:
            print("No data to analyze")
            return
        
        print("🔍 Starting comprehensive correlation and pattern analysis...")
        
        # 1. Correlation Analysis
        self._analyze_correlations()
        
        # 2. Pattern Discovery
        self._discover_patterns()
        
        # 3. Cluster Analysis
        self._perform_cluster_analysis()
        
        # 4. Risk-Reward Analysis
        self._analyze_risk_reward()
        
        # 5. Performance Segmentation
        self._segment_performance()
        
        # 6. Transaction Pattern Analysis
        self._analyze_transaction_patterns()
        
        # 7. Create Visualizations
        self._create_correlation_visualizations()
        
        # 8. Generate Insights Report
        self._generate_insights_report()
        
        print(f"✅ Analysis complete! Results saved to: {self.output_dir}")
    
    def _analyze_correlations(self):
        """Analyze correlations between all numeric variables"""
        print("📊 Analyzing correlations...")
        
        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_df = self.df[numeric_cols].copy()
        
        # Remove columns with too many zeros or constant values
        valid_cols = []
        for col in numeric_cols:
            if numeric_df[col].nunique() > 1 and numeric_df[col].std() > 0:
                valid_cols.append(col)
        
        numeric_df = numeric_df[valid_cols]
        
        # Calculate correlation matrix
        self.correlation_matrix = numeric_df.corr()
        
        # Find significant correlations (|r| > 0.3)
        significant_correlations = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                col1 = self.correlation_matrix.columns[i]
                col2 = self.correlation_matrix.columns[j]
                corr_value = self.correlation_matrix.iloc[i, j]
                
                if abs(corr_value) > 0.3:
                    significant_correlations.append({
                        'variable1': col1,
                        'variable2': col2,
                        'correlation': corr_value,
                        'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate' if abs(corr_value) > 0.5 else 'weak'
                    })
        
        # Sort by absolute correlation value
        significant_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        self.pattern_insights['correlations'] = {
            'matrix': self.correlation_matrix.to_dict(),
            'significant_correlations': significant_correlations,
            'strongest_positive': max(significant_correlations, key=lambda x: x['correlation']) if significant_correlations else None,
            'strongest_negative': min(significant_correlations, key=lambda x: x['correlation']) if significant_correlations else None
        }
        
        print(f"📈 Found {len(significant_correlations)} significant correlations")
    
    def _discover_patterns(self):
        """Discover patterns in the data"""
        print("🔍 Discovering patterns...")
        
        patterns = {}
        
        # 1. Performance patterns by risk level
        risk_performance = self.df.groupby('risk_level')['fdv_change_pct'].agg(['mean', 'std', 'count']).round(2)
        patterns['risk_performance'] = risk_performance.to_dict()
        
        # 2. Performance patterns by momentum level
        momentum_performance = self.df.groupby('momentum_level')['fdv_change_pct'].agg(['mean', 'std', 'count']).round(2)
        patterns['momentum_performance'] = momentum_performance.to_dict()
        
        # 3. Risk patterns by performance category
        performance_risk = self.df.groupby('pattern')['risk_score'].agg(['mean', 'std', 'count']).round(2)
        patterns['performance_risk'] = performance_risk.to_dict()
        
        # 4. Transaction patterns by performance
        if 'total_transactions' in self.df.columns:
            transaction_performance = self.df.groupby('pattern')['total_transactions'].agg(['mean', 'std', 'count']).round(2)
            patterns['transaction_performance'] = transaction_performance.to_dict()
        
        # 5. Buy pressure patterns
        if 'buy_pressure' in self.df.columns:
            buy_pressure_performance = self.df.groupby('pattern')['buy_pressure'].agg(['mean', 'std', 'count']).round(2)
            patterns['buy_pressure_performance'] = buy_pressure_performance.to_dict()
        
        # 6. Size effect (start FDV vs performance)
        if 'start_fdv' in self.df.columns:
            # Create size categories
            self.df['size_category'] = pd.cut(self.df['start_fdv'], 
                                            bins=[0, 10000, 50000, 100000, np.inf], 
                                            labels=['Micro', 'Small', 'Medium', 'Large'])
            size_performance = self.df.groupby('size_category')['fdv_change_pct'].agg(['mean', 'std', 'count']).round(2)
            patterns['size_performance'] = size_performance.to_dict()
        
        self.pattern_insights['patterns'] = patterns
        print("🎯 Pattern discovery complete")
    
    def _perform_cluster_analysis(self):
        """Perform K-means clustering to identify token groups"""
        print("🎪 Performing cluster analysis...")
        
        # Select features for clustering
        cluster_features = ['fdv_change_pct', 'risk_score', 'momentum_score']
        if 'total_transactions' in self.df.columns:
            cluster_features.append('total_transactions')
        if 'buy_pressure' in self.df.columns:
            cluster_features.append('buy_pressure')
        
        # Prepare data
        cluster_data = self.df[cluster_features].copy()
        cluster_data = cluster_data.fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)
        
        # Perform clustering with different numbers of clusters
        best_k = 3  # Start with 3 clusters
        best_score = -np.inf
        
        for k in range(2, 6):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(cluster_data_scaled)
            score = kmeans.inertia_
            if score > best_score:
                best_score = score
                best_k = k
        
        # Final clustering
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(cluster_data_scaled)
        
        # Add cluster labels to dataframe
        self.df['cluster'] = cluster_labels
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(best_k):
            cluster_mask = self.df['cluster'] == cluster_id
            cluster_data_subset = self.df[cluster_mask]
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data_subset),
                'percentage': len(cluster_data_subset) / len(self.df) * 100,
                'avg_fdv_change': cluster_data_subset['fdv_change_pct'].mean(),
                'avg_risk_score': cluster_data_subset['risk_score'].mean(),
                'avg_momentum_score': cluster_data_subset['momentum_score'].mean(),
                'common_patterns': cluster_data_subset['pattern'].value_counts().head(3).to_dict(),
                'sample_tokens': cluster_data_subset['token_name'].head(5).tolist()
            }
        
        self.cluster_analysis = {
            'n_clusters': best_k,
            'cluster_details': cluster_analysis,
            'cluster_labels': cluster_labels.tolist()
        }
        
        print(f"🎪 Clustering complete: {best_k} clusters identified")
    
    def _analyze_risk_reward(self):
        """Analyze risk-reward relationships"""
        print("⚖️ Analyzing risk-reward relationships...")
        
        risk_reward_insights = {}
        
        # 1. Risk-reward scatter analysis
        risk_reward_data = self.df[['risk_score', 'fdv_change_pct']].copy()
        risk_reward_data = risk_reward_data.dropna()
        
        if len(risk_reward_data) > 0:
            # Calculate risk-adjusted returns
            risk_reward_data['risk_adjusted_return'] = risk_reward_data['fdv_change_pct'] / (risk_reward_data['risk_score'] + 1)
            
            # Find best risk-adjusted performers
            best_risk_adjusted = risk_reward_data.nlargest(5, 'risk_adjusted_return')
            worst_risk_adjusted = risk_reward_data.nsmallest(5, 'risk_adjusted_return')
            
            risk_reward_insights['risk_adjusted_performance'] = {
                'best_5': best_risk_adjusted.to_dict('records'),
                'worst_5': worst_risk_adjusted.to_dict('records'),
                'avg_risk_adjusted_return': risk_reward_data['risk_adjusted_return'].mean()
            }
        
        # 2. Risk categories analysis
        risk_categories = {
            'low_risk': self.df[self.df['risk_score'] <= 3],
            'medium_risk': self.df[(self.df['risk_score'] > 3) & (self.df['risk_score'] <= 7)],
            'high_risk': self.df[self.df['risk_score'] > 7]
        }
        
        risk_category_analysis = {}
        for category, data in risk_categories.items():
            if len(data) > 0:
                risk_category_analysis[category] = {
                    'count': len(data),
                    'avg_fdv_change': data['fdv_change_pct'].mean(),
                    'success_rate': len(data[data['fdv_change_pct'] > 0]) / len(data) * 100,
                    'avg_momentum': data['momentum_score'].mean() if 'momentum_score' in data.columns else 0
                }
        
        risk_reward_insights['risk_categories'] = risk_category_analysis
        
        # 3. Risk-reward efficiency frontier
        if len(risk_reward_data) > 0:
            # Find Pareto optimal points (best risk-reward combinations)
            pareto_points = []
            for _, point in risk_reward_data.iterrows():
                is_pareto = True
                for _, other_point in risk_reward_data.iterrows():
                    if (other_point['risk_score'] <= point['risk_score'] and 
                        other_point['fdv_change_pct'] > point['fdv_change_pct']):
                        is_pareto = False
                        break
                if is_pareto:
                    pareto_points.append(point.to_dict())
            
            risk_reward_insights['pareto_frontier'] = pareto_points
        
        self.pattern_insights['risk_reward'] = risk_reward_insights
        print("⚖️ Risk-reward analysis complete")
    
    def _segment_performance(self):
        """Segment tokens by performance characteristics"""
        print("📊 Segmenting performance...")
        
        # Create performance segments
        self.df['performance_segment'] = pd.cut(self.df['fdv_change_pct'], 
                                              bins=[-np.inf, -50, -20, 0, 20, 50, 100, np.inf],
                                              labels=['Died', 'Significant Drop', 'Moderate Drop', 
                                                     'Stable', 'Moderate Rise', 'Strong Rise', 'Moon Shot'])
        
        # Analyze each segment
        segment_analysis = {}
        for segment in self.df['performance_segment'].unique():
            if pd.notna(segment):
                segment_data = self.df[self.df['performance_segment'] == segment]
                segment_analysis[segment] = {
                    'count': len(segment_data),
                    'percentage': len(segment_data) / len(self.df) * 100,
                    'avg_risk_score': segment_data['risk_score'].mean(),
                    'avg_momentum_score': segment_data['momentum_score'].mean(),
                    'avg_start_fdv': segment_data['start_fdv'].mean(),
                    'common_patterns': segment_data['pattern'].value_counts().head(3).to_dict()
                }
        
        self.pattern_insights['performance_segments'] = segment_analysis
        print("📊 Performance segmentation complete")
    
    def _analyze_transaction_patterns(self):
        """Analyze transaction patterns and their relationship to performance"""
        print("💱 Analyzing transaction patterns...")
        
        if 'total_transactions' not in self.df.columns:
            print("⚠️ Transaction data not available")
            return
        
        transaction_insights = {}
        
        # 1. Transaction volume vs performance
        transaction_performance = self.df[['total_transactions', 'fdv_change_pct']].copy()
        transaction_performance = transaction_performance.dropna()
        
        if len(transaction_performance) > 0:
            # Calculate correlation
            corr = transaction_performance['total_transactions'].corr(transaction_performance['fdv_change_pct'])
            transaction_insights['volume_performance_correlation'] = corr
            
            # High vs low volume analysis
            median_transactions = transaction_performance['total_transactions'].median()
            high_volume = transaction_performance[transaction_performance['total_transactions'] > median_transactions]
            low_volume = transaction_performance[transaction_performance['total_transactions'] <= median_transactions]
            
            transaction_insights['volume_analysis'] = {
                'high_volume_count': len(high_volume),
                'low_volume_count': len(low_volume),
                'high_volume_avg_performance': high_volume['fdv_change_pct'].mean(),
                'low_volume_avg_performance': low_volume['fdv_change_pct'].mean(),
                'volume_performance_difference': high_volume['fdv_change_pct'].mean() - low_volume['fdv_change_pct'].mean()
            }
        
        # 2. Buy pressure analysis
        if 'buy_pressure' in self.df.columns:
            buy_pressure_data = self.df[['buy_pressure', 'fdv_change_pct']].copy()
            buy_pressure_data = buy_pressure_data.dropna()
            
            if len(buy_pressure_data) > 0:
                buy_pressure_corr = buy_pressure_data['buy_pressure'].corr(buy_pressure_data['fdv_change_pct'])
                transaction_insights['buy_pressure_correlation'] = buy_pressure_corr
                
                # Buy pressure categories
                buy_pressure_data['pressure_category'] = pd.cut(buy_pressure_data['buy_pressure'],
                                                              bins=[-np.inf, -20, 0, 20, np.inf],
                                                              labels=['Strong Sell', 'Sell', 'Buy', 'Strong Buy'])
                
                pressure_analysis = buy_pressure_data.groupby('pressure_category')['fdv_change_pct'].agg(['mean', 'count'])
                transaction_insights['buy_pressure_analysis'] = pressure_analysis.to_dict()
        
        self.pattern_insights['transaction_patterns'] = transaction_insights
        print("💱 Transaction pattern analysis complete")
    
    def _create_correlation_visualizations(self):
        """Create visualizations for correlation analysis"""
        print("📊 Creating correlation visualizations...")
        
        # 1. Correlation heatmap
        plt.figure(figsize=(12, 10))
        if self.correlation_matrix is not None:
            mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))
            sns.heatmap(self.correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            plt.title('Token Metrics Correlation Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Scatter plot matrix for key variables
        key_vars = ['fdv_change_pct', 'risk_score', 'momentum_score']
        if 'total_transactions' in self.df.columns:
            key_vars.append('total_transactions')
        
        key_data = self.df[key_vars].dropna()
        if len(key_data) > 0:
            plt.figure(figsize=(15, 12))
            sns.pairplot(key_data, diag_kind='kde')
            plt.suptitle('Key Variables Scatter Plot Matrix', y=1.02, fontsize=16, fontweight='bold')
            plt.savefig(self.output_dir / 'scatter_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Cluster visualization
        if 'cluster' in self.df.columns:
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(self.df['risk_score'], self.df['fdv_change_pct'], 
                                c=self.df['cluster'], cmap='viridis', alpha=0.7, s=100)
            plt.xlabel('Risk Score')
            plt.ylabel('FDV Change (%)')
            plt.title('Token Clusters: Risk vs Performance', fontsize=16, fontweight='bold')
            plt.colorbar(scatter, label='Cluster')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / 'cluster_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Performance by risk level
        plt.figure(figsize=(12, 8))
        risk_performance = self.df.groupby('risk_level')['fdv_change_pct'].agg(['mean', 'std']).reset_index()
        bars = plt.bar(risk_performance['risk_level'], risk_performance['mean'], 
                      yerr=risk_performance['std'], capsize=5, alpha=0.7,
                      color=['green', 'orange', 'red'])
        plt.xlabel('Risk Level')
        plt.ylabel('Average FDV Change (%)')
        plt.title('Performance by Risk Level', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, mean_val in zip(bars, risk_performance['mean']):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{mean_val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_by_risk.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Correlation visualizations created")
    
    def _generate_insights_report(self):
        """Generate a comprehensive insights report"""
        print("📝 Generating insights report...")
        
        report = []
        report.append("=" * 80)
        report.append("TOKEN CORRELATION AND PATTERN ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        report.append("📊 SUMMARY STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Tokens Analyzed: {len(self.df)}")
        report.append(f"Average FDV Change: {self.df['fdv_change_pct'].mean():.2f}%")
        report.append(f"Average Risk Score: {self.df['risk_score'].mean():.2f}/10")
        report.append(f"Average Momentum Score: {self.df['momentum_score'].mean():.2f}/3")
        report.append("")
        
        # Key correlations
        if 'correlations' in self.pattern_insights:
            report.append("🔗 KEY CORRELATIONS")
            report.append("-" * 40)
            correlations = self.pattern_insights['correlations']
            
            if correlations['strongest_positive']:
                pos = correlations['strongest_positive']
                report.append(f"Strongest Positive: {pos['variable1']} ↔ {pos['variable2']} (r = {pos['correlation']:.3f})")
            
            if correlations['strongest_negative']:
                neg = correlations['strongest_negative']
                report.append(f"Strongest Negative: {neg['variable1']} ↔ {neg['variable2']} (r = {neg['correlation']:.3f})")
            
            report.append(f"Total Significant Correlations: {len(correlations['significant_correlations'])}")
            report.append("")
        
        # Risk-reward insights
        if 'risk_reward' in self.pattern_insights:
            report.append("⚖️ RISK-REWARD INSIGHTS")
            report.append("-" * 40)
            risk_reward = self.pattern_insights['risk_reward']
            
            if 'risk_categories' in risk_reward:
                categories = risk_reward['risk_categories']
                for category, data in categories.items():
                    if category in data:
                        report.append(f"{category.title()}: {data[category]['count']} tokens, "
                                   f"Avg Performance: {data[category]['avg_fdv_change']:.1f}%, "
                                   f"Success Rate: {data[category]['success_rate']:.1f}%")
            report.append("")
        
        # Performance segments
        if 'performance_segments' in self.pattern_insights:
            report.append("📈 PERFORMANCE SEGMENTS")
            report.append("-" * 40)
            segments = self.pattern_insights['performance_segments']
            for segment, data in segments.items():
                report.append(f"{segment}: {data['count']} tokens ({data['percentage']:.1f}%), "
                           f"Avg Risk: {data['avg_risk_score']:.1f}, "
                           f"Avg Momentum: {data['avg_momentum_score']:.1f}")
            report.append("")
        
        # Cluster analysis
        if self.cluster_analysis:
            report.append("🎪 CLUSTER ANALYSIS")
            report.append("-" * 40)
            report.append(f"Number of Clusters: {self.cluster_analysis['n_clusters']}")
            for cluster_id, details in self.cluster_analysis['cluster_details'].items():
                report.append(f"{cluster_id}: {details['size']} tokens ({details['percentage']:.1f}%), "
                           f"Avg Performance: {details['avg_fdv_change']:.1f}%, "
                           f"Avg Risk: {details['avg_risk_score']:.1f}")
            report.append("")
        
        # Transaction insights
        if 'transaction_patterns' in self.pattern_insights:
            report.append("💱 TRANSACTION INSIGHTS")
            report.append("-" * 40)
            tx_patterns = self.pattern_insights['transaction_patterns']
            
            if 'volume_performance_correlation' in tx_patterns:
                report.append(f"Volume-Performance Correlation: {tx_patterns['volume_performance_correlation']:.3f}")
            
            if 'buy_pressure_correlation' in tx_patterns:
                report.append(f"Buy Pressure-Performance Correlation: {tx_patterns['buy_pressure_correlation']:.3f}")
            
            if 'volume_analysis' in tx_patterns:
                vol_analysis = tx_patterns['volume_analysis']
                report.append(f"High vs Low Volume Performance Difference: {vol_analysis['volume_performance_difference']:.1f}%")
            report.append("")
        
        # Key insights and recommendations
        report.append("💡 KEY INSIGHTS & RECOMMENDATIONS")
        report.append("-" * 40)
        
        # Generate insights based on analysis
        insights = self._generate_key_insights()
        for insight in insights:
            report.append(f"• {insight}")
        
        report.append("")
        report.append("=" * 80)
        
        # Save report
        report_file = self.output_dir / 'correlation_analysis_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"📝 Insights report saved to: {report_file}")
    
    def _generate_key_insights(self):
        """Generate key insights based on the analysis"""
        insights = []
        
        # Performance insights
        positive_tokens = len(self.df[self.df['fdv_change_pct'] > 0])
        success_rate = positive_tokens / len(self.df) * 100
        insights.append(f"Success Rate: {success_rate:.1f}% of tokens show positive performance")
        
        # Risk insights
        if 'risk_reward' in self.pattern_insights:
            risk_reward = self.pattern_insights['risk_reward']
            if 'risk_categories' in risk_reward:
                categories = risk_reward['risk_categories']
                if 'low_risk' in categories and 'high_risk' in categories:
                    low_risk_perf = categories['low_risk']['avg_fdv_change']
                    high_risk_perf = categories['high_risk']['avg_fdv_change']
                    if low_risk_perf > high_risk_perf:
                        insights.append("Low-risk tokens outperform high-risk tokens on average")
                    else:
                        insights.append("High-risk tokens show higher average returns (higher risk, higher reward)")
        
        # Correlation insights
        if 'correlations' in self.pattern_insights:
            correlations = self.pattern_insights['correlations']
            if correlations['significant_correlations']:
                strongest = correlations['significant_correlations'][0]
                insights.append(f"Strongest relationship: {strongest['variable1']} and {strongest['variable2']} "
                             f"({strongest['correlation']:.3f} correlation)")
        
        # Transaction insights
        if 'transaction_patterns' in self.pattern_insights:
            tx_patterns = self.pattern_insights['transaction_patterns']
            if 'volume_analysis' in tx_patterns:
                vol_analysis = tx_patterns['volume_analysis']
                if vol_analysis['volume_performance_difference'] > 0:
                    insights.append("Higher transaction volume correlates with better performance")
                else:
                    insights.append("Lower transaction volume correlates with better performance")
        
        # Cluster insights
        if self.cluster_analysis:
            cluster_details = self.cluster_analysis['cluster_details']
            largest_cluster = max(cluster_details.values(), key=lambda x: x['size'])
            insights.append(f"Largest cluster contains {largest_cluster['size']} tokens "
                         f"with {largest_cluster['avg_fdv_change']:.1f}% average performance")
        
        return insights

def main():
    """Main function to run correlation analysis"""
    try:
        # Initialize analyzer
        analyzer = TokenCorrelationAnalyzer()
        
        # Run comprehensive analysis
        analyzer.run_comprehensive_analysis()
        
        print("✅ Correlation and pattern analysis completed successfully!")
        print(f"📁 Check the 'output/correlation_analysis' folder for detailed results")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
