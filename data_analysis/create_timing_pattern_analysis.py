#!/usr/bin/env python3
"""
Timing Pattern Analysis for Memecoin Lifecycle
Analyzes when different patterns occur after token registration
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime, timedelta
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

class TimingPatternAnalyzer:
    """Analyzes timing patterns in memecoin lifecycle"""
    
    def __init__(self, results_file: str = "output/adapted_timeseries_analysis.json"):
        self.results_file = Path(results_file)
        self.output_dir = Path("output/timing_pattern_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.results = self._load_results()
        self.df = self._create_dataframe()
        
        # Pattern definitions with emojis
        self.patterns = {
            'moon_shot': '🚀 Moon Shot',
            'strong_rise': '📈 Strong Rise',
            'moderate_rise': '📊 Moderate Rise',
            'stable': '⚖️ Stable',
            'moderate_drop': '📉 Moderate Drop',
            'significant_drop': '💸 Significant Drop',
            'died': '⚰️ Died'
        }
        
        # Color scheme for patterns
        self.pattern_colors = {
            'moon_shot': '#FF6B6B',
            'strong_rise': '#4ECDC4',
            'moderate_rise': '#45B7D1',
            'stable': '#96CEB4',
            'moderate_drop': '#FFEAA7',
            'significant_drop': '#DDA0DD',
            'died': '#A8E6CF'
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
        """Convert results to pandas DataFrame with timing data"""
        if not self.results or 'results' not in self.results:
            return pd.DataFrame()
        
        rows = []
        for address, result in self.results['results'].items():
            if result.get('status') == 'analyzed':
                # Extract basic metrics
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
                
                # Extract timing data
                data_summary = result.get('data_summary', {})
                if 'date_range' in data_summary:
                    date_range = data_summary['date_range']
                    start_time = date_range.get('start')
                    end_time = date_range.get('end')
                    
                    if start_time and end_time:
                        try:
                            start_dt = pd.to_datetime(start_time)
                            end_dt = pd.to_datetime(end_time)
                            
                            # Calculate timing metrics
                            row['start_time'] = start_dt
                            row['end_time'] = end_dt
                            row['duration_hours'] = (end_dt - start_dt).total_seconds() / 3600
                            row['duration_days'] = row['duration_hours'] / 24
                            
                            # Time since registration (assuming start_time is registration)
                            now = pd.Timestamp.now()
                            row['time_since_registration_hours'] = (now - start_dt).total_seconds() / 3600
                            row['time_since_registration_days'] = row['time_since_registration_hours'] / 24
                            
                        except Exception as e:
                            print(f"Error parsing dates for {address}: {e}")
                            continue
                
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
        
        # Add derived timing metrics
        if not df.empty and 'duration_hours' in df.columns:
            df['is_short_lived'] = df['duration_hours'] < 24  # Less than 1 day
            df['is_medium_lived'] = (df['duration_hours'] >= 24) & (df['duration_hours'] < 168)  # 1 day to 1 week
            df['is_long_lived'] = df['duration_hours'] >= 168  # More than 1 week
            
            # Pattern timing categories
            df['pattern_timing'] = df.apply(self._categorize_pattern_timing, axis=1)
            
            # Lifecycle stage
            df['lifecycle_stage'] = df.apply(self._categorize_lifecycle_stage, axis=1)
        
        return df
    
    def _categorize_pattern_timing(self, row):
        """Categorize when patterns occur in the token lifecycle"""
        if pd.isna(row.get('duration_hours')):
            return 'unknown'
        
        duration = row['duration_hours']
        pattern = row['pattern']
        
        if duration < 1:  # Less than 1 hour
            if pattern in ['moon_shot', 'strong_rise']:
                return 'instant_moon'
            elif pattern in ['died', 'significant_drop']:
                return 'instant_death'
            else:
                return 'instant_other'
        elif duration < 6:  # Less than 6 hours
            if pattern in ['moon_shot', 'strong_rise']:
                return 'early_moon'
            elif pattern in ['died', 'significant_drop']:
                return 'early_death'
            else:
                return 'early_other'
        elif duration < 24:  # Less than 1 day
            if pattern in ['moon_shot', 'strong_rise']:
                return 'day_one_moon'
            elif pattern in ['died', 'significant_drop']:
                return 'day_one_death'
            else:
                return 'day_one_other'
        elif duration < 168:  # Less than 1 week
            if pattern in ['moon_shot', 'strong_rise']:
                return 'week_one_moon'
            elif pattern in ['died', 'significant_drop']:
                return 'week_one_death'
            else:
                return 'week_one_other'
        else:  # More than 1 week
            if pattern in ['moon_shot', 'strong_rise']:
                return 'long_term_moon'
            elif pattern in ['died', 'significant_drop']:
                return 'long_term_death'
            else:
                return 'long_term_other'
    
    def _categorize_lifecycle_stage(self, row):
        """Categorize the lifecycle stage of the token"""
        if pd.isna(row.get('duration_hours')):
            return 'unknown'
        
        duration = row['duration_hours']
        
        if duration < 1:
            return 'instant'
        elif duration < 6:
            return 'early_hours'
        elif duration < 24:
            return 'day_one'
        elif duration < 168:
            return 'week_one'
        else:
            return 'long_term'
    
    def create_timing_analysis(self):
        """Create comprehensive timing pattern analysis"""
        print("⏰ Creating timing pattern analysis...")
        
        if self.df.empty or 'duration_hours' not in self.df.columns:
            print("❌ No timing data available for analysis")
            return
        
        # Generate all timing reports
        reports = {}
        
        # 1. Pattern Timing Analysis
        reports['pattern_timing'] = self._analyze_pattern_timing()
        
        # 2. Lifecycle Stage Analysis
        reports['lifecycle_stages'] = self._analyze_lifecycle_stages()
        
        # 3. Duration Distribution Analysis
        reports['duration_distribution'] = self._analyze_duration_distribution()
        
        # 4. Pattern Evolution Analysis
        reports['pattern_evolution'] = self._analyze_pattern_evolution()
        
        # 5. Optimal Entry/Exit Timing
        reports['optimal_timing'] = self._analyze_optimal_timing()
        
        # 6. Time-based Success Rates
        reports['time_based_success'] = self._analyze_time_based_success()
        
        # Create visualizations
        self._create_timing_visualizations()
        
        # Save reports
        self._save_timing_reports(reports)
        
        # Display summary
        self._display_timing_summary(reports)
        
        return reports
    
    def _analyze_pattern_timing(self):
        """Analyze when different patterns occur"""
        analysis = {
            'title': '⏰ PATTERN TIMING ANALYSIS',
            'description': 'When different patterns occur after token registration',
            'insights': []
        }
        
        # Analyze each pattern's timing
        for pattern in self.df['pattern'].unique():
            pattern_data = self.df[self.df['pattern'] == pattern]
            pattern_count = len(pattern_data)
            
            if pattern_count == 0:
                continue
            
            # Timing statistics
            avg_duration_hours = pattern_data['duration_hours'].mean()
            avg_duration_days = avg_duration_hours / 24
            
            # Timing distribution
            timing_dist = pattern_data['pattern_timing'].value_counts()
            
            # Success rate by timing
            success_rate = (pattern_data['fdv_change_pct'] > 0).mean() * 100
            
            insight = {
                'pattern': pattern,
                'emoji': self.patterns.get(pattern, '❓'),
                'total_tokens': pattern_count,
                'avg_duration_hours': f"{avg_duration_hours:.1f}",
                'avg_duration_days': f"{avg_duration_days:.2f}",
                'success_rate': f"{success_rate:.1f}%",
                'timing_distribution': timing_dist.to_dict(),
                'recommendation': self._get_timing_recommendation(pattern, avg_duration_hours, success_rate)
            }
            
            analysis['insights'].append(insight)
        
        return analysis
    
    def _analyze_lifecycle_stages(self):
        """Analyze performance across different lifecycle stages"""
        analysis = {
            'title': '🔄 LIFECYCLE STAGE ANALYSIS',
            'description': 'How tokens perform at different stages of their lifecycle',
            'insights': []
        }
        
        # Analyze each lifecycle stage
        for stage in self.df['lifecycle_stage'].unique():
            stage_data = self.df[self.df['lifecycle_stage'] == stage]
            stage_count = len(stage_data)
            
            if stage_count == 0:
                continue
            
            # Performance metrics
            avg_performance = stage_data['fdv_change_pct'].mean()
            success_rate = (stage_data['fdv_change_pct'] > 0).mean() * 100
            moon_shot_rate = (stage_data['fdv_change_pct'] > 100).mean() * 100
            death_rate = (stage_data['fdv_change_pct'] < -80).mean() * 100
            
            # Pattern distribution
            pattern_dist = stage_data['pattern'].value_counts()
            
            insight = {
                'lifecycle_stage': stage,
                'total_tokens': stage_count,
                'avg_performance': f"{avg_performance:+.1f}%",
                'success_rate': f"{success_rate:.1f}%",
                'moon_shot_rate': f"{moon_shot_rate:.1f}%",
                'death_rate': f"{death_rate:.1f}%",
                'pattern_distribution': pattern_dist.to_dict(),
                'recommendation': self._get_lifecycle_recommendation(stage, avg_performance, success_rate)
            }
            
            analysis['insights'].append(insight)
        
        return analysis
    
    def _analyze_duration_distribution(self):
        """Analyze the distribution of token durations"""
        analysis = {
            'title': '📊 DURATION DISTRIBUTION ANALYSIS',
            'description': 'How long tokens typically live and when patterns occur',
            'insights': []
        }
        
        # Duration categories
        duration_categories = {
            'Instant (< 1 hour)': self.df[self.df['duration_hours'] < 1],
            'Early Hours (1-6 hours)': self.df[(self.df['duration_hours'] >= 1) & (self.df['duration_hours'] < 6)],
            'Day One (6-24 hours)': self.df[(self.df['duration_hours'] >= 6) & (self.df['duration_hours'] < 24)],
            'Week One (1-7 days)': self.df[(self.df['duration_hours'] >= 24) & (self.df['duration_hours'] < 168)],
            'Long Term (> 7 days)': self.df[self.df['duration_hours'] >= 168]
        }
        
        for category_name, category_data in duration_categories.items():
            if len(category_data) == 0:
                continue
            
            category_count = len(category_data)
            category_percentage = (category_count / len(self.df)) * 100
            
            # Performance metrics
            avg_performance = category_data['fdv_change_pct'].mean()
            success_rate = (category_data['fdv_change_pct'] > 0).mean() * 100
            moon_shot_rate = (category_data['fdv_change_pct'] > 100).mean() * 100
            death_rate = (category_data['fdv_change_pct'] < -80).mean() * 100
            
            # Pattern distribution
            pattern_dist = category_data['pattern'].value_counts()
            
            insight = {
                'duration_category': category_name,
                'total_tokens': category_count,
                'percentage': f"{category_percentage:.1f}%",
                'avg_performance': f"{avg_performance:+.1f}%",
                'success_rate': f"{success_rate:.1f}%",
                'moon_shot_rate': f"{moon_shot_rate:.1f}%",
                'death_rate': f"{death_rate:.1f}%",
                'pattern_distribution': pattern_dist.to_dict(),
                'recommendation': self._get_duration_recommendation(category_name, avg_performance, success_rate)
            }
            
            analysis['insights'].append(insight)
        
        return analysis
    
    def _analyze_pattern_evolution(self):
        """Analyze how patterns evolve over time"""
        analysis = {
            'title': '🔄 PATTERN EVOLUTION ANALYSIS',
            'description': 'How patterns change and evolve throughout token lifecycle',
            'insights': []
        }
        
        # Pattern evolution by timing
        evolution_data = {}
        
        for pattern in self.df['pattern'].unique():
            pattern_data = self.df[self.df['pattern'] == pattern]
            
            # Group by timing categories
            timing_groups = pattern_data.groupby('pattern_timing')
            
            for timing, group in timing_groups:
                if timing not in evolution_data:
                    evolution_data[timing] = {}
                
                evolution_data[timing][pattern] = {
                    'count': len(group),
                    'avg_performance': group['fdv_change_pct'].mean(),
                    'success_rate': (group['fdv_change_pct'] > 0).mean() * 100
                }
        
        analysis['evolution_data'] = evolution_data
        
        # Key insights
        analysis['key_insights'] = []
        
        # Instant patterns
        if 'instant_moon' in evolution_data:
            analysis['key_insights'].append({
                'insight': '🚀 Instant Moon Shots',
                'description': f"Tokens that moon immediately: {evolution_data['instant_moon'].get('moon_shot', {}).get('count', 0)} tokens",
                'timing': 'Within 1 hour of registration',
                'action': 'Quick entry required - these are extremely fast'
            })
        
        if 'instant_death' in evolution_data:
            analysis['key_insights'].append({
                'insight': '⚰️ Instant Deaths',
                'description': f"Tokens that die immediately: {evolution_data['instant_death'].get('died', {}).get('count', 0)} tokens",
                'timing': 'Within 1 hour of registration',
                'action': 'Avoid completely - these are instant rug pulls'
            })
        
        # Early patterns
        if 'early_moon' in evolution_data:
            analysis['key_insights'].append({
                'insight': '📈 Early Moon Shots',
                'description': f"Tokens that moon in first 6 hours: {evolution_data['early_moon'].get('moon_shot', {}).get('count', 0)} tokens",
                'timing': '1-6 hours after registration',
                'action': 'Good entry window - still early but not instant'
            })
        
        return analysis
    
    def _analyze_optimal_timing(self):
        """Analyze optimal entry and exit timing"""
        analysis = {
            'title': '🎯 OPTIMAL TIMING ANALYSIS',
            'description': 'Best times to enter and exit based on pattern analysis',
            'insights': []
        }
        
        # Optimal entry timing for positive patterns
        positive_patterns = self.df[self.df['fdv_change_pct'] > 0]
        
        if len(positive_patterns) > 0:
            # Group by duration to find optimal entry windows
            entry_timing = positive_patterns.groupby('lifecycle_stage')['fdv_change_pct'].agg(['mean', 'count'])
            
            # Find best entry stage
            best_entry_stage = entry_timing.loc[entry_timing['mean'].idxmax()]
            
            analysis['best_entry_timing'] = {
                'stage': entry_timing['mean'].idxmax(),
                'avg_performance': f"{best_entry_stage['mean']:+.1f}%",
                'token_count': int(best_entry_stage['count']),
                'recommendation': f"Best entry timing: {entry_timing['mean'].idxmax()} stage"
            }
        
        # Optimal exit timing
        moon_shots = self.df[self.df['pattern'] == 'moon_shot']
        if len(moon_shots) > 0:
            avg_moon_duration = moon_shots['duration_hours'].mean()
            analysis['optimal_exit_timing'] = {
                'avg_moon_duration_hours': f"{avg_moon_duration:.1f}",
                'avg_moon_duration_days': f"{avg_moon_duration/24:.2f}",
                'recommendation': f"Moon shots typically peak within {avg_moon_duration:.1f} hours"
            }
        
        # Time-based risk assessment
        risk_by_time = self.df.groupby('lifecycle_stage')['risk_score'].mean()
        analysis['risk_by_timing'] = {
            'lowest_risk_stage': risk_by_time.idxmin(),
            'highest_risk_stage': risk_by_time.idxmax(),
            'risk_timing_insight': f"Lowest risk: {risk_by_time.idxmin()}, Highest risk: {risk_by_time.idxmax()}"
        }
        
        return analysis
    
    def _analyze_time_based_success(self):
        """Analyze success rates based on timing"""
        analysis = {
            'title': '📈 TIME-BASED SUCCESS RATES',
            'description': 'Success rates and performance metrics by timing',
            'insights': []
        }
        
        # Success rate by lifecycle stage
        success_by_stage = self.df.groupby('lifecycle_stage').apply(
            lambda x: (x['fdv_change_pct'] > 0).mean() * 100
        ).sort_values(ascending=False)
        
        analysis['success_by_stage'] = success_by_stage.to_dict()
        
        # Performance by timing
        performance_by_timing = self.df.groupby('lifecycle_stage')['fdv_change_pct'].agg(['mean', 'count'])
        analysis['performance_by_timing'] = performance_by_timing.to_dict()
        
        # Pattern success by timing
        pattern_timing_success = {}
        for pattern in self.df['pattern'].unique():
            pattern_data = self.df[self.df['pattern'] == pattern]
            if len(pattern_data) > 0:
                timing_success = pattern_data.groupby('lifecycle_stage').apply(
                    lambda x: (x['fdv_change_pct'] > 0).mean() * 100
                )
                pattern_timing_success[pattern] = timing_success.to_dict()
        
        analysis['pattern_timing_success'] = pattern_timing_success
        
        return analysis
    
    def _get_timing_recommendation(self, pattern, avg_duration, success_rate):
        """Get timing recommendation for a pattern"""
        if pattern in ['moon_shot', 'strong_rise']:
            if avg_duration < 1:
                return "🚀 Instant moon - Requires immediate entry (< 1 hour)"
            elif avg_duration < 6:
                return "📈 Early moon - Good entry window (1-6 hours)"
            elif avg_duration < 24:
                return "📊 Day one moon - Standard entry window (6-24 hours)"
            else:
                return "⏰ Long-term moon - Patient entry strategy (> 1 day)"
        elif pattern in ['died', 'significant_drop']:
            if avg_duration < 1:
                return "⚰️ Instant death - Avoid completely (< 1 hour)"
            elif avg_duration < 6:
                return "💸 Early death - Very high risk (1-6 hours)"
            else:
                return "📉 Delayed death - Exit quickly if detected"
        else:
            return "⚖️ Stable pattern - Standard timing considerations"
    
    def _get_lifecycle_recommendation(self, stage, avg_performance, success_rate):
        """Get recommendation for a lifecycle stage"""
        if stage == 'instant':
            if success_rate > 50:
                return "🚀 High-risk, high-reward - Quick entry/exit required"
            else:
                return "❌ Extremely high risk - Avoid completely"
        elif stage == 'early_hours':
            if success_rate > 50:
                return "📈 Good risk/reward - Early entry recommended"
            else:
                return "⚠️ High risk - Proceed with extreme caution"
        elif stage == 'day_one':
            if success_rate > 50:
                return "📊 Balanced risk/reward - Standard entry window"
            else:
                return "⚠️ Moderate risk - Careful analysis required"
        elif stage == 'week_one':
            if success_rate > 50:
                return "⏰ Lower risk, moderate reward - Patient strategy"
            else:
                return "📉 Declining performance - Exit if possible"
        else:  # long_term
            if success_rate > 50:
                return "🕰️ Long-term hold - Low risk, steady returns"
            else:
                return "💀 Long-term decline - Exit immediately"
    
    def _get_duration_recommendation(self, category, avg_performance, success_rate):
        """Get recommendation for a duration category"""
        if 'Instant' in category:
            if success_rate > 50:
                return "🚀 Extremely fast - Requires instant decision making"
            else:
                return "❌ Instant death - Avoid completely"
        elif 'Early Hours' in category:
            if success_rate > 50:
                return "📈 Fast action - Good for aggressive strategies"
            else:
                return "⚠️ High risk - Quick exit if needed"
        elif 'Day One' in category:
            if success_rate > 50:
                return "📊 Standard timing - Balanced approach"
            else:
                return "⚠️ Moderate risk - Standard risk management"
        elif 'Week One' in category:
            if success_rate > 50:
                return "⏰ Patient strategy - Lower risk, steady returns"
            else:
                return "📉 Declining - Exit if possible"
        else:  # Long Term
            if success_rate > 50:
                return "🕰️ Long-term hold - Conservative strategy"
            else:
                return "💀 Long-term decline - Exit immediately"
    
    def _create_timing_visualizations(self):
        """Create timing pattern visualizations"""
        print("🎨 Creating timing pattern visualizations...")
        
        # 1. Pattern Duration Distribution
        self._create_duration_distribution_chart()
        
        # 2. Pattern Timing Heatmap
        self._create_pattern_timing_heatmap()
        
        # 3. Lifecycle Performance Chart
        self._create_lifecycle_performance_chart()
        
        # 4. Pattern Evolution Timeline
        self._create_pattern_evolution_timeline()
        
        # 5. Success Rate by Timing
        self._create_success_rate_timing_chart()
        
        print("✅ Timing visualizations created")
    
    def _create_duration_distribution_chart(self):
        """Create duration distribution chart"""
        if self.df.empty or 'duration_hours' not in self.df.columns:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Duration histogram
        ax1.hist(self.df['duration_hours'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('📊 Token Duration Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Duration (Hours)')
        ax1.set_ylabel('Number of Tokens')
        ax1.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='1 Hour')
        ax1.axvline(x=24, color='orange', linestyle='--', alpha=0.7, label='1 Day')
        ax1.axvline(x=168, color='green', linestyle='--', alpha=0.7, label='1 Week')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Duration by pattern
        pattern_durations = []
        pattern_names = []
        pattern_colors = []
        
        for pattern in self.df['pattern'].unique():
            pattern_data = self.df[self.df['pattern'] == pattern]
            if len(pattern_data) > 0:
                pattern_durations.append(pattern_data['duration_hours'].mean())
                pattern_names.append(self.patterns.get(pattern, pattern))
                pattern_colors.append(self.pattern_colors.get(pattern, '#CCCCCC'))
        
        bars = ax2.bar(range(len(pattern_durations)), pattern_durations, color=pattern_colors)
        ax2.set_title('⏰ Average Duration by Pattern', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Pattern')
        ax2.set_ylabel('Average Duration (Hours)')
        ax2.set_xticks(range(len(pattern_names)))
        ax2.set_xticklabels(pattern_names, rotation=45, ha='right')
        
        # Add value labels
        for bar, duration in zip(bars, pattern_durations):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{duration:.1f}h', ha='center', va='bottom', fontweight='bold')
        
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'duration_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_pattern_timing_heatmap(self):
        """Create pattern timing heatmap"""
        if self.df.empty or 'pattern_timing' not in self.df.columns:
            return
        
        # Create pivot table for heatmap
        timing_pivot = pd.crosstab(self.df['pattern'], self.df['lifecycle_stage'], values=self.df['fdv_change_pct'], aggfunc='mean')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(timing_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
                   cbar_kws={'label': 'Average FDV Change (%)'})
        plt.title('🔥 Pattern Performance by Lifecycle Stage', fontsize=16, fontweight='bold')
        plt.xlabel('Lifecycle Stage')
        plt.ylabel('Pattern')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pattern_timing_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_lifecycle_performance_chart(self):
        """Create lifecycle performance chart"""
        if self.df.empty or 'lifecycle_stage' not in self.df.columns:
            return
        
        lifecycle_performance = self.df.groupby('lifecycle_stage')['fdv_change_pct'].agg(['mean', 'count']).sort_values('mean')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Performance by lifecycle stage
        bars = ax1.bar(range(len(lifecycle_performance)), lifecycle_performance['mean'], 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax1.set_title('📈 Performance by Lifecycle Stage', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Lifecycle Stage')
        ax1.set_ylabel('Average FDV Change (%)')
        ax1.set_xticks(range(len(lifecycle_performance)))
        ax1.set_xticklabels([stage.replace('_', ' ').title() for stage in lifecycle_performance.index], rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars, lifecycle_performance['mean']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -15),
                    f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold')
        
        ax1.grid(True, alpha=0.3)
        
        # Token count by lifecycle stage
        ax2.pie(lifecycle_performance['count'], labels=[stage.replace('_', ' ').title() for stage in lifecycle_performance.index],
                autopct='%1.1f%%', startangle=90, colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax2.set_title('📊 Token Distribution by Lifecycle Stage', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'lifecycle_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_pattern_evolution_timeline(self):
        """Create pattern evolution timeline"""
        if self.df.empty or 'pattern_timing' not in self.df.columns:
            return
        
        # Create timeline data
        timeline_data = []
        for pattern in self.df['pattern'].unique():
            pattern_data = self.df[self.df['pattern'] == pattern]
            if len(pattern_data) > 0:
                for timing in pattern_data['pattern_timing'].unique():
                    timing_data = pattern_data[pattern_data['pattern_timing'] == timing]
                    if len(timing_data) > 0:
                        timeline_data.append({
                            'pattern': pattern,
                            'timing': timing,
                            'count': len(timing_data),
                            'avg_performance': timing_data['fdv_change_pct'].mean()
                        })
        
        if not timeline_data:
            return
        
        timeline_df = pd.DataFrame(timeline_data)
        
        # Create scatter plot
        plt.figure(figsize=(14, 8))
        
        for pattern in timeline_df['pattern'].unique():
            pattern_data = timeline_df[timeline_df['pattern'] == pattern]
            plt.scatter(pattern_data['timing'], pattern_data['avg_performance'], 
                       s=pattern_data['count']*50, alpha=0.7, 
                       label=self.patterns.get(pattern, pattern),
                       color=self.pattern_colors.get(pattern, '#CCCCCC'))
        
        plt.title('🔄 Pattern Evolution Timeline', fontsize=16, fontweight='bold')
        plt.xlabel('Pattern Timing')
        plt.ylabel('Average Performance (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pattern_evolution_timeline.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_success_rate_timing_chart(self):
        """Create success rate by timing chart"""
        if self.df.empty or 'lifecycle_stage' not in self.df.columns:
            return
        
        # Calculate success rates by timing
        success_by_timing = self.df.groupby('lifecycle_stage').apply(
            lambda x: (x['fdv_change_pct'] > 0).mean() * 100
        ).sort_values(ascending=False)
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(success_by_timing)), success_by_timing.values,
                      color=['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#FF6B6B'])
        plt.title('📊 Success Rate by Lifecycle Stage', fontsize=16, fontweight='bold')
        plt.xlabel('Lifecycle Stage')
        plt.ylabel('Success Rate (%)')
        plt.xticks(range(len(success_by_timing)), 
                  [stage.replace('_', ' ').title() for stage in success_by_timing.index], 
                  rotation=45, ha='right')
        
        # Add value labels
        for bar, rate in zip(bars, success_by_timing.values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'success_rate_timing.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _save_timing_reports(self, reports):
        """Save timing analysis reports"""
        # Save detailed report
        detailed_report = self._format_timing_report(reports)
        with open(self.output_dir / 'detailed_timing_report.txt', 'w') as f:
            f.write(detailed_report)
        
        # Save summary report
        summary_report = self._format_timing_summary(reports)
        with open(self.output_dir / 'summary_timing_report.txt', 'w') as f:
            f.write(summary_report)
        
        # Save JSON version
        with open(self.output_dir / 'timing_analysis.json', 'w') as f:
            json.dump(reports, f, indent=2, default=str)
        
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
                summary_lines.append(f"   Stage: {timing['best_entry_timing']['stage']}")
                summary_lines.append(f"   Performance: {timing['best_entry_timing']['avg_performance']}")
                summary_lines.append(f"   Tokens: {timing['best_entry_timing']['token_count']}")
                summary_lines.append("")
        
        # Pattern timing insights
        if 'pattern_timing' in reports:
            summary_lines.append("⏰ PATTERN TIMING INSIGHTS:")
            pattern_insights = reports['pattern_timing']['insights']
            for insight in pattern_insights[:3]:  # Top 3
                summary_lines.append(f"   {insight['emoji']} {insight['pattern'].replace('_', ' ').title()}")
                summary_lines.append(f"      Duration: {insight['avg_duration_days']} days")
                summary_lines.append(f"      Success Rate: {insight['success_rate']}")
                summary_lines.append("")
        
        return "\n".join(summary_lines)
    
    def _display_timing_summary(self, reports):
        """Display summary of timing analysis"""
        print("\n" + "="*80)
        print("⏰ TIMING PATTERN ANALYSIS COMPLETE!")
        print("="*80)
        
        # Display key insights
        if 'optimal_timing' in reports:
            timing = reports['optimal_timing']
            if 'best_entry_timing' in timing:
                print(f"🎯 BEST ENTRY TIMING: {timing['best_entry_timing']['stage']}")
                print(f"   Performance: {timing['best_entry_timing']['avg_performance']}")
                print("")
        
        # Display pattern timing
        if 'pattern_timing' in reports:
            print("⏰ PATTERN TIMING INSIGHTS:")
            pattern_insights = reports['pattern_timing']['insights']
            for insight in pattern_insights[:3]:
                print(f"   {insight['emoji']} {insight['pattern'].replace('_', ' ').title()}")
                print(f"      Duration: {insight['avg_duration_days']} days")
                print(f"      Success Rate: {insight['success_rate']}")
            print("")
        
        print("📁 Check the 'output/timing_pattern_analysis' folder for detailed reports")
        print("📊 Summary report: summary_timing_report.txt")
        print("📝 Detailed report: detailed_timing_report.txt")
        print("🔧 JSON data: timing_analysis.json")

def main():
    """Main function to create timing pattern analysis"""
    print("⏰ Starting Timing Pattern Analysis...")
    
    # Create analyzer
    analyzer = TimingPatternAnalyzer()
    
    # Generate timing analysis
    reports = analyzer.create_timing_analysis()
    
    print("\n" + "="*80)
    print("🎉 TIMING ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
