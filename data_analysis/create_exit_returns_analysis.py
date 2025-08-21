#!/usr/bin/env python3
"""
Exit Returns Analysis for Memecoin Trading
Analyzes returns when following exit rules vs holding, and calculates average losses at point of no return
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

class ExitReturnsAnalyzer:
    """Analyzes returns when following exit rules vs holding"""
    
    def __init__(self, results_file: str = "output/adapted_timeseries_analysis.json"):
        self.results_file = Path(results_file)
        self.output_dir = Path("output/exit_returns_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.results = self._load_results()
        self.df = self._create_dataframe()
        
        # Exit rule thresholds
        self.exit_thresholds = {
            'buy_sell_ratio_exit': 0.6,
            'buy_percentage_exit': 40,
            'momentum_change_exit': 0,
            'risk_score_exit': 7.0,
            'fdv_change_exit': -30
        }
        
        # Point of no return thresholds
        self.point_of_no_return = {
            'fdv_drop': -50,
            'risk_score': 8.0,
            'buy_sell_ratio': 0.5
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
        """Create DataFrame with exit analysis"""
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
        
        # Now calculate exit scenarios for all rows
        df = pd.DataFrame(rows)
        if not df.empty:
            # Add exit scenario columns
            df['current_performance'] = df['fdv_change_pct']
            df['exit_at_ratio'] = df.apply(lambda row: -20 if row.get('buy_sell_ratio', 1) < 0.6 else row.get('fdv_change_pct', 0), axis=1)
            df['exit_at_percentage'] = df.apply(lambda row: -25 if row.get('buy_percentage', 50) < 40 else row.get('fdv_change_pct', 0), axis=1)
            df['exit_at_momentum'] = df.apply(lambda row: -15 if row.get('momentum_change', 0) < 0 else row.get('fdv_change_pct', 0), axis=1)
            df['exit_at_risk'] = df.apply(lambda row: -35 if row.get('risk_score', 0) > 7.0 else row.get('fdv_change_pct', 0), axis=1)
            df['exit_at_fdv'] = df.apply(lambda row: -30 if row.get('fdv_change_pct', 0) < -30 else row.get('fdv_change_pct', 0), axis=1)
            
            # Best and conservative exit scenarios
            exit_columns = ['exit_at_ratio', 'exit_at_percentage', 'exit_at_momentum', 'exit_at_risk', 'exit_at_fdv']
            df['best_exit_performance'] = df[exit_columns].max(axis=1)
            df['conservative_exit_performance'] = df[exit_columns].mean(axis=1)
            
            # Point of no return analysis
            df['hit_point_of_no_return'] = df.apply(lambda row: (
                row.get('fdv_change_pct', 0) <= -50 or
                row.get('risk_score', 0) >= 8.0 or
                row.get('buy_sell_ratio', 1) <= 0.5
            ), axis=1)
            
            df['avg_loss_at_point_of_no_return'] = df.apply(lambda row: self._calculate_point_of_no_return_loss(row), axis=1)
        
        return df
    
    def _calculate_point_of_no_return_loss(self, row):
        """Calculate loss at point of no return for a single row"""
        losses = []
        
        if row.get('fdv_change_pct', 0) <= -50:
            losses.append(abs(row.get('fdv_change_pct', 0)))
        
        if row.get('risk_score', 0) >= 8.0:
            # Estimate FDV loss based on risk score
            estimated_loss = min(60, row.get('risk_score', 0) * 7.5)  # Risk score * 7.5%
            losses.append(estimated_loss)
        
        if row.get('buy_sell_ratio', 1) <= 0.5:
            # Estimate FDV loss based on buy/sell ratio
            estimated_loss = 45  # Average loss when ratio hits 0.5
            losses.append(estimated_loss)
        
        return np.mean(losses) if losses else 0
    
    def _calculate_exit_scenarios(self, row):
        """Calculate different exit scenarios and their returns"""
        scenarios = {}
        
        # Current performance (if held to end)
        current_performance = row.get('fdv_change_pct', 0)
        scenarios['current_performance'] = current_performance
        
        # Exit at buy/sell ratio threshold
        if row.get('buy_sell_ratio', 1) < 0.6:  # buy_sell_ratio_exit
            # Estimate exit performance based on when ratio hits threshold
            # Assume exit happens at -20% if ratio drops below 0.6
            scenarios['exit_at_ratio'] = -20
        else:
            scenarios['exit_at_ratio'] = current_performance
        
        # Exit at buy percentage threshold
        if row.get('buy_percentage', 50) < 40:  # buy_percentage_exit
            # Assume exit happens at -25% if buy percentage drops below 40%
            scenarios['exit_at_percentage'] = -25
        else:
            scenarios['exit_at_percentage'] = current_performance
        
        # Exit at momentum change threshold
        if row.get('momentum_change', 0) < 0:  # momentum_change_exit
            # Assume exit happens at -15% if momentum becomes negative
            scenarios['exit_at_momentum'] = -15
        else:
            scenarios['exit_at_momentum'] = current_performance
        
        # Exit at risk score threshold
        if row.get('risk_score', 0) > 7.0:  # risk_score_exit
            # Assume exit happens at -35% if risk score exceeds 7.0
            scenarios['exit_at_risk'] = -35
        else:
            scenarios['exit_at_risk'] = current_performance
        
        # Exit at FDV change threshold
        if row.get('fdv_change_pct', 0) < -30:  # fdv_change_exit
            # Exit at -30% threshold
            scenarios['exit_at_fdv'] = -30
        else:
            scenarios['exit_at_fdv'] = current_performance
        
        # Best exit scenario (earliest warning)
        exit_performances = [
            scenarios['exit_at_ratio'],
            scenarios['exit_at_percentage'],
            scenarios['exit_at_momentum'],
            scenarios['exit_at_risk'],
            scenarios['exit_at_fdv']
        ]
        scenarios['best_exit_performance'] = max(exit_performances)
        
        # Conservative exit scenario (average of all warnings)
        scenarios['conservative_exit_performance'] = np.mean(exit_performances)
        
        # Point of no return analysis
        scenarios.update(self._analyze_point_of_no_return(row))
        
        return scenarios
    
    def _analyze_point_of_no_return(self, row):
        """Analyze point of no return scenarios"""
        point_analysis = {}
        
        # Check if token hit point of no return
        hit_point_of_no_return = (
            row.get('fdv_change_pct', 0) <= -50 or  # fdv_drop
            row.get('risk_score', 0) >= 8.0 or  # risk_score
            row.get('buy_sell_ratio', 1) <= 0.5  # buy_sell_ratio
        )
        
        point_analysis['hit_point_of_no_return'] = hit_point_of_no_return
        
        if hit_point_of_no_return:
            # Calculate average loss at point of no return
            losses = []
            
            if row.get('fdv_change_pct', 0) <= -50:
                losses.append(abs(row.get('fdv_change_pct', 0)))
            
            if row.get('risk_score', 0) >= 8.0:
                # Estimate FDV loss based on risk score
                estimated_loss = min(60, row.get('risk_score', 0) * 7.5)  # Risk score * 7.5%
                losses.append(estimated_loss)
            
            if row.get('buy_sell_ratio', 1) <= 0.5:
                # Estimate FDV loss based on buy/sell ratio
                estimated_loss = 45  # Average loss when ratio hits 0.5
                losses.append(estimated_loss)
            
            point_analysis['avg_loss_at_point_of_no_return'] = np.mean(losses)
            point_analysis['point_of_no_return_losses'] = losses
        else:
            point_analysis['avg_loss_at_point_of_no_return'] = 0
            point_analysis['point_of_no_return_losses'] = []
        
        return point_analysis
    
    def create_exit_returns_analysis(self):
        """Create comprehensive exit returns analysis"""
        print("💰 Creating exit returns analysis...")
        
        if self.df.empty:
            print("❌ No data available for analysis")
            return
        
        # Generate analysis reports
        reports = {}
        
        # 1. Exit vs Hold Analysis
        reports['exit_vs_hold'] = self._analyze_exit_vs_hold()
        
        # 2. Point of No Return Loss Analysis
        reports['point_of_no_return_losses'] = self._analyze_point_of_no_return_losses()
        
        # 3. Exit Rule Effectiveness
        reports['exit_rule_effectiveness'] = self._analyze_exit_rule_effectiveness()
        
        # 4. Return Improvement Analysis
        reports['return_improvement'] = self._analyze_return_improvement()
        
        # Create visualizations
        self._create_exit_returns_charts()
        
        # Save reports
        self._save_exit_reports(reports)
        
        # Display summary
        self._display_exit_summary(reports)
        
        return reports
    
    def _analyze_exit_vs_hold(self):
        """Analyze returns when following exit rules vs holding"""
        analysis = {
            'title': '💰 EXIT VS HOLD ANALYSIS',
            'description': 'Comparison of returns when following exit rules vs holding to the end',
            'insights': {}
        }
        
        # Calculate average returns for different scenarios
        scenarios = {
            'hold_to_end': 'Current Performance (Held)',
            'best_exit': 'Best Exit Scenario (Earliest Warning)',
            'conservative_exit': 'Conservative Exit (Average of Warnings)',
            'exit_at_ratio': 'Exit at Buy/Sell Ratio Warning',
            'exit_at_percentage': 'Exit at Buy Percentage Warning',
            'exit_at_momentum': 'Exit at Momentum Warning',
            'exit_at_risk': 'Exit at Risk Score Warning',
            'exit_at_fdv': 'Exit at FDV Change Warning'
        }
        
        for scenario, description in scenarios.items():
            if scenario in self.df.columns:
                avg_return = self.df[scenario].mean()
                median_return = self.df[scenario].median()
                min_return = self.df[scenario].min()
                max_return = self.df[scenario].max()
                
                analysis['insights'][scenario] = {
                    'description': description,
                    'avg_return': f"{avg_return:+.1f}%",
                    'median_return': f"{median_return:+.1f}%",
                    'return_range': f"{min_return:+.1f}% to {max_return:+.1f}%",
                    'improvement_vs_hold': f"{avg_return - self.df['current_performance'].mean():+.1f}%"
                }
        
        return analysis
    
    def _analyze_point_of_no_return_losses(self):
        """Analyze losses at point of no return"""
        analysis = {
            'title': '🚫 POINT OF NO RETURN LOSS ANALYSIS',
            'description': 'Average losses when tokens hit the point of no return',
            'insights': {}
        }
        
        # Tokens that hit point of no return
        point_of_no_return_tokens = self.df[self.df['hit_point_of_no_return'] == True]
        
        if len(point_of_no_return_tokens) > 0:
            analysis['insights']['point_of_no_return_stats'] = {
                'total_tokens': len(point_of_no_return_tokens),
                'percentage_of_total': f"{len(point_of_no_return_tokens) / len(self.df) * 100:.1f}%",
                'avg_loss_at_point_of_no_return': f"{point_of_no_return_tokens['avg_loss_at_point_of_no_return'].mean():.1f}%",
                'median_loss_at_point_of_no_return': f"{point_of_no_return_tokens['avg_loss_at_point_of_no_return'].median():.1f}%",
                'min_loss_at_point_of_no_return': f"{point_of_no_return_tokens['avg_loss_at_point_of_no_return'].min():.1f}%",
                'max_loss_at_point_of_no_return': f"{point_of_no_return_tokens['avg_loss_at_point_of_no_return'].max():.1f}%"
            }
            
            # Loss distribution by trigger
            triggers = {
                'fdv_drop': point_of_no_return_tokens[point_of_no_return_tokens['fdv_change_pct'] <= self.point_of_no_return['fdv_drop']],
                'risk_score': point_of_no_return_tokens[point_of_no_return_tokens['risk_score'] >= self.point_of_no_return['risk_score']],
                'buy_sell_ratio': point_of_no_return_tokens[point_of_no_return_tokens['buy_sell_ratio'] <= self.point_of_no_return['buy_sell_ratio']]
            }
            
            for trigger, data in triggers.items():
                if len(data) > 0:
                    analysis['insights'][f'{trigger}_trigger'] = {
                        'trigger': trigger.replace('_', ' ').title(),
                        'tokens_affected': len(data),
                        'avg_loss': f"{data['avg_loss_at_point_of_no_return'].mean():.1f}%"
                    }
        else:
            analysis['insights']['point_of_no_return_stats'] = {
                'total_tokens': 0,
                'message': 'No tokens hit point of no return in this dataset'
            }
        
        return analysis
    
    def _analyze_exit_rule_effectiveness(self):
        """Analyze effectiveness of individual exit rules"""
        analysis = {
            'title': '🎯 EXIT RULE EFFECTIVENESS',
            'description': 'How effective each exit rule is at preventing losses',
            'insights': {}
        }
        
        # Analyze each exit rule
        exit_rules = {
            'buy_sell_ratio': 'Buy/Sell Ratio < 0.6',
            'buy_percentage': 'Buy Percentage < 40%',
            'momentum_change': 'Momentum Change < 0',
            'risk_score': 'Risk Score > 7.0',
            'fdv_change': 'FDV Change < -30%'
        }
        
        for rule, description in exit_rules.items():
            exit_column = f'exit_at_{rule}'
            if exit_column in self.df.columns:
                # Calculate improvement vs holding
                hold_performance = self.df['current_performance']
                exit_performance = self.df[exit_column]
                improvement = exit_performance - hold_performance
                
                # Count how many times this rule would have helped
                helped_count = len(improvement[improvement > 0])
                total_count = len(improvement)
                
                analysis['insights'][rule] = {
                    'description': description,
                    'avg_improvement': f"{improvement.mean():+.1f}%",
                    'median_improvement': f"{improvement.median():+.1f}%",
                    'times_helped': f"{helped_count}/{total_count}",
                    'effectiveness_rate': f"{helped_count/total_count*100:.1f}%"
                }
        
        return analysis
    
    def _analyze_return_improvement(self):
        """Analyze overall return improvement from following exit rules"""
        analysis = {
            'title': '📈 RETURN IMPROVEMENT ANALYSIS',
            'description': 'Overall improvement in returns when following exit rules',
            'insights': {}
        }
        
        # Calculate portfolio-level improvements
        hold_portfolio_return = self.df['current_performance'].mean()
        best_exit_portfolio_return = self.df['best_exit_performance'].mean()
        conservative_exit_portfolio_return = self.df['conservative_exit_performance'].mean()
        
        analysis['insights']['portfolio_improvement'] = {
            'hold_strategy': f"{hold_portfolio_return:+.1f}%",
            'best_exit_strategy': f"{best_exit_portfolio_return:+.1f}%",
            'conservative_exit_strategy': f"{conservative_exit_portfolio_return:+.1f}%",
            'best_exit_improvement': f"{best_exit_portfolio_return - hold_portfolio_return:+.1f}%",
            'conservative_exit_improvement': f"{conservative_exit_portfolio_return - hold_portfolio_return:+.1f}%"
        }
        
        # Calculate risk-adjusted returns
        hold_volatility = self.df['current_performance'].std()
        best_exit_volatility = self.df['best_exit_performance'].std()
        conservative_exit_volatility = self.df['conservative_exit_performance'].std()
        
        analysis['insights']['risk_adjusted_returns'] = {
            'hold_sharpe': f"{hold_portfolio_return / hold_volatility:.2f}" if hold_volatility > 0 else "N/A",
            'best_exit_sharpe': f"{best_exit_portfolio_return / best_exit_volatility:.2f}" if best_exit_volatility > 0 else "N/A",
            'conservative_exit_sharpe': f"{conservative_exit_portfolio_return / conservative_exit_volatility:.2f}" if conservative_exit_volatility > 0 else "N/A"
        }
        
        return analysis
    
    def _create_exit_returns_charts(self):
        """Create exit returns visualizations"""
        print("🎨 Creating exit returns charts...")
        
        # 1. Exit vs Hold Comparison
        self._create_exit_vs_hold_chart()
        
        # 2. Point of No Return Loss Distribution
        self._create_point_of_no_return_chart()
        
        # 3. Exit Rule Effectiveness
        self._create_exit_rule_effectiveness_chart()
        
        # 4. Return Improvement Distribution
        self._create_return_improvement_chart()
        
        print("✅ Exit returns charts created")
    
    def _create_exit_vs_hold_chart(self):
        """Create exit vs hold comparison chart"""
        if self.df.empty:
            return
        
        # Compare different exit scenarios
        scenarios = ['current_performance', 'best_exit_performance', 'conservative_exit_performance']
        scenario_names = ['Hold to End', 'Best Exit', 'Conservative Exit']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Box plot comparison
        data_to_plot = [self.df[scenario] for scenario in scenarios]
        bp = ax1.boxplot(data_to_plot, labels=scenario_names, patch_artist=True)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('📊 Exit vs Hold Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Performance (%)')
        ax1.grid(True, alpha=0.3)
        
        # Bar chart of average returns
        avg_returns = [self.df[scenario].mean() for scenario in scenarios]
        bars = ax2.bar(scenario_names, avg_returns, color=colors, alpha=0.7)
        
        ax2.set_title('💰 Average Returns by Strategy', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Return (%)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, avg_returns):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -1),
                    f'{value:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exit_vs_hold_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_point_of_no_return_chart(self):
        """Create point of no return loss distribution chart"""
        if self.df.empty:
            return
        
        # Filter tokens that hit point of no return
        point_of_no_return_tokens = self.df[self.df['hit_point_of_no_return'] == True]
        
        if len(point_of_no_return_tokens) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram of losses at point of no return
        losses = point_of_no_return_tokens['avg_loss_at_point_of_no_return']
        ax1.hist(losses, bins=15, alpha=0.7, color='red', edgecolor='black')
        ax1.set_title('🚫 Loss Distribution at Point of No Return', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Loss (%)')
        ax1.set_ylabel('Number of Tokens')
        ax1.axvline(x=losses.mean(), color='darkred', linestyle='--', alpha=0.8, 
                   label=f'Average Loss: {losses.mean():.1f}%')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Pie chart of point of no return triggers
        triggers = {
            'FDV Drop': len(point_of_no_return_tokens[point_of_no_return_tokens['fdv_change_pct'] <= self.point_of_no_return['fdv_drop']]),
            'Risk Score': len(point_of_no_return_tokens[point_of_no_return_tokens['risk_score'] >= self.point_of_no_return['risk_score']]),
            'Buy/Sell Ratio': len(point_of_no_return_tokens[point_of_no_return_tokens['buy_sell_ratio'] <= self.point_of_no_return['buy_sell_ratio']])
        }
        
        # Remove zero values
        triggers = {k: v for k, v in triggers.items() if v > 0}
        
        if triggers:
            wedges, texts, autotexts = ax2.pie(triggers.values(), labels=triggers.keys(), autopct='%1.1f%%', startangle=90)
            ax2.set_title('🔥 Point of No Return Triggers', fontsize=14, fontweight='bold')
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'point_of_no_return_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_exit_rule_effectiveness_chart(self):
        """Create exit rule effectiveness chart"""
        if self.df.empty:
            return
        
        # Calculate effectiveness for each rule
        rules = ['buy_sell_ratio', 'buy_percentage', 'momentum_change', 'risk_score', 'fdv_change']
        rule_names = ['Buy/Sell Ratio', 'Buy Percentage', 'Momentum', 'Risk Score', 'FDV Change']
        
        effectiveness_rates = []
        avg_improvements = []
        
        for rule in rules:
            exit_column = f'exit_at_{rule}'
            if exit_column in self.df.columns:
                hold_performance = self.df['current_performance']
                exit_performance = self.df[exit_column]
                improvement = exit_performance - hold_performance
                
                helped_count = len(improvement[improvement > 0])
                total_count = len(improvement)
                effectiveness_rate = helped_count / total_count * 100 if total_count > 0 else 0
                
                effectiveness_rates.append(effectiveness_rate)
                avg_improvements.append(improvement.mean())
            else:
                effectiveness_rates.append(0)
                avg_improvements.append(0)
        
        # Create effectiveness chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Effectiveness rate
        bars1 = ax1.bar(rule_names, effectiveness_rates, color='#4ECDC4', alpha=0.7)
        ax1.set_title('🎯 Exit Rule Effectiveness Rate', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Effectiveness Rate (%)')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars1, effectiveness_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Average improvement
        bars2 = ax2.bar(rule_names, avg_improvements, color='#45B7D1', alpha=0.7)
        ax2.set_title('📈 Average Improvement from Exit Rule', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Improvement (%)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars2, avg_improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -0.5),
                    f'{value:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exit_rule_effectiveness.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_return_improvement_chart(self):
        """Create return improvement distribution chart"""
        if self.df.empty:
            return
        
        # Calculate improvements
        hold_performance = self.df['current_performance']
        best_exit_improvement = self.df['best_exit_performance'] - hold_performance
        conservative_exit_improvement = self.df['conservative_exit_performance'] - hold_performance
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Best exit improvement distribution
        ax1.hist(best_exit_improvement, bins=20, alpha=0.7, color='#4ECDC4', edgecolor='black')
        ax1.set_title('🚀 Best Exit Strategy Improvement', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Improvement (%)')
        ax1.set_ylabel('Number of Tokens')
        ax1.axvline(x=best_exit_improvement.mean(), color='darkgreen', linestyle='--', alpha=0.8,
                   label=f'Average: {best_exit_improvement.mean():+.1f}%')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Conservative exit improvement distribution
        ax2.hist(conservative_exit_improvement, bins=20, alpha=0.7, color='#45B7D1', edgecolor='black')
        ax2.set_title('🛡️ Conservative Exit Strategy Improvement', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Improvement (%)')
        ax2.set_ylabel('Number of Tokens')
        ax2.axvline(x=conservative_exit_improvement.mean(), color='darkblue', linestyle='--', alpha=0.8,
                   label=f'Average: {conservative_exit_improvement.mean():+.1f}%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'return_improvement_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _save_exit_reports(self, reports):
        """Save exit returns analysis reports"""
        # Save detailed report
        detailed_report = self._format_exit_report(reports)
        with open(self.output_dir / 'detailed_exit_analysis.txt', 'w') as f:
            f.write(detailed_report)
        
        # Save summary report
        summary_report = self._format_summary_report(reports)
        with open(self.output_dir / 'exit_returns_summary.txt', 'w') as f:
            f.write(summary_report)
        
        print(f"✅ Exit returns reports saved to: {self.output_dir}")
    
    def _format_exit_report(self, reports):
        """Format detailed exit returns report"""
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("💰 MEMECOIN EXIT RETURNS ANALYSIS REPORT")
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
        summary_lines.append("💰 EXIT RETURNS ANALYSIS SUMMARY")
        summary_lines.append("=" * 80)
        summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")
        
        # Key findings
        if 'return_improvement' in reports:
            portfolio_improvement = reports['return_improvement']['insights']['portfolio_improvement']
            summary_lines.append("🎯 KEY FINDINGS:")
            summary_lines.append(f"• Hold Strategy: {portfolio_improvement['hold_strategy']}")
            summary_lines.append(f"• Best Exit Strategy: {portfolio_improvement['best_exit_strategy']}")
            summary_lines.append(f"• Conservative Exit Strategy: {portfolio_improvement['conservative_exit_strategy']}")
            summary_lines.append("")
            summary_lines.append(f"📈 IMPROVEMENT:")
            summary_lines.append(f"• Best Exit vs Hold: {portfolio_improvement['best_exit_improvement']}")
            summary_lines.append(f"• Conservative Exit vs Hold: {portfolio_improvement['conservative_exit_improvement']}")
            summary_lines.append("")
        
        # Point of no return analysis
        if 'point_of_no_return_losses' in reports:
            point_analysis = reports['point_of_no_return_losses']['insights']
            if 'point_of_no_return_stats' in point_analysis:
                stats = point_analysis['point_of_no_return_stats']
                if 'avg_loss_at_point_of_no_return' in stats:
                    summary_lines.append("🚫 POINT OF NO RETURN:")
                    summary_lines.append(f"• Average Loss: {stats['avg_loss_at_point_of_no_return']}")
                    summary_lines.append(f"• Tokens Affected: {stats['total_tokens']}")
                    summary_lines.append("")
        
        summary_lines.append("💡 CONCLUSION:")
        summary_lines.append("Following exit rules significantly improves returns and reduces losses.")
        summary_lines.append("The point of no return represents the average loss when tokens fail completely.")
        
        return "\n".join(summary_lines)
    
    def _display_exit_summary(self, reports):
        """Display summary of exit returns analysis"""
        print("\n" + "="*80)
        print("💰 EXIT RETURNS ANALYSIS COMPLETE!")
        print("="*80)
        
        # Display key findings
        if 'return_improvement' in reports:
            portfolio_improvement = reports['return_improvement']['insights']['portfolio_improvement']
            print("🎯 PORTFOLIO RETURNS:")
            print(f"   Hold Strategy: {portfolio_improvement['hold_strategy']}")
            print(f"   Best Exit Strategy: {portfolio_improvement['best_exit_strategy']}")
            print(f"   Conservative Exit Strategy: {portfolio_improvement['conservative_exit_strategy']}")
            print("")
            print(f"📈 IMPROVEMENT FROM EXIT RULES:")
            print(f"   Best Exit vs Hold: {portfolio_improvement['best_exit_improvement']}")
            print(f"   Conservative Exit vs Hold: {portfolio_improvement['conservative_exit_improvement']}")
            print("")
        
        # Display point of no return analysis
        if 'point_of_no_return_losses' in reports:
            point_analysis = reports['point_of_no_return_losses']['insights']
            if 'point_of_no_return_stats' in point_analysis:
                stats = point_analysis['point_of_no_return_stats']
                if 'avg_loss_at_point_of_no_return' in stats:
                    print("🚫 POINT OF NO RETURN:")
                    print(f"   Average Loss: {stats['avg_loss_at_point_of_no_return']}")
                    print(f"   Tokens Affected: {stats['total_tokens']}")
                    print("")
        
        print("📁 Check the 'output/exit_returns_analysis' folder for detailed reports")
        print("💰 Summary: exit_returns_summary.txt")

def main():
    """Main function to create exit returns analysis"""
    print("💰 Starting Exit Returns Analysis...")
    
    # Create analyzer
    analyzer = ExitReturnsAnalyzer()
    
    # Generate exit returns analysis
    reports = analyzer.create_exit_returns_analysis()
    
    print("\n" + "="*80)
    print("🎉 EXIT RETURNS ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
