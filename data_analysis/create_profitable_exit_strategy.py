#!/usr/bin/env python3
"""
Profitable Exit Strategy Creator
Comprehensive exit strategy for profitable memecoin trades
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

class ProfitableExitStrategyCreator:
    """Creates comprehensive exit strategies for profitable trades"""
    
    def __init__(self, results_file: str = "output/adapted_timeseries_analysis.json"):
        self.results_file = Path(results_file)
        self.output_dir = Path("output/profitable_exit_strategy")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.results = self._load_results()
        self.df = self._create_dataframe()
        
        # Color scheme for visualizations
        self.colors = {
            'profit': '#00FF00',      # Green
            'loss': '#FF0000',        # Red
            'neutral': '#FFFF00',     # Yellow
            'take_profit': '#32CD32', # Lime green
            'trailing_stop': '#FFD700' # Gold
        }
        
        # Exit strategy parameters
        self.exit_strategies = {
            'conservative': {
                'take_profit_levels': [25, 50, 100, 200],  # % gains
                'position_sizing': [0.4, 0.3, 0.2, 0.1],  # % of position to sell
                'trailing_stop': 15,  # % below peak
                'time_based_exit': 48  # hours
            },
            'moderate': {
                'take_profit_levels': [50, 100, 200, 500],
                'position_sizing': [0.3, 0.3, 0.25, 0.15],
                'trailing_stop': 20,
                'time_based_exit': 72
            },
            'aggressive': {
                'take_profit_levels': [100, 200, 500, 1000],
                'position_sizing': [0.2, 0.3, 0.3, 0.2],
                'trailing_stop': 25,
                'time_based_exit': 120
            }
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
        """Create DataFrame with exit strategy analysis"""
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
                
                # Calculate exit strategy metrics
                row.update(self._calculate_exit_metrics(row))
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _calculate_exit_metrics(self, row):
        """Calculate exit strategy metrics for a token"""
        metrics = {}
        
        current_return = row.get('fdv_change_pct', 0)
        risk_score = row.get('risk_score', 0)
        momentum_change = row.get('momentum_change', 0)
        buy_sell_ratio = row.get('buy_sell_ratio', 1)
        
        # 1. Profit Category
        if current_return > 100:
            metrics['profit_category'] = 'mega_profit'
        elif current_return > 50:
            metrics['profit_category'] = 'high_profit'
        elif current_return > 25:
            metrics['profit_category'] = 'moderate_profit'
        elif current_return > 10:
            metrics['profit_category'] = 'low_profit'
        elif current_return > 0:
            metrics['profit_category'] = 'break_even'
        else:
            metrics['profit_category'] = 'loss'
        
        # 2. Exit Urgency Score (0-10, higher = more urgent to exit)
        urgency_factors = []
        
        # Risk-based urgency
        if risk_score > 8:
            urgency_factors.append(3)  # High risk = urgent exit
        elif risk_score > 6:
            urgency_factors.append(2)  # Medium risk = moderate urgency
        else:
            urgency_factors.append(0)  # Low risk = no urgency
        
        # Momentum-based urgency
        if momentum_change < -0.5:
            urgency_factors.append(3)  # Declining momentum = urgent
        elif momentum_change < 0:
            urgency_factors.append(2)  # Flat momentum = moderate urgency
        else:
            urgency_factors.append(0)  # Rising momentum = no urgency
        
        # Buy pressure urgency
        if buy_sell_ratio < 0.6:
            urgency_factors.append(2)  # Low buy pressure = urgent
        elif buy_sell_ratio < 0.8:
            urgency_factors.append(1)  # Moderate buy pressure = slight urgency
        else:
            urgency_factors.append(0)  # High buy pressure = no urgency
        
        # Profit-taking urgency (higher profits = more urgent to secure gains)
        if current_return > 200:
            urgency_factors.append(3)  # Mega profits = very urgent
        elif current_return > 100:
            urgency_factors.append(2)  # High profits = urgent
        elif current_return > 50:
            urgency_factors.append(1)  # Moderate profits = slight urgency
        else:
            urgency_factors.append(0)  # Low profits = no urgency
        
        metrics['exit_urgency_score'] = min(sum(urgency_factors), 10)
        
        # 3. Recommended Exit Strategy
        if metrics['exit_urgency_score'] >= 8:
            metrics['recommended_strategy'] = 'immediate_exit'
        elif metrics['exit_urgency_score'] >= 6:
            metrics['recommended_strategy'] = 'aggressive_take_profit'
        elif metrics['exit_urgency_score'] >= 4:
            metrics['recommended_strategy'] = 'moderate_take_profit'
        elif metrics['exit_urgency_score'] >= 2:
            metrics['recommended_strategy'] = 'conservative_take_profit'
        else:
            metrics['recommended_strategy'] = 'hold_and_monitor'
        
        # 4. Take Profit Levels
        if current_return > 0:
            # Calculate optimal take profit levels based on current performance
            base_levels = [current_return * 0.5, current_return * 0.75, current_return, current_return * 1.25]
            metrics['take_profit_levels'] = [max(level, 10) for level in base_levels]  # Minimum 10%
        else:
            metrics['take_profit_levels'] = [25, 50, 100, 200]  # Default levels
        
        # 5. Trailing Stop Level
        if current_return > 0:
            # Dynamic trailing stop based on profit level
            if current_return > 200:
                metrics['trailing_stop'] = 30  # 30% below peak for mega profits
            elif current_return > 100:
                metrics['trailing_stop'] = 25  # 25% below peak for high profits
            elif current_return > 50:
                metrics['trailing_stop'] = 20  # 20% below peak for moderate profits
            else:
                metrics['trailing_stop'] = 15  # 15% below peak for low profits
        else:
            metrics['trailing_stop'] = 20  # Default trailing stop
        
        # 6. Position Sizing for Exit
        if current_return > 0:
            # More aggressive exit for higher profits
            if current_return > 200:
                metrics['exit_position_sizing'] = [0.4, 0.3, 0.2, 0.1]  # Exit faster
            elif current_return > 100:
                metrics['exit_position_sizing'] = [0.3, 0.3, 0.25, 0.15]
            else:
                metrics['exit_position_sizing'] = [0.2, 0.3, 0.3, 0.2]  # Exit slower
        else:
            metrics['exit_position_sizing'] = [0.25, 0.25, 0.25, 0.25]  # Default
        
        # 7. Time-Based Exit
        if metrics['exit_urgency_score'] >= 8:
            metrics['time_based_exit_hours'] = 6  # Very urgent = exit within 6 hours
        elif metrics['exit_urgency_score'] >= 6:
            metrics['time_based_exit_hours'] = 24  # Urgent = exit within 24 hours
        elif metrics['exit_urgency_score'] >= 4:
            metrics['time_based_exit_hours'] = 48  # Moderate = exit within 48 hours
        elif metrics['exit_urgency_score'] >= 2:
            metrics['time_based_exit_hours'] = 72  # Low urgency = exit within 72 hours
        else:
            metrics['time_based_exit_hours'] = 120  # Very low urgency = exit within 5 days
        
        return metrics
    
    def create_exit_strategy_guide(self):
        """Create comprehensive exit strategy guide"""
        print("🎯 Creating Profitable Exit Strategy Guide...")
        
        if self.df.empty:
            print("❌ No data available for analysis")
            return
        
        # Generate strategy sections
        strategy = {}
        
        # 1. Exit Strategy Overview
        strategy['overview'] = self._create_exit_strategy_overview()
        
        # 2. Profit-Based Exit Strategies
        strategy['profit_based'] = self._create_profit_based_strategies()
        
        # 3. Risk-Based Exit Strategies
        strategy['risk_based'] = self._create_risk_based_strategies()
        
        # 4. Momentum-Based Exit Strategies
        strategy['momentum_based'] = self._create_momentum_based_strategies()
        
        # 5. Time-Based Exit Strategies
        strategy['time_based'] = self._create_time_based_strategies()
        
        # 6. Practical Implementation
        strategy['implementation'] = self._create_implementation_guide()
        
        # Create visualizations
        self._create_exit_strategy_charts()
        
        # Save strategy
        self._save_exit_strategy(strategy)
        
        # Display summary
        self._display_exit_strategy_summary(strategy)
        
        return strategy
    
    def _create_exit_strategy_overview(self):
        """Create overview of exit strategies"""
        overview = {
            'title': '🎯 PROFITABLE EXIT STRATEGY OVERVIEW',
            'description': 'Comprehensive exit strategies for locking in profits',
            'strategies': {}
        }
        
        # Define exit strategy types
        overview['strategies'] = {
            'immediate_exit': {
                'name': 'Immediate Exit',
                'description': 'Exit entire position immediately when conditions are met',
                'when_to_use': 'High urgency score (8-10), extreme risk, or mega profits',
                'risk_level': 'Low (locks in gains)',
                'potential_downside': 'May miss further upside'
            },
            'aggressive_take_profit': {
                'name': 'Aggressive Take Profit',
                'description': 'Exit 40-30-20-10% of position at profit levels',
                'when_to_use': 'High urgency score (6-7), high profits, declining momentum',
                'risk_level': 'Low-Medium',
                'potential_downside': 'Reduces position size quickly'
            },
            'moderate_take_profit': {
                'name': 'Moderate Take Profit',
                'description': 'Exit 30-30-25-15% of position at profit levels',
                'when_to_use': 'Medium urgency score (4-5), moderate profits, stable momentum',
                'risk_level': 'Medium',
                'potential_downside': 'Balanced approach'
            },
            'conservative_take_profit': {
                'name': 'Conservative Take Profit',
                'description': 'Exit 20-30-30-20% of position at profit levels',
                'when_to_use': 'Low urgency score (2-3), low profits, rising momentum',
                'risk_level': 'Medium-High',
                'potential_downside': 'May hold too long in declining markets'
            },
            'hold_and_monitor': {
                'name': 'Hold and Monitor',
                'description': 'Maintain position with tight trailing stops',
                'when_to_use': 'Very low urgency score (0-1), strong fundamentals, rising momentum',
                'risk_level': 'High',
                'potential_downside': 'May give back significant profits'
            }
        }
        
        return overview
    
    def _create_profit_based_strategies(self):
        """Create profit-based exit strategies"""
        strategies = {
            'title': '💰 PROFIT-BASED EXIT STRATEGIES',
            'description': 'Exit strategies based on profit levels',
            'insights': {}
        }
        
        # Analyze profit categories
        profit_categories = self.df['profit_category'].value_counts()
        strategies['insights']['profit_distribution'] = profit_categories.to_dict()
        
        # Calculate optimal exit points for each profit category
        profit_strategies = {}
        
        for category in ['mega_profit', 'high_profit', 'moderate_profit', 'low_profit']:
            category_data = self.df[self.df['profit_category'] == category]
            if len(category_data) > 0:
                avg_return = category_data['fdv_change_pct'].mean()
                avg_urgency = category_data['exit_urgency_score'].mean()
                
                profit_strategies[category] = {
                    'avg_return': f"{avg_return:.1f}%",
                    'avg_urgency': f"{avg_urgency:.1f}/10",
                    'recommended_exit': category_data['recommended_strategy'].mode().iloc[0] if len(category_data) > 0 else 'unknown',
                    'take_profit_levels': category_data['take_profit_levels'].iloc[0] if len(category_data) > 0 else [25, 50, 100, 200],
                    'trailing_stop': f"{category_data['trailing_stop'].iloc[0]:.0f}%" if len(category_data) > 0 else 20
                }
        
        strategies['insights']['profit_strategies'] = profit_strategies
        
        # Profit vs Exit Urgency correlation
        if len(self.df) > 1:
            correlation = self.df['fdv_change_pct'].corr(self.df['exit_urgency_score'])
            strategies['insights']['profit_urgency_correlation'] = f"{correlation:.3f}"
        
        return strategies
    
    def _create_risk_based_strategies(self):
        """Create risk-based exit strategies"""
        strategies = {
            'title': '⚠️ RISK-BASED EXIT STRATEGIES',
            'description': 'Exit strategies based on risk assessment',
            'insights': {}
        }
        
        # Analyze risk levels
        risk_levels = pd.cut(self.df['risk_score'], bins=[0, 4, 6, 8, 10], labels=['Low', 'Medium', 'High', 'Critical'])
        risk_distribution = risk_levels.value_counts()
        strategies['insights']['risk_distribution'] = risk_distribution.to_dict()
        
        # Risk-based exit recommendations
        risk_strategies = {}
        
        for risk_level in ['Low', 'Medium', 'High', 'Critical']:
            if risk_level in risk_distribution:
                risk_data = self.df[risk_levels == risk_level]
                if len(risk_data) > 0:
                    avg_return = risk_data['fdv_change_pct'].mean()
                    avg_urgency = risk_data['exit_urgency_score'].mean()
                    
                    risk_strategies[risk_level] = {
                        'avg_return': f"{avg_return:.1f}%",
                        'avg_urgency': f"{avg_urgency:.1f}/10",
                        'recommended_exit': risk_data['recommended_strategy'].mode().iloc[0] if len(risk_data) > 0 else 'unknown',
                        'trailing_stop': f"{risk_data['trailing_stop'].iloc[0]:.0f}%" if len(risk_data) > 0 else 20,
                        'time_based_exit': f"{risk_data['time_based_exit_hours'].iloc[0]:.0f} hours" if len(risk_data) > 0 else 48
                    }
        
        strategies['insights']['risk_strategies'] = risk_strategies
        
        return strategies
    
    def _create_momentum_based_strategies(self):
        """Create momentum-based exit strategies"""
        strategies = {
            'title': '📈 MOMENTUM-BASED EXIT STRATEGIES',
            'description': 'Exit strategies based on momentum indicators',
            'insights': {}
        }
        
        # Analyze momentum changes
        momentum_changes = self.df['momentum_change']
        strategies['insights']['momentum_stats'] = {
            'avg_momentum_change': f"{momentum_changes.mean():.2f}",
            'positive_momentum': (momentum_changes > 0).sum(),
            'negative_momentum': (momentum_changes < 0).sum(),
            'flat_momentum': (momentum_changes == 0).sum()
        }
        
        # Momentum-based exit recommendations
        momentum_strategies = {}
        
        # Positive momentum
        positive_data = self.df[momentum_changes > 0]
        if len(positive_data) > 0:
            momentum_strategies['positive_momentum'] = {
                'count': len(positive_data),
                'avg_return': f"{positive_data['fdv_change_pct'].mean():.1f}%",
                'avg_urgency': f"{positive_data['exit_urgency_score'].mean():.1f}/10",
                'recommended_exit': positive_data['recommended_strategy'].mode().iloc[0] if len(positive_data) > 0 else 'unknown'
            }
        
        # Negative momentum
        negative_data = self.df[momentum_changes < 0]
        if len(negative_data) > 0:
            momentum_strategies['negative_momentum'] = {
                'count': len(negative_data),
                'avg_return': f"{negative_data['fdv_change_pct'].mean():.1f}%",
                'avg_urgency': f"{negative_data['exit_urgency_score'].mean():.1f}/10",
                'recommended_exit': negative_data['recommended_strategy'].mode().iloc[0] if len(negative_data) > 0 else 'unknown'
            }
        
        strategies['insights']['momentum_strategies'] = momentum_strategies
        
        return strategies
    
    def _create_time_based_strategies(self):
        """Create time-based exit strategies"""
        strategies = {
            'title': '⏰ TIME-BASED EXIT STRATEGIES',
            'description': 'Exit strategies based on time horizons',
            'insights': {}
        }
        
        # Analyze time-based exit recommendations
        time_exits = self.df['time_based_exit_hours']
        strategies['insights']['time_exit_stats'] = {
            'avg_exit_time': f"{time_exits.mean():.1f} hours",
            'immediate_exits': (time_exits <= 6).sum(),
            'day_trades': (time_exits <= 24).sum(),
            'swing_trades': (time_exits <= 72).sum(),
            'position_trades': (time_exits > 72).sum()
        }
        
        # Time-based strategy recommendations
        time_strategies = {}
        
        time_categories = {
            'immediate': time_exits <= 6,
            'day_trade': (time_exits > 6) & (time_exits <= 24),
            'swing': (time_exits > 24) & (time_exits <= 72),
            'position': time_exits > 72
        }
        
        for category, condition in time_categories.items():
            category_data = self.df[condition]
            if len(category_data) > 0:
                time_strategies[category] = {
                    'count': len(category_data),
                    'avg_return': f"{category_data['fdv_change_pct'].mean():.1f}%",
                    'avg_urgency': f"{category_data['exit_urgency_score'].mean():.1f}/10",
                    'recommended_exit': category_data['recommended_strategy'].mode().iloc[0] if len(category_data) > 0 else 'unknown'
                }
        
        strategies['insights']['time_strategies'] = time_strategies
        
        return strategies
    
    def _create_implementation_guide(self):
        """Create practical implementation guide"""
        guide = {
            'title': '🚀 PRACTICAL IMPLEMENTATION GUIDE',
            'description': 'Step-by-step guide to implement exit strategies',
            'steps': {}
        }
        
        guide['steps']['setup'] = {
            'step': 'Setup Exit Strategy',
            'actions': [
                'Set up take-profit orders at calculated levels',
                'Configure trailing stop-loss orders',
                'Set time-based exit reminders',
                'Prepare position sizing for partial exits'
            ]
        }
        
        guide['steps']['monitoring'] = {
            'step': 'Continuous Monitoring',
            'actions': [
                'Track exit urgency score changes',
                'Monitor momentum indicators',
                'Watch for risk score increases',
                'Check profit level achievements'
            ]
        }
        
        guide['steps']['execution'] = {
            'step': 'Strategy Execution',
            'actions': [
                'Execute partial exits at take-profit levels',
                'Activate trailing stops when profits exceed 50%',
                'Implement time-based exits for urgent situations',
                'Scale out positions based on urgency score'
            ]
        }
        
        guide['steps']['risk_management'] = {
            'step': 'Risk Management',
            'actions': [
                'Never exit more than 50% of position at once',
                'Always maintain trailing stops for remaining positions',
                'Consider market conditions before major exits',
                'Have backup exit plans for extreme scenarios'
            ]
        }
        
        return guide
    
    def _create_exit_strategy_charts(self):
        """Create visualizations for exit strategies"""
        print("🎨 Creating exit strategy charts...")
        
        # 1. Exit Urgency Distribution
        self._create_urgency_chart()
        
        # 2. Profit vs Exit Strategy
        self._create_profit_exit_chart()
        
        # 3. Risk vs Exit Strategy
        self._create_risk_exit_chart()
        
        # 4. Exit Strategy Dashboard
        self._create_exit_dashboard()
        
        print("✅ Exit strategy charts created")
    
    def _create_urgency_chart(self):
        """Create exit urgency distribution chart"""
        if self.df.empty:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Exit urgency distribution
        urgency_scores = self.df['exit_urgency_score']
        ax1.hist(urgency_scores, bins=11, alpha=0.7, color='orange', edgecolor='black')
        ax1.set_title('🚨 Exit Urgency Score Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Exit Urgency Score (0-10)')
        ax1.set_ylabel('Number of Tokens')
        ax1.axvline(x=urgency_scores.mean(), color='red', linestyle='--', alpha=0.8,
                   label=f'Average: {urgency_scores.mean():.1f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Urgency vs Return
        ax2.scatter(urgency_scores, self.df['fdv_change_pct'], 
                   c=['green' if x > 0 else 'red' for x in self.df['fdv_change_pct']], alpha=0.7, s=80)
        ax2.set_title('📊 Exit Urgency vs Return', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Exit Urgency Score')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exit_urgency_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_profit_exit_chart(self):
        """Create profit vs exit strategy chart"""
        if self.df.empty:
            return
        
        plt.figure(figsize=(14, 8))
        
        # Group by profit category and exit strategy
        profit_exit_data = self.df.groupby(['profit_category', 'recommended_strategy']).size().unstack(fill_value=0)
        
        # Create stacked bar chart
        profit_exit_data.plot(kind='bar', stacked=True, figsize=(14, 8), 
                            color=['red', 'orange', 'yellow', 'lightgreen', 'darkgreen'])
        plt.title('💰 Profit Category vs Exit Strategy', fontsize=16, fontweight='bold')
        plt.xlabel('Profit Category')
        plt.ylabel('Number of Tokens')
        plt.xticks(rotation=45)
        plt.legend(title='Exit Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'profit_exit_strategy.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_risk_exit_chart(self):
        """Create risk vs exit strategy chart"""
        if self.df.empty:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create risk categories
        risk_categories = pd.cut(self.df['risk_score'], bins=[0, 4, 6, 8, 10], labels=['Low', 'Medium', 'High', 'Critical'])
        
        # Group by risk category and exit strategy
        risk_exit_data = self.df.groupby([risk_categories, 'recommended_strategy']).size().unstack(fill_value=0)
        
        # Create stacked bar chart
        risk_exit_data.plot(kind='bar', stacked=True, figsize=(12, 8), 
                           color=['darkgreen', 'lightgreen', 'yellow', 'orange', 'red'])
        plt.title('⚠️ Risk Level vs Exit Strategy', fontsize=16, fontweight='bold')
        plt.xlabel('Risk Level')
        plt.ylabel('Number of Tokens')
        plt.xticks(rotation=45)
        plt.legend(title='Exit Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_exit_strategy.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_exit_dashboard(self):
        """Create comprehensive exit strategy dashboard"""
        if self.df.empty:
            return
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Exit Urgency Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        urgency_scores = self.df['exit_urgency_score']
        ax1.hist(urgency_scores, bins=11, alpha=0.7, color='orange', edgecolor='black')
        ax1.set_title('🚨 Exit Urgency Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Urgency Score')
        
        # 2. Profit Category Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        profit_categories = self.df['profit_category'].value_counts()
        ax2.pie(profit_categories.values, labels=profit_categories.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('💰 Profit Categories', fontsize=12, fontweight='bold')
        
        # 3. Exit Strategy Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        exit_strategies = self.df['recommended_strategy'].value_counts()
        ax3.bar(exit_strategies.index, exit_strategies.values, color='lightblue', alpha=0.7)
        ax3.set_title('🎯 Recommended Exit Strategies', fontsize=12, fontweight='bold')
        ax3.set_xticklabels(exit_strategies.index, rotation=45)
        
        # 4. Urgency vs Return
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(urgency_scores, self.df['fdv_change_pct'], alpha=0.7, s=60)
        ax4.set_title('📊 Urgency vs Return', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Urgency Score')
        ax4.set_ylabel('Return (%)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Risk vs Urgency
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.scatter(self.df['risk_score'], urgency_scores, alpha=0.7, s=60)
        ax5.set_title('⚠️ Risk vs Urgency', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Risk Score')
        ax5.set_ylabel('Exit Urgency')
        ax5.grid(True, alpha=0.3)
        
        # 6. Trailing Stop Distribution
        ax6 = fig.add_subplot(gs[1, 2])
        trailing_stops = self.df['trailing_stop']
        ax6.hist(trailing_stops, bins=10, alpha=0.7, color='gold', edgecolor='black')
        ax6.set_title('📉 Trailing Stop Levels', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Trailing Stop (%)')
        
        # 7. Summary Statistics
        ax7 = fig.add_subplot(gs[2:, :])
        ax7.axis('off')
        
        # Create summary text
        summary_text = f"""
        🎯 EXIT STRATEGY DASHBOARD SUMMARY
        
        📊 OVERVIEW:
        • Total Tokens Analyzed: {len(self.df)}
        • Average Exit Urgency: {urgency_scores.mean():.1f}/10
        • Profitable Tokens: {(self.df['fdv_change_pct'] > 0).sum()}
        
        🚨 URGENCY ANALYSIS:
        • High Urgency (8-10): {(urgency_scores >= 8).sum()} tokens
        • Medium Urgency (4-7): {((urgency_scores >= 4) & (urgency_scores < 8)).sum()} tokens
        • Low Urgency (0-3): {(urgency_scores < 4).sum()} tokens
        
        💰 PROFIT ANALYSIS:
        • Mega Profits (>200%): {(self.df['fdv_change_pct'] > 200).sum()} tokens
        • High Profits (100-200%): {((self.df['fdv_change_pct'] > 100) & (self.df['fdv_change_pct'] <= 200)).sum()} tokens
        • Moderate Profits (50-100%): {((self.df['fdv_change_pct'] > 50) & (self.df['fdv_change_pct'] <= 100)).sum()} tokens
        
        ⚠️ RISK ANALYSIS:
        • High Risk (>6): {(self.df['risk_score'] > 6).sum()} tokens
        • Critical Risk (>8): {(self.df['risk_score'] > 8).sum()} tokens
        
        💡 KEY INSIGHTS:
        • Exit urgency correlates with profit levels and risk
        • Higher profits require more aggressive exit strategies
        • Risk management is crucial for protecting gains
        • Time-based exits complement profit-based strategies
        """
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.suptitle('🎯 COMPREHENSIVE EXIT STRATEGY DASHBOARD', fontsize=18, fontweight='bold')
        plt.savefig(self.output_dir / 'exit_strategy_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _save_exit_strategy(self, strategy):
        """Save exit strategy guide"""
        # Save summary guide
        summary = self._format_exit_strategy_summary(strategy)
        with open(self.output_dir / 'exit_strategy_summary.txt', 'w') as f:
            f.write(summary)
        
        print(f"✅ Exit strategy guide saved to: {self.output_dir}")
    
    def _format_exit_strategy_summary(self, strategy):
        """Format exit strategy summary"""
        lines = []
        lines.append("=" * 80)
        lines.append("🎯 PROFITABLE EXIT STRATEGY GUIDE SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Strategy overview
        if 'overview' in strategy:
            lines.append("🎯 EXIT STRATEGY TYPES:")
            strategies = strategy['overview']['strategies']
            for key, strat in strategies.items():
                lines.append(f"• {strat['name']}: {strat['description']}")
                lines.append(f"  When to use: {strat['when_to_use']}")
                lines.append(f"  Risk level: {strat['risk_level']}")
                lines.append("")
        
        # Key insights
        if 'profit_based' in strategy:
            profit_stats = strategy['profit_based']['insights']['profit_strategies']
            lines.append("💰 PROFIT-BASED STRATEGIES:")
            for category, data in profit_stats.items():
                lines.append(f"• {category}: {data['avg_return']} return, {data['avg_urgency']} urgency")
                lines.append(f"  Recommended: {data['recommended_exit']}")
                lines.append("")
        
        if 'risk_based' in strategy:
            risk_stats = strategy['risk_based']['insights']['risk_strategies']
            lines.append("⚠️ RISK-BASED STRATEGIES:")
            for level, data in risk_stats.items():
                lines.append(f"• {level} Risk: {data['avg_return']} return, {data['avg_urgency']} urgency")
                lines.append(f"  Exit time: {data['time_based_exit']}")
                lines.append("")
        
        lines.append("💡 IMPLEMENTATION TIPS:")
        lines.append("• Set take-profit orders at calculated levels")
        lines.append("• Use trailing stops to protect profits")
        lines.append("• Monitor exit urgency scores continuously")
        lines.append("• Scale out positions based on risk and momentum")
        
        return "\n".join(lines)
    
    def _display_exit_strategy_summary(self, strategy):
        """Display exit strategy summary"""
        print("\n" + "="*80)
        print("🎯 PROFITABLE EXIT STRATEGY GUIDE COMPLETE!")
        print("="*80)
        
        # Display key strategies
        if 'overview' in strategy:
            print("🎯 EXIT STRATEGY TYPES COVERED:")
            strategies = strategy['overview']['strategies']
            for key, strat in strategies.items():
                print(f"   ✅ {strat['name']}")
            print("")
        
        # Display key insights
        if 'profit_based' in strategy:
            profit_stats = strategy['profit_based']['insights']['profit_strategies']
            print("💰 PROFIT-BASED INSIGHTS:")
            for category, data in profit_stats.items():
                print(f"   {category}: {data['avg_return']} return, {data['avg_urgency']} urgency")
            print("")
        
        if 'risk_based' in strategy:
            risk_stats = strategy['risk_based']['insights']['risk_strategies']
            print("⚠️ RISK-BASED INSIGHTS:")
            for level, data in risk_stats.items():
                print(f"   {level} Risk: {data['avg_return']} return, {data['avg_urgency']} urgency")
            print("")
        
        print("📁 Check the 'output/profitable_exit_strategy' folder for complete guide")
        print("🎯 Summary: exit_strategy_summary.txt")

def main():
    """Main function to create exit strategy guide"""
    print("🎯 Starting Profitable Exit Strategy Creation...")
    
    # Create guide
    creator = ProfitableExitStrategyCreator()
    
    # Generate guide
    strategy = creator.create_exit_strategy_guide()
    
    print("\n" + "="*80)
    print("🎉 PROFITABLE EXIT STRATEGY GUIDE COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
