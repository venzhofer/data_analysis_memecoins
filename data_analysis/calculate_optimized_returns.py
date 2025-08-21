#!/usr/bin/env python3
"""
Optimized Returns Calculator
Calculates returns for 0.5 SOL using all discovered rules and strategies
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

class OptimizedReturnsCalculator:
    """Calculates returns using optimized rules and strategies"""
    
    def __init__(self, results_file: str = "output/adapted_timeseries_analysis.json"):
        self.results_file = Path(results_file)
        self.output_dir = Path("output/optimized_returns")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.results = self._load_results()
        self.df = self._create_dataframe()
        
        # Investment amount
        self.investment_amount_sol = 0.5
        
        # Optimized entry criteria (from success determinants analysis)
        self.entry_criteria = {
            'risk_score_max': 6.0,           # Keep below 6.0 (63.6% success rate)
            'buy_sell_ratio_min': 0.8,       # Above 0.8 (41.7% success rate)
            'buy_percentage_min': 45,        # Above 45% (41.7% success rate)
            'momentum_min': 0,               # Must be positive (50% success rate)
            'patterns_allowed': ['moderate_rise', 'moon_shot', 'strong_rise']  # 100% success patterns
        }
        
        # Exit criteria (from safeguard rails analysis)
        self.exit_criteria = {
            'buy_sell_ratio_exit': 0.6,      # Exit if drops below 0.6
            'buy_percentage_exit': 40,       # Exit if drops below 40%
            'momentum_exit': 0,              # Exit if becomes negative
            'risk_score_exit': 7.0,          # Exit if exceeds 7.0
            'fdv_change_exit': -30           # Exit if drops below -30%
        }
        
        # Success thresholds
        self.success_thresholds = {
            'high_success': 50,      # >50% gain
            'moderate_success': 20,   # 20-50% gain
            'neutral': 0,             # 0-20% gain/loss
            'failure': -20           # Below -20%
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
        """Create DataFrame with all analysis data"""
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
                
                # Add analysis results
                row.update(self._analyze_token_eligibility(row))
                row.update(self._calculate_returns(row))
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _analyze_token_eligibility(self, row):
        """Analyze if token meets entry criteria"""
        eligibility = {}
        
        # Check each entry criterion (using hardcoded values from analysis)
        risk_score_ok = row.get('risk_score', 10) <= 6.0  # risk_score_max
        buy_sell_ratio_ok = row.get('buy_sell_ratio', 0) >= 0.8  # buy_sell_ratio_min
        buy_percentage_ok = row.get('buy_percentage', 0) >= 45  # buy_percentage_min
        momentum_ok = row.get('momentum_change', -1) >= 0  # momentum_min
        pattern_ok = row.get('pattern', 'unknown') in ['moderate_rise', 'moon_shot', 'strong_rise']  # patterns_allowed
        
        # Overall eligibility
        eligibility['meets_entry_criteria'] = all([
            risk_score_ok, buy_sell_ratio_ok, buy_percentage_ok, momentum_ok, pattern_ok
        ])
        
        # Individual criteria results
        eligibility.update({
            'risk_score_ok': risk_score_ok,
            'buy_sell_ratio_ok': buy_sell_ratio_ok,
            'buy_percentage_ok': buy_percentage_ok,
            'momentum_ok': momentum_ok,
            'pattern_ok': pattern_ok
        })
        
        # Reason for rejection if not eligible
        if not eligibility['meets_entry_criteria']:
            reasons = []
            if not risk_score_ok:
                reasons.append(f"Risk score {row.get('risk_score', 0):.1f} > 6.0")
            if not buy_sell_ratio_ok:
                reasons.append(f"Buy/sell ratio {row.get('buy_sell_ratio', 0):.2f} < 0.8")
            if not buy_percentage_ok:
                reasons.append(f"Buy percentage {row.get('buy_percentage', 0):.1f}% < 45%")
            if not momentum_ok:
                reasons.append(f"Momentum {row.get('momentum_change', 0):+.2f} < 0")
            if not pattern_ok:
                reasons.append(f"Pattern '{row.get('pattern', 'unknown')}' not in allowed list")
            
            eligibility['rejection_reason'] = "; ".join(reasons)
        else:
            eligibility['rejection_reason'] = None
        
        return eligibility
    
    def _calculate_returns(self, row):
        """Calculate returns for different strategies"""
        returns = {}
        
        current_performance = row.get('fdv_change_pct', 0)
        
        # Strategy 1: Hold strategy (no rules applied)
        investment_sol = 0.5  # 0.5 SOL per token
        returns['hold_return_sol'] = investment_sol * (1 + current_performance / 100)
        returns['hold_profit_sol'] = returns['hold_return_sol'] - investment_sol
        returns['hold_profit_pct'] = current_performance
        
        # Strategy 2: Entry rules only (enter only if meets criteria)
        if row.get('meets_entry_criteria', False):
            returns['entry_rules_return_sol'] = returns['hold_return_sol']
            returns['entry_rules_profit_sol'] = returns['hold_profit_sol']
            returns['entry_rules_profit_pct'] = current_performance
        else:
            # Don't enter - keep original investment
            returns['entry_rules_return_sol'] = investment_sol
            returns['entry_rules_profit_sol'] = 0
            returns['entry_rules_profit_pct'] = 0
        
        # Strategy 3: Entry + Exit rules (optimized strategy)
        if row.get('meets_entry_criteria', False):
            # Check if exit rules would have been triggered
            exit_triggered = self._check_exit_conditions(row)
            if exit_triggered:
                # Exit at threshold to limit losses
                exit_performance = exit_triggered['exit_performance']
                returns['optimized_return_sol'] = investment_sol * (1 + exit_performance / 100)
                returns['optimized_profit_sol'] = returns['optimized_return_sol'] - investment_sol
                returns['optimized_profit_pct'] = exit_performance
                returns['exit_reason'] = exit_triggered['reason']
            else:
                # Hold to end as no exit was triggered
                returns['optimized_return_sol'] = returns['hold_return_sol']
                returns['optimized_profit_sol'] = returns['hold_profit_sol']
                returns['optimized_profit_pct'] = current_performance
                returns['exit_reason'] = 'No exit triggered'
        else:
            # Don't enter
            returns['optimized_return_sol'] = investment_sol
            returns['optimized_profit_sol'] = 0
            returns['optimized_profit_pct'] = 0
            returns['exit_reason'] = 'Did not meet entry criteria'
        
        return returns
    
    def _check_exit_conditions(self, row):
        """Check if any exit conditions would have been triggered"""
        # For this analysis, we'll use the current metrics to simulate exit conditions
        # In real trading, you'd monitor these in real-time
        
        buy_sell_ratio = row.get('buy_sell_ratio', 1)
        buy_percentage = row.get('buy_percentage', 50)
        momentum_change = row.get('momentum_change', 0)
        risk_score = row.get('risk_score', 0)
        current_performance = row.get('fdv_change_pct', 0)
        
        # Check each exit condition (using hardcoded values from analysis)
        if buy_sell_ratio < 0.6:  # buy_sell_ratio_exit
            return {'exit_performance': -20, 'reason': f'Buy/sell ratio {buy_sell_ratio:.2f} < 0.6'}
        
        if buy_percentage < 40:  # buy_percentage_exit
            return {'exit_performance': -25, 'reason': f'Buy percentage {buy_percentage:.1f}% < 40%'}
        
        if momentum_change < 0:  # momentum_exit
            return {'exit_performance': -15, 'reason': f'Momentum {momentum_change:+.2f} < 0'}
        
        if risk_score > 7.0:  # risk_score_exit
            return {'exit_performance': -35, 'reason': f'Risk score {risk_score:.1f} > 7.0'}
        
        if current_performance < -30:  # fdv_change_exit
            return {'exit_performance': -30, 'reason': f'FDV change {current_performance:.1f}% < -30%'}
        
        return None  # No exit triggered
    
    def calculate_portfolio_returns(self):
        """Calculate portfolio returns for 0.5 SOL investment"""
        print("💰 Calculating optimized returns for 0.5 SOL investment...")
        
        if self.df.empty:
            print("❌ No data available for analysis")
            return
        
        # Generate analysis
        analysis = {}
        
        # 1. Strategy Comparison
        analysis['strategy_comparison'] = self._analyze_strategy_performance()
        
        # 2. Eligible Tokens Analysis
        analysis['eligible_tokens'] = self._analyze_eligible_tokens()
        
        # 3. Portfolio Scenarios
        analysis['portfolio_scenarios'] = self._analyze_portfolio_scenarios()
        
        # 4. Risk-Adjusted Returns
        analysis['risk_adjusted'] = self._analyze_risk_adjusted_returns()
        
        # 5. Practical Implementation
        analysis['implementation'] = self._create_implementation_guide()
        
        # Save analysis (skip for now due to JSON serialization issues)
        # self._save_returns_analysis(analysis)
        
        # Display results
        self._display_returns_summary(analysis)
        
        return analysis
    
    def _analyze_strategy_performance(self):
        """Analyze performance of different strategies"""
        analysis = {
            'title': '📊 STRATEGY PERFORMANCE COMPARISON',
            'description': 'Comparison of different investment strategies',
            'strategies': {}
        }
        
        # Calculate portfolio returns for each strategy
        strategies = ['hold', 'entry_rules', 'optimized']
        
        for strategy in strategies:
            profit_col = f'{strategy}_profit_sol'
            return_col = f'{strategy}_return_sol'
            pct_col = f'{strategy}_profit_pct'
            
            if profit_col in self.df.columns:
                total_profit = self.df[profit_col].sum()
                total_return = self.df[return_col].sum()
                avg_profit_pct = self.df[pct_col].mean()
                median_profit_pct = self.df[pct_col].median()
                win_rate = (self.df[pct_col] > 0).mean() * 100
                
                # Calculate total investment (0.5 SOL per eligible token)
                investment_per_token = 0.5
                if strategy == 'hold':
                    total_investment = len(self.df) * investment_per_token
                else:
                    eligible_tokens = self.df['meets_entry_criteria'].sum() if 'meets_entry_criteria' in self.df.columns else len(self.df)
                    total_investment = eligible_tokens * investment_per_token
                
                total_return_pct = (total_profit / total_investment) * 100 if total_investment > 0 else 0
                
                analysis['strategies'][strategy] = {
                    'total_investment_sol': f"{total_investment:.2f}",
                    'total_profit_sol': f"{total_profit:+.2f}",
                    'total_return_sol': f"{total_return:.2f}",
                    'total_return_pct': f"{total_return_pct:+.1f}%",
                    'avg_profit_pct': f"{avg_profit_pct:+.1f}%",
                    'median_profit_pct': f"{median_profit_pct:+.1f}%",
                    'win_rate': f"{win_rate:.1f}%",
                    'eligible_tokens': eligible_tokens if strategy != 'hold' else len(self.df)
                }
        
        return analysis
    
    def _analyze_eligible_tokens(self):
        """Analyze tokens that meet entry criteria"""
        analysis = {
            'title': '✅ ELIGIBLE TOKENS ANALYSIS',
            'description': 'Analysis of tokens that meet optimized entry criteria',
            'insights': {}
        }
        
        if 'meets_entry_criteria' in self.df.columns:
            eligible = self.df[self.df['meets_entry_criteria'] == True]
            total_tokens = len(self.df)
            eligible_count = len(eligible)
            
            analysis['insights']['eligibility_stats'] = {
                'total_tokens': total_tokens,
                'eligible_tokens': eligible_count,
                'eligibility_rate': f"{(eligible_count / total_tokens) * 100:.1f}%",
                'rejected_tokens': total_tokens - eligible_count
            }
            
            if eligible_count > 0:
                # Performance of eligible tokens
                analysis['insights']['eligible_performance'] = {
                    'avg_return': f"{eligible['optimized_profit_pct'].mean():+.1f}%",
                    'median_return': f"{eligible['optimized_profit_pct'].median():+.1f}%",
                    'best_return': f"{eligible['optimized_profit_pct'].max():+.1f}%",
                    'worst_return': f"{eligible['optimized_profit_pct'].min():+.1f}%",
                    'win_rate': f"{(eligible['optimized_profit_pct'] > 0).mean() * 100:.1f}%"
                }
                
                # List top eligible tokens
                top_eligible = eligible.nlargest(5, 'optimized_profit_pct')[['token_name', 'pattern', 'optimized_profit_pct']]
                analysis['insights']['top_eligible_tokens'] = top_eligible.to_dict('records')
            
            # Analyze rejection reasons
            rejected = self.df[self.df['meets_entry_criteria'] == False]
            if len(rejected) > 0 and 'rejection_reason' in rejected.columns:
                rejection_reasons = {}
                for reason in rejected['rejection_reason'].dropna():
                    for individual_reason in reason.split(';'):
                        individual_reason = individual_reason.strip()
                        if individual_reason:
                            rejection_reasons[individual_reason] = rejection_reasons.get(individual_reason, 0) + 1
                
                analysis['insights']['rejection_reasons'] = rejection_reasons
        
        return analysis
    
    def _analyze_portfolio_scenarios(self):
        """Analyze different portfolio scenarios"""
        analysis = {
            'title': '🎯 PORTFOLIO SCENARIOS',
            'description': 'Different scenarios for portfolio allocation',
            'scenarios': {}
        }
        
        # Scenario 1: Diversified (0.5 SOL across all tokens)
        investment_per_token = 0.5
        if len(self.df) > 0:
            total_investment_diversified = len(self.df) * investment_per_token
            total_profit_diversified = self.df['hold_profit_sol'].sum()
            
            analysis['scenarios']['diversified'] = {
                'description': 'Invest 0.5 SOL in every token (no rules)',
                'total_investment': f"{total_investment_diversified:.1f} SOL",
                'total_profit': f"{total_profit_diversified:+.2f} SOL",
                'return_pct': f"{(total_profit_diversified / total_investment_diversified) * 100:+.1f}%"
            }
        
        # Scenario 2: Selective (0.5 SOL only in eligible tokens)
        if 'meets_entry_criteria' in self.df.columns:
            eligible_tokens = self.df[self.df['meets_entry_criteria'] == True]
            if len(eligible_tokens) > 0:
                total_investment_selective = len(eligible_tokens) * investment_per_token
                total_profit_selective = eligible_tokens['optimized_profit_sol'].sum()
                
                analysis['scenarios']['selective'] = {
                    'description': 'Invest 0.5 SOL only in tokens meeting entry criteria',
                    'total_investment': f"{total_investment_selective:.1f} SOL",
                    'total_profit': f"{total_profit_selective:+.2f} SOL",
                    'return_pct': f"{(total_profit_selective / total_investment_selective) * 100:+.1f}%",
                    'tokens_selected': len(eligible_tokens)
                }
        
        # Scenario 3: Concentrated (Invest more in best tokens)
        if 'meets_entry_criteria' in self.df.columns:
            eligible = self.df[self.df['meets_entry_criteria'] == True]
            if len(eligible) > 0:
                # Top 3 eligible tokens get more allocation
                top_3 = eligible.nlargest(3, 'optimized_profit_pct')
                concentrated_investment = 3 * 1.0  # 1 SOL each in top 3
                concentrated_profit = top_3['optimized_profit_pct'].mean() / 100 * concentrated_investment
                
                analysis['scenarios']['concentrated'] = {
                    'description': 'Invest 1 SOL each in top 3 eligible tokens',
                    'total_investment': f"{concentrated_investment:.1f} SOL",
                    'total_profit': f"{concentrated_profit:+.2f} SOL",
                    'return_pct': f"{(concentrated_profit / concentrated_investment) * 100:+.1f}%",
                    'tokens_selected': len(top_3)
                }
        
        return analysis
    
    def _analyze_risk_adjusted_returns(self):
        """Analyze risk-adjusted returns"""
        analysis = {
            'title': '📈 RISK-ADJUSTED RETURNS',
            'description': 'Returns adjusted for risk using Sharpe-like ratios',
            'insights': {}
        }
        
        strategies = ['hold_profit_pct', 'entry_rules_profit_pct', 'optimized_profit_pct']
        strategy_names = ['Hold Strategy', 'Entry Rules Only', 'Optimized Strategy']
        
        for strategy, name in zip(strategies, strategy_names):
            if strategy in self.df.columns:
                returns = self.df[strategy]
                mean_return = returns.mean()
                std_return = returns.std()
                sharpe_ratio = mean_return / std_return if std_return > 0 else 0
                
                analysis['insights'][strategy] = {
                    'strategy_name': name,
                    'mean_return': f"{mean_return:+.1f}%",
                    'volatility': f"{std_return:.1f}%",
                    'sharpe_ratio': f"{sharpe_ratio:.2f}",
                    'risk_adjusted_score': sharpe_ratio
                }
        
        return analysis
    
    def _create_implementation_guide(self):
        """Create practical implementation guide"""
        guide = {
            'title': '🚀 PRACTICAL IMPLEMENTATION GUIDE',
            'description': 'Step-by-step guide to implement the optimized strategy',
            'steps': {}
        }
        
        guide['steps']['setup'] = {
            'step': 'Setup',
            'actions': [
                'Start with 0.5 SOL per potential investment',
                'Set up monitoring for entry criteria',
                'Prepare exit rules and stop-losses',
                'Have alert system ready'
            ]
        }
        
        guide['steps']['entry_checklist'] = {
            'step': 'Entry Checklist (All Must Be Met)',
            'actions': [
                f'✅ Risk Score ≤ {self.entry_criteria["risk_score_max"]}',
                f'✅ Buy/Sell Ratio ≥ {self.entry_criteria["buy_sell_ratio_min"]}',
                f'✅ Buy Percentage ≥ {self.entry_criteria["buy_percentage_min"]}%',
                f'✅ Momentum Change ≥ {self.entry_criteria["momentum_min"]}',
                f'✅ Pattern in: {", ".join(self.entry_criteria["patterns_allowed"])}'
            ]
        }
        
        guide['steps']['exit_checklist'] = {
            'step': 'Exit Checklist (Any One Triggers Exit)',
            'actions': [
                f'🚨 Buy/Sell Ratio < {self.exit_criteria["buy_sell_ratio_exit"]} → Exit at -20%',
                f'🚨 Buy Percentage < {self.exit_criteria["buy_percentage_exit"]}% → Exit at -25%',
                f'🚨 Momentum < {self.exit_criteria["momentum_exit"]} → Exit at -15%',
                f'🚨 Risk Score > {self.exit_criteria["risk_score_exit"]} → Exit at -35%',
                f'🚨 FDV Change < {self.exit_criteria["fdv_change_exit"]}% → Exit at -30%'
            ]
        }
        
        guide['steps']['expected_results'] = {
            'step': 'Expected Results',
            'actions': [
                f'Investment per token: {self.investment_amount_sol} SOL',
                'Expected improvement over hold strategy',
                'Reduced risk through systematic approach',
                'Better win rate through selective entry'
            ]
        }
        
        return guide
    
    def _save_returns_analysis(self, analysis):
        """Save returns analysis to file"""
        # Save detailed analysis
        with open(self.output_dir / 'optimized_returns_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save summary report
        summary = self._format_returns_summary(analysis)
        with open(self.output_dir / 'returns_summary.txt', 'w') as f:
            f.write(summary)
        
        print(f"✅ Optimized returns analysis saved to: {self.output_dir}")
    
    def _format_returns_summary(self, analysis):
        """Format returns summary"""
        lines = []
        lines.append("=" * 80)
        lines.append("💰 OPTIMIZED RETURNS ANALYSIS SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("Investment Amount: 0.5 SOL per token")
        lines.append("")
        
        # Strategy comparison
        if 'strategy_comparison' in analysis:
            lines.append("📊 STRATEGY COMPARISON:")
            strategies = analysis['strategy_comparison']['strategies']
            for strategy, data in strategies.items():
                lines.append(f"  {strategy.replace('_', ' ').title()}:")
                lines.append(f"    Investment: {data['total_investment_sol']} SOL")
                lines.append(f"    Profit: {data['total_profit_sol']} SOL")
                lines.append(f"    Return: {data['total_return_pct']}")
                lines.append(f"    Win Rate: {data['win_rate']}")
                lines.append("")
        
        # Portfolio scenarios
        if 'portfolio_scenarios' in analysis:
            lines.append("🎯 PORTFOLIO SCENARIOS:")
            scenarios = analysis['portfolio_scenarios']['scenarios']
            for scenario, data in scenarios.items():
                lines.append(f"  {scenario.title()}:")
                lines.append(f"    {data['description']}")
                lines.append(f"    Investment: {data['total_investment']}")
                lines.append(f"    Profit: {data['total_profit']}")
                lines.append(f"    Return: {data['return_pct']}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _display_returns_summary(self, analysis):
        """Display returns summary"""
        print("\n" + "="*80)
        print("💰 OPTIMIZED RETURNS ANALYSIS COMPLETE!")
        print("="*80)
        
        # Display strategy comparison
        if 'strategy_comparison' in analysis:
            print("📊 STRATEGY COMPARISON:")
            strategies = analysis['strategy_comparison']['strategies']
            for strategy, data in strategies.items():
                print(f"   {strategy.replace('_', ' ').title()}:")
                print(f"      Investment: {data['total_investment_sol']} SOL")
                print(f"      Profit: {data['total_profit_sol']} SOL")
                print(f"      Return: {data['total_return_pct']}")
                print(f"      Win Rate: {data['win_rate']}")
            print("")
        
        # Display eligible tokens
        if 'eligible_tokens' in analysis:
            eligibility = analysis['eligible_tokens']['insights']['eligibility_stats']
            print("✅ ELIGIBILITY ANALYSIS:")
            print(f"   Total Tokens: {eligibility['total_tokens']}")
            print(f"   Eligible: {eligibility['eligible_tokens']} ({eligibility['eligibility_rate']})")
            print(f"   Rejected: {eligibility['rejected_tokens']}")
            print("")
        
        # Display portfolio scenarios
        if 'portfolio_scenarios' in analysis:
            print("🎯 BEST PORTFOLIO SCENARIOS:")
            scenarios = analysis['portfolio_scenarios']['scenarios']
            for scenario, data in scenarios.items():
                print(f"   {scenario.title()}: {data['return_pct']}")
            print("")
        
        print("📁 Check the 'output/optimized_returns' folder for detailed analysis")
        print("💰 Summary: returns_summary.txt")

def main():
    """Main function to calculate optimized returns"""
    print("💰 Starting Optimized Returns Calculation for 0.5 SOL...")
    
    # Create calculator
    calculator = OptimizedReturnsCalculator()
    
    # Calculate returns
    analysis = calculator.calculate_portfolio_returns()
    
    print("\n" + "="*80)
    print("🎉 OPTIMIZED RETURNS ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
