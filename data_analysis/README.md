# Token Performance Analyzer

This tool analyzes token trading data to determine which tokens died, rose, or had no point of return after launch. It uses webhook APIs to fetch token information and 5-second trading data for comprehensive analysis.

## Features

- **Token Discovery**: Fetches all available tokens from the Being Labs bridge
- **Trading Data Analysis**: Retrieves 5-second interval trading data for each token
- **Performance Classification**: Categorizes tokens into:
  - **DIED**: Tokens that dropped >50% (no point of return)
  - **ROSE**: Tokens that gained >5% after launch
  - **DROPPED**: Tokens that lost 5-50%
  - **STABLE**: Tokens with minimal price movement
- **Visual Analytics**: Generates charts and graphs for pattern analysis
- **Detailed Reporting**: Saves comprehensive results to JSON files

## Installation

1. Install Python 3.8+ if not already installed
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Analysis
```bash
python token_analyzer.py
```

This will:
- Fetch token information from the webhook
- Analyze up to 50 tokens (configurable)
- Generate performance charts
- Save detailed results to `token_analysis_results.json`
- Display summary statistics

### Custom Analysis
You can modify the `main()` function in `token_analyzer.py` to:
- Change the number of tokens analyzed (`max_tokens` parameter)
- Adjust performance thresholds
- Modify the analysis criteria

## Webhook Endpoints

- **Tokens Info**: `https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4`
- **Trading Data**: `https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0?token={token_address}`

## Output Files

1. **`token_analysis_results.json`**: Complete analysis results
2. **`token_analysis_charts.png`**: Performance visualization charts

## Analysis Criteria

### Performance Patterns
- **strong_rise**: >20% price increase
- **moderate_rise**: 5-20% price increase
- **stable**: -5% to +5% price change
- **moderate_drop**: -5% to -20% price change
- **significant_drop**: -20% to -50% price change
- **died**: >-50% price change (no recovery)

### Key Metrics
- Initial vs. final price
- Maximum gain/loss percentages
- Price volatility
- Recovery patterns
- Data point count

## Example Output

```
TOKEN ANALYSIS SUMMARY
============================================================
Total tokens analyzed: 45
Success rate: 90.0%

PERFORMANCE CATEGORIES:
------------------------------
DIED: 12 tokens (26.7%)
ROSE: 18 tokens (40.0%)
DROPPED: 10 tokens (22.2%)
STABLE: 5 tokens (11.1%)

KEY INSIGHTS:
============================================================
• 26.7% of tokens DIED (no point of return)
• 40.0% of tokens ROSE after launch
• 22.2% of tokens DROPPED significantly
• 11.1% of tokens were STABLE
```

## Configuration

You can modify the following parameters in the code:
- `max_tokens`: Maximum number of tokens to analyze
- Performance thresholds for pattern classification
- API request delays and timeouts
- Recovery threshold percentages

## Error Handling

The tool includes robust error handling for:
- Network timeouts and failures
- Invalid data formats
- Missing token information
- API rate limiting

## Dependencies

- `requests`: HTTP requests to webhook APIs
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `matplotlib`: Chart generation
- `seaborn`: Enhanced plotting styles
- `plotly`: Interactive visualizations

## Troubleshooting

1. **Network Errors**: Check internet connection and webhook availability
2. **Data Format Issues**: Verify webhook response structure
3. **Memory Issues**: Reduce `max_tokens` for large datasets
4. **API Rate Limiting**: Increase delays between requests

## License

This tool is provided as-is for educational and analysis purposes.
