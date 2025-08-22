# 🚀 Trading Rules, Calculations & Dataset Analysis
## Advanced Pattern Analysis for Memecoin Trading

---

## 📊 Dataset Overview

### **Data Sources**
- **Tokens Webhook:** `https://bridge.being-labs.com/webhook/590bfec6-4a94-4db1-aaf5-7f874bb6dcf4`
- **Trading Webhook:** `https://bridge.being-labs.com/webhook/95db9432-d7c8-4b07-a0cc-949f7765ebf0`

### **Dataset Statistics**
- **Total Tokens Analyzed:** 26
- **Total Data Points:** 170,000+ historical price records
- **Total Patterns Identified:** 33,405
- **Time Range:** August 14-16, 2025
- **Data Frequency:** High-frequency (multiple data points per minute)

### **Token Breakdown by Data Volume**
| Token | Data Points | Patterns Found | Success Rate |
|-------|-------------|----------------|--------------|
| Fork Chain | 21,974 | 19,190 | 89.8% ⭐ |
| Chin | 22,846 | 14,031 | 14.3% |
| Cabal | 22,614 | 4,178 | 9.8% |
| Goonpocalypse | 22,280 | 5,878 | 10.7% |
| Anal Intelligence | 22,267 | 3,663 | 11.2% |
| a retarded token | 22,189 | 9,052 | 9.7% |
| Gake | 21,949 | 11,635 | 9.0% |
| This will bait the retail | 21,914 | 4,300 | 8.9% |
| deadly weapon | 15,607 | 9,897 | N/A |
| Albert | 15,155 | 5,216 | N/A |

---

## 🎯 Trading Pattern Rules & Calculations

### **1. Golden Cross Pattern** ⭐
**Average Return:** 325.19% | **Success Rate:** 25.4%

#### **Entry Conditions:**
```python
if (current['price'] > current['sma_5'] > current['sma_20'] and 
    current['rsi'] < 70 and current['rsi'] > 30):
    signal = "BULLISH"
```

#### **Technical Indicators:**
- **SMA 5:** 5-period Simple Moving Average
- **SMA 20:** 20-period Simple Moving Average  
- **RSI:** Relative Strength Index (14-period)
- **Price Position:** Must be above both moving averages

#### **Calculation:**
```python
# Moving Averages
sma_5 = price.rolling(window=5).mean()
sma_20 = price.rolling(window=20).mean()

# RSI
delta = price.diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
```

#### **Trading Logic:**
- **Entry:** When price crosses above both SMAs with RSI in neutral zone
- **Target:** 325% average return (based on historical data)
- **Stop Loss:** Below SMA 20
- **Risk:** Medium (25.4% success rate)

---

### **2. Death Cross Pattern** ⭐
**Average Return:** 287.08% | **Success Rate:** 74.6% 🏆

#### **Entry Conditions:**
```python
if (current['price'] < current['sma_5'] < current['sma_20'] and 
    current['rsi'] > 30):
    signal = "BEARISH"
```

#### **Technical Indicators:**
- **SMA 5:** 5-period Simple Moving Average
- **SMA 20:** 20-period Simple Moving Average
- **RSI:** Relative Strength Index (14-period)
- **Price Position:** Must be below both moving averages

#### **Trading Logic:**
- **Entry:** When price crosses below both SMAs with RSI above oversold
- **Target:** 287% average return (based on historical data)
- **Stop Loss:** Above SMA 20
- **Risk:** Low (74.6% success rate - HIGHEST RELIABILITY)

---

### **3. Breakout Above Bollinger Bands**
**Average Return:** 173.35% | **Success Rate:** 18.9%

#### **Entry Conditions:**
```python
if (current['price'] > current['bb_upper'] * 0.98 and 
    current['rsi'] > 60):
    signal = "BULLISH"
```

#### **Technical Indicators:**
- **Bollinger Bands:** 20-period with 2 standard deviations
- **RSI:** Relative Strength Index (14-period)
- **Price Position:** Must break above upper band

#### **Calculation:**
```python
# Bollinger Bands
sma = price.rolling(window=20).mean()
std = price.rolling(window=20).std()
upper = sma + (std * 2)
lower = sma - (std * 2)

# Breakout detection
breakout = price > upper * 0.98  # 2% tolerance
```

#### **Trading Logic:**
- **Entry:** When price breaks above upper Bollinger Band with strong RSI
- **Target:** 173% average return
- **Stop Loss:** Below middle band (SMA 20)
- **Risk:** High (18.9% success rate)

---

### **4. Bounce Off Support**
**Average Return:** 171.66% | **Success Rate:** 17.2%

#### **Entry Conditions:**
```python
if (current['price_vs_support'] < 5 and 
    current['rsi'] < 40):
    signal = "BULLISH"
```

#### **Technical Indicators:**
- **Support Level:** 20-period rolling minimum
- **RSI:** Relative Strength Index (14-period)
- **Price Position:** Within 5% of support level

#### **Calculation:**
```python
# Support Level
support_level = price.rolling(window=20).min()

# Price vs Support
price_vs_support = (price / support_level - 1) * 100

# Bounce detection
bounce = price_vs_support < 5 and rsi < 40
```

#### **Trading Logic:**
- **Entry:** When price bounces off support with oversold RSI
- **Target:** 172% average return
- **Stop Loss:** Below support level
- **Risk:** High (17.2% success rate)

---

### **5. MACD Bearish Crossover**
**Average Return:** 162.94% | **Success Rate:** 83.4% ⭐

#### **Entry Conditions:**
```python
if (current['macd'] < current['macd_signal'] and 
    current['macd_histogram'] < prev['macd_histogram']):
    signal = "BEARISH"
```

#### **Technical Indicators:**
- **MACD:** 12, 26, 9 exponential moving averages
- **MACD Signal:** 9-period EMA of MACD
- **MACD Histogram:** MACD - Signal line

#### **Calculation:**
```python
# MACD
ema_fast = price.ewm(span=12).mean()
ema_slow = price.ewm(span=26).mean()
macd = ema_fast - ema_slow
macd_signal = macd.ewm(span=9).mean()
macd_histogram = macd - macd_signal

# Bearish crossover
bearish = macd < macd_signal and histogram_decreasing
```

#### **Trading Logic:**
- **Entry:** When MACD crosses below signal line with decreasing histogram
- **Target:** 163% average return
- **Stop Loss:** Above MACD signal line
- **Risk:** Low (83.4% success rate - SECOND HIGHEST RELIABILITY)

---

## ⏰ Optimal Entry Timing Analysis

### **Hourly Success Rates (UTC)**
| Hour | Success Rate | Average Return | Pattern Count |
|------|--------------|----------------|---------------|
| **05:00** | **54.2%** ⭐ | **295.47%** | 225 |
| **04:00** | **41.4%** | **304.85%** | 2,075 |
| **03:00** | **28.4%** | **227.76%** | 2,370 |
| **02:00** | **27.1%** | **277.73%** | 2,675 |
| **00:00** | **24.9%** | **142.79%** | 2,891 |

### **Timing Rules:**
1. **Best Entry Time:** 05:00 UTC (54.2% success rate)
2. **Second Best:** 04:00 UTC (41.4% success rate)
3. **Avoid:** 17:00 UTC (3.0% success rate)
4. **High Volume Hours:** 19:00-23:00 UTC (most patterns)

---

## 🏆 Token Performance Rankings

### **Top 5 Most Reliable Tokens**

#### **1. Fork Chain** 🥇
- **Success Rate:** 89.8%
- **Average Return:** 1,371.69%
- **Patterns Analyzed:** 19,190
- **Risk Level:** Very Low
- **Best Patterns:** All patterns show high success rates

#### **2. Louber Gronger** 🥈
- **Success Rate:** 33.4%
- **Average Return:** -30.91%
- **Patterns Analyzed:** 1,399
- **Risk Level:** Medium
- **Note:** High success rate but negative returns

#### **3. Chin** 🥉
- **Success Rate:** 14.3%
- **Average Return:** -52.89%
- **Patterns Analyzed:** 14,031
- **Risk Level:** High
- **Note:** Large dataset but low success rate

---

## 📈 Risk Management & Position Sizing

### **Success Rate Categories**
- **High Reliability (70%+):** Death Cross, MACD Bearish Crossover
- **Medium Reliability (25-50%):** Golden Cross, Louber Gronger token
- **Low Reliability (<25%):** Breakout patterns, Support bounces

### **Position Sizing Recommendations**
```python
# Position size based on success rate
if success_rate >= 70:
    position_size = "LARGE"      # 10-15% of portfolio
elif success_rate >= 40:
    position_size = "MEDIUM"     # 5-10% of portfolio
else:
    position_size = "SMALL"      # 2-5% of portfolio
```

### **Stop Loss Guidelines**
- **Golden Cross:** Below SMA 20 (20-period moving average)
- **Death Cross:** Above SMA 20
- **Bollinger Breakout:** Below middle band
- **Support Bounce:** Below support level
- **MACD Crossover:** Above/below signal line

---

## 🔍 Technical Indicator Calculations

### **Moving Averages**
```python
# Simple Moving Average
def calculate_sma(prices, window):
    return prices.rolling(window=window).mean()

# Exponential Moving Average  
def calculate_ema(prices, span):
    return prices.ewm(span=span).mean()
```

### **RSI (Relative Strength Index)**
```python
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
```

### **Bollinger Bands**
```python
def calculate_bollinger_bands(prices, window=20, num_std=2):
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower
```

### **MACD (Moving Average Convergence Divergence)**
```python
def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    return macd, macd_signal
```

---

## 💰 Profitability Analysis

### **Return Distribution by Pattern**
| Pattern | Mean Return | Std Dev | Min | Max | Count |
|---------|-------------|---------|-----|-----|-------|
| Golden Cross | 325.19% | 1,177.24% | -99.63% | 7,994.24% | 690 |
| Death Cross | 287.08% | 1,103.03% | -99.03% | 10,565.11% | 418 |
| Breakout Above BB | 173.35% | 763.20% | -99.67% | 9,275.78% | 8,104 |
| Bounce Off Support | 171.66% | 915.67% | -99.63% | 13,489.83% | 14,779 |
| MACD Bearish | 162.94% | 919.02% | -99.68% | 9,610.89% | 2,977 |

### **Key Insights:**
1. **Golden Cross** provides highest average returns (325%)
2. **Death Cross** has highest success rate (74.6%)
3. **Bearish signals** are more reliable than bullish (82.3% vs 17.2%)
4. **Fork Chain** is the most profitable token overall

---

## 🎯 Trading Strategy Summary

### **Primary Strategy: Bearish Momentum Trading**
1. **Focus on Death Cross patterns** (74.6% success rate)
2. **Trade Fork Chain token** (89.8% success rate)
3. **Enter at 05:00 UTC** (54.2% success rate)
4. **Use MACD bearish crossovers** for confirmation (83.4% success rate)

### **Secondary Strategy: Bullish Reversals**
1. **Golden Cross entries** (25.4% success, 325% average return)
2. **Support level bounces** (17.2% success, 172% average return)
3. **Bollinger Band breakouts** (18.9% success, 173% average return)

### **Risk Management:**
- **Position Size:** 5-15% based on pattern reliability
- **Stop Loss:** Technical levels (SMAs, Bollinger Bands, Support/Resistance)
- **Time Filter:** Focus on 03:00-05:00 UTC for best success rates
- **Token Selection:** Prioritize Fork Chain and other high-success tokens

---

## 📊 Data Quality & Limitations

### **Data Strengths:**
- **Large Dataset:** 170,000+ data points across 26 tokens
- **High Frequency:** Multiple data points per minute
- **Recent Data:** August 2025 (current market conditions)
- **Multiple Timeframes:** 5-minute to 24-hour analysis

### **Data Limitations:**
- **Limited Historical Data:** Only 3 days of data
- **Market Conditions:** Bull market period may skew results
- **Token Selection:** Focus on newer memecoins
- **Success Rate:** Overall 23.8% success rate indicates high volatility

### **Recommendations for Use:**
1. **Combine multiple patterns** for higher success probability
2. **Use stop losses** to manage risk
3. **Monitor market conditions** for pattern effectiveness
4. **Regular re-analysis** as more data becomes available

---

*This analysis is based on historical data from August 14-16, 2025. Past performance does not guarantee future results. Always use proper risk management and consider market conditions before trading.*
