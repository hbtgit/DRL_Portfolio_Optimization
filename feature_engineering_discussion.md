# Feature Engineering Discussion

This document outlines the feature engineering process performed on the historical financial data for the 32 selected assets (27 equities and 5 cryptocurrencies).

## 1. Data Preprocessing
- **Source Alignment**: Data was loaded from raw CSV files in `data/equities/` and `data/crypto/`.
- **Messy Headers**: Raw files contained a multi-line header structure (3 rows of metadata). The loading logic was adapted to skip these and manually assign standard column names: `Date`, `Close`, `High`, `Low`, `Open`, `Volume`.
- **Handling Missing Values**:
    - Technical indicators like SMA and RSI require a warm-up period (e.g., 30 days for SMA-30).
    - To preserve the start of the dataset (2015-01-02), we used a per-ticker **Forward-Fill then Backward-Fill** approach. This ensures that the first 30 days have constant values based on the first available calculation, preventing the DRL agent from seeing `NaN` values.

## 2. Feature Extraction
A comprehensive set of technical indicators was computed for each asset to provide the DRL agent with trend, momentum, and volatility signals:

| Category | Indicator | Description |
| :--- | :--- | :--- |
| **Trend** | SMA (10, 30) | Simple Moving Averages for short and medium-term trend detection. |
| **Trend/Momentum** | MACD | Moving Average Convergence Divergence (12, 26 EMA) with signal line (9 EMA). |
| **Momentum** | RSI (14) | Relative Strength Index to identify overbought or oversold conditions. |
| **Volatility** | Bollinger Bands | Middle (20-day SMA), Std Dev, Upper, and Lower bands. |
| **Price/Reward** | Log Returns | Natural log of price changes ($ln(P_t / P_{t-1})$) for scale-invariant reward signals. |

## 3. Normalization Strategy
- **Per-Ticker Z-Score**: Financial data across different assets (e.g., AAPL at ~$25 vs BTC at ~$30,000) has vastly different scales. 
- We applied **Z-score normalization** ($z = (x - \mu) / \sigma$) independently for each ticker.
- This transforms all features into a standard range where 0 represents the ticker's average and $\pm 1, 2, ...$ represent standard deviations from that average.
- This is crucial for Deep Reinforcement Learning (DRL) stability, ensuring that high-priced assets do not dominate the gradient updates compared to low-priced assets.

## 4. Verification Results
The final processed dataset is saved in two formats:
1. `data/processed/portfolio_data_raw.csv`: All indicators in their original scale.
2. `data/processed/portfolio_data_normalized.csv`: All indicators scaled for the DRL agent.

**Dataset Statistics:**
- **Total Rows**: 73,451
- **Asset Classes**: Equities (27) and Crypto (5)
- **Time Range**: 2015-01-01 to ~2024 (Exact end date varies by download).
- **Features per Row**: 11 normalized technical features + Date/Ticker metadata.

## Next Steps
The data is now ready for the DRL environment. We will proceed to implement the `TradingEnv` and start the agent training phase.
