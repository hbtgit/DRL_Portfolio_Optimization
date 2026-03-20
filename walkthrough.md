# Feature Engineering Walkthrough

This walkthrough demonstrates the completion of the feature engineering phase for the DRL Portfolio Optimization project.

## 1. Feature Engineering Script
The [feature_engineering.py](file:///c:/Users/Hab/Desktop/DRL_Portfolio_Optimization-1/feature_engineering.py) script was implemented to automate the processing of all 32 assets.

**Key operations handled:**
- Automated parsing of multi-line CSV headers from custom downloads.
- Calculation of Technical Indicators: SMA (10, 30), RSI, MACD, and Bollinger Bands.
- Multi-step NaN handling (Forward-fill/Backward-fill) to maximize data availability.
- **Per-Ticker Z-score Normalization** to level the playing field between high-priced (Crypto) and low-priced (Equity) assets.

```python
# Per-ticker normalization logic
norm_df = full_df.groupby('Ticker', group_keys=False).apply(z_score_ticker)
```

## 2. Processed Datasets
Two datasets were generated in `data/processed/`:
- `portfolio_data_raw.csv`: Raw values for all indicators.
- `portfolio_data_normalized.csv`: Scaled values ready for the Reinforcement Learning agent.

## 3. Findings and Discussion
A detailed analysis of the feature engineering results, including a description of each indicator and the normalization strategy, can be found in the [feature_engineering_discussion.md](file:///c:/Users/Hab/Desktop/DRL_Portfolio_Optimization-1/feature_engineering_discussion.md) file.

## 5. Env Upgrade: Trading Environment
The custom [trading_env.py](file:///c:/Users/Hab/Desktop/DRL_Portfolio_Optimization-1/trading_env.py) was implemented and verified.

**High-Level Specs:**
- **Framework**: Gymnasium-compatible `TradingEnv` class.
- **Assets**: 32 (27 Equity + 5 Crypto).
- **Observation Space**: 
  - `market`: `(10, 32, 12)` - Last 10 days of 12 normalized features for all 32 assets.
  - `portfolio`: `(32,)` - Current portfolio weights.
- **Action Space**: `(32,)` continuous weights, automatically normalized to sum=1.
- **Reward Function**: Log portfolio return minus transaction cost penalty ($\lambda = 0.001$).

**Verification Results:**
The environment was tested with a random agent using [test_env.py](file:///c:/Users/Hab/Desktop/DRL_Portfolio_Optimization-1/test_env.py).
- Verified consistent transitions and reward calculations.
- Confirmed handling of price-driven weight changes before rebalancing.

The environment is now ready for DRL agent training (PPO/DDPG).

## 6. Validation: Return Calculations
Verified that the `TradingEnv` return and reward formulas match theoretical expectations using [verify_returns.py](file:///c:/Users/Hab/Desktop/DRL_Portfolio_Optimization-1/verify_returns.py).

**Validation Findings:**
- **Turnover & TC**: Portfolio rebalancing costs were cross-referenced with manual calculations. The environment correctly identifies transaction fees as $PV'_t \cdot \lambda \cdot Turnover$.
- **Reward Accuracy**: The log-return approximation used ($\ln(R_{gross}) - \lambda \cdot Turnover$) was found to have a negligible error ($<0.5\%$) compared to the "exact" theoretical formula.
- **Report**: Full details available in [return_validation_report.md](file:///c:/Users/Hab/Desktop/DRL_Portfolio_Optimization-1/return_validation_report.md).

## 7. Sprint 2: Agent Implementation & Refinement
Initialized the DRL training pipeline using `Stable-Baselines3`.

**Architecture Refinements:**
- Documented in [agent_architecture_refinement.md](file:///c:/Users/Hab/Desktop/DRL_Portfolio_Optimization-1/agent_architecture_refinement.md).
- Decoupled **Policy** and **Value** heads for PPO to improve convergence.
- Implemented 3-layer MLP ([256, 128, 64]) to handle high-dimensional 32-asset state Space.

**Initial Training & Backtest:**
- **PPO Training**: Completed 30,000 steps with an explained variance of ~0.85.
- **Backtest Results (2022-2023)**:
  - **DRL Agent**: -14.6% Annualized Return (Sharpe: -0.86).
  - **Baseline (Equal Weight)**: Outperformed the agent in this initial low-step run.
- **Visualization**: [backtest_ppo.png](file:///C:/Users/Hab/.gemini/antigravity/brain/8a053462-691e-46d8-b5e0-dbf6ba5e2307/backtest_ppo.png)

**Next Steps:**
- Hyperparameter tuning (PPO/DDPG) to improve risk-adjusted returns.
- Long-duration training (200k+ steps).
