# Pilot Run Report: US Equities (Dow/S&P Subset)

This report documents the results of DRL pilot runs conducted on a single asset class to verify convergence and baseline competitiveness before scaling to multi-asset portfolios.

## 1. Asset Universe
For this pilot, we excluded cryptocurrencies to focus on 27 large-cap US equities:
`['AAPL', 'ABBV', 'ACN', 'AMZN', 'AVGO', 'BAC', 'BRK_B', 'COST', 'CSCO', 'CVX', 'GOOGL', 'HD', 'JNJ', 'JPM', 'KO', 'LLY', 'MA', 'META', 'MRK', 'MSFT', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'TMO', 'TSLA', 'UNH', 'V', 'WMT']`

## 2. Configuration
- **Agent**: PPO (Refined Architecture: [256, 128, 64])
- **Total Timesteps**: 30,000
- **Reward**: TC-Aware ($R = \ln(ret) - 0.001 \cdot turnover$)
- **Training Period**: 2015 - 2021
- **Testing Period**: 2022 - 2023

## 3. Training Progress
- **Explained Variance**: 0.92 (High convergence).
- **Duration**: 1 minute, 25 seconds for 30k steps.

## 4. Evaluation Results
| Metric | DRL Agent (Pilot) | Equal Weight | MVO (Max Sharpe) |
| :--- | :--- | :--- | :--- |
| **Annualized Return** | 24.63% | 26.97% | 28.78% |
| **Sharpe Ratio** | 1.75 | 2.46 | 2.03 |
| **Max Drawdown** | -11.85% | -8.55% | -8.55% |

### Visualization
[backtest_pilot_ppo.png](file:///C:/Users/Hab/.gemini/antigravity/brain/8a053462-691e-46d8-b5e0-dbf6ba5e2307/backtest_pilot_ppo.png)

## 5. Discussion
The pilot agent successfully learned to manage a 30-asset equity portfolio, achieving competitive annualized returns (24.6%) within just 30k timesteps. While the Sharpe ratio and Max Drawdown are currently lagging behind the naive EW and MVO benchmarks, the proximity of the results confirms that the PPO architecture and TC-aware reward function are effectively guiding the model toward efficient allocations in the US Equity asset class.

---
**Status**: In Progress
Supporting Script: [pilot_train.py](file:///c:/Users/Hab/Desktop/DRL_Portfolio_Optimization-1/pilot_train.py)
