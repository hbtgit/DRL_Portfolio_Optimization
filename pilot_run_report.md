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
*Training logs and convergence metrics will be added here.*

## 4. Evaluation Results
| Metric | DRL Agent (Pilot) | Equal Weight | MVO (Max Sharpe) |
| :--- | :--- | :--- | :--- |
| **Annualized Return** | | | |
| **Sharpe Ratio** | | | |
| **Max Drawdown** | | | |

## 5. Discussion
Preliminary findings on the agent's behavior in a single-asset class environment.

---
**Status**: In Progress
Supporting Script: [pilot_train.py](file:///c:/Users/Hab/Desktop/DRL_Portfolio_Optimization-1/pilot_train.py)
