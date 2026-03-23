# Baseline Implementation Details

This document describes the standardized implementation of benchmark strategies for the DRL portfolio optimization project.

## 1. Equal-Weight (EW) Portfolio
The Equal-Weight strategy serves as a "naive" diversification baseline.
- **Formulation**: $w_i = \frac{1}{N}$ for all $i \in \{1, \dots, N\}$, where $N=32$.
- **Rebalancing**: The portfolio is rebalanced at every step to maintain equal weights, accounting for transaction costs.

## 2. Mean-Variance Optimization (MVO)
MVO follows Modern Portfolio Theory (MPT) to find the optimal trade-off between risk and return.
- **Objective**: Maximize the Sharpe Ratio.
- **Input Data**: Historical returns over a **252-day lookback window** (1 trading year).
- **Optimization**:
    - **Expected Returns**: Calculated using the Mean Historical Return.
    - **Risk**: Estimated using the Sample Covariance matrix.
- **Constraints**:
    - $\sum w_i = 1$ (Fully invested)
    - $w_i \ge 0$ (Long-only)

## 3. Implementation Tools
We use [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/) to solve the MVO quadratic programming problem.

---
**Status**: Implementation in Progress
Supporting Script: [baselines.py](file:///c:/Users/Hab/Desktop/DRL_Portfolio_Optimization-1/baselines.py)
