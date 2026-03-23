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

"""
metrics_aggregator.py
=====================
Automated collection and persistence of step-level turnover and
transaction cost data produced during any rollout.

Usage (inside a rollout loop)
------------------------------
    from metrics_aggregator import MetricsCollector, aggregate_and_plot

    collector = MetricsCollector(strategy="PPO_Pilot", run_id="bench_001")
    for step in episode:
        ...env.step(action)...
        collector.record(
            step=step,
            turnover=info["turnover"],
            tc_penalty=info["tc_penalty"],
            net_return=info["net_return"],
            gross_return=info["portfolio_return"],
        )
    collector.save()          # writes per-strategy CSV + appends to summary
    aggregate_and_plot()      # regenerates all diagnostic plots
"""
