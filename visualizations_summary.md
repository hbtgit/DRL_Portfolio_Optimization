# Visualizations Summary

This document explains the various charts and data visualizations generated at the end of the project during the metric aggregation and visual analysis phases. These plots are critical for evaluating algorithm performance, portfolio selection, and transaction cost-awareness.

## 1. Sprint 4: Visual Analysis (`results/sprint4/`)

These charts are specifically tailored for publication and showcase the final testing outcomes.

### 1.1 Equity Curves (`equity_curves.png`)
* **What it shows**: The cumulative net return trajectories (portfolio value starting at 1.0) of all trained DRL models alongside the established baselines (Equal Weight, MVO Max Sharpe) over the 2022-2023 out-of-sample test period.
* **Key Interpretations**: Look for the highest ending multiplier and lowest volatility. The shaded area below the zero-line tracks the *Maximum Drawdown* (percentage drop from peak) of the most performant DRL model. A robust model will have smoother upward trajectories and shallower drawdowns during market turbulence compared to benchmarks.

### 1.2 Weight Allocation Heatmaps
* **What it shows**: The evolving portfolio composition determined by the agent at each trading step. Tickers are listed on the y-axis, trading days on the x-axis. Warmer/darker colors indicate heavier percentage allocations.
  * **PPO Pilot (`weight_heatmap_pilot.png`)**: Model operating strictly on 30 equities.
  * **PPO Full (`weight_heatmap_full.png`)**: Model operating on the full 32-asset universe (equities + cryptocurrencies).
* **Key Interpretations**: Reveals the agent's internal logic. Does the agent favor a few specific "winning" stocks, or does it diversify evenly? Frequent horizontal color changes imply high daily turnover (frequent buying/selling).

### 1.3 Cost vs Return Scatter Plot (`cost_vs_return_scatter.png`)
* **What it shows**: A scatter plot comparing the *Cumulative Transaction Cost Drag (%)* on the x-axis against the *Net Annualized Return (%)* on the y-axis across all strategies. Points are uniquely styled to differentiate DRL models from baselines.
* **Key Interpretations**: This illustrates the fundamental trade-off in algorithmic trading: excessive trading (high TC drag) destroys net returns. A downward trend line confirms this relationship. The most desirable models sit in the "Low-cost zone" (top-left quadrant) offering high returns with minimal trading friction.

---

## 2. Metric Aggregation Charts (`results/metrics/`)

These functional plots delve deeper into the raw statistics behind the final results, explicitly tracking step-level turnover and costs.

### 2.1 Benchmark Bar Chart (`benchmark_metrics_bar.png`)
* **What it shows**: A straightforward side-by-side grouped bar chart comparing key traditional performance metrics: Net Annualized Return (ARR), Sharpe Ratio, and Max Drawdown across all models.
* **Key Interpretations**: Summarizes absolute performance metrics quickly without time-series noise.

### 2.2 Transaction Cost Timeseries (`tc_timeseries.png`)
* **What it shows**: A line chart mapping the transaction cost penalty (or sheer volume traded) experienced by the portfolio at every single time step.
* **Key Interpretations**: Highlights periods of intense trading activity. Spikes typically correlate with market crashes or high-volatility shifts where the agent panic-sells or rapidly rebalances.

### 2.3 Turnover Distribution (`turnover_distribution.png`)
* **What it shows**: A violin/box plot summarizing the statistical distribution of daily portfolio turnover rates for each model.
* **Key Interpretations**: Indicates the strategy's "churn rate". A fat tail or wide distribution implies erratic, aggressive trading behavior. Models explicitly penalized for high turnover during training should exhibit a tighter, lower distribution profile compared to naive models.

### 2.4 TC Summary Bar (`tc_summary_bar.png`)
* **What it shows**: A bar chart quantifying the absolute total cumulative transaction costs paid by each strategy at the end of the testing period.
* **Key Interpretations**: Highly correlated with the turnover distribution. Serves as a direct diagnostic for how expensive it was to operate the strategy in the real world.
