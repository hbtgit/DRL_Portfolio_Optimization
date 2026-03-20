# TC-Aware Reward Implementation

This document details the implementation and verification of the transaction cost (TC) aware reward function $R_t = \ln(\text{return}) - \lambda \cdot \text{turnover}$.

## 1. Mathematical Formulation
To maximize the long-term cumulative wealth, the reward function is designed using the **Logarithmic Net Return**.

Let:
- $w_{t-1}$: Portfolio weights at the end of the previous step.
- $r_t$: Asset returns (price ratios $\frac{P_t}{P_{t-1}}$).
- $R_{gross} = \sum w_{t-1, i} \cdot r_{t, i}$: Gross portfolio return.
- $w'_{t-1} = \frac{w_{t-1} \odot r_t}{R_{gross}}$: Weights shifted by market movement before rebalancing.
- $w_t$: Target weights (agent action).
- $Turnover = \sum |w_t - w'_{t-1}|$.
- $\lambda$: Transaction cost rate (e.g., 0.001).

The net return after transaction costs is $R_{net} = R_{gross} \cdot (1 - \lambda \cdot Turnover)$.
The log reward is:
$$R_t = \ln(R_{net}) = \ln(R_{gross}) + \ln(1 - \lambda \cdot Turnover)$$

Using the first-order Taylor approximation $\ln(1 - x) \approx -x$ for small $x$:
$$R_t \approx \ln(R_{gross}) - \lambda \cdot Turnover$$

## 2. Implementation in `TradingEnv`
The code in `trading_env.py` implements this as:
```python
# 1. Gross Return
portfolio_return = np.sum(self.portfolio_weights * asset_returns)

# 2. Market Drift
weights_after_price_change = (self.portfolio_weights * asset_returns) / portfolio_return

# 3. Turnover & Penalty
turnover = np.sum(np.abs(target_weights - weights_after_price_change))
tc_penalty = self.transaction_cost * turnover

# 4. Final Reward
reward = np.log(portfolio_return) - tc_penalty
```

## 3. Verification & Debugging
A dedicated test script `test_reward.py` is used to:
1. Verify turnover calculation across known weights.
2. Ensure the $\ln(\text{return}) - \lambda \cdot \text{turnover}$ approximation holds for various $\lambda$ values.
3. Check for edge cases (e.g., zero turnover, high volatility).

---
**Status**: Verified
Supporting Script: [test_reward.py](file:///c:/Users/Hab/Desktop/DRL_Portfolio_Optimization-1/test_reward.py)
