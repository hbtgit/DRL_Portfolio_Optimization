# Env Upgrade: DRL Trading Environment Details

This document details the "Env Upgrade" stage, focusing on the implementation of a custom **Gymnasium-compatible** environment for portfolio optimization.

## 1. MDP Formulation
The portfolio optimization problem is modeled as a Markov Decision Process (MDP):

- **State ($s_t$)**: A tuple containing:
    - **Historical Window**: A 2D/3D tensor of normalized features (Close, RSI, MACD, etc.) for the last $N$ days (e.g., $N=30$).
    - **Portfolio State**: The current weights $w_{t-1}$ of the assets.
- **Action ($a_t$)**: A vector of target weights $w_t \in [0, 1]^M$, where $M=32$. The actions are projected onto a simplex so that $\sum w_{t,i} = 1$.
- **Reward ($r_t$)**: Transaction-cost adjusted net log return.

## 2. Environment Architecture (`TradingEnv`)
We will implement a custom class inheriting from `gymnasium.Env`.

### Observation Space
The observation provides the agent with both market context and current portfolio status.
- **Market Features**: `(Window_Size, Num_Assets, Num_Features)`.
    - Features (11): Close, Volume, SMA_10, SMA_30, RSI, MACD, MACD_Signal, BB_Middle, BB_Std, BB_Upper, BB_Lower.
- **Internal State**: `(Num_Assets,)` representing $w_{t-1}$.

### Action Space
- **Type**: `Box(0, 1, shape=(32,))`.
- **Constraint**: Softmax or projection to ensure sum is 1.

### Reward Function
The reward directly penalizes excessive rebalancing to discourage churning:
$$R_t = \ln(\frac{PV_t}{PV_{t-1}}) - \lambda \cdot \sum_{i=1}^{M} |w_{t,i} - w'_{t-1,i}|$$
Where:
- $PV$ is the Portfolio Value.
- $\lambda$ is the transaction cost coefficient (e.g., 0.001 for 0.1%).
- $w'$ is the weight vector *after* price change but *before* rebalancing.

## 3. Transaction Cost Model
- **Trading Fee**: Proportional cost based on turnover.
- **Objective**: Identify "no-trade regions" where the expected gain from rebalancing does not justify the cost.

## 4. Implementation Steps
1. Define the `TradingEnv` class.
2. Implement `reset()` to initialize with a random window from the historical data.
3. Implement `step()` to:
    - Apply weights.
    - Calculate portfolio return.
    - Subtract transaction costs.
    - Update internal state and move to next time step.
4. Verify with a "Buy and Hold" and "Random" baseline.
