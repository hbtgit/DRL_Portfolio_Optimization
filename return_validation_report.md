# Return Calculation Validation Report

This report documents the validation of the net return and reward calculations within the `TradingEnv`.

## 1. Methodology
We compared the environment's internal logic against a manual theoretical calculation using a controlled scenario:
- **Transaction Cost ($\lambda$)**: 0.01 (1%)
- **Initial Weights**: [0.5, 0.5]
- **Market Movement**: Asset A (+10%), Asset B (-10%)
- **Action**: Full rebalance to Asset A [1.0, 0.0]

## 2. Theoretical Formulation
The theoretical net Portfolio Value ($PV_t$) after rebalancing is:
$$PV_t = PV_{t-1} \cdot R_{gross} \cdot (1 - \lambda \cdot Turnover)$$
Where:
- $R_{gross} = \sum w_{t-1, i} \cdot \frac{P_{t,i}}{P_{t-1,i}}$
- $Turnover = \sum |w_{t,i} - w'_{t-1,i}|$
- $w'_{t-1}$ is the weight after price change but before rebalance.

**Manual Calculation for Scenario:**
1. Gross Return: $0.5 \cdot 1.1 + 0.5 \cdot 0.9 = 1.0$
2. Intermediate weights: $[0.55, 0.45]$
3. Turnover: $|1.0 - 0.55| + |0.0 - 0.45| = 0.9$
4. Total Cost: $1.0 \cdot 0.01 \cdot 0.9 = 0.009$
5. Net PV: $1.0 - 0.009 = 0.991$
6. **Log Net Return**: $\ln(0.991) \approx -0.009041$

## 3. Environment Implementation
The `TradingEnv` uses the following approximation for computational efficiency and gradient stability:
$$Reward = \ln(R_{gross}) - \lambda \cdot Turnover$$

**Environment Result for Scenario:**
- Reward: $\ln(1.0) - (0.01 \cdot 0.9) = 0 - 0.009 = -0.009$

## 4. Comparison Results
| Metric | Theoretical | Environment | Error |
| :--- | :--- | :--- | :--- |
| **Transaction Cost** | 0.009000 | 0.009000 | 0.00% |
| **Reward (Log)** | -0.009041 | -0.009000 | 0.45% |

### Conclusion
The environment's logic matches the theoretical expectation with high precision. The minor 0.45% deviation in the log-reward is a byproduct of the standard approximation $\ln(1-x) \approx -x$, which is commonly used in DRL for financial models to maintain linearity in cost penalties. For realistic transaction costs (0.1%), this error drops to $<0.05\%$.

**Status: VERIFIED**
Supporting Script: [verify_returns.py](file:///c:/Users/Hab/Desktop/DRL_Portfolio_Optimization-1/verify_returns.py)
