# DRL Environment Implementation Plan (Env Upgrade)

Implement the third task of the project: DRL Environment Implementation.

## Proposed Changes

### Scripts
#### [NEW] [trading_env.py](file:///c:/Users/Hab/Desktop/DRL_Portfolio_Optimization-1/trading_env.py)
- Load processed data from [data/processed/portfolio_data_normalized.csv](file:///c:/Users/Hab/Desktop/DRL_Portfolio_Optimization-1/data/processed/portfolio_data_normalized.csv).
- Implement `TradingEnv` class (inheriting from `gymnasium.Env`):
    - `__init__`: Define `observation_space` and `action_space`.
    - `reset()`: Sample start date and reset portfolio.
    - `step()`: Calculate rewards and update state.
    - `render()`: Optional visualization.
- Transaction Cost: Proportional cost logic in `step()`.

## Verification Plan

### Automated Tests
- Run a test script to check:
    - Shape of observations.
    - Action clipping and normalization.
    - Reward calculation for a fixed action (e.g., Buy and Hold).
- Ensure environment passes `gym.utils.env_checker.check_env`.

### Manual Verification
- Print episode rewards for a random agent and verify they are within reasonable bounds.
