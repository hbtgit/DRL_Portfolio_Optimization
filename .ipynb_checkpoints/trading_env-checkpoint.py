import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

class TradingEnv(gym.Env):
    """
    A custom Gymnasium environment for portfolio optimization.
    Features:
    - 32 Assets (27 Equity + 5 Crypto)
    - Transaction cost awareness
    - Normalized technical indicators as observation
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, data, window_size=10, transaction_cost=0.001):
        super(TradingEnv, self).__init__()
        
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        
        # Load data if path is provided, otherwise use DataFrame directly
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Get unique tickers and dates
        self.tickers = sorted(self.df['Ticker'].unique())
        self.dates = sorted(self.df['Date'].unique())
        self.num_assets = len(self.tickers)
        
        # Features to use for observation (normalized ones)
        self.feature_cols = [
            'Close', 'Volume', 'sma_10', 'sma_30', 'rsi_14', 
            'macd', 'macd_signal', 'bb_middle', 'bb_std', 'bb_upper', 'bb_lower', 'log_return', 'raw_return'
        ]
        self.num_features = len(self.feature_cols)
        
        # Pivot data for easy indexing: (Date, Ticker, Features)
        print("Pivoting data for environment...")
        self.data_tensor = self._prepare_data()
        
        # Action space: weights for each asset (0 to 1)
        # We will normalize these to sum to 1 in the step function
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        
        # Observation space: 
        # 1. Market Features: (window_size, num_assets, num_features - 1) 
        # We exclude raw_return from observation
        self.observation_space = spaces.Dict({
            "market": spaces.Box(low=-np.inf, high=np.inf, 
                                shape=(self.window_size, self.num_assets, self.num_features - 1), 
                                dtype=np.float32),
            "portfolio": spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        })
        
        self.current_step = 0
        self.portfolio_weights = np.ones(self.num_assets) / self.num_assets

    def _prepare_data(self):
        """Converts the long-form dataframe into a 3D numpy array (Time, Asset, Feature)."""
        # Create a mapping for faster lookup
        date_map = {date: i for i, date in enumerate(self.dates)}
        ticker_map = {ticker: i for i, ticker in enumerate(self.tickers)}
        
        data_tensor = np.zeros((len(self.dates), self.num_assets, self.num_features), dtype=np.float32)
        
        for _, row in self.df.iterrows():
            d_idx = date_map[row['Date']]
            t_idx = ticker_map[row['Ticker']]
            features = row[self.feature_cols].values.astype(np.float32)
            data_tensor[d_idx, t_idx, :] = features
            
        return data_tensor

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        # In RL training, it's often better to start at different points in time
        # to increase sample diversity. We'll start between window_size and 80% of data.
        if len(self.dates) > self.window_size + 100:
            max_start = int(len(self.dates) * 0.8)
            self.current_step = np.random.randint(self.window_size, max_start)
        else:
            self.current_step = self.window_size
            
        self.portfolio_weights = np.ones(self.num_assets) / self.num_assets
        
        observation = self._get_observation()
        info = {}
        return observation, info

    def _get_observation(self):
        # Exclude raw_return (last index) from the market window provided to agent
        market_window = self.data_tensor[self.current_step - self.window_size : self.current_step, :, :-1]
        return {
            "market": market_window,
            "portfolio": self.portfolio_weights.astype(np.float32)
        }

    def step(self, action):
        # 1. Normalize actions to sum to 1 (Simplex constraint)
        if np.sum(action) > 0:
            target_weights = action / np.sum(action)
        else:
            target_weights = np.ones(self.num_assets) / self.num_assets
        
        # 2. Get asset returns for the current step
        # Use 'raw_return' which is the actual ln(P_t/P_t-1)
        raw_return_idx = self.feature_cols.index('raw_return')
        log_returns = self.data_tensor[self.current_step, :, raw_return_idx]
        asset_returns = np.exp(log_returns) # Price ratio P_t / P_t-1
        
        # 3. Calculate Portfolio Return (before rebalancing)
        # Portfolio value changes based on the weights at the *end* of previous step
        portfolio_return = np.sum(self.portfolio_weights * asset_returns)
        
        # Weights change internally due to price movement
        weights_after_price_change = (self.portfolio_weights * asset_returns) / portfolio_return
        
        # 4. Calculate Transaction Costs (Turnover)
        turnover = np.sum(np.abs(target_weights - weights_after_price_change))
        tc_penalty = self.transaction_cost * turnover
        
        # 5. Net Log Reward
        # Reward = ln(Net Portfolio Return)
        # Actually, using net return directly is common: R = log(port_return * (1 - cost))
        # Or R = log(port_return) - cost (as approximation)
        reward = np.log(portfolio_return) - tc_penalty
        
        # 6. Update state
        self.portfolio_weights = target_weights
        self.current_step += 1
        
        # Check if done
        terminated = self.current_step >= len(self.dates) - 1
        truncated = False
        
        observation = self._get_observation()
        info = {
            "portfolio_return": portfolio_return,
            "turnover": turnover,
            "tc_penalty": tc_penalty,
            "net_return": portfolio_return - tc_penalty
        }
        
        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass
