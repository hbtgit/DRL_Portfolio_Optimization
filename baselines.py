import numpy as np
import pandas as pd
from pypfopt import base_optimizer, expected_returns, risk_models, EfficientFrontier

class BaselineStrategy:
    """Base class for all baseline strategies."""
    def get_action(self, env, current_step):
        raise NotImplementedError

class EqualWeightStrategy(BaselineStrategy):
    """Naive 1/N allocation."""
    def get_action(self, env, current_step):
        num_assets = env.num_assets
        return np.ones(num_assets) / num_assets

class MVOStrategy(BaselineStrategy):
    """
    Mean-Variance Optimization (Markowitz) baseline.
    Requires raw price data to compute expected returns and covariance accurately.
    """
    def __init__(self, raw_data_path, lookback_window=252):
        self.lookback_window = lookback_window
        # Load RAW prices directly as MVO requires actual levels/returns
        raw_df = pd.read_csv(raw_data_path)
        raw_df['Date'] = pd.to_datetime(raw_df['Date'])
        self.prices_df = raw_df.pivot(index='Date', columns='Ticker', values='Close')
        # Ensure we keep the same sort order for tickers as the environment
        self.tickers = sorted(raw_df['Ticker'].unique())
        
    def get_action(self, env, current_step):
        current_date = env.dates[current_step]
        
        # Get historical slice ending just before current_step
        # In environmental terms, current_step is the one we are about to take
        # So we use [current_step - lookback : current_step]
        start_idx = max(0, current_step - self.lookback_window)
        hist_prices = self.prices_df.iloc[start_idx : current_step]
        
        if len(hist_prices) < 30: # Need enough data
            return np.ones(env.num_assets) / env.num_assets
            
        try:
            # 1. Calculate Expected Returns and Sample Covariance
            # Since inputs are already normalized/Z-scored in env? 
            # NO, we use the raw prices we pivoted from full_df (if they were raw)
            # Wait, TradingEnv's self.df is likely the normalized one.
            # We should ensure we have raw prices here.
            mu = expected_returns.mean_historical_return(hist_prices)
            S = risk_models.sample_cov(hist_prices)
            
            # 2. Optimize for Maximum Sharpe Ratio
            ef = EfficientFrontier(mu, S)
            weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights() # Small weights to 0
            
            # 3. Convert dict to array in correct ticker order
            action = np.array([cleaned_weights[ticker] for ticker in env.tickers])
            return action
            
        except Exception as e:
            # Fallback to EW on optimization failure
            # print(f"MVO Error at step {current_step}: {e}")
            return np.ones(env.num_assets) / env.num_assets
