import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models, EfficientFrontier, objective_functions
from trading_env import TradingEnv

# 1. Configuration
DATA_PATH = "data/processed/portfolio_data_normalized.csv"
RAW_DATA_PATH = "data/processed/portfolio_data_raw.csv"
RESULTS_DIR = "results/sensitivity"
os.makedirs(RESULTS_DIR, exist_ok=True)

class TCAwareMVO:
    def __init__(self, raw_data_path, lookback_window=252):
        self.lookback_window = lookback_window
        raw_df = pd.read_csv(raw_data_path)
        raw_df['Date'] = pd.to_datetime(raw_df['Date'])
        self.prices_df = raw_df.pivot(index='Date', columns='Ticker', values='Close')
        self.tickers = sorted(raw_df['Ticker'].unique())

    def get_action(self, env, current_step, w_prev, tc_lambda):
        start_idx = max(0, current_step - self.lookback_window)
        hist_prices = self.prices_df.iloc[start_idx : current_step]
        
        if len(hist_prices) < 60:
            return np.ones(env.num_assets) / env.num_assets
            
        try:
            mu = expected_returns.mean_historical_return(hist_prices)
            S = risk_models.sample_cov(hist_prices)
            
            mu = mu.fillna(0)
            S = S.fillna(0)
            S = risk_models.fix_nonpositive_semidefinite(S)
            
            ef = EfficientFrontier(mu, S)
            ef.add_objective(objective_functions.transaction_cost, w_prev=w_prev, k=tc_lambda)
            weights = ef.max_quadratic_utility(risk_aversion=1)
            cleaned_weights = ef.clean_weights()
            
            action = np.array([cleaned_weights[ticker] for ticker in env.tickers])
            action = action / np.sum(action) if np.sum(action) > 0 else action
            
            if current_step < 5:
                print(f"  Step {current_step} - mu mean: {mu.mean():.6f}, Action sum: {action.sum():.2f}")
                print(f"  Action non-zero: {np.count_nonzero(action > 1e-3)}")

            return action
            
        except Exception as e:
            if current_step % 100 == 0:
                print(f"  Optimization error at step {current_step}: {e}")
            return w_prev

def run_experiment(tc_lambda):
    print(f"Running experiment with lambda = {tc_lambda}")
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Use only the last 6 months (approx 126 trading days) for speed
    test_df = df[df['Date'] >= '2023-01-01'].copy()
    
    env = TradingEnv(test_df, transaction_cost=tc_lambda)
    strategy = TCAwareMVO(RAW_DATA_PATH)
    
    obs, _ = env.reset()
    done = False
    
    turnovers = []
    trades = [] # 1 if turnover > epsilon, 0 otherwise
    net_returns = []
    
    w_prev = env.portfolio_weights
    step_count = 0
    
    action_changes = []
    
    while not done and step_count < 10:
        # Get optimal action considering TC
        action = strategy.get_action(env, env.current_step, w_prev, tc_lambda)
        
        # Calculate how much the action (target weights) changed from previous action
        action_change = np.sum(np.abs(action - w_prev))
        action_changes.append(action_change)
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        turnover = info['turnover']
        turnovers.append(turnover)
        trades.append(1 if action_change > 1e-5 else 0)
        net_returns.append(info['net_return'])
        
        w_prev = action # Agent sets target weights
        done = terminated or truncated
        step_count += 1
        if step_count % 50 == 0:
            print(f"  Step {step_count}...")
        
    print(f"  Experiment complete. Avg Turnover: {np.mean(turnovers):.4f}, No-Trade Freq: {1 - np.mean(trades):.4f}")
    return {
        "avg_turnover": np.mean(turnovers),
        "no_trade_freq": 1 - np.mean(trades),
        "cum_return": np.prod(net_returns),
        "turnovers": turnovers
    }

if __name__ == "__main__":
    lambdas = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    results = []
    
    for l in lambdas:
        res = run_experiment(l)
        res['lambda'] = l
        results.append(res)
        
    res_df = pd.DataFrame(results)
    print("\nResults Summary:")
    print(res_df[['lambda', 'avg_turnover', 'no_trade_freq', 'cum_return']])
    
    # --- Plotting ---
    plt.figure(figsize=(12, 5))
    
    # 1. Turnover vs Lambda
    plt.subplot(1, 2, 1)
    plt.plot(res_df['lambda'], res_df['avg_turnover'], marker='o', color='royalblue')
    plt.xscale('log')
    plt.xlabel("Transaction Cost Lambda (log scale)")
    plt.ylabel("Average Daily Turnover")
    plt.title("Turnover Reduction as Lambda Increases")
    plt.grid(True, alpha=0.3)
    
    # 2. No-Trade Frequency vs Lambda
    plt.subplot(1, 2, 2)
    plt.plot(res_df['lambda'], res_df['no_trade_freq'], marker='s', color='darkorange')
    plt.xscale('log')
    plt.xlabel("Transaction Cost Lambda (log scale)")
    plt.ylabel("No-Trade Frequency")
    plt.title("Emergence of No-Trade Region")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "no_trade_emergence.png"))
    print(f"\nMain plot saved to {os.path.join(RESULTS_DIR, 'no_trade_emergence.png')}")
    
    # 3. Save Results CSV
    res_df.to_csv(os.path.join(RESULTS_DIR, "sensitivity_results.csv"), index=False)
