import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DDPG
from trading_env import TradingEnv
from baselines import EqualWeightStrategy, MVOStrategy

# 1. Configuration
DATA_PATH = "data/processed/portfolio_data_normalized.csv"
RAW_DATA_PATH = "data/processed/portfolio_data_raw.csv"
MODEL_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def calculate_metrics(cumulative_returns):
    """Calculates key financial metrics from a cumulative return series."""
    cum_ret = np.array(cumulative_returns)
    daily_rets = cum_ret[1:] / cum_ret[:-1] - 1.0
    
    # Annualized Return
    total_ret = cum_ret[-1] / cum_ret[0]
    num_years = len(cum_ret) / 252
    ann_ret = (total_ret ** (1/num_years)) - 1 if num_years > 0 else -1
    
    # Sharpe Ratio
    std_ret = np.std(daily_rets) * np.sqrt(252)
    sharpe = (ann_ret / std_ret) if std_ret != 0 else 0
    
    # Max Drawdown
    peak = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - peak) / peak
    max_dd = np.min(drawdown)
    
    return ann_ret, sharpe, max_dd

def evaluate_agent(algo="ppo", model_path=None, model_type="full"):
    print(f"--- Evaluating {algo.upper()} Agent ({model_type}) ---")
    
    # Load data for testing split (2022-2023)
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter tickers if it's a pilot run (Equities only)
    if model_type == "pilot":
        # Keep only assets present in the pilot (which were all except crypto)
        crypto_tickers = ['BTC_USD', 'ETH_USD']
        df = df[~df['Ticker'].isin(crypto_tickers)].copy()
        
    test_df = df[df['Date'] >= '2022-01-01'].copy()
    
    # Create Environment
    env = TradingEnv(test_df)
    
    # Load Model
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, f"{algo}_final.zip")
        
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return
        
    model = PPO.load(model_path) if algo.lower() == "ppo" else DDPG.load(model_path)
    
    # --- 1. Agent Backtest ---
    obs, info = env.reset()
    done = False
    agent_cum = [1.0]
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        agent_cum.append(agent_cum[-1] * info['net_return'])
        done = terminated or truncated
        
    # --- 2. Equal Weight Baseline ---
    env_ew = TradingEnv(test_df)
    obs, _ = env_ew.reset()
    ew_strategy = EqualWeightStrategy()
    ew_cum = [1.0]
    done = False
    while not done:
        action = ew_strategy.get_action(env_ew, env_ew.current_step)
        _, _, terminated, truncated, info = env_ew.step(action)
        ew_cum.append(ew_cum[-1] * info['net_return'])
        done = terminated or truncated
        
    # --- 3. MVO Baseline ---
    env_mvo = TradingEnv(test_df)
    obs, _ = env_mvo.reset()
    # MVO needs the raw data path for price extraction
    mvo_strategy = MVOStrategy(RAW_DATA_PATH)
    mvo_cum = [1.0]
    done = False
    while not done:
        action = mvo_strategy.get_action(env_mvo, env_mvo.current_step)
        _, _, terminated, truncated, info = env_mvo.step(action)
        mvo_cum.append(mvo_cum[-1] * info['net_return'])
        done = terminated or truncated
        
    # --- 4. Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(agent_cum, label=f"DRL Agent ({model_type.upper()})", linewidth=2)
    plt.plot(ew_cum, label="Equal Weight", linestyle="--")
    plt.plot(mvo_cum, label="MVO (Max Sharpe)", linestyle=":")
    plt.title(f"Cumulative Returns - {model_type.upper()} Mode (2022-2023)")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value (Normalized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_name = f"backtest_{model_type}_{algo}.png"
    plt.savefig(os.path.join(RESULTS_DIR, plot_name))
    print(f"Results plot saved to {os.path.join(RESULTS_DIR, plot_name)}")
    
    # --- 5. Metrics ---
    metrics = {
        "Agent": calculate_metrics(agent_cum),
        "EW": calculate_metrics(ew_cum),
        "MVO": calculate_metrics(mvo_cum)
    }
    
    print("\n" + "="*40)
    print(f"{'Strategy':<10} | {'Ann. Ret':<10} | {'Sharpe':<8} | {'Max DD':<8}")
    print("-"*40)
    for name, (ret, sh, dd) in metrics.items():
        print(f"{name:<10} | {ret:>9.2%} | {sh:>8.2f} | {dd:>8.2%}")
    print("="*40)

if __name__ == "__main__":
    # Check if pilot run exists, if so evaluate it
    pilot_path = "models/pilot/ppo_pilot_final.zip"
    if os.path.exists(pilot_path):
        evaluate_agent("ppo", model_path=pilot_path, model_type="pilot")
    else:
        evaluate_agent("ppo")
