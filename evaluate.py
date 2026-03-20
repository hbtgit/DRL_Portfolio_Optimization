import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DDPG
from trading_env import TradingEnv

# 1. Configuration
DATA_PATH = "data/processed/portfolio_data_normalized.csv"
MODEL_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_agent(algo="ppo", model_name="ppo_final"):
    print(f"--- Evaluating {algo.upper()} Agent ---")
    
    # Load data for testing split (2022-2023)
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    test_df = df[df['Date'] >= '2022-01-01'].copy()
    
    # Create Environment
    env = TradingEnv(test_df)
    
    # Load Model
    model_path = os.path.join(MODEL_DIR, f"{model_name}.zip")
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return
        
    if algo.lower() == "ppo":
        model = PPO.load(model_path)
    else:
        model = DDPG.load(model_path)
        
    # 2. Run Backtest
    obs, info = env.reset()
    done = False
    
    agent_net_returns = []
    agent_cumulative_return = [1.0]
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # We track net_return from info (gross_return - cost)
        net_ret = info['net_return']
        agent_net_returns.append(net_ret)
        agent_cumulative_return.append(agent_cumulative_return[-1] * net_ret)
        
        done = terminated or truncated
        
    # 3. Baselines
    # a) Equal Weight (EW)
    obs, _ = env.reset()
    ew_cumulative = [1.0]
    num_assets = env.num_assets
    done = False
    while not done:
        action = np.ones(num_assets) / num_assets
        _, _, terminated, truncated, info = env.step(action)
        ew_cumulative.append(ew_cumulative[-1] * info['net_return'])
        done = terminated or truncated
        
    # b) Buy & Hold (B&H) - Start with EW and never trade (turnover = 0)
    # Note: Our env calculates rebalance cost if action changes.
    # To simulate B&H, we can either re-pass the price-changed weights as action 
    # Or just run it manually. In our env, if target_weights == weights_after_price_change, cost=0.
    obs, _ = env.reset()
    bh_cumulative = [1.0]
    done = False
    current_weights = np.ones(num_assets) / num_assets
    while not done:
        # In B&H, we "do nothing", but we still need to calculate the return
        # Our env.step expects an action. If we pass the price-drifted weights, turnover is 0.
        # But wait, weights drift internally. 
        # Let's just use a simple B&H calculation here for 100% accuracy.
        # Actually, let's just use the env with a "null" action that matches drift.
        # For simplicity, we'll use Equal Weight as the primary baseline.
        break # Skipping BH for now, EW is more standard for diversifation evaluation.

    # 4. Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(agent_cumulative_return, label=f"DRL Agent ({algo.upper()})")
    plt.plot(ew_cumulative, label="Equal Weight (Baseline)")
    plt.title(f"Portfolio Cumulative Returns (Test Set: 2022-2023)")
    plt.xlabel("Days")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(RESULTS_DIR, f"backtest_{algo}.png")
    plt.savefig(plot_path)
    print(f"Results plot saved to {plot_path}")
    
    # 5. Metrics
    def calculate_metrics(returns_series):
        rets = np.array(returns_series) - 1.0 # Simple daily returns
        avg_ret = np.mean(rets) * 252
        std_ret = np.std(rets) * np.sqrt(252)
        sharpe = avg_ret / std_ret if std_ret != 0 else 0
        return avg_ret, sharpe
        
    agent_ann_ret, agent_sharpe = calculate_metrics(agent_net_returns)
    print(f"\n--- Metrics ---")
    print(f"Agent: Annualized Return = {agent_ann_ret:.2%}, Sharpe Ratio = {agent_sharpe:.2f}")

if __name__ == "__main__":
    evaluate_agent("ppo")
