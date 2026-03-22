import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import TradingEnv

# 1. Configuration
DATA_PATH = "data/processed/portfolio_data_normalized.csv"
BASE_MODEL_DIR = "models/sprint3"
LOG_DIR = "logs/sprint3"
os.makedirs(BASE_MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Define architecture (Refined in Sprint 2)
ppo_policy_kwargs = dict(
    net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])
)

def get_regime_data(df, regime):
    crypto_tickers = ['BTC_USD', 'ETH_USD']
    if regime == "equities":
        return df[~df['Ticker'].isin(crypto_tickers)].copy()
    elif regime == "crypto":
        return df[df['Ticker'].isin(crypto_tickers)].copy()
    else: # combined
        return df.copy()

def train_regime(regime, total_timesteps=100000):
    print(f"\n--- Starting Sprint 3 Training: {regime.upper()} Regime ---")
    
    # Load and filter data
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    regime_df = get_regime_data(df, regime)
    
    # Train/Test Split (2015-2021 for training)
    train_df = regime_df[regime_df['Date'] < '2022-01-01'].copy()
    
    tickers = sorted(regime_df['Ticker'].unique())
    print(f"Tickers ({len(tickers)}): {tickers}")
    
    # Create Environment
    env = TradingEnv(train_df)
    env = DummyVecEnv([lambda: env])
    
    # Initialize Model
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=os.path.join(LOG_DIR, regime),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        policy_kwargs=ppo_policy_kwargs
    )
    
    # Train
    model.learn(total_timesteps=total_timesteps)
    
    # Save model
    regime_dir = os.path.join(BASE_MODEL_DIR, regime)
    os.makedirs(regime_dir, exist_ok=True)
    model_path = os.path.join(regime_dir, f"ppo_{regime}_final.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=str, choices=["equities", "crypto", "combined"], default="equities")
    parser.add_argument("--steps", type=int, default=100000)
    args = parser.parse_args()
    
    train_regime(args.regime, args.steps)
