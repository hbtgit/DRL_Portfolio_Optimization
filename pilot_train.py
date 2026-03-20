import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import TradingEnv

# 1. Configuration
DATA_PATH = "data/processed/portfolio_data_normalized.csv"
MODEL_DIR = "models/pilot"
LOG_DIR = "logs/pilot"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Define architecture (Refined in Sprint 2)
ppo_policy_kwargs = dict(
    net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])
)

def train_pilot(total_timesteps=30000):
    print("--- Starting PILOT RUN (Equities Only) ---")
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter for US Equities only (Exclude Crypto)
    crypto_tickers = ['BTC_USD', 'ETH_USD']
    df = df[~df['Ticker'].isin(crypto_tickers)].copy()
    
    # Train/Test Split (Same as before)
    train_df = df[df['Date'] < '2022-01-01'].copy()
    
    print(f"Tickers for Pilot: {sorted(df['Ticker'].unique())}")
    print(f"Number of Assets: {len(df['Ticker'].unique())}")
    
    # Create Environment
    env = TradingEnv(train_df)
    env = DummyVecEnv([lambda: env])
    
    # Initialize Model
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        policy_kwargs=ppo_policy_kwargs
    )
    
    # Train
    model.learn(total_timesteps=total_timesteps)
    
    # Save pilot model
    model_path = os.path.join(MODEL_DIR, "ppo_pilot_final.zip")
    model.save(model_path)
    print(f"Pilot model saved to {model_path}")

if __name__ == "__main__":
    train_pilot(total_timesteps=30000)
