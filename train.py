import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from trading_env import TradingEnv

# 1. Configuration
DATA_PATH = "data/processed/portfolio_data_normalized.csv"
MODEL_DIR = "models"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Define custom network architectures for high-dimensional action space (32 assets)
# Use deeper networks with decoupled heads for better feature separation
ppo_policy_kwargs = dict(
    net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])
)

ddpg_policy_kwargs = dict(
    net_arch=[256, 128, 64]
)

def get_ppo_model(env):
    return PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128, # Increased for 32-asset stability
        policy_kwargs=ppo_policy_kwargs
    )

def get_ddpg_model(env):
    return DDPG(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=1e-3,
        buffer_size=100000,
        batch_size=128,
        tau=0.005,
        policy_kwargs=ddpg_policy_kwargs
    )

def train_agent(algo="ppo", total_timesteps=50000):
    print(f"--- Training {algo.upper()} Agent ---")
    
    # Load data for training split (2015-2021)
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    train_df = df[df['Date'] < '2022-01-01'].copy()
    
    # Create Environment
    env = TradingEnv(train_df)
    env = DummyVecEnv([lambda: env])
    
    # Initialize Model
    if algo.lower() == "ppo":
        model = get_ppo_model(env)
    elif algo.lower() == "ddpg":
        model = get_ddpg_model(env)
    else:
        raise ValueError("Unsupported algorithm")
        
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=os.path.join(MODEL_DIR, algo),
        name_prefix=f"{algo}_model"
    )
    
    # Train
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    # Save final model
    model_path = os.path.join(MODEL_DIR, f"{algo}_final.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # For initial run, we do a shorter training to verify logic
    train_agent("ppo", total_timesteps=30000)
    # train_agent("ddpg", total_timesteps=30000)
