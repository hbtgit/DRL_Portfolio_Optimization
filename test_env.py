import numpy as np
from trading_env import TradingEnv

def test_random_agent():
    print("Testing TradingEnv with a random agent...")
    
    data_path = "data/processed/portfolio_data_normalized.csv"
    env = TradingEnv(data_path=data_path, window_size=10, transaction_cost=0.001)
    
    obs, info = env.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"Market shape: {obs['market'].shape}") # Should be (10, 32, 12)
    print(f"Portfolio shape: {obs['portfolio'].shape}") # Should be (32,)
    
    done = False
    step_count = 0
    total_reward = 0
    
    while not done and step_count < 100:
        # Sample a random action (weights)
        action = np.random.uniform(0, 1, size=(env.num_assets,))
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        step_count += 1
        
        if step_count % 20 == 0:
            print(f"Step {step_count}: Reward = {reward:.6f}, Net Return = {info['net_return']:.6f}")
    
    print(f"Test complete. Total reward (100 steps): {total_reward:.6f}")
    print("Environment verified.")

if __name__ == "__main__":
    test_random_agent()
