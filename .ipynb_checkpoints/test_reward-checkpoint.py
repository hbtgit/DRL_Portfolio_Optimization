import numpy as np

def test_reward_math():
    print("--- TC-Aware Reward Verification ---")
    
    # Setup parameters
    transaction_cost = 0.001 # 0.1%
    initial_weights = np.array([0.5, 0.5])
    asset_returns = np.array([1.1, 0.9]) # Asset A +10%, Asset B -10%
    target_weights = np.array([1.0, 0.0]) # Fully rebalance to Asset A
    
    print(f"Initial Weights: {initial_weights}")
    print(f"Asset Returns: {asset_returns}")
    print(f"Target Weights: {target_weights}")
    print(f"TC Rate: {transaction_cost}")
    
    # 1. Calculate Gross Return
    portfolio_return = np.sum(initial_weights * asset_returns)
    print(f"\n1. Gross Portfolio Return: {portfolio_return:.4f}")
    
    # 2. Intermediate weights after market drift
    weights_after_drift = (initial_weights * asset_returns) / portfolio_return
    print(f"2. Weights after drift (before rebalance): {weights_after_drift}")
    
    # 3. Turnover calculation
    turnover = np.sum(np.abs(target_weights - weights_after_drift))
    print(f"3. Turnover: {turnover:.4f}")
    
    # 4. Transaction Cost (TC) Penalty
    tc_penalty = transaction_cost * turnover
    print(f"4. TC Penalty (Cost / PortValue): {tc_penalty:.6f}")
    
    # 5. Reward Calculation (Log-Net Return approximation)
    reward = np.log(portfolio_return) - tc_penalty
    print(f"5. Final Reward (ln(R_gross) - TC): {reward:.6f}")
    
    # 6. Exact Log Net Return Comparison
    exact_net_return = portfolio_return * (1 - tc_penalty)
    exact_log_reward = np.log(exact_net_return)
    print(f"6. Exact Log Net Return: {exact_log_reward:.6f}")
    
    # 7. Verification Result
    error = abs(reward - exact_log_reward) / abs(exact_log_reward) * 100
    print(f"\nRelative Approximation Error: {error:.6f}%")
    
    if error < 0.1:
        print("SUCCESS: REWARD MATH VERIFIED.")
    else:
        print("WARNING: DEVIATION DETECTED.")

if __name__ == "__main__":
    test_reward_math()
