import numpy as np

def theoretical_verification():
    print("--- Theoretical Return Verification ---")
    
    # 1. Setup Scenario
    transaction_cost = 0.01 # 1%
    initial_weights = np.array([0.5, 0.5])
    asset_returns = np.array([1.1, 0.9]) # Asset A +10%, Asset B -10%
    target_weights = np.array([1.0, 0.0]) # Rebalance fully to Asset A
    
    print(f"Initial Weights: {initial_weights}")
    print(f"Asset Price Ratios (P_t / P_t-1): {asset_returns}")
    print(f"Target Weights: {target_weights}")
    print(f"Transaction Cost Rate: {transaction_cost}")
    
    # 2. Step 1: Market movement (Gross Return)
    portfolio_return_gross = np.sum(initial_weights * asset_returns)
    print(f"\n1. Gross Portfolio Return: {portfolio_return_gross:.4f}")
    
    # 3. Step 2: Weights after price change
    weights_after_price_change = (initial_weights * asset_returns) / portfolio_return_gross
    print(f"2. Weights after price change (before rebalance): {weights_after_price_change}")
    
    # 4. Step 3: Turnover calculation
    turnover = np.sum(np.abs(target_weights - weights_after_price_change))
    print(f"3. Turnover: {turnover:.4f}")
    
    # 5. Step 4: Transaction Cost Calculation
    # Theoretical: Cost is applied to the current portfolio value
    tc_theoretical = portfolio_return_gross * (transaction_cost * turnover)
    net_return_theoretical = portfolio_return_gross - tc_theoretical
    log_net_return_theoretical = np.log(net_return_theoretical)
    
    print(f"4. Theoretical Transaction Cost: {tc_theoretical:.6f}")
    print(f"5. Theoretical Net Return: {net_return_theoretical:.6f}")
    print(f"6. Theoretical Log Net Return: {log_net_return_theoretical:.6f}")
    
    # 6. Step 5: Environment Implementation (Current Code)
    # tc_penalty = self.transaction_cost * turnover
    # reward = np.log(portfolio_return) - tc_penalty
    tc_penalty_env = transaction_cost * turnover
    reward_env = np.log(portfolio_return_gross) - tc_penalty_env
    net_return_env = portfolio_return_gross - tc_penalty_env # From info dict
    
    print("\n--- Environment Logic Results ---")
    print(f"Env TC Penalty: {tc_penalty_env:.6f}")
    print(f"Env Net Return (info): {net_return_env:.6f}")
    print(f"Env Reward (log): {reward_env:.6f}")
    
    # 7. Comparison
    error_percent = abs(reward_env - log_net_return_theoretical) / abs(log_net_return_theoretical) * 100
    print(f"\nError in Reward (Log Return): {error_percent:.4f}%")
    
    if error_percent < 0.5:
        print("\nSUCCESS: Environment logic matches theoretical expectation (within approx bounds).")
    else:
        print("\nWARNING: Deviation detected. Reviewing precision requirements.")

if __name__ == "__main__":
    theoretical_verification()
