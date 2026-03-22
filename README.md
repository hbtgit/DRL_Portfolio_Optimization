# DRL_Portfolio_Optimization
Transaction Cost-Aware Deep Reinforcement Learning for Financial Portfolio Optimization

This repository contains the implementation of a Deep Reinforcement Learning (DRL) framework designed for dynamic portfolio optimization. This project specifically addresses the challenge of transaction costs in financial markets, implementing a cost-aware reward function that balances return generation with trading friction. The framework is evaluated across diverse asset classes, including equities (Dow Jones) and cryptocurrencies, over an 8-year historical period.
🏗 Project Architecture

The system is built upon a modular DRL architecture:

    Environment (env.py): A custom Gymnasium environment simulating the portfolio management lifecycle, enforcing simplex constraints on asset weights and calculating transaction costs.

    Agent (agent.py): Implementation of Actor-Critic based algorithms (PPO/DDPG) designed to optimize portfolios in continuous action spaces.

    Neural Network (models/): Hybrid Actor-Critic networks utilizing Convolutional Neural Networks (CNN) to extract temporal patterns from financial price tensors, combined with Dense layers for weight allocation.

🚀 Key Features

    Transaction Cost Awareness: The reward function Rt​=ln(wt−1T​yt​)+ln(1−λ⋅turnovert​) directly penalizes excessive rebalancing.

    Hardware Acceleration: Optimized for NVIDIA GPU execution, significantly reducing training time for high-frequency historical data.

    Modular Benchmarking: Includes scripts to compare the DRL agent performance against standard Baselines:

        Buy-and-Hold

        Mean-Variance Optimization (MVO)

        Equal Weight

🛠 Installation

    Clone the repository:
    Bash

    git clone https://github.com/hbtgit/drl_portfolio_optimization
    cd drl_portfolio_optimization

    Install dependencies:
    Bash

    pip install -r requirements.txt

    Configure Environment:
    Ensure you have your historical OHLCV data in the data/ directory.

📈 Usage
Training the Agent

To train the agent on your specific market data, run the training script:
Bash

python agent.py --algo PPO --epochs 100 --learning_rate 0.0003

Evaluating Performance

Once the agent is trained, evaluate its performance and visualize metrics (Sharpe Ratio, Max Drawdown):
Bash

python evaluate.py --model_path models/ppo_final.zip --test_data data/2025_test.csv

📊 Research Methodology

This project implements the research outlined in my thesis proposal, focusing on the following research objectives and questions:

### Objectives
- **Quantify Cost Impact**: Analyze the detrimental effects of transaction costs on non-cost-aware DRL agents.
- **Design Cost-Aware Reward**: Implement a modified reward function that penalizes excessive trading.
- **Comparative Analysis**: Compare the TC-aware DRL framework against traditional Mean-Variance Optimization (MVO) and standard DRL models.
- **Cross-Asset Evaluation**: Test performance on both stable equities and volatile cryptocurrencies.

### Research Questions
- **RQ1**: Does a cost-aware reward function successfully create a "no-trade region" for the agent?
- **RQ2**: How does the DRL agent compare to classical Mean-Variance Optimization in high-volatility regimes?
- **RQ3**: Does the inclusion of transaction costs in the reward function lead to more realistic and stable trading strategies?

🏆 Key Empirical Findings (2022-2023 Out-of-Sample Test)

1. **Transaction Costs Dominate**: Standard 32-asset cross-regime PPO models generated a staggering 54% cumulative transaction cost drag, completely erasing gross returns (-13.14% net ARR).
2. **Cost-Aware DRL Can Outperform**: The TC-aware PPO Pilot model, trained exclusively on a 30-asset equities regime, achieved a **+13.81% Net Annualized Return** (Sharpe 0.73), significantly outperforming Equal Weight and MVO baselines (+8.70%).
3. **Turnover Reduction is Critical**: The equities pilot model learned to restrict its trading to a moderate 0.40 average daily turnover (vs 0.88 for full cross-asset models), proving that proportional penalties ($R_t = \ln(\text{return}) - \lambda \cdot \text{turnover}$) can enforce realistic trading behaviors if the asset universe is well-bounded.

📄 License

This repository is associated with the thesis project: "Transaction Cost-Aware Deep Reinforcement Learning for Financial Portfolio Optimization" at Adama Science and Technology University.
