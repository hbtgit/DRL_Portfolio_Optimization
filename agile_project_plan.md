# 1-Month Agile Project Plan: TC-Aware DRL Research

This plan outlines the implementation, experimentation, and documentation phases for the "Transaction Cost-Aware Deep Reinforcement Learning for Financial Portfolio Optimization" project.

## Project Timeline (4 Sprints / 1 Week Each)

### Sprint 1: Data Mastery & Environment Refinement
**Goal**: Establish a robust data pipeline and a cost-accurate simulation environment.
- [x] **Data Acquisition**: Finalize ingestion scripts for Yahoo Finance and Crypto APIs (2015–2023).
- [x] **Feature Engineering**: Implement technical indicators and time-windowed price tensors.
- [x] **Env Upgrade**: Refine `env.py` to strictly enforce proportional transaction costs ($\lambda$).
- [x] **Validation**: Verify that the "net return" calculation matches theoretical expectations.

### Sprint 2: Agent Implementation & Initial Training
**Goal**: Build and tune core DRL agents against baseline strategies.
- [x] **Algorithm Refinement**: Fine-tune PPO and DDPG architectures for continuous action spaces.
- [x] **Baseline Implementation**: Standardize Mean-Variance Optimization (MVO) and Equal-Weight benchmarks.
- [x] **TC-Aware Reward**: Implement and debug the penalty-based reward function $R_t = \ln(\text{return}) - \lambda \cdot \text{turnover}$.
- [x] **Pilot Runs**: Conduct initial training on a single asset class (e.g., Dow 30).

### Sprint 3: Full-Scale Experimentation & Analysis
**Goal**: Run comprehensive tests across diverse markets and analyze agent behavior.
- [x] **Cross-Asset Training**: Execute large-scale training on volatile (Crypto) vs. stable (Equities) regimes.
- [x] **Sensitivity Analysis**: Run experiments with varying $\lambda$ values to observe "no-trade region" emergence.
- [x] **Performance Benchmarking**: Compare net ARR, Sharpe Ratio, and Max Drawdown across all models.
- [x] **Metric Aggregation**: Automate the collection of turnover and transaction cost data.

### Sprint 4: Visualization, Documentation & Synthesis
**Goal**: Finalize research results and prepare the thesis/repository for delivery.
- [x] **Visual Analysis**: Generate equity curves, weight allocation heatmaps, and cost-vs-return scatter plots.
- [x] **Research Synthesis**: Formally answer RQ1, RQ2, and RQ3 based on experimental evidence.
- [x] **Documentation**: Complete the `Thesis_proposal.md` with final results and update `README.md`.
- [x] **Final Review**: Perform code cleanup and ensure the repository is reproducible via `requirements.txt`.


---

## Sprint Cadence
- **Monday**: Sprint Planning (Define specific tickets for the week).
- **Wednesday**: Mid-Sprint Review (Check training convergence/data issues).
- **Friday**: Sprint Demo & Retro (Review visualizations and metrics).
