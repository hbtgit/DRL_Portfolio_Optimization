# Transaction Cost-Aware Deep Reinforcement Learning for Financial Portfolio Optimization

**Habtamu Tadese Sefera**

A thesis proposal submitted to the Department of Computer Science and Engineering  
School of Electrical Engineering and Computing  

Office of Graduate Studies  
Adama Science and Technology University  

January, 2026  
Adama, Ethiopia  

---

## Abstract

This research addresses the challenge of transaction costs in Deep Reinforcement Learning (DRL) for financial portfolio optimization. Traditional models often ignore trading friction, leading to unrealistic strategies with high turnover. This project develops a cost-aware DRL framework that incorporates transaction cost penalties into the reward function, creating "no-trade regions" and improving net profitability. The framework is evaluated using 8 years of historical data across equities and cryptocurrencies.

---

## Contents

- [1. Introduction](#1-introduction)
- [2. Literature Review](#2-literature-review)
- [3. Design and Methodology](#3-design-and-methodology)
- [4. Work and Budget Plan](#4-work-and-budget-plan)

---

# CHAPTER ONE

## 1. INTRODUCTION

### 1.1 Background of the Study

Financial portfolio management aims to allocate capital across assets to maximize returns while minimizing risk. Modern Portfolio Theory (MPT), introduced by Markowitz (1952), established the efficient frontier using mean-variance optimization.

However, MPT assumes market efficiency, normal return distributions, and linear correlations. Real markets violate these assumptions due to volatility clustering, non-stationarity, and behavioral biases.

### 1.2 Motivation

Deep Reinforcement Learning (DRL) has shown strong performance in simulations but often ignores transaction costs. This leads to excessive trading (“churning”), unrealistic profits, and poor real-world performance. This research introduces cost-aware learning to address this gap.

### 1.3 Problem Statement

Traditional DRL models optimize gross returns while ignoring transaction costs. This results in high turnover strategies and poor net performance when deployed in realistic market environments where trading is not free.

### 1.4 Research Questions

1. How do transaction costs affect DRL performance?
2. Can cost-aware DRL outperform traditional strategies?
3. Does it reduce trading frequency?

### 1.5 Objectives

#### General Objective
To develop, implement, and assess a Transaction Cost-Aware Deep Reinforcement Learning (DRL) framework for effective portfolio optimization in a realistic market model.

#### Specific Objectives
- **Quantify Impact**: Quantify the detrimental impacts of transaction costs on the performance of non-cost-aware DRL agents.
- **Reward Engineering**: Design a modified reward function that directly penalizes excessive trading while preserving returns.
- **Comparative Study**: Compare the proposed TC-aware DRL framework against traditional Portfolio Optimization (e.g., Mean-Variance Optimization) and standard non-TC-aware DRL models.
- **Robustness Testing**: Evaluate the framework across diverse asset classes, including equities and cryptocurrencies, under various market conditions.

### 1.6 Significance

- Bridges the gap between theoretical DRL performance and practical trading.
- Improves the reliability and profitability of autonomous trading systems.
- Provides insights into the behavior of DRL agents in the presence of friction.

### 1.7 Scope

- **DRL Models**: PPO (Proximal Policy Optimization) and DDPG (Deep Deterministic Policy Gradient).
- **Assets**: Diverse classes including Dow Jones stocks and Cryptocurrencies.
- **Data Period**: 2015–2023 (8 years of historical daily data).

---

# CHAPTER TWO

## 2. LITERATURE REVIEW

### 2.1 Evolution of Portfolio Theory

- **Markowitz (1952)**: Introduced Mean-Variance Optimization.
- **Later work**: Focused on transaction costs and the concept of "no-trade regions" to minimize friction.

### 2.2 DRL in Finance

DRL handles high-dimensional data and learns adaptive strategies. Popular frameworks like FinRL and Stable-Baselines3 provide the infrastructure for implementing complex agents like PPO and SAC.

### 2.3 Cost-Aware DRL

Key idea: Penalize turnover in the reward function. This discourages unnecessary rebalancing, improves the Sharpe ratio, and leads to more realistic strategies.

---

# CHAPTER THREE

## 3. DESIGN AND METHODOLOGY

### 3.1 Problem Formulation

Portfolio management is modeled as a Markov Decision Process (MDP).

#### 3.1.1 State Space
The state $s_t$ includes historical price features extracted using time-step windows and the current portfolio weights $w_{t-1}$.

#### 3.1.2 Action Space
The action $w_t$ represents the target portfolio weights, satisfying simplex constraints (sum to 1, no short selling).

#### 3.1.3 Transaction Cost Model
Cost is modeled as a proportional fee: $Cost_t = \lambda \sum |w_t - w'_{t-1}|$.

### 3.2 Reward Function

The modified reward function incorporates a penalty for transaction costs:
$R_t = \ln(\text{return}) - \lambda \times \text{turnover}$

### 3.3 Algorithms

- **PPO**: Proximal Policy Optimization for stable policy updates.
- **DDPG**: Deep Deterministic Policy Gradient for continuous action spaces.

### 3.4 Data

- **Source**: Yahoo Finance.
- **Period**: 2015–2023.
- **Assets**: Dow Jones Industrial Average components and major Cryptocurrencies.

---

# CHAPTER FOUR

## 4. WORK AND BUDGET PLAN

### 4.1 Time Schedule

- Literature Review & Data Collection
- Model Development & Reward Engineering
- Evaluation & Comparison
- Thesis Writing & Defense

### 4.2 Budget

Total estimated budget: 25,000 ETB.

---

# References

- Markowitz (1952)
- Liang et al. (2018)
- Jiang et al. (2017)
- Liu et al. (2022)
- Bai et al. (2024)

---

# CHAPTER FIVE

## 5. EXPERIMENTAL RESULTS AND SYNTHESIS

Based on the 8-year historical backtest (2015-2023) and out-of-sample testing (2022-2023) across multiple PPO agents and baselines, the research questions are addressed as follows:

### 5.1 RQ1: How do transaction costs affect DRL performance?
**Transaction costs are the primary bottleneck for DRL profitability in realistic markets.** During the 2022-2023 test period, the standard 32-asset cross-regime PPO models generated a staggering **50% to 54% cumulative transaction cost drag**. This extreme trading friction entirely erased the agent's gross returns, resulting in net annualized returns of -13.14% compared to the baseline's +8.70%. The empirical evidence confirms that without strict turnover constraints or extreme predictive accuracy, standard DRL architectures will spend themselves into unprofitability via constant microscopic rebalancing.

### 5.2 RQ2: Can cost-aware DRL outperform traditional strategies?
**Yes, under the right conditions.** The TC-aware "PPO Pilot" model, trained exclusively on the 30-asset equities regime, successfully captured market trends while managing costs. It achieved a **+13.81% Net Annualized Return** with a Sharpe Ratio of 0.73, significantly outperforming both the Equal Weight (1/N) and Mean-Variance Optimization (MVO) baselines, which yielded +8.70% (Sharpe 0.55). This demonstrates that proportional cost penalties ($R_t = \ln(\text{return}) - \lambda \cdot \text{turnover}$) can force the agent to find a profitable sub-space. However, scaling this to highly volatile cross-asset universes (mixing crypto and equities) destabilizes the agent, indicating that successful cost-aware DRL requires either regime-specific training or significantly higher compute allocation (e.g., millions of timesteps).

### 5.3 RQ3: Does it reduce trading frequency?
**Yes, but it requires precise tuning of the penalty parameter ($\lambda$).** The TC-aware reward function successfully created a continuum of trading frequencies. The pilot equities model learned to restrict its trading to a moderate **0.40 average daily turnover** (translating to ~15% cumulative drag, which was easily offset by its gross returns). Contrastingly, the full cross-asset models failed to find this balance, trading at ~0.88 turnover daily as they endlessly chased volatile cryptocurrency spikes. Furthermore, sensitivity analysis utilizing a TC-aware MVO proxy confirmed the theoretical existence of a "no-trade region" that widens exponentially as $\lambda$ increases.

### 5.4 Conclusion
Integrating transaction costs directly into the MDP reward function is non-negotiable for deploying DRL in financial markets. This research validates that a heavily penalized DRL agent can outperform Markowitz MPT baselines (RQ2), but it also highlights the fragility of these agents in multi-regime environments (RQ1, RQ3) where the allure of volatile assets can overpower the agent's learned cost-aversion.
