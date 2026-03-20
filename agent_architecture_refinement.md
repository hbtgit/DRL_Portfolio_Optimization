# Agent Architecture Refinement

This document outlines the refined neural network architectures for the PPO and DDPG agents, optimized for a continuous action space with 32 financial assets.

## 1. Motivation for Refinement
Portfolio optimization involves high-dimensional action spaces (32 weights) and complex state representations (10-day windows of 12 features per asset). Standard small networks (e.g., 64x64) may lack the capacity to capture cross-asset correlations and multi-temporal features.

## 2. PPO Architecture
**Policy Network (Actor) & Value Network (Critic):**
- **Type**: Multi-layer Perceptron (MLP) with shared feature extraction or decoupled heads.
- **Refinement**: We use **decoupled heads** with 3 hidden layers each.
- **Layers**: `[256, 128, 64]`
- **Activation**: `ReLU` for hidden layers, `Softmax` for policy output (to ensure $\sum w_i = 1$). *Note: Softmax is implemented via the environment's action normalization or a custom policy head.*
- **Hyperparameters**:
    - `learning_rate`: 3e-4
    - `n_steps`: 2048 (balanced for temporal correlation)
    - `batch_size`: 128 (increased for gradient stability)

## 3. DDPG Architecture
**Actor & Critic Networks:**
- **Refinement**: Larger capacity for the deterministic policy search.
- **Layers**: `[256, 128, 64]`
- **Activation**: `ReLU` for all hidden layers.
- **Hyperparameters**:
    - `learning_rate`: 1e-3
    - `buffer_size`: 100,000
    - `batch_size`: 128
    - `tau`: 0.005 (soft update coefficient)

## 4. Input Processing
Both agents use a `CombinedExtractor` to handle the `Dict` observation space:
1.  **Market Features**: `(10, 32, 12)` is flattened into a 3,840-dimensional vector.
2.  **Portfolio Weights**: `(32,)` is concatenated.
3.  **Total Flattened Input**: 3,872 dimensions.

---
**Status**: Ready for Implementation (Sprint 2)
Supporting Script: [train.py](file:///c:/Users/Hab/Desktop/DRL_Portfolio_Optimization-1/train.py)
