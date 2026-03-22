"""
benchmark_comparison.py
========================
Comprehensive performance benchmarking across:
  - All trained DRL PPO models (pilot, full, checkpoints)
  - Equal Weight baseline
  - MVO (Max Sharpe) baseline

Outputs
-------
  results/benchmark_table.csv         - Metrics for every strategy
  results/benchmark_cumreturns.png    - Cumulative return curves
  results/benchmark_metrics_bar.png   - Side-by-side metric bar charts
"""

import os
import sys
import warnings
import numpy as np

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from stable_baselines3 import PPO
from trading_env import TradingEnv
from baselines import EqualWeightStrategy, MVOStrategy
from metrics_aggregator import MetricsCollector, aggregate_and_plot

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH     = "data/processed/portfolio_data_normalized.csv"
RAW_DATA_PATH = "data/processed/portfolio_data_raw.csv"
RESULTS_DIR   = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model catalogue: (display_label, zip_path, asset_filter)
#   asset_filter = "all" | "equities"
MODEL_CATALOGUE = [
    ("PPO Pilot (equities)",    "models/pilot/ppo_pilot_final.zip",       "equities"),
    ("PPO Full (30k steps)",    "models/ppo_final.zip",                    "all"),
    ("PPO Checkpoint 10k",      "models/ppo/ppo_model_10000_steps.zip",    "all"),
    ("PPO Checkpoint 20k",      "models/ppo/ppo_model_20000_steps.zip",    "all"),
    ("PPO Checkpoint 30k",      "models/ppo/ppo_model_30000_steps.zip",    "all"),
]

CRYPTO_TICKERS = ["BTC_USD", "ETH_USD"]
TEST_START     = "2022-01-01"
RISK_FREE      = 0.0          # annualised, for Sharpe calculation


# ─── Metric helpers ───────────────────────────────────────────────────────────

def calculate_metrics(cum_series, turnovers=None):
    """
    Returns a dict with:
      ann_ret   – Annualised Net Return
      sharpe    – Annualised Sharpe (rf=0)
      max_dd    – Maximum Drawdown (negative)
      avg_to    – Average Daily Turnover (if supplied)
    """
    cum = np.array(cum_series, dtype=float)
    daily = cum[1:] / cum[:-1] - 1.0
    n_days = len(cum)

    # Annualised return
    total = cum[-1] / cum[0]
    n_years = n_days / 252.0
    ann_ret = (total ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0

    # Sharpe
    vol = np.std(daily, ddof=1) * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE) / vol if vol > 0 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(np.min(dd))

    avg_to = float(np.mean(turnovers)) if turnovers is not None else np.nan

    return dict(ann_ret=ann_ret, sharpe=sharpe, max_dd=max_dd, avg_turnover=avg_to)


# ─── Runner helpers ───────────────────────────────────────────────────────────

def _make_test_env(df_all, asset_filter):
    if asset_filter == "equities":
        df = df_all[~df_all["Ticker"].isin(CRYPTO_TICKERS)].copy()
    else:
        df = df_all.copy()
    return TradingEnv(df)


def rollout_ppo(model_path, label, df_all, asset_filter, collector: MetricsCollector = None):
    """Run a deterministic PPO rollout and return (cum_returns, turnovers)."""
    if not os.path.exists(model_path):
        print(f"  [SKIP] {label}: file not found -> {model_path}")
        return None, None

    print(f"  Rolling out {label} ...")
    env = _make_test_env(df_all, asset_filter)
    model = PPO.load(model_path, device="cpu")

    obs, _ = env.reset(seed=42)
    done = False
    cum = [1.0]
    tos = []
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        cum.append(cum[-1] * info["net_return"])
        tos.append(info["turnover"])
        if collector is not None:
            collector.record(
                step=step,
                turnover=info["turnover"],
                tc_penalty=info["tc_penalty"],
                net_return=info["net_return"],
                gross_return=info["portfolio_return"],
            )
        done = terminated or truncated
        step += 1

    if collector is not None:
        collector.save()

    return cum, tos


def rollout_equal_weight(df_all, asset_filter, collector: MetricsCollector = None):
    print("  Rolling out Equal Weight ...")
    env = _make_test_env(df_all, asset_filter)
    strat = EqualWeightStrategy()
    obs, _ = env.reset(seed=42)
    done = False
    cum = [1.0]
    tos = []
    step = 0
    while not done:
        action = strat.get_action(env, env.current_step)
        _, _, terminated, truncated, info = env.step(action)
        cum.append(cum[-1] * info["net_return"])
        tos.append(info["turnover"])
        if collector is not None:
            collector.record(
                step=step,
                turnover=info["turnover"],
                tc_penalty=info["tc_penalty"],
                net_return=info["net_return"],
                gross_return=info["portfolio_return"],
            )
        done = terminated or truncated
        step += 1
    if collector is not None:
        collector.save()
    return cum, tos


def rollout_mvo(df_all, asset_filter, collector: MetricsCollector = None):
    print("  Rolling out MVO (Max Sharpe) ...")
    env = _make_test_env(df_all, asset_filter)
    strat = MVOStrategy(RAW_DATA_PATH)
    obs, _ = env.reset(seed=42)
    done = False
    cum = [1.0]
    tos = []
    step = 0
    while not done:
        action = strat.get_action(env, env.current_step)
        _, _, terminated, truncated, info = env.step(action)
        cum.append(cum[-1] * info["net_return"])
        tos.append(info["turnover"])
        if collector is not None:
            collector.record(
                step=step,
                turnover=info["turnover"],
                tc_penalty=info["tc_penalty"],
                net_return=info["net_return"],
                gross_return=info["portfolio_return"],
            )
        done = terminated or truncated
        step += 1
    if collector is not None:
        collector.save()
    return cum, tos


# ─── Plotting ─────────────────────────────────────────────────────────────────

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]

def plot_cumulative(results, save_path):
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (label, data) in enumerate(results.items()):
        cum = data["cum"]
        ls = "--" if "Equal" in label else (":" if "MVO" in label else "-")
        lw = 2.5 if "Pilot" in label or "Full" in label else 1.5
        ax.plot(cum, label=label, color=PALETTE[i % len(PALETTE)],
                linestyle=ls, linewidth=lw)

    ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_title("Cumulative Portfolio Value — Test Period (2022–2023)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trading Days", fontsize=11)
    ax.set_ylabel("Portfolio Value (normalised to 1.0)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}×"))
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_metrics_bar(metrics_df, save_path):
    labels   = metrics_df["Strategy"].tolist()
    arr      = metrics_df["Ann. Return (%)"].tolist()
    sharpe   = metrics_df["Sharpe Ratio"].tolist()
    max_dd   = [abs(v) for v in metrics_df["Max Drawdown (%)"].tolist()]

    x   = np.arange(len(labels))
    w   = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x - w,   arr,    w, label="Ann. Return (%)", color="#4C72B0", zorder=3)
    b2 = ax.bar(x,       sharpe, w, label="Sharpe Ratio",    color="#55A868", zorder=3)
    b3 = ax.bar(x + w,   max_dd, w, label="|Max DD| (%)",    color="#C44E52", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title("Performance Metrics — All Strategies (2022–2023 Test)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Value", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PERFORMANCE BENCHMARKING")
    print("  Test period: 2022-01-01 -> end of dataset")
    print("=" * 60)

    # Load data once
    df_all = pd.read_csv(DATA_PATH)
    df_all["Date"] = pd.to_datetime(df_all["Date"])
    df_all = df_all[df_all["Date"] >= TEST_START].copy()
    print(f"\nTest set: {df_all['Date'].min().date()} → {df_all['Date'].max().date()}"
          f"  |  {df_all['Ticker'].nunique()} tickers")

    # ── Shared run_id for this benchmark session ──────────────────────────────
    import datetime
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {}     # label -> { cum:[], metrics:{} }

    # 1. DRL models
    for label, path, filt in MODEL_CATALOGUE:
        coll = MetricsCollector(strategy=label, run_id=run_id)
        cum, tos = rollout_ppo(path, label, df_all, filt, collector=coll)
        if cum is not None:
            results[label] = {"cum": cum, "metrics": calculate_metrics(cum, tos)}

    # 2. Equal Weight (full universe)
    ew_coll = MetricsCollector(strategy="Equal Weight (1/N)", run_id=run_id)
    cum_ew, tos_ew = rollout_equal_weight(df_all, "all", collector=ew_coll)
    results["Equal Weight (1/N)"] = {"cum": cum_ew, "metrics": calculate_metrics(cum_ew, tos_ew)}

    # 3. MVO
    mvo_coll = MetricsCollector(strategy="MVO (Max Sharpe)", run_id=run_id)
    cum_mvo, tos_mvo = rollout_mvo(df_all, "all", collector=mvo_coll)
    results["MVO (Max Sharpe)"] = {"cum": cum_mvo, "metrics": calculate_metrics(cum_mvo, tos_mvo)}

    # ── Build metrics table ──────────────────────────────────────────────────
    rows = []
    for label, data in results.items():
        m = data["metrics"]
        rows.append({
            "Strategy":          label,
            "Ann. Return (%)":   round(m["ann_ret"] * 100, 2),
            "Sharpe Ratio":      round(m["sharpe"],  3),
            "Max Drawdown (%)":  round(m["max_dd"] * 100, 2),
            "Avg Turnover":      round(m["avg_turnover"], 5) if not np.isnan(m["avg_turnover"]) else "—",
        })

    metrics_df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print(metrics_df.to_string(index=False))
    print("=" * 70)

    csv_path = os.path.join(RESULTS_DIR, "benchmark_table.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n  Metrics saved → {csv_path}")

    print("\n  Generating plots ...")
    plot_cumulative(results,   os.path.join(RESULTS_DIR, "benchmark_cumreturns.png"))
    plot_metrics_bar(metrics_df, os.path.join(RESULTS_DIR, "benchmark_metrics_bar.png"))

    # ── Metric Aggregation plots ─────────────────────────────────────────────
    print("\n  Running metric aggregation plots ...")
    aggregate_and_plot()

    print("\n  Done! All outputs in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
