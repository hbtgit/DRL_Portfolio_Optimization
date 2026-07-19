import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from trading_env import TradingEnv
from baselines import EqualWeightStrategy


def sharpe_annret_maxdd(cum):
    cum = np.asarray(cum, dtype=float)
    daily = cum[1:] / cum[:-1] - 1.0
    years = len(cum) / 252.0
    ann_ret = (cum[-1] / cum[0]) ** (1 / years) - 1 if years > 0 else -1.0
    std = np.std(daily) * np.sqrt(252)
    sharpe = ann_ret / std if std != 0 else 0.0
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.min((cum - peak) / peak))
    return float(sharpe), float(ann_ret), max_dd


def backtest(model, env, timed=False):
    obs, _ = env.reset()
    cum, latencies, done = [1.0], [], False
    while not done:
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        if timed:
            latencies.append((time.perf_counter() - t0) * 1000.0)
        obs, _, terminated, truncated, info = env.step(action)
        cum.append(cum[-1] * info["net_return"])
        done = terminated or truncated
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    return cum, p95


def baseline_backtest(env):
    obs, _ = env.reset()
    strat, cum, done = EqualWeightStrategy(), [1.0], False
    while not done:
        action = strat.get_action(env, env.current_step)
        _, _, terminated, truncated, info = env.step(action)
        cum.append(cum[-1] * info["net_return"])
        done = terminated or truncated
    return cum


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--min-sharpe", type=float, required=True)
    p.add_argument("--max-drawdown", type=float, required=True)
    p.add_argument("--max-latency-ms", type=float, required=True)
    p.add_argument("--report", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    df["Date"] = pd.to_datetime(df["Date"])
    test_df = df[df["Date"] >= "2022-01-01"].copy()

    model = PPO.load(args.model)
    agent_cum, p95_latency = backtest(model, TradingEnv(test_df), timed=True)
    agent_sharpe, agent_ret, agent_dd = sharpe_annret_maxdd(agent_cum)

    ew_cum = baseline_backtest(TradingEnv(test_df))
    ew_sharpe, _, _ = sharpe_annret_maxdd(ew_cum)

    checks = {
        "sharpe_above_min": agent_sharpe >= args.min_sharpe,
        "beats_equal_weight": agent_sharpe >= ew_sharpe,
        "drawdown_within_limit": agent_dd >= args.max_drawdown,
        "latency_within_limit": p95_latency <= args.max_latency_ms,
    }
    report = {
        "agent": {
            "sharpe": agent_sharpe,
            "annual_return": agent_ret,
            "max_drawdown": agent_dd,
            "p95_predict_latency_ms": p95_latency,
        },
        "equal_weight": {"sharpe": ew_sharpe},
        "thresholds": {
            "min_sharpe": args.min_sharpe,
            "max_drawdown": args.max_drawdown,
            "max_latency_ms": args.max_latency_ms,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }

    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    if not report["passed"]:
        failed = [k for k, v in checks.items() if not v]
        print(f"VALIDATION FAILED: {failed}", file=sys.stderr)
        sys.exit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
