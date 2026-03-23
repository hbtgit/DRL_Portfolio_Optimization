import os
import sys
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy import stats
from stable_baselines3 import PPO
from trading_env import TradingEnv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH       = "data/processed/portfolio_data_normalized.csv"
METRICS_DIR     = "results/metrics"
BENCHMARK_CSV   = "results/benchmark_table.csv"
SUMMARY_CSV     = os.path.join(METRICS_DIR, "aggregated_summary.csv")
OUT_DIR         = "results/sprint4"
os.makedirs(OUT_DIR, exist_ok=True)

CRYPTO_TICKERS  = ["BTC_USD", "ETH_USD"]
TEST_START      = "2022-01-01"

PALETTE = {
    "PPO Pilot (equities)": "#4C72B0",
    "PPO Full (30k steps)": "#DD8452",
    "PPO Checkpoint 10k":   "#937860",
    "PPO Checkpoint 20k":   "#8C8C8C",
    "PPO Checkpoint 30k":   "#8172B2",
    "Equal Weight (1/N)":   "#55A868",
    "MVO (Max Sharpe)":     "#C44E52",
}


# ══════════════════════════════════════════════════════════════════════════════
# Chart 1 – Equity Curves
# ══════════════════════════════════════════════════════════════════════════════

def _build_cum_from_csv(csv_path: str) -> np.ndarray:
    """Reconstruct cumulative returns from step-level metric CSV."""
    df  = pd.read_csv(csv_path)
    cum = np.cumprod(df["net_return"].values)
    return np.concatenate([[1.0], cum])


def _latest_csv(metrics_dir: str, safe_name_pattern: str) -> str | None:
    """Return the most recent CSV file matching a pattern."""
    candidates = [
        f for f in os.listdir(metrics_dir)
        if re.match(safe_name_pattern + r"_\d{8}_\d{6}\.csv", f)
    ]
    if not candidates:
        return None
    candidates.sort()
    return os.path.join(metrics_dir, candidates[-1])


# Map display label -> regex-safe filename prefix
LABEL_TO_FILE = {
    "PPO Pilot (equities)": r"PPO_Pilot__equities_",
    "PPO Full (30k steps)": r"PPO_Full__30k_steps_",
    "PPO Checkpoint 10k":   r"PPO_Checkpoint_10k",
    "PPO Checkpoint 20k":   r"PPO_Checkpoint_20k",
    "PPO Checkpoint 30k":   r"PPO_Checkpoint_30k",
    "Equal Weight (1/N)":   r"Equal_Weight__1_N_",
    "MVO (Max Sharpe)":     r"MVO__Max_Sharpe_",
}


def plot_equity_curves(save_path: str) -> None:
    print("  Building equity curves ...")
    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    highlight = "PPO Pilot (equities)"          # drawdown shading only for best DRL

    for label, pat in LABEL_TO_FILE.items():
        csv = _latest_csv(METRICS_DIR, pat)
        if csv is None:
            print(f"    [skip] no CSV found for {label}")
            continue

        cum   = _build_cum_from_csv(csv)
        color = PALETTE.get(label, "#333333")
        lw    = 2.2 if label == highlight else 1.4
        ls    = "--" if "Equal" in label or "MVO" in label else "-"
        alpha = 1.0 if label in (highlight, "Equal Weight (1/N)", "MVO (Max Sharpe)") else 0.65

        ax_eq.plot(cum, color=color, linewidth=lw, linestyle=ls,
                   alpha=alpha, label=label, zorder=3)

        # Drawdown shading for the highlighted model
        if label == highlight:
            peak = np.maximum.accumulate(cum)
            dd   = (cum - peak) / peak
            ax_dd.fill_between(range(len(dd)), dd, 0,
                               color=color, alpha=0.30, label=label)
            ax_dd.plot(dd, color=color, linewidth=1.0, alpha=0.7)

        # Annotate final value on right edge
        ax_eq.annotate(
            f"  {cum[-1]:.2f}x",
            xy=(len(cum) - 1, cum[-1]),
            fontsize=7, color=color, va="center",
        )

    ax_eq.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4, zorder=1)
    ax_eq.set_ylabel("Portfolio Value (1.0   =   Start)", fontsize=11)
    ax_eq.set_title("Equity Curves — Test Period 2022-2023\n(DRL models vs Baselines)",
                    fontsize=13, fontweight="bold")
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}x"))
    ax_eq.legend(fontsize=8, framealpha=0.88, loc="lower left")
    ax_eq.grid(True, alpha=0.2)

    ax_dd.axhline(0, color="black", linewidth=0.7)
    ax_dd.set_ylabel("Drawdown", fontsize=9)
    ax_dd.set_xlabel("Trading Days", fontsize=11)
    ax_dd.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_dd.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    print(f"  Saved -> {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Chart 2 – Weight Allocation Heatmaps
# ══════════════════════════════════════════════════════════════════════════════

def _rollout_weights(model_path: str, df_all: pd.DataFrame,
                     asset_filter: str, max_steps: int = 400):
    """Run a deterministic PPO rollout and record target_weights at each step."""
    if not os.path.exists(model_path):
        print(f"    [skip] model not found: {model_path}")
        return None, None

    if asset_filter == "equities":
        df = df_all[~df_all["Ticker"].isin(CRYPTO_TICKERS)].copy()
    else:
        df = df_all.copy()

    env   = TradingEnv(df)
    model = PPO.load(model_path, device="cpu")
    obs, _ = env.reset(seed=42)

    weight_history = []
    done  = False
    steps = 0

    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        action     = action.astype(float)
        if action.sum() > 0:
            action /= action.sum()
        weight_history.append(action.copy())
        obs, _, terminated, truncated, _ = env.step(action)
        done  = terminated or truncated
        steps += 1

    return np.array(weight_history), env.tickers   # (T, N), [ticker list]


def plot_weight_heatmap(weight_matrix: np.ndarray, tickers: list,
                        title: str, save_path: str) -> None:
    T, N = weight_matrix.shape

    # Sort tickers by average weight (most allocated at top)
    avg_w   = weight_matrix.mean(axis=0)
    order   = np.argsort(avg_w)[::-1]
    W       = weight_matrix[:, order]
    labels  = [f"{tickers[i]}  ({avg_w[i]:.3f})" for i in order]

    # Only show top-30 tickers for readability
    top_n = min(30, N)
    W     = W[:, :top_n]
    labels = labels[:top_n]

    fig, ax = plt.subplots(figsize=(14, max(6, top_n * 0.35)))
    im = ax.imshow(W.T, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=min(0.5, W.max()),
                   interpolation="nearest")

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Trading Day", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Portfolio Weight", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    print(f"  Saved -> {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Chart 3 – Cost vs Return Scatter
# ══════════════════════════════════════════════════════════════════════════════

def plot_cost_return_scatter(save_path: str) -> None:
    print("  Building cost-vs-return scatter ...")

    # Load data
    bench   = pd.read_csv(BENCHMARK_CSV)  # Strategy, Ann. Return(%), ...
    summary = pd.read_csv(SUMMARY_CSV)
    # Keep latest run per strategy
    summary = summary.sort_values("timestamp").drop_duplicates("strategy", keep="last")

    # Normalise strategy name for merge (lowercase strip)
    bench["_key"]   = bench["Strategy"].str.strip().str.lower()
    summary["_key"] = summary["strategy"].str.strip().str.lower()
    merged = pd.merge(bench, summary, on="_key", how="inner")

    if merged.empty:
        print("    [skip] no matching rows for scatter (check strategy name alignment)")
        return

    x = merged["total_tc_pct"].values.astype(float)
    y = merged["Ann. Return (%)"].values.astype(float)
    labels = merged["Strategy"].values

    fig, ax = plt.subplots(figsize=(11, 7))

    # Scatter
    drl_mask = ~merged["Strategy"].str.contains("Equal|MVO", na=False)
    ax.scatter(x[drl_mask],  y[drl_mask],  s=110, zorder=4,
               color="#4C72B0", edgecolors="white", linewidths=0.8, label="DRL Models")
    ax.scatter(x[~drl_mask], y[~drl_mask], s=110, zorder=4,
               color="#55A868", edgecolors="white", linewidths=0.8,
               marker="D", label="Baselines")

    # Labels
    try:
        from adjustText import adjust_text
        texts = []
        yrange = max(y) - min(y) if len(y) > 0 else 10
        y_offset = yrange * 0.03
        for xi, yi, lbl in zip(x, y, labels):
            short = lbl.replace("PPO Pilot (equities)", "Pilot").replace(
                    "PPO Full (30k steps)", "Full").replace("PPO Checkpoint ", "Ckpt ")
            texts.append(ax.text(xi, yi + y_offset, short, fontsize=8, color="#333333",
                                 ha="center", va="bottom"))
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5), force_points=(0.2, 0.5))
        # Add margins to prevent labels from being cut off at the edges
        ax.margins(y=0.15, x=0.15)
    except ImportError:
        for xi, yi, lbl in zip(x, y, labels):
            short = lbl.replace("PPO Pilot (equities)", "Pilot").replace(
                    "PPO Full (30k steps)", "Full").replace("PPO Checkpoint ", "Ckpt ")
            ax.annotate(short, (xi, yi),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color="#333333")

    # Regression line
    if len(x) >= 3:
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xfit = np.linspace(x.min() * 0.9, x.max() * 1.05, 100)
        ax.plot(xfit, slope * xfit + intercept, "--", color="grey",
                linewidth=1.3, alpha=0.7,
                label=f"Trend  (r={r:.2f}, p={p:.2f})")

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xlabel("Cumulative TC Drag  (%)", fontsize=12)
    ax.set_ylabel("Net Annualised Return  (%)", fontsize=12)
    ax.set_title("Transaction Cost Drag vs Net Return\n(2022-2023 Test Period)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.88)
    ax.grid(True, alpha=0.2)

    # Shade "efficient" region (low TC, positive return)
    ax.axvspan(0, 20, alpha=0.05, color="green", label="_nolegend_")
    ax.text(1, ax.get_ylim()[0] * 0.85, "Low-cost zone",
            fontsize=8, color="darkgreen", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    print(f"  Saved -> {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  SPRINT 4: VISUAL ANALYSIS")
    print("=" * 60)

    # ── Chart 1: Equity Curves ─────────────────────────────────────────────
    print("\n[1/4] Equity Curves ...")
    plot_equity_curves(os.path.join(OUT_DIR, "equity_curves.png"))

    # ── Charts 2a & 2b: Weight Heatmaps ───────────────────────────────────
    print("\n[2/4] Loading data for weight heatmaps ...")
    df_all = pd.read_csv(DATA_PATH)
    df_all["Date"] = pd.to_datetime(df_all["Date"])
    df_all = df_all[df_all["Date"] >= TEST_START].copy()

    print("  Running PPO Pilot rollout ...")
    W_pilot, tickers_pilot = _rollout_weights(
        "models/pilot/ppo_pilot_final.zip", df_all, "equities", max_steps=388
    )
    if W_pilot is not None:
        plot_weight_heatmap(
            W_pilot, list(tickers_pilot),
            title="Weight Allocation Heatmap — PPO Pilot (Equities, 2022-2023)",
            save_path=os.path.join(OUT_DIR, "weight_heatmap_pilot.png"),
        )

    print("  Running PPO Full rollout ...")
    W_full, tickers_full = _rollout_weights(
        "models/ppo_final.zip", df_all, "all", max_steps=400
    )
    if W_full is not None:
        plot_weight_heatmap(
            W_full, list(tickers_full),
            title="Weight Allocation Heatmap — PPO Full (32 Assets, 2022-2023)",
            save_path=os.path.join(OUT_DIR, "weight_heatmap_full.png"),
        )

    # ── Chart 3: Cost vs Return Scatter ───────────────────────────────────
    print("\n[3/4] Cost-vs-Return Scatter ...")
    plot_cost_return_scatter(os.path.join(OUT_DIR, "cost_vs_return_scatter.png"))

    print(f"\n  All charts saved in: {OUT_DIR}/")
    print("  Done!")


if __name__ == "__main__":
    main()
