"""
metrics_aggregator.py
=====================
Automated collection and persistence of step-level turnover and
transaction cost data produced during any rollout.

Usage (inside a rollout loop)
------------------------------
    from metrics_aggregator import MetricsCollector, aggregate_and_plot

    collector = MetricsCollector(strategy="PPO_Pilot", run_id="bench_001")
    for step in episode:
        ...env.step(action)...
        collector.record(
            step=step,
            turnover=info["turnover"],
            tc_penalty=info["tc_penalty"],
            net_return=info["net_return"],
            gross_return=info["portfolio_return"],
        )
    collector.save()          # writes per-strategy CSV + appends to summary
    aggregate_and_plot()      # regenerates all diagnostic plots
"""

import os
import sys
import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

METRICS_DIR = os.path.join("results", "metrics")
SUMMARY_CSV = os.path.join(METRICS_DIR, "aggregated_summary.csv")


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Collects step-level turnover and TC data for one rollout episode.

    Parameters
    ----------
    strategy : str
        Human-readable strategy name, used as file prefix.
    run_id : str, optional
        Unique run identifier.  Defaults to an ISO timestamp.
    metrics_dir : str, optional
        Root directory for metric CSVs.  Defaults to ``results/metrics``.
    """

    def __init__(self, strategy: str, run_id: str = None,
                 metrics_dir: str = METRICS_DIR):
        self.strategy    = strategy
        self.run_id      = run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_dir = metrics_dir
        os.makedirs(self.metrics_dir, exist_ok=True)

        self._rows: list[dict] = []

    # ------------------------------------------------------------------
    def record(self,
               step:          int,
               turnover:      float,
               tc_penalty:    float,
               net_return:    float,
               gross_return:  float) -> None:
        """Append one step's metrics.  Call once per env.step()."""
        self._rows.append({
            "step":         step,
            "turnover":     float(turnover),
            "tc_penalty":   float(tc_penalty),
            "net_return":   float(net_return),
            "gross_return": float(gross_return),
            # Derived
            "tc_drag":      float(gross_return - net_return),   # absolute cost
        })

    # ------------------------------------------------------------------
    def save(self) -> str:
        """
        Persist step-level data and append one row to the shared summary CSV.

        Returns
        -------
        str
            Path to the per-strategy CSV that was written.
        """
        if not self._rows:
            print(f"  [MetricsCollector] No rows to save for '{self.strategy}'.")
            return ""

        df = pd.DataFrame(self._rows)

        # ── per-strategy step-level CSV ──────────────────────────────────
        # Sanitize: replace spaces, parens, slashes -> underscores
        import re
        safe_name = re.sub(r"[^\w\-]", "_", self.strategy)
        fname    = f"{safe_name}_{self.run_id}.csv"
        out_path = os.path.join(self.metrics_dir, fname)
        df.to_csv(out_path, index=False)

        # ── summary row ──────────────────────────────────────────────────
        n_days = len(df)
        tc_total_pct = df["tc_penalty"].sum() * 100        # cumulative TC %

        summary_row = {
            "strategy":        self.strategy,
            "run_id":          self.run_id,
            "timestamp":       datetime.datetime.now().isoformat(timespec="seconds"),
            "n_steps":         n_days,
            "avg_turnover":    round(df["turnover"].mean(), 6),
            "std_turnover":    round(df["turnover"].std(),  6),
            "max_turnover":    round(df["turnover"].max(),  6),
            "p95_turnover":    round(df["turnover"].quantile(0.95), 6),
            "avg_tc_penalty":  round(df["tc_penalty"].mean(), 6),
            "total_tc_pct":    round(tc_total_pct, 4),
            "avg_tc_drag":     round(df["tc_drag"].mean(), 6),
            "total_tc_drag":   round(df["tc_drag"].sum(),  6),
        }

        if os.path.exists(SUMMARY_CSV):
            summary_df = pd.read_csv(SUMMARY_CSV)
        else:
            summary_df = pd.DataFrame()

        summary_df = pd.concat(
            [summary_df, pd.DataFrame([summary_row])], ignore_index=True
        )
        summary_df.to_csv(SUMMARY_CSV, index=False)

        print(f"  [MetricsCollector] Saved {n_days} steps -> {out_path}")
        print(f"  [MetricsCollector] Summary updated -> {SUMMARY_CSV}")
        return out_path

    # ------------------------------------------------------------------
    @property
    def dataframe(self) -> pd.DataFrame:
        """Return collected data as a DataFrame (before save)."""
        return pd.DataFrame(self._rows)


# ---------------------------------------------------------------------------
# aggregate_and_plot
# ---------------------------------------------------------------------------

def _load_all_step_csvs(metrics_dir: str = METRICS_DIR) -> pd.DataFrame:
    """Read every per-strategy step CSV and stack them with a 'strategy' column."""
    frames = []
    for fname in sorted(os.listdir(metrics_dir)):
        if fname == "aggregated_summary.csv" or not fname.endswith(".csv"):
            continue
        path = os.path.join(metrics_dir, fname)
        try:
            df   = pd.read_csv(path)
            # Derive strategy name: filename is <strategy_name>_<YYYYMMDD>_<HHMMSS>.csv
            # The last two underscore-tokens are the timestamp; everything before is the strategy.
            name_no_ext = fname[:-4]   # strip .csv
            parts = name_no_ext.rsplit("_", 2)  # split off YYYYMMDD and HHMMSS
            strat = parts[0].replace("_", " ")
            df["strategy"] = strat
            frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_and_plot(metrics_dir: str = METRICS_DIR) -> None:
    """
    Reads all persisted metric CSVs and regenerates three diagnostic plots:

    1. tc_timeseries.png    – daily TC penalty per strategy
    2. turnover_distribution.png – KDE of daily turnover per strategy
    3. tc_summary_bar.png   – total TC drag % vs mean turnover
    """
    os.makedirs(metrics_dir, exist_ok=True)

    # ── Load step-level data ─────────────────────────────────────────────────
    df = _load_all_step_csvs(metrics_dir)
    if df.empty:
        print("  [aggregate_and_plot] No step-level CSVs found. Run a rollout first.")
        return

    # ── Load summary ─────────────────────────────────────────────────────────
    if os.path.exists(SUMMARY_CSV):
        summary = pd.read_csv(SUMMARY_CSV)
        # Keep only the latest run per strategy
        summary = summary.sort_values("timestamp").drop_duplicates(
            subset="strategy", keep="last"
        )
    else:
        summary = pd.DataFrame()

    strategies = df["strategy"].unique()
    palette    = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52",
        "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
    ]
    colors = {s: palette[i % len(palette)] for i, s in enumerate(strategies)}

    # ════════════════════════════════════════════════════════════════════════
    # Plot 1 – TC Timeseries
    # ════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(13, 5))
    for strat, grp in df.groupby("strategy"):
        # Rolling 5-day mean to smooth noise
        smoothed = grp["tc_penalty"].rolling(5, min_periods=1).mean().values
        ax.plot(smoothed, label=strat, color=colors[strat], linewidth=1.4)

    ax.set_title("Daily Transaction Cost Penalty (5-day rolling mean)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Trading Day", fontsize=11)
    ax.set_ylabel("TC Penalty (fraction of portfolio)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=3))
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    p1 = os.path.join(metrics_dir, "tc_timeseries.png")
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"  Saved -> {p1}")

    # ════════════════════════════════════════════════════════════════════════
    # Plot 2 – Turnover Distribution (violin)
    # ════════════════════════════════════════════════════════════════════════
    strat_list = sorted(strategies)
    data_list  = [df[df["strategy"] == s]["turnover"].dropna().values for s in strat_list]

    fig, ax = plt.subplots(figsize=(max(10, len(strat_list) * 2), 5))
    parts = ax.violinplot(data_list, positions=range(len(strat_list)),
                          showmedians=True, showextrema=False)

    for i, (pc, strat) in enumerate(zip(parts["bodies"], strat_list)):
        pc.set_facecolor(colors[strat])
        pc.set_alpha(0.75)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.8)

    ax.set_xticks(range(len(strat_list)))
    ax.set_xticklabels(strat_list, rotation=25, ha="right", fontsize=9)
    ax.set_title("Daily Turnover Distribution by Strategy", fontsize=13, fontweight="bold")
    ax.set_ylabel("Turnover (fraction of portfolio)", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    p2 = os.path.join(metrics_dir, "turnover_distribution.png")
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"  Saved -> {p2}")

    # ════════════════════════════════════════════════════════════════════════
    # Plot 3 – TC Summary Bar (requires summary CSV)
    # ════════════════════════════════════════════════════════════════════════
    if not summary.empty and "total_tc_pct" in summary.columns:
        labels    = summary["strategy"].tolist()
        total_tc  = summary["total_tc_pct"].tolist()
        avg_to    = (summary["avg_turnover"] * 100).tolist()   # scale to %

        x = np.arange(len(labels))
        w = 0.35

        fig, ax1 = plt.subplots(figsize=(max(10, len(labels) * 2), 5))
        ax2 = ax1.twinx()

        b1 = ax1.bar(x - w / 2, total_tc, w, label="Cumulative TC (%)", color="#C44E52", alpha=0.85, zorder=3)
        b2 = ax2.bar(x + w / 2, avg_to,   w, label="Avg Daily Turnover (%)", color="#4C72B0", alpha=0.85, zorder=3)

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax1.set_ylabel("Cumulative TC drag (%)", color="#C44E52", fontsize=11)
        ax2.set_ylabel("Avg Daily Turnover (%)",  color="#4C72B0", fontsize=11)
        ax1.set_title("Total Transaction Cost Drag vs Average Turnover", fontsize=13, fontweight="bold")
        ax1.axhline(0, color="black", linewidth=0.7)
        ax1.grid(axis="y", alpha=0.2, zorder=0)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

        plt.tight_layout()
        p3 = os.path.join(metrics_dir, "tc_summary_bar.png")
        plt.savefig(p3, dpi=150)
        plt.close()
        print(f"  Saved -> {p3}")

    # ── Print summary table ──────────────────────────────────────────────────
    if not summary.empty:
        cols = ["strategy", "n_steps", "avg_turnover", "p95_turnover",
                "avg_tc_penalty", "total_tc_pct", "total_tc_drag"]
        available = [c for c in cols if c in summary.columns]
        print("\n" + "=" * 80)
        print(summary[available].to_string(index=False))
        print("=" * 80)
