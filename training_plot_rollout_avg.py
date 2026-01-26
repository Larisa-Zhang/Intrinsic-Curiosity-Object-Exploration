#!/usr/bin/env python
"""
training_plot_rollout_avg.py

Plots selected columns from training_log.csv and record.csv,
averaged per rollout (rollout_size rows per rollout).

- For training_log.csv: groups every rollout_size rows.
- For record.csv: groups every rollout_size rows too (matches your trigger cadence).

Optional: show ±1 std shading.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
ROLLOUT_SIZE = int(os.getenv("ROLLOUT_SIZE", "50"))

TRAINING_CSV = os.getenv("TRAINING_CSV", "training_log.csv")
RECORD_CSV   = os.getenv("RECORD_CSV", "record.csv")

# pick whatever you want
TRAINING_COLS = [
     "forward_loss_each",
     "inverse_loss_each",
    # "icm_loss_each",
     "policy_loss_each",
    # "value_loss_each",
    # "entropy_each",
    # "total_loss_each",
    # "intrinsic_reward",
    # "inverse_acc_each",
    # "inverse_acc_rollout",  # note: this is constant within a rollout in our logger
    # "inv_grad_mean_abs",
    # "inv_grad_max_abs",
]

RECORD_COLS = [
    # "reward",
    # "delta_yaw",
    # "delta_pitch",
    # "prob_0", "prob_1", "prob_2", "prob_3",
]

# =========================
# Helpers
# =========================
def coerce_numeric(df: pd.DataFrame, cols):
    cols = [c for c in cols if c in df.columns]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return cols

def add_rollout_index(df: pd.DataFrame, rollout_size: int) -> pd.DataFrame:
    """
    Adds:
      - row_idx: 0..N-1
      - rollout_idx: floor(row_idx / rollout_size)
      - within_rollout: row_idx % rollout_size
    """
    df = df.copy()
    df["row_idx"] = np.arange(len(df), dtype=int)
    df["rollout_idx"] = (df["row_idx"] // rollout_size).astype(int)
    df["within_rollout"] = (df["row_idx"] % rollout_size).astype(int)
    return df

def rollout_aggregate(df: pd.DataFrame, cols):
    """
    Returns a dataframe indexed by rollout_idx with mean and std for each col.
    """
    g = df.groupby("rollout_idx")[cols]
    mean_df = g.mean().add_suffix("_mean")
    std_df  = g.std(ddof=0).fillna(0.0).add_suffix("_std")
    out = pd.concat([mean_df, std_df], axis=1).reset_index()
    return out

def plot_rollout_means(agg_df: pd.DataFrame, cols, title: str, show_std=True):
    """
    agg_df: has columns like <col>_mean and <col>_std
    """
    if agg_df.empty:
        print(f"[WARN] Nothing to plot for: {title}")
        return

    x = agg_df["rollout_idx"].to_numpy()

    plt.figure(figsize=(11, 6))
    for c in cols:
        mcol = f"{c}_mean"
        scol = f"{c}_std"
        if mcol not in agg_df.columns:
            continue

        y = agg_df[mcol].to_numpy()
        plt.plot(x, y, marker="o", linewidth=1.5, label=c)

        if show_std and scol in agg_df.columns:
            s = agg_df[scol].to_numpy()
            plt.fill_between(x, y - s, y + s, alpha=0.15)

    plt.title(title)
    plt.xlabel(f"Rollout index (each = {ROLLOUT_SIZE} rows)")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =========================
# Main
# =========================
def main():
    # -------- training_log.csv --------
    if os.path.exists(TRAINING_CSV):
        df_t = pd.read_csv(TRAINING_CSV)

        # Clean: keep only rows where at least one selected col exists & is numeric
        tcols = coerce_numeric(df_t, TRAINING_COLS)
        if tcols:
            df_t = df_t.dropna(subset=tcols, how="all").reset_index(drop=True)
            df_t = add_rollout_index(df_t, ROLLOUT_SIZE)
            agg_t = rollout_aggregate(df_t, tcols)

            plot_rollout_means(
                agg_t,
                cols=tcols,
                title=f"training_log.csv averaged per rollout (N={ROLLOUT_SIZE})",
                show_std=True,
            )
        else:
            print("[WARN] None of TRAINING_COLS found in training_log.csv")
    else:
        print(f"[WARN] {TRAINING_CSV} not found")

    # -------- record.csv --------
    if os.path.exists(RECORD_CSV):
        df_r = pd.read_csv(RECORD_CSV)

        # Your record.csv includes init rows with actionId=-1; drop those for per-step stats
        if "actionId" in df_r.columns:
            df_r = df_r[df_r["actionId"].astype(str) != "-1"].reset_index(drop=True)

        rcols = coerce_numeric(df_r, RECORD_COLS)
        if rcols:
            df_r = df_r.dropna(subset=rcols, how="all").reset_index(drop=True)
            df_r = add_rollout_index(df_r, ROLLOUT_SIZE)
            agg_r = rollout_aggregate(df_r, rcols)

            plot_rollout_means(
                agg_r,
                cols=rcols,
                title=f"record.csv averaged per rollout (N={ROLLOUT_SIZE})",
                show_std=True,
            )
        else:
            print("[WARN] None of RECORD_COLS found in record.csv")
    else:
        print(f"[WARN] {RECORD_CSV} not found")


if __name__ == "__main__":
    main()
