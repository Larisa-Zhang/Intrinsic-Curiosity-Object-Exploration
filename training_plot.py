#!/usr/bin/env python
"""
training_plot_simple.py

Plots selected columns from training_log.csv row-by-row (no smoothing).
"""

import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "training_log.csv"
#D:\Users\Public\Documents\testailatest\data\Forth offical run

# Put the EXACT column names from your CSV here:
LOSS_COLS = [
    "forward_loss_each",
    #"inverse_acc_each",
    #"inverse_acc_rollout",
    #"inverse_loss_each",
    #"policy_loss_each",
    #"total_loss_each",
    #"intrinsic_reward",
    #"entropy_each"
]

def main():
    df = pd.read_csv(CSV_PATH)

    # keep only columns that actually exist
    plot_cols = [c for c in LOSS_COLS if c in df.columns]
    if not plot_cols:
        raise ValueError(
            f"None of these columns were found: {LOSS_COLS}\n"
            f"Your CSV columns are: {list(df.columns)}"
        )

    # convert to numeric + keep rows where at least one plot col is valid
    for c in plot_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=plot_cols, how="all").reset_index(drop=True)

    x = df.index  # 0,1,2,... one point per row

    plt.figure(figsize=(10, 6))
    for c in plot_cols:
        plt.plot(x, df[c], marker=".", linestyle="-", label=c)

    plt.title("Forward loss on 5 degree rotation training")
    plt.xlabel("Row index")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
