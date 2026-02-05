#!/usr/bin/env python
"""
training_plot_simple.py

Plots selected columns from training_log.csv row-by-row (no smoothing).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "data/training6.0beta0.5/training_log.csv"
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

COLOR_MAP = {
    "forward_loss_each": "tab:blue",
    "inverse_acc_rollout": "tab:purple",
    "inverse_loss_each": "tab:orange",
    "policy_loss_each": "tab:red",
    "total_loss_each": "tab:gray",
    "intrinsic_reward": "tab:cyan",
    "entropy_each": "tab:brown",
}

SPLIT_ROW = 40000
AVG_WINDOW = 1000  # average every 100 rows

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
        # before split
        plt.plot(
            x[:SPLIT_ROW],
            df[c].iloc[:SPLIT_ROW],
            linestyle="-",
            marker=".",
            color=COLOR_MAP.get(c, "black"), #default to black if the column is missing
            alpha=0.7,
            label=f"{c} (before {SPLIT_ROW})"
        )

        # after split
        plt.plot(
            x[SPLIT_ROW:],
            df[c].iloc[SPLIT_ROW:],
            linestyle="-",
            marker=".",
            color=COLOR_MAP.get(c, "black"),
            alpha=0.3,
            label=f"{c} (after {SPLIT_ROW})"
        )
        
                # ===== block-averaged trend line =====
        y = df[c].values

        # number of complete blocks
        n_blocks = len(y) // AVG_WINDOW

        # truncate so it divides evenly
        y_trim = y[:n_blocks * AVG_WINDOW]

        # reshape and average
        y_block_mean = y_trim.reshape(n_blocks, AVG_WINDOW).mean(axis=1)

        # x position = center of each block
        x_block = (
            np.arange(n_blocks) * AVG_WINDOW
            + AVG_WINDOW // 2
        )

        plt.plot(
            x_block,
            y_block_mean,
            color="black",        # 👈 distinct trend color
            linewidth=1.5,
            alpha=0.9,
            label=f"{c} (avg / {AVG_WINDOW})"
        )
        

    plt.axvline(
        x=SPLIT_ROW,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label="Split @ row 38252"
        )

    plt.text(
        SPLIT_ROW + 50,
        plt.ylim()[1] * 0.95,
        "second round of training starts here",
        rotation=90,
        va="top",
        ha="left",
        fontsize=9,
        alpha=0.7
        )


    plt.title(" Entropy Training + Testing (multinomial) (average = 1000 rows)")
    plt.xlabel("Row index")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
