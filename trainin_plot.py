#!/usr/bin/env python
"""
training_plot_simple.py

Plots selected columns from training_log.csv row-by-row (no smoothing).

- Raw per-row points (colored)
- Black block-averaged trend line (AVG_WINDOW)
- NEW: label every 10th block-mean point (i.e., every 10,000 rows) on top of the black line
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "training_log.csv"
# D:\Users\Public\Documents\testailatest\data\Forth offical run

LOSS_COLS = [
    "forward_loss_each",
    #"inverse_acc_each",
    "inverse_acc_rollout",
    "inverse_loss_each",
    "policy_loss_each",
    "total_loss_each",
    "intrinsic_reward",
    "entropy_each"
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

TITLE_NAME_MAP = {
    "forward_loss_each": "Forward Loss",
    "inverse_acc_rollout": "Inverse Accuracy Rollout",
    "inverse_loss_each": "Inverse Loss",
    "policy_loss_each": "Policy Loss",
    "total_loss_each": "Total Loss",
    "intrinsic_reward": "Intrinsic Reward",
    "entropy_each": "Entropy",
}

AVG_WINDOW = 1000                 # black trend line resolution
LABEL_EVERY_N_BLOCKS = 10         # every 10 blocks => every 10,000 rows
DRAW_10K_LINES = True
ANNOTATE_BLOCK_MEANS = True
MAX_LABELS_PER_METRIC = 20        # avoid clutter


def main():
    df = pd.read_csv(CSV_PATH)

    plot_cols = [c for c in LOSS_COLS if c in df.columns]
    if not plot_cols:
        raise ValueError(
            f"None of these columns were found: {LOSS_COLS}\n"
            f"Your CSV columns are: {list(df.columns)}"
        )

    for c in plot_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=plot_cols, how="all").reset_index(drop=True)

    x = df.index
    n_rows = len(df)

    # plot all metrics in LOSS_COLS that are present in the CSV one by one
    for c in plot_cols:
        plt.figure(figsize=(10, 6))

        # Optional: draw faint boundaries at every 10,000 rows
        if DRAW_10K_LINES and n_rows >= (AVG_WINDOW * LABEL_EVERY_N_BLOCKS):
            n_full_10k = n_rows // (AVG_WINDOW * LABEL_EVERY_N_BLOCKS)
            for w in range(1, n_full_10k):
                boundary = w * AVG_WINDOW * LABEL_EVERY_N_BLOCKS
                plt.axvline(boundary, color="black", alpha=0.08, linewidth=1.0, zorder=0)

        # --- raw per-row points (colored) ---
        plt.plot(
            x,
            df[c],
            linestyle="-",
            marker=".",
            color=COLOR_MAP.get(c, "black"),
            alpha=0.7,
            label=f"{c}"
        )

        y = df[c].values
        n_blocks = len(y) // AVG_WINDOW
        if n_blocks >= 1:
            y_trim = y[:n_blocks * AVG_WINDOW]
            y_block_mean = y_trim.reshape(n_blocks, AVG_WINDOW).mean(axis=1)
            x_block = np.arange(n_blocks) * AVG_WINDOW + AVG_WINDOW // 2

            # black trend line
            plt.plot(
                x_block,
                y_block_mean,
                color="black",
                linewidth=1.5,
                alpha=0.9,
                label=f"{c} (avg / {AVG_WINDOW})"
            )

            # label every 10th block-mean + highlight
            if ANNOTATE_BLOCK_MEANS:
                idxs = np.unique(
                    np.concatenate([
                        np.array([0], dtype=int),
                        np.arange(LABEL_EVERY_N_BLOCKS - 1, n_blocks, LABEL_EVERY_N_BLOCKS)
                    ])
                )

                y_std = float(np.nanstd(y_block_mean))
                y_offset = 0.02 * y_std if y_std > 0 else 0.0

                for i in idxs:
                    xi = x_block[i]
                    yi = y_block_mean[i]
                    if not np.isfinite(yi):
                        continue

                    plt.scatter([xi], [yi], s=10, color="black", alpha=0.9, zorder=5)
                    plt.text(
                        xi, yi + y_offset, f"{yi:.3g}",
                        fontsize=8, alpha=0.75, ha="center", va="bottom", zorder=6
                    )

                    start = i * AVG_WINDOW
                    end = (i + 1) * AVG_WINDOW - 1
                    print(f"{c} | block_mean {start}–{end}: {yi:.6g}")

        pretty_name = TITLE_NAME_MAP.get(c, c)
        plt.title(f"{pretty_name} in Training (block avg = {AVG_WINDOW} rows)")
        plt.xlabel("Row index")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
        main()
