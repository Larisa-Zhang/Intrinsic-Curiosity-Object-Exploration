"""
action_plot.py

1) Scatter plot: greedy_action (0/1/2/3) vs step
2) Overall percentage of each action
3) NEW: Probability curves (prob_0..prob_3) over training steps
"""

import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "record1.csv"
ACTION_COL = "greedy_action"

# If your prob column names differ, change here:
PROB_COLS = ["prob_0", "prob_1", "prob_2", "prob_3"]

ACTION_COLORS = {0: "red", 1: "blue", 2: "green", 3: "orange"}

def main():
    df = pd.read_csv(CSV_PATH)

    # ---------------- Clean action data ----------------
    df = df[df[ACTION_COL].notna()]
    df[ACTION_COL] = df[ACTION_COL].astype(int)
    df = df[df[ACTION_COL].between(0, 3)].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid actions found (expected 0-3).")

    # ---------------- Overall action percentages ----------------
    counts = df[ACTION_COL].value_counts().reindex([0, 1, 2, 3], fill_value=0)
    total = int(counts.sum())
    percentages = (counts / total) * 100

    print("\nOverall action distribution:")
    for a in [0, 1, 2, 3]:
        print(f"  Action {a}: {counts[a]:>8} steps  ({percentages[a]:6.2f}%)")
    print(f"  Total:    {total:>8} steps\n")

    # ---------------- Scatter plot: action vs step ----------------
    plt.figure(figsize=(14, 4))
    for action in [0, 1, 2, 3]:
        mask = df[ACTION_COL] == action
        steps = df.index[mask]
        plt.scatter(
            steps,
            [action] * len(steps),
            s=8,
            color=ACTION_COLORS[action],
            label=f"Action {action}"
        )

    plt.yticks([0, 1, 2, 3])
    plt.ylim(-0.5, 3.5)
    plt.xlabel("Training step")
    plt.ylabel("Greedy Action")
    plt.title(f"{ACTION_COL} in {len(df)} training steps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------- Bar chart: action percentages ----------------
    plt.figure(figsize=(6, 4))
    bars = plt.bar([0, 1, 2, 3], percentages.values,
                   color=[ACTION_COLORS[a] for a in [0, 1, 2, 3]])

    for bar, pct in zip(bars, percentages.values):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{pct:.1f}%",
                 ha="center", va="bottom", fontsize=10)

    plt.xticks([0, 1, 2, 3], ["0", "1", "2", "3"])
    plt.xlabel("Action")
    plt.ylabel("Percent of steps (%)")
    plt.title(f"{ACTION_COL} percentage in {len(df)} steps")
    plt.tight_layout()
    plt.show()

    # ---------------- SoftmaxProbability curves + black block-mean trend ----------------
    AVG_WINDOW = 1000          # black trend line resolution (block size)
    LABEL_EVERY = 10           # label every 10th block mean (plus the first)

    # Ensure probability columns exist
    for col in PROB_COLS:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV.")

    steps = df.index.to_numpy()
    n = len(df)
    n_blocks = n // AVG_WINDOW  # full blocks only (ignore remainder for the trend)

    if n_blocks < 1:
        raise ValueError(f"Not enough rows ({n}) for AVG_WINDOW={AVG_WINDOW}.")

    plt.figure(figsize=(14, 5))

    for i, col in enumerate(PROB_COLS):
        y = df[col].astype(float).to_numpy()

        # --- raw probability curve (colored) ---
        plt.plot(steps, y, color=ACTION_COLORS[i], linewidth=0.8, alpha=0.35, label=f"P(action {i}) raw")

        # --- block means (black trend) ---
        y_trim = y[: n_blocks * AVG_WINDOW]
        y_blocks = y_trim.reshape(n_blocks, AVG_WINDOW).mean(axis=1)  # shape: (n_blocks,)

        # Make a per-step black trend line by repeating each block mean across the block
        y_trend = y_blocks.repeat(AVG_WINDOW)
        steps_trend = steps[: len(y_trend)]

        plt.plot(steps_trend, y_trend, color="black", linewidth=1.8, alpha=0.95, label=f"P(action {i}) trend (avg {AVG_WINDOW})" if i == 0 else None)

        # --- label markers: first block + every 10th block ---
        # block center x coordinate for placing marker/label
        block_centers = ( (pd.Series(range(n_blocks)) * AVG_WINDOW) + (AVG_WINDOW // 2) ).to_numpy()

        # indices of blocks to label: 0, 10, 20, ...
        label_blocks = set([0] + list(range(0, n_blocks, LABEL_EVERY)))
        for b in sorted(label_blocks):
            x = int(block_centers[b])
            yb = float(y_blocks[b])

            # highlight marker on black trend
            plt.scatter([x], [yb], s=55, color="black", edgecolors="white", linewidths=1.0, zorder=5)

            # text label slightly above the point
            plt.text(
                x, yb + 0.03,
                f"{yb:.2f}",
                ha="center", va="bottom",
                fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", alpha=0.75),
                zorder=6
            )

    plt.xlabel("Training step")
    plt.ylabel("Action probability")
    plt.title(f"Action probabilities over training (raw + block-mean trend, AVG_WINDOW={AVG_WINDOW})")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.25)

    # Legend: keep it readable (raw curves per action + one trend entry)
    handles, labels = plt.gca().get_legend_handles_labels()
    # Remove duplicate "trend" labels (we only set it for i==0)
    plt.legend(handles, labels, ncol=2, fontsize=9)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
