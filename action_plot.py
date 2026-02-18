"""
action_plot.py

1) Scatter plot: action (0/1/2/3) vs step with different colors
2) Overall percentage of each action across the entire training run
"""

import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "record.csv"
ACTION_COL = "greedy_action"  # column in record.csv that contains the action taken (0,1,2,3)

ACTION_COLORS = {0: "red", 1: "blue", 2: "green", 3: "orange"}

def main():
    df = pd.read_csv(CSV_PATH)

    # Keep only valid actions
    df = df[df[ACTION_COL].notna()]
    df[ACTION_COL] = df[ACTION_COL].astype(int)
    df = df[df[ACTION_COL].between(0, 3)].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid actions found (expected actionId in {0,1,2,3}).")

    # ---------- Overall action percentages ----------
    counts = df[ACTION_COL].value_counts().reindex([0, 1, 2, 3], fill_value=0)
    total = int(counts.sum())
    percentages = (counts / total) * 100

    print("\nOverall action distribution:")
    for a in [0, 1, 2, 3]:
        print(f"  Action {a}: {counts[a]:>8} steps  ({percentages[a]:6.2f}%)")
    print(f"  Total:    {total:>8} steps\n")

    # ---------- Scatter plot: action vs step ----------
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
    plt.xlabel("Training step (row index in record.csv)")
    plt.ylabel("Action taken")
    plt.title(f"{ACTION_COL} in {len(df)} training steps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------- Optional: Bar chart of overall percentages ----------
    plt.figure(figsize=(6, 4))
    bars = plt.bar([0, 1, 2, 3], percentages.values, color=[ACTION_COLORS[a] for a in [0, 1, 2, 3]])
    #state each action's percentage on top of the bar
    for i, (bar, pct) in enumerate(zip(bars, percentages.values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{pct:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.xticks([0, 1, 2, 3], ["0", "1", "2", "3"])
    plt.xlabel("Action")
    plt.ylabel("Percent of steps (%)")
    plt.title(f"{ACTION_COL} percentage in {len(df)} Training steps")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
