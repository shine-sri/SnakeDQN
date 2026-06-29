import matplotlib.pyplot as plt
import csv
import os
import numpy as np


def plot_training_data():
    file_path = "training_log.csv"

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Train first.")
        return

    episodes, scores, avg_scores = [], [], []
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                episodes.append(int(row["episode"]))
                scores.append(int(row["score"]))
                avg_scores.append(float(row["avg_score"]))
            except (ValueError, KeyError):
                continue

    if not episodes:
        print("Log file is empty.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(episodes, scores,     alpha=0.3, color="#2ca02c", label="Score per episode")
    ax.plot(episodes, avg_scores, linewidth=2, color="#1f77b4", label=f"Avg score (last 50 ep)")

    ax.set_title("Snake DQN — Learning Progression", fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Episodes", fontsize=12)
    ax.set_ylabel("Score (food eaten)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    out = "learning_curve.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_training_data()
