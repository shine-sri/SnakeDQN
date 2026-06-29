"""
metrics.py — Snake DQN Metrics Tracker
=======================================
Measures and logs all key RL metrics for the Snake DQN agent.

Usage:
    python metrics.py                    # run evaluation with best model
    python metrics.py --episodes 100     # evaluate over 100 games
    python metrics.py --model snake_brain.pth  # use a specific checkpoint
"""

import argparse
import os
import csv
import time
import numpy as np
import torch
from collections import deque

from environment import SnakeEnv, GRID_H, GRID_W
from agent import SnakeAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_agent(model_path: str) -> SnakeAgent:
    """Load a trained SnakeAgent from a checkpoint file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    agent = SnakeAgent()
    checkpoint = torch.load(model_path, weights_only=True)
    agent.model.load_state_dict(checkpoint["model_state_dict"])
    agent.epsilon = 0.0          # pure exploitation during evaluation
    agent.model.eval()
    print(f"Loaded : {model_path}")
    print(f"  Trained episode : {checkpoint.get('episode', '?')}")
    print(f"  Best score      : {checkpoint.get('score', '?')}")
    return agent


def run_episode(env: SnakeEnv, agent: SnakeAgent):
    """
    Play one full episode and collect per-step data.

    Returns a dict with raw episode data:
        score          : int   – food pellets eaten
        steps          : int   – total moves taken
        survival_ratio : float – steps / (GRID_H * GRID_W)
        rewards        : list  – reward at each step
        death_cause    : str   – 'wall', 'self', 'starvation', 'loop', 'win', 'unknown'
        efficiency     : float – score / steps (0 if steps == 0)
        final_length   : int   – snake length at end of episode
        board_coverage : float – final_length / grid_area
    """
    state = env.reset()
    done  = False
    steps = 0
    rewards = []
    death_cause = "unknown"

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        rewards.append(reward)
        steps += 1
        state  = next_state

        if done:
            # Infer death cause from last reward
            if reward >= 200:
                death_cause = "win"
            elif reward == -10:
                # Collision: check if wall or self
                head_r, head_c = env.snake[0]
                if (head_r < 0 or head_r >= GRID_H or
                        head_c < 0 or head_c >= GRID_W):
                    death_cause = "wall"
                else:
                    death_cause = "self"
            elif reward == -30:
                death_cause = "loop"
            elif reward == -15:
                death_cause = "starvation"

    grid_area      = GRID_H * GRID_W
    final_length   = len(env.snake)
    efficiency     = env.score / steps if steps > 0 else 0.0
    survival_ratio = steps / grid_area

    return {
        "score":          env.score,
        "steps":          steps,
        "survival_ratio": survival_ratio,
        "rewards":        rewards,
        "death_cause":    death_cause,
        "efficiency":     efficiency,
        "final_length":   final_length,
        "board_coverage": final_length / grid_area,
    }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(episodes_data: list) -> dict:
    """
    Aggregate per-episode data into final RL metrics.

    Metrics computed
    ────────────────
    Performance:
        avg_score             – mean food eaten per episode
        max_score             – best single-episode score
        min_score             – worst single-episode score
        score_std             – standard deviation of scores
        score_p25/p50/p75     – score percentiles

    Survival:
        avg_steps             – mean steps per episode
        avg_survival_ratio    – avg steps / grid area
        max_steps             – longest episode
        min_steps             – shortest episode

    Reward:
        avg_total_reward      – mean cumulative reward per episode
        avg_step_reward       – mean per-step reward across all episodes
        reward_std            – std of cumulative rewards

    Efficiency:
        avg_efficiency        – mean (score / steps)
        best_efficiency       – max (score / steps) across episodes

    Board Usage:
        avg_board_coverage    – mean fraction of board filled at end
        max_board_coverage    – best board coverage achieved

    Death Causes:
        death_wall_pct        – % episodes ended by wall collision
        death_self_pct        – % episodes ended by self collision
        death_starvation_pct  – % episodes ended by starvation
        death_loop_pct        – % episodes ended by loop penalty
        win_pct               – % episodes where snake won (board full)

    Stability:
        score_iqr             – interquartile range of scores (robustness)
        score_cv              – coefficient of variation (std / mean)
        episodes_above_avg    – % episodes scoring above their mean

    Food Eating Rate:
        food_per_100_steps    – normalised eating rate
    """
    scores          = np.array([e["score"]          for e in episodes_data])
    steps_arr       = np.array([e["steps"]          for e in episodes_data])
    surv_ratios     = np.array([e["survival_ratio"] for e in episodes_data])
    efficiencies    = np.array([e["efficiency"]     for e in episodes_data])
    coverages       = np.array([e["board_coverage"] for e in episodes_data])
    total_rewards   = np.array([sum(e["rewards"])   for e in episodes_data])
    all_step_rew    = np.concatenate([e["rewards"]  for e in episodes_data])
    death_causes    = [e["death_cause"]             for e in episodes_data]
    n               = len(episodes_data)

    def pct(cause): return 100.0 * death_causes.count(cause) / n

    avg_score = float(np.mean(scores))

    metrics = {
        # --- Performance ---
        "avg_score":            avg_score,
        "max_score":            int(np.max(scores)),
        "min_score":            int(np.min(scores)),
        "score_std":            float(np.std(scores)),
        "score_p25":            float(np.percentile(scores, 25)),
        "score_p50":            float(np.percentile(scores, 50)),
        "score_p75":            float(np.percentile(scores, 75)),

        # --- Survival ---
        "avg_steps":            float(np.mean(steps_arr)),
        "max_steps":            int(np.max(steps_arr)),
        "min_steps":            int(np.min(steps_arr)),
        "avg_survival_ratio":   float(np.mean(surv_ratios)),

        # --- Reward ---
        "avg_total_reward":     float(np.mean(total_rewards)),
        "avg_step_reward":      float(np.mean(all_step_rew)),
        "reward_std":           float(np.std(total_rewards)),

        # --- Efficiency ---
        "avg_efficiency":       float(np.mean(efficiencies)),
        "best_efficiency":      float(np.max(efficiencies)),

        # --- Board Usage ---
        "avg_board_coverage":   float(np.mean(coverages)),
        "max_board_coverage":   float(np.max(coverages)),

        # --- Death Causes ---
        "death_wall_pct":       pct("wall"),
        "death_self_pct":       pct("self"),
        "death_starvation_pct": pct("starvation"),
        "death_loop_pct":       pct("loop"),
        "win_pct":              pct("win"),

        # --- Stability ---
        "score_iqr":            float(np.percentile(scores, 75) - np.percentile(scores, 25)),
        "score_cv":             float(np.std(scores) / avg_score) if avg_score > 0 else 0.0,
        "episodes_above_avg":   float(100.0 * np.sum(scores > avg_score) / n),

        # --- Food Rate ---
        "food_per_100_steps":   float(
            np.sum(scores) / np.sum(steps_arr) * 100
        ) if np.sum(steps_arr) > 0 else 0.0,
    }
    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

SECTION_WIDTH = 54

def print_metrics(metrics: dict, n_episodes: int, elapsed: float):
    div  = "─" * SECTION_WIDTH
    head = "═" * SECTION_WIDTH

    def row(label, value, unit=""):
        print(f"  {label:<32} {value:>10}{unit}")

    print(f"\n{'═' * SECTION_WIDTH}")
    print(f"  SNAKE DQN — EVALUATION METRICS  ({n_episodes} episodes)")
    print(head)

    print(f"\n  {'PERFORMANCE':}")
    print(div)
    row("Avg Score (food eaten)",   f"{metrics['avg_score']:.2f}")
    row("Max Score",                metrics['max_score'])
    row("Min Score",                metrics['min_score'])
    row("Score Std Dev",            f"{metrics['score_std']:.2f}")
    row("Score P25 / P50 / P75",
        f"{metrics['score_p25']:.0f} / {metrics['score_p50']:.0f} / {metrics['score_p75']:.0f}")

    print(f"\n  {'SURVIVAL':}")
    print(div)
    row("Avg Steps per Episode",    f"{metrics['avg_steps']:.1f}")
    row("Max Steps",                metrics['max_steps'])
    row("Min Steps",                metrics['min_steps'])
    row("Avg Survival Ratio",       f"{metrics['avg_survival_ratio']:.4f}")

    print(f"\n  {'REWARD':}")
    print(div)
    row("Avg Total Reward",         f"{metrics['avg_total_reward']:.2f}")
    row("Avg Per-Step Reward",      f"{metrics['avg_step_reward']:.4f}")
    row("Reward Std Dev",           f"{metrics['reward_std']:.2f}")

    print(f"\n  {'EFFICIENCY':}")
    print(div)
    row("Avg Efficiency (score/steps)", f"{metrics['avg_efficiency']:.5f}")
    row("Best Efficiency",          f"{metrics['best_efficiency']:.5f}")
    row("Food per 100 Steps",       f"{metrics['food_per_100_steps']:.3f}")

    print(f"\n  {'BOARD USAGE':}")
    print(div)
    row("Avg Board Coverage",       f"{metrics['avg_board_coverage']:.2%}")
    row("Max Board Coverage",       f"{metrics['max_board_coverage']:.2%}")

    print(f"\n  {'DEATH CAUSES':}")
    print(div)
    row("Wall Collision",           f"{metrics['death_wall_pct']:.1f}", "%")
    row("Self Collision",           f"{metrics['death_self_pct']:.1f}", "%")
    row("Starvation",               f"{metrics['death_starvation_pct']:.1f}", "%")
    row("Loop Penalty",             f"{metrics['death_loop_pct']:.1f}", "%")
    row("Win (board full)",         f"{metrics['win_pct']:.1f}", "%")

    print(f"\n  {'STABILITY':}")
    print(div)
    row("Score IQR",                f"{metrics['score_iqr']:.2f}")
    row("Coeff of Variation",       f"{metrics['score_cv']:.4f}")
    row("Episodes Above Average",   f"{metrics['episodes_above_avg']:.1f}", "%")

    print(f"\n  Evaluation time: {elapsed:.1f}s")
    print(head)


def save_metrics_csv(metrics: dict, n_episodes: int, out_path: str = "eval_metrics.csv"):
    """Save the flat metrics dict to a one-row CSV."""
    metrics_with_meta = {"n_episodes": n_episodes, **metrics}
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_with_meta.keys()))
        writer.writeheader()
        writer.writerow(metrics_with_meta)
    print(f"\nMetrics saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(model_path: str = None, n_episodes: int = 50):
    # Pick model file
    if model_path is None:
        if os.path.exists("snake_best.pth"):
            model_path = "snake_best.pth"
        elif os.path.exists("snake_brain.pth"):
            model_path = "snake_brain.pth"
        else:
            raise FileNotFoundError(
                "No model found. Train first with train.py, "
                "or pass --model <path>."
            )

    agent = load_agent(model_path)
    env   = SnakeEnv(num_food=1)   # single food for fair evaluation

    print(f"\nEvaluating over {n_episodes} episodes …")
    t0 = time.time()

    episodes_data = []
    for ep in range(1, n_episodes + 1):
        data = run_episode(env, agent)
        episodes_data.append(data)
        if ep % max(1, n_episodes // 10) == 0:
            done_pct = 100.0 * ep / n_episodes
            print(f"  [{done_pct:5.1f}%] ep {ep:>4}/{n_episodes}  "
                  f"score={data['score']:>3}  steps={data['steps']:>5}  "
                  f"death={data['death_cause']}")

    elapsed = time.time() - t0
    metrics = compute_metrics(episodes_data)
    print_metrics(metrics, n_episodes, elapsed)
    save_metrics_csv(metrics, n_episodes)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Snake DQN agent.")
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of evaluation episodes (default: 50)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to model checkpoint (default: snake_best.pth or snake_brain.pth)"
    )
    args = parser.parse_args()
    evaluate(model_path=args.model, n_episodes=args.episodes)
