import torch
import csv
import os
import time
import numpy as np
from environment import SnakeEnv
from agent import SnakeAgent, device


LOG_FILE        = "training_log.csv"
CHECKPOINT_FILE = "snake_brain.pth"
BEST_MODEL_FILE = "snake_best.pth"
LOG_INTERVAL    = 50
SAVE_INTERVAL   = 500


def num_food_schedule(episode, total):
    """
    Taper 50 → 5 over 80% of training, then hold at 5 forever.
    Never drops to 1 — keeps enough food density that the agent
    always has a clear signal to chase.
    """
    taper_end = int(total * 0.8)
    if episode >= taper_end:
        return 5
    frac = episode / taper_end
    return max(5, round(50 - 45 * frac))  # 50 → 5


def train_engine(episodes=5000):
    env   = SnakeEnv(num_food=50)
    agent = SnakeAgent()

    best_score   = 0
    score_window = []
    start_time   = time.time()

    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "score", "avg_score", "epsilon", "num_food"])

    print(f"Device       : {device}")
    print(f"Episodes     : {episodes}")
    print(f"Batch size   : {agent.batch_size}")
    print(f"Grid         : {env.grid_h}x{env.grid_w}")
    print(f"Vision       : 7x7 local CNN")
    print(f"Food taper   : 50 → 5 over first 80% of episodes, then held at 5")
    print("-" * 65)

    for e in range(1, episodes + 1):
        nf           = num_food_schedule(e, episodes)
        env.num_food = nf
        state        = env.reset()

        while True:
            action                   = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            if done:
                break

        score = env.score
        score_window.append(score)
        if len(score_window) > LOG_INTERVAL:
            score_window.pop(0)

        if score > best_score:
            best_score = score
            torch.save({
                'episode':          e,
                'score':            score,
                'model_state_dict': agent.model.state_dict(),
            }, BEST_MODEL_FILE)

        if e % LOG_INTERVAL == 0:
            avg_score = np.mean(score_window)
            elapsed_h = (time.time() - start_time) / 3600
            eta_h     = elapsed_h / e * (episodes - e)
            print(
                f"Ep {e:>5}/{episodes} | "
                f"Score: {score:>3} | "
                f"Best: {best_score:>3} | "
                f"Avg: {avg_score:>5.2f} | "
                f"Eps: {agent.epsilon:.4f} | "
                f"Food: {nf:>2} | "
                f"Mem: {len(agent.memory):>6} | "
                f"Elapsed: {elapsed_h:.1f}h | "
                f"ETA: {eta_h:.1f}h"
            )
            with open(LOG_FILE, "a", newline="") as f:
                csv.writer(f).writerow(
                    [e, score, round(avg_score, 2), round(agent.epsilon, 4), nf]
                )

        if e % SAVE_INTERVAL == 0:
            torch.save({
                'episode':          e,
                'model_state_dict': agent.model.state_dict(),
                'epsilon':          agent.epsilon,
                'steps_done':       agent.steps_done,
            }, CHECKPOINT_FILE)

    total_h = (time.time() - start_time) / 3600
    print(f"\nTraining complete in {total_h:.2f} hours.")
    print(f"Best score    : {best_score}")
    print(f"Best model    : {BEST_MODEL_FILE}")
    print(f"Checkpoint    : {CHECKPOINT_FILE}")


if __name__ == "__main__":
    train_engine()
