import torch
import os
import time
from environment import SnakeEnv
from agent import SnakeAgent


def watch_ai(use_best=True, games=5):
    env   = SnakeEnv()
    agent = SnakeAgent()

    model_file = "snake_best.pth" if use_best and os.path.exists("snake_best.pth") \
                 else "snake_brain.pth"

    if not os.path.exists(model_file):
        print("No model found. Run train.py first.")
        return

    checkpoint = torch.load(model_file, weights_only=True)
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded: {model_file} | ep {checkpoint.get('episode','?')} | best {checkpoint.get('score','?')}\n")

    agent.epsilon = 0.0
    agent.model.eval()

    SYMBOLS    = {0: ".", 1: "O", 2: "@", 3: "*"}
    move_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    for g in range(1, games + 1):
        state = env.reset()
        done  = False
        moves = 0
        print(f"═══ Game {g} ═══")

        while not done:
            grid = env.get_grid()
            print(f"Score: {env.score}  Moves: {moves}")
            for row in grid:
                print(" ".join(SYMBOLS[v] for v in row))
            print()

            action = agent.act(state)
            print(f"→ {move_names[action]}\n")
            state, _, done = env.step(action)
            moves += 1
            time.sleep(0.05)

        print(f"Game {g} over. Score: {env.score}  Moves: {moves}\n")


if __name__ == "__main__":
    watch_ai()
