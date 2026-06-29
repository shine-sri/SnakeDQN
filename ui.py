import pygame
import torch
import os
import time
import numpy as np
from environment import SnakeEnv, GRID_W, GRID_H
from agent import SnakeAgent

CELL  = 28
PANEL = 60

COLORS = {
    "bg":      (15,  15,  25),
    "grid":    (25,  25,  40),
    "snake_h": (0,  220,  80),
    "snake_b": (0,  160,  50),
    "food":    (255,  60,  60),
    "text":    (220, 220, 220),
}


def draw(screen, env, font, best_score):
    grid = env.get_grid()
    h, w = grid.shape
    screen.fill(COLORS["bg"])

    score_surf = font.render(
        f"Score: {env.score}   Best: {best_score}", True, COLORS["text"]
    )
    screen.blit(score_surf, (10, 15))

    for r in range(h + 1):
        pygame.draw.line(screen, COLORS["grid"],
                         (0, PANEL + r * CELL), (w * CELL, PANEL + r * CELL))
    for c in range(w + 1):
        pygame.draw.line(screen, COLORS["grid"],
                         (c * CELL, PANEL), (c * CELL, PANEL + h * CELL))

    for r in range(h):
        for c in range(w):
            val = grid[r][c]
            if val == 0:
                continue
            rect = pygame.Rect(c * CELL + 1, PANEL + r * CELL + 1, CELL - 2, CELL - 2)
            if val == 2:
                pygame.draw.rect(screen, COLORS["snake_h"], rect, border_radius=5)
            elif val == 1:
                pygame.draw.rect(screen, COLORS["snake_b"], rect, border_radius=3)
            elif val == 3:
                pygame.draw.ellipse(screen, COLORS["food"], rect)

    pygame.display.flip()


def play_gui(use_best=True, delay=0.08):
    pygame.init()
    screen = pygame.display.set_mode((GRID_W * CELL, GRID_H * CELL + PANEL))
    pygame.display.set_caption("Snake — DQN Agent (Vision CNN)")
    font = pygame.font.SysFont("arial", 22, bold=True)

    env   = SnakeEnv()
    agent = SnakeAgent()

    model_file = "snake_best.pth" if use_best and os.path.exists("snake_best.pth") \
                 else "snake_brain.pth"

    if not os.path.exists(model_file):
        print("No model found. Run train.py first.")
        pygame.quit()
        return

    checkpoint = torch.load(model_file, weights_only=True)
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded: {model_file} | ep {checkpoint.get('episode','?')} | best {checkpoint.get('score','?')}")

    agent.epsilon = 0.0
    agent.model.eval()

    best_score = 0
    game_count = 0

    while True:
        state     = env.reset()
        done      = False
        game_count += 1

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            draw(screen, env, font, best_score)
            time.sleep(delay)

            action = agent.act(state)
            state, _, done = env.step(action)

        if env.score > best_score:
            best_score = env.score
        print(f"Game {game_count} | Score: {env.score} | Best: {best_score}")


if __name__ == "__main__":
    play_gui()
