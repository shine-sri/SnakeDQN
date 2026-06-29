import numpy as np
import random
from collections import deque

# Grid size
GRID_W  = 20
GRID_H  = 20
VISION  = 7   # local vision window around head (must be odd)

# Actions
UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3

DIR_MAP  = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


class SnakeEnv:
    """
    Snake on a GRID_H x GRID_W board.

    State: tuple (grid, extras)
        grid   : float32 array (3, VISION, VISION)
                   ch0 = wall/body danger
                   ch1 = food (any food pellet)
                   ch2 = head marker (always centre)
        extras : float32 array (5,)
                   [direction one-hot x4, normalised length]

    num_food: how many food pellets are on the board at once.
              Use ~10 early in training so the agent discovers the +100
              reward quickly, then taper toward 1 as it learns.
    """

    def __init__(self, num_food=10):
        self.grid_h   = GRID_H
        self.grid_w   = GRID_W
        self.num_food = num_food
        self.reset()

    def reset(self):
        mid_r = self.grid_h // 2
        mid_c = self.grid_w // 2
        self.snake = deque([
            (mid_r, mid_c),
            (mid_r, mid_c - 1),
            (mid_r, mid_c - 2),
        ])
        self.direction        = RIGHT
        self.score            = 0
        self.steps_since_food = 0
        self.state_seen       = {}
        self.loop_counts      = {}
        self.total_steps      = 0

        # Set of food positions — replenished after each eat
        self.foods = set()
        self._fill_foods()
        return self._get_state()

    # ------------------------------------------------------------------
    # Food management
    # ------------------------------------------------------------------

    def _empty_cells(self):
        occupied = set(self.snake) | self.foods
        return [
            (r, c)
            for r in range(self.grid_h)
            for c in range(self.grid_w)
            if (r, c) not in occupied
        ]

    def _fill_foods(self):
        """Top up foods to self.num_food (or as many as fit)."""
        empty = self._empty_cells()
        random.shuffle(empty)
        while len(self.foods) < self.num_food and empty:
            self.foods.add(empty.pop())

    # ------------------------------------------------------------------
    # Collision / state
    # ------------------------------------------------------------------

    def _is_collision(self, r, c):
        return (
            r < 0 or r >= self.grid_h or
            c < 0 or c >= self.grid_w or
            (r, c) in set(self.snake)
        )

    def _nearest_food(self, r, c):
        """Manhattan distance to the closest food pellet."""
        if not self.foods:
            return 0
        return min(abs(r - fr) + abs(c - fc) for fr, fc in self.foods)

    def _get_state(self):
        head_r, head_c = self.snake[0]
        half   = VISION // 2
        body   = set(self.snake)

        grid = np.zeros((3, VISION, VISION), dtype=np.float32)

        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                r, c   = head_r + dr, head_c + dc
                vr, vc = dr + half,   dc + half

                # Channel 0: wall or body
                if r < 0 or r >= self.grid_h or c < 0 or c >= self.grid_w:
                    grid[0, vr, vc] = 1.0
                elif (r, c) in body:
                    grid[0, vr, vc] = 1.0

                # Channel 1: any food pellet
                if (r, c) in self.foods:
                    grid[1, vr, vc] = 1.0

        # Channel 2: head at centre
        grid[2, half, half] = 1.0

        extras = np.array([
            float(self.direction == UP),
            float(self.direction == DOWN),
            float(self.direction == LEFT),
            float(self.direction == RIGHT),
            len(self.snake) / (self.grid_h * self.grid_w),
        ], dtype=np.float32)

        return grid, extras

    # ------------------------------------------------------------------
    # Loop detection
    # ------------------------------------------------------------------

    def _check_loop(self, head_pos):
        fp = (head_pos, self.direction)
        if fp in self.state_seen:
            self.loop_counts[fp] = self.loop_counts.get(fp, 0) + 1
        else:
            self.state_seen[fp]  = self.total_steps
            self.loop_counts[fp] = 0
        return self.loop_counts[fp]

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        if action == OPPOSITE[self.direction]:
            action = self.direction
        self.direction = action

        dy, dx         = DIR_MAP[self.direction]
        head_r, head_c = self.snake[0]
        new_r, new_c   = head_r + dy, head_c + dx

        if self._is_collision(new_r, new_c):
            return self._get_state(), -10.0, True

        self.total_steps      += 1
        self.steps_since_food += 1

        # Distance shaping toward nearest food
        dist_before = self._nearest_food(head_r, head_c)
        dist_after  = self._nearest_food(new_r,  new_c)

        self.snake.appendleft((new_r, new_c))

        if (new_r, new_c) in self.foods:
            self.score            += 1
            self.steps_since_food  = 0
            self.state_seen.clear()
            self.loop_counts.clear()
            self.foods.discard((new_r, new_c))
            self._fill_foods()          # immediately replace eaten pellet
            reward = 100.0
            # Board full — snake wins
            if not self.foods and not self._empty_cells():
                return self._get_state(), 200.0, True
            return self._get_state(), reward, False

        self.snake.pop()

        # Directional shaping
        reward = 0.5 if dist_after < dist_before else -0.5

        # Near-but-didn't-eat penalty (nearest food)
        if dist_after == 1:
            reward -= 2.0
        elif dist_after == 2:
            reward -= 0.5

        # Loop detection
        loop_count = self._check_loop((new_r, new_c))
        if loop_count == 1:
            reward -= 5.0
        elif loop_count == 2:
            reward -= 10.0
        elif loop_count >= 3:
            return self._get_state(), -30.0, True

        # Progressive starvation
        snake_len      = len(self.snake)
        grid_area      = self.grid_h * self.grid_w
        starvation_cap = max(grid_area // 2, grid_area * 2 // snake_len)
        if self.steps_since_food > starvation_cap:
            return self._get_state(), -15.0, True

        return self._get_state(), reward, False

    # ------------------------------------------------------------------
    # Rendering helper
    # ------------------------------------------------------------------

    def get_grid(self):
        """Full 20x20 grid. 0=empty 1=body 2=head 3=food."""
        grid = np.zeros((self.grid_h, self.grid_w), dtype=int)
        for i, (r, c) in enumerate(self.snake):
            grid[r][c] = 2 if i == 0 else 1
        for fr, fc in self.foods:
            grid[fr][fc] = 3
        return grid

    # Legacy single-food property so watch.py / ui.py don't break
    @property
    def food(self):
        return next(iter(self.foods)) if self.foods else None
